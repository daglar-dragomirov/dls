from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from collections import deque
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import RelevanceAgent
from nlp_llm_agents.data_io import load_relevance_data
from nlp_llm_agents.instructor_backend import instructor_available, LLMUnavailable
from nlp_llm_agents.models import BaselineRelevanceModel

ARTIFACTS = ROOT / "artifacts"
BASELINE_PATH = ARTIFACTS / "baseline.joblib"
REFERENCE_PATH = ARTIFACTS / "retrieval_reference.jsonl"
METRICS_PATH = ARTIFACTS / "official_metrics.json"
LLM_METRICS_PATH = ARTIFACTS / "official_llm_metrics.json"
VERSION = "3.0.1"
LLM_SLOTS = threading.BoundedSemaphore(2)
LLM_REQUESTS: deque[float] = deque()
LLM_RATE_LOCK = threading.Lock()


class OrganizationCard(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    organization_name: str = Field(min_length=1, max_length=300)
    category: str = Field(default="", max_length=500)
    address: str = Field(default="", max_length=1000)
    prices_summarized: str = Field(default="", max_length=4000)
    review_snippets: str = Field(default="", max_length=6000)
    permalink: str = Field(default="", max_length=2000)
    use_llm: bool = True
    search_mode: Literal["auto", "always", "off"] = "auto"
    # Совместимость со старым клиентом, который передавал checkbox.
    use_web_search: bool | None = None


def _load_agent() -> RelevanceAgent:
    baseline = BaselineRelevanceModel.load(str(BASELINE_PATH))
    reference = load_relevance_data(REFERENCE_PATH) if REFERENCE_PATH.exists() else None
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    weight = float(metrics.get("validation", {}).get("selected_retrieval_weight", 0.25))
    return RelevanceAgent(baseline, reference, retrieval_weight=weight)


agent = _load_agent()
app = FastAPI(title="DLS NLP: агент релевантности организаций", version=VERSION)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": "nlp_llm_agents", "version": VERSION}


@app.get("/capabilities")
def capabilities() -> dict:
    return {"llm_available": instructor_available(), "model": os.getenv("OPENROUTER_MODEL", "tencent/hy3"),
            "search_modes": ["auto", "always", "off"]}


@app.get("/metrics")
def metrics() -> dict:
    result = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    if LLM_METRICS_PATH.exists():
        result["official_llm"] = json.loads(LLM_METRICS_PATH.read_text(encoding="utf-8"))
    result["scope"] = "Archived July experiment, not the current interactive agent"
    return result


def _as_series(card: OrganizationCard) -> pd.Series:
    data = card.model_dump(exclude={"use_llm", "use_web_search", "search_mode"})
    return pd.Series({**data, "label": 0.0, "query_id": 0})


@app.post("/score")
def score(card: OrganizationCard) -> dict:
    if card.use_llm and not instructor_available():
        raise HTTPException(503, "LLM не настроена. Проверьте OPENROUTER_API_KEY и USE_INSTRUCTOR на сервере.")
    mode = card.search_mode
    if card.use_web_search is not None:
        mode = "always" if card.use_web_search else "off"
    # Baseline не использует веб-доказательства: не имитируем эту возможность.
    if not card.use_llm:
        mode = "off"
    acquired = False
    if card.use_llm:
        acquired = LLM_SLOTS.acquire(blocking=False)
        if not acquired:
            raise HTTPException(429, "Два запроса уже выполняются. Попробуйте позже.")
        with LLM_RATE_LOCK:
            now = time.monotonic()
            while LLM_REQUESTS and LLM_REQUESTS[0] < now - 3600:
                LLM_REQUESTS.popleft()
            if len(LLM_REQUESTS) >= int(os.getenv("LLM_REQUESTS_PER_HOUR", "60")):
                LLM_SLOTS.release()
                raise HTTPException(429, "Часовой лимит LLM-запросов исчерпан. Доступен режим Baseline.")
            LLM_REQUESTS.append(now)
    try:
        return agent.score_one(_as_series(card), use_llm=card.use_llm, search_mode=mode)
    except LLMUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    finally:
        if acquired:
            LLM_SLOTS.release()


@app.post("/batch_score")
def batch_score(cards: Annotated[list[OrganizationCard], Field(max_length=10)]) -> dict:
    if any(card.use_llm for card in cards) and not instructor_available():
        raise HTTPException(503, "LLM не настроена на сервере.")
    return {"items": [score(card) for card in cards]}
