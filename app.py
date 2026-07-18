from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import RelevanceAgent
from nlp_llm_agents.data_io import load_relevance_data
from nlp_llm_agents.models import BaselineRelevanceModel


ARTIFACTS = ROOT / "artifacts"
BASELINE_PATH = ARTIFACTS / "baseline.joblib"
REFERENCE_PATH = ARTIFACTS / "retrieval_reference.jsonl"
METRICS_PATH = ARTIFACTS / "official_metrics.json"
LLM_METRICS_PATH = ARTIFACTS / "official_llm_metrics.json"


class OrganizationCard(BaseModel):
    query: str = Field(..., examples=["ресторан с верандой"])
    organization_name: str = Field(..., examples=["Северный сад"])
    category: str = Field("", examples=["Ресторан"])
    address: str = ""
    prices_summarized: str = ""
    review_snippets: str = ""
    permalink: str = ""
    use_llm: bool = False
    use_web_search: bool = False


def _load_agent() -> RelevanceAgent:
    if not BASELINE_PATH.exists():
        raise RuntimeError("The trained baseline artifact is missing")
    baseline = BaselineRelevanceModel.load(str(BASELINE_PATH))
    reference = load_relevance_data(REFERENCE_PATH) if REFERENCE_PATH.exists() else None
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    weight = float(metrics.get("validation", {}).get("selected_retrieval_weight", 0.25))
    return RelevanceAgent(baseline, reference, retrieval_weight=weight)


agent = _load_agent()
app = FastAPI(title="DLS NLP: агент релевантности организаций", version="2.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return r"""
<!doctype html><html lang="ru"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Агент релевантности организаций</title><style>
:root{--ink:#17212b;--muted:#66717e;--line:#d8e0e7;--accent:#087f5b;--accent2:#e76f51;--paper:#fff;--bg:#f4f7f8}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,sans-serif;color:var(--ink);background:var(--bg);letter-spacing:0}
header{background:#142b32;color:#fff;padding:24px max(20px,calc((100% - 1120px)/2));border-bottom:4px solid var(--accent2)}
h1{font-size:27px;margin:0 0 6px}header p{margin:0;color:#c9d6d9}.layout{max-width:1120px;margin:22px auto;padding:0 18px 36px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}
.band{grid-column:1/-1;background:var(--paper);border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:16px 18px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:18px}
h2{font-size:18px;margin:0 0 14px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:12px}.metric{border-left:4px solid var(--accent);padding:7px 10px;background:#f1faf7}.metric b{display:block;font-size:20px}.metric span{font-size:12px;color:var(--muted)}
label{display:block;font-weight:650;font-size:13px;margin:11px 0 5px}input,textarea{width:100%;border:1px solid #b9c5cf;border-radius:5px;padding:9px 10px;font:inherit;background:#fff}textarea{min-height:78px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{margin-top:13px;border:0;border-radius:5px;padding:10px 14px;font-weight:700;cursor:pointer;background:var(--accent);color:#fff}button:disabled{cursor:wait;opacity:.65}button.secondary{background:#e7edf0;color:#20323a;margin-left:6px}pre{margin:0;min-height:460px;max-height:650px;overflow:auto;background:#111b20;color:#d9f2e8;border-radius:6px;padding:14px;white-space:pre-wrap;font-size:13px}
.status{font-size:13px;color:var(--muted);margin-top:8px}@media(max-width:800px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.row{grid-template-columns:1fr}}
</style></head><body><header><h1>Агент релевантности организаций</h1><p>Baseline, поиск доказательств и похожие размеченные примеры</p></header>
<main class="layout"><section class="band"><div class="metrics" id="metrics"></div></section>
<section class="panel"><h2>Карточка организации</h2><label>Поисковый запрос</label><input id="query" value="ресторан с верандой"/>
<label>Название</label><input id="organization_name" value="Северный сад"/><div class="row"><div><label>Рубрика</label><input id="category" value="Ресторан"/></div><div><label>Адрес</label><input id="address" value="Москва"/></div></div>
<label>Цены и услуги</label><textarea id="prices_summarized">Летнее меню, ужины, бронирование столиков</textarea><label>Сводка отзывов</label><textarea id="review_snippets">Посетители хвалят просторную летнюю веранду, спокойную атмосферу и обслуживание.</textarea>
<label><input id="use_web_search" type="checkbox" style="width:auto;margin-right:7px"/>Выполнить реальный поиск в интернете</label>
<button id="evaluate" onclick="runAgent()">Оценить</button><button class="secondary" onclick="loadNegative()">Сложный пример</button><div class="status" id="status">Заполните карточку и нажмите «Оценить».</div></section>
<section class="panel" id="result-panel"><h2>Решение и трасса инструментов</h2><pre id="output">Здесь появятся прогноз, вероятности и вызовы инструментов.</pre></section></main>
<script>
const ids=['query','organization_name','category','address','prices_summarized','review_snippets'];
async function runAgent(){const status=document.getElementById('status'),button=document.getElementById('evaluate'),output=document.getElementById('output');button.disabled=true;button.textContent='Оцениваю…';status.textContent='Агент анализирует карточку…';const body={};ids.forEach(id=>body[id]=document.getElementById(id).value);body.use_web_search=document.getElementById('use_web_search').checked;try{const response=await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();output.textContent=JSON.stringify(data,null,2);if(!response.ok)throw new Error(data.detail||'Ошибка запроса');const confidence=Number(data.confidence||0).toFixed(3);status.textContent=`Готово: класс ${data.predicted_relevance}, уверенность ${confidence}`;if(window.innerWidth<=800)document.getElementById('result-panel').scrollIntoView({behavior:'smooth',block:'start'});}catch(error){status.textContent=`Ошибка: ${error.message}`;output.textContent=status.textContent;}finally{button.disabled=false;button.textContent='Оценить';}}
function loadNegative(){document.getElementById('query').value='кальянная для мероприятий';document.getElementById('organization_name').value='PioNero';document.getElementById('category').value='Кафе';document.getElementById('address').value='Санкт-Петербург';document.getElementById('prices_summarized').value='Пицца, паста, завтраки и десерты';document.getElementById('review_snippets').value='Отзывы хвалят итальянскую кухню и уют, но кальяны не упоминаются.';runAgent();}
fetch('/metrics').then(r=>r.json()).then(m=>{const e=m.official_eval||{};const l=m.official_llm?.metrics||{};const values=[['Baseline accuracy',e.baseline?.accuracy],['LLM-agent accuracy',l.accuracy],['LLM-agent macro F1',l.macro_f1],['Structured output',m.official_llm?.structured_output_rate]];document.getElementById('metrics').innerHTML=values.map(([k,v])=>`<div class="metric"><b>${Number(v||0).toFixed(3)}</b><span>${k}</span></div>`).join('')});
</script></body></html>"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": "nlp_llm_agents", "version": "2.0.0"}


@app.get("/metrics")
def metrics() -> dict:
    result = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    if LLM_METRICS_PATH.exists():
        result["official_llm"] = json.loads(LLM_METRICS_PATH.read_text(encoding="utf-8"))
    return result


def _as_series(card: OrganizationCard) -> pd.Series:
    data = card.model_dump(exclude={"use_llm", "use_web_search"})
    data["label"] = 0.0
    data["query_id"] = 0
    return pd.Series(data)


@app.post("/score")
def score(card: OrganizationCard) -> dict:
    if card.use_llm and not os.getenv("USE_INSTRUCTOR"):
        raise HTTPException(status_code=400, detail="LLM-review is disabled on this deployment")
    return agent.score_one(
        _as_series(card),
        use_llm=card.use_llm,
        use_web_search=card.use_web_search,
    )


@app.post("/batch_score")
def batch_score(cards: list[OrganizationCard]) -> dict:
    if not cards:
        return {"items": []}
    frame = pd.DataFrame([_as_series(card) for card in cards])
    scored = agent.score_frame(frame, use_llm=False)
    columns = ["predicted_relevance", "confidence", "used_search", "tool_calls", "rationale"]
    return {"items": scored[columns].to_dict(orient="records")}
