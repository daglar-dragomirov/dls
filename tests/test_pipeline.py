from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import RelevanceAgent
from nlp_llm_agents.data_io import LABEL_VALUES, load_relevance_data
from nlp_llm_agents.metrics import multiclass_metrics
from nlp_llm_agents.models import BaselineRelevanceModel


def small_frame(repeats: int = 16) -> pd.DataFrame:
    rows = []
    templates = [
        ("купить сигары", "Табачный дом", "Магазин табака", "Большой выбор сигар", 1.0),
        ("кальянная для мероприятий", "Пицца бар", "Кафе", "Пицца и завтраки, кальянов нет", 0.0),
        ("ресторан с верандой", "Летний сад", "Ресторан", "Есть небольшая сезонная веранда", 0.1),
    ]
    for index in range(repeats):
        for query, name, rubric, review, label in templates:
            rows.append(
                {
                    "query": f"{query} {index % 3}",
                    "organization_name": name,
                    "category": rubric,
                    "address": "Москва",
                    "prices_summarized": "",
                    "review_snippets": review,
                    "permalink": str(index),
                    "query_id": index,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_three_class_baseline_and_agent_outputs():
    train = small_frame()
    baseline = BaselineRelevanceModel.fit(train)
    agent = RelevanceAgent(baseline, train, retrieval_weight=0.3)
    scored = agent.score_frame(train.head(9))
    assert set(scored["predicted_relevance"]).issubset(set(LABEL_VALUES))
    assert scored["confidence"].between(0, 1).all()
    assert scored["used_search"].all()
    assert scored["tool_calls"].map(len).eq(2).all()


def test_agent_exposes_real_web_search_trace(monkeypatch):
    train = small_frame()
    agent = RelevanceAgent(BaselineRelevanceModel.fit(train), train, retrieval_weight=0.3)
    monkeypatch.setattr(
        agent.web_search_tool,
        "search",
        lambda query: {
            "status": "ok",
            "query": query,
            "search_url": "https://example.test/search",
            "results": [{"title": "Летняя веранда", "url": "https://example.test/place"}],
        },
    )
    result = agent.score_one(train.iloc[0], use_web_search=True)
    assert result["used_web_search"] is True
    assert result["tool_calls"][-1]["tool"] == "public_web_search"
    assert result["tool_calls"][-1]["result"]["status"] == "ok"


def test_metrics_include_all_three_classes():
    train = small_frame()
    baseline = BaselineRelevanceModel.fit(train)
    metrics = multiclass_metrics(train["label"], baseline.predict_proba(train))
    assert metrics["accuracy"] > 0.9
    assert set(metrics["per_class"]) == {"0.0", "0.1", "1.0"}


def test_official_column_adapter_preserves_partial_class(tmp_path):
    path = tmp_path / "official.jsonl"
    pd.DataFrame(
        [
            {
                "Text": "ресторан с верандой",
                "name": "Тест",
                "normalized_main_rubric_name_ru": "Ресторан",
                "reviews_summarized": "Есть веранда",
                "relevance": 0.1,
            }
        ]
    ).to_json(path, orient="records", lines=True, force_ascii=False)
    frame = load_relevance_data(path, target_column="relevance")
    assert frame.loc[0, "query"] == "ресторан с верандой"
    assert frame.loc[0, "label"] == 0.1
    assert frame.loc[0, "label_id"] == 1
