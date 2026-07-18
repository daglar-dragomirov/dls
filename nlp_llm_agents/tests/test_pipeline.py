from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import RelevanceAgent
from nlp_llm_agents.data import generate_dataset
from nlp_llm_agents.metrics import binary_metrics
from nlp_llm_agents.models import BaselineRelevanceModel
from nlp_llm_agents.data_io import load_relevance_data


def test_agent_outputs_parseable_scores():
    train = generate_dataset(400, seed=1)
    eval_frame = generate_dataset(80, seed=2)
    baseline = BaselineRelevanceModel.fit(train)
    agent = RelevanceAgent(baseline)
    scored = agent.score_frame(eval_frame)
    assert scored["agent_score"].between(0, 1).all()
    assert set(scored["agent_decision"].unique()).issubset({0, 1})
    assert scored["used_search"].mean() > 0.2


def test_agent_not_worse_than_baseline_on_controlled_split():
    train = generate_dataset(1000, seed=10)
    eval_frame = generate_dataset(200, seed=11)
    baseline = BaselineRelevanceModel.fit(train)
    eval_frame["baseline_score"] = baseline.predict_proba(eval_frame)
    scored = RelevanceAgent(baseline).score_frame(eval_frame)
    base_f1 = binary_metrics(scored["label"], scored["baseline_score"])["f1"]
    agent_f1 = binary_metrics(scored["label"], scored["agent_score"])["f1"]
    assert agent_f1 >= base_f1


def test_agent_handles_manual_russian_and_english_queries():
    train = generate_dataset(700, seed=110)
    baseline = BaselineRelevanceModel.fit(train)
    agent = RelevanceAgent(baseline)
    rows = pd.DataFrame(
        [
            {
                "query": "restaurant with a terrace",
                "organization_name": "Atlas Garden",
                "category": "restaurant",
                "public_description": "restaurant good food summer terrace reservation",
                "public_tags": "restaurant terrace good_food reservation",
                "review_snippets": "Reviews mention a terrace, dinner and good service.",
                "hidden_tags": "restaurant terrace good_food",
            },
            {
                "query": "кафе куда можно с собакой",
                "organization_name": "Север Daily",
                "category": "cafe",
                "public_description": "кофе десерты но с животными нельзя",
                "public_tags": "cafe no_pets desserts",
                "review_snippets": "В отзывах пишут: собак не пускают, питомцам нельзя.",
                "hidden_tags": "no_pets cafe",
            },
        ]
    )
    scored = agent.score_frame(rows)
    assert scored.loc[0, "agent_decision"] == 1
    assert scored.loc[1, "agent_decision"] == 0


def test_real_data_adapter_accepts_common_mentor_columns(tmp_path):
    path = tmp_path / "mentor.csv"
    pd.DataFrame([{"request": "cafe", "name": "A", "description": "quiet", "target": 1}]).to_csv(path, index=False)
    frame = load_relevance_data(path)
    assert frame.loc[0, "query"] == "cafe"
    assert frame.loc[0, "organization_name"] == "A"
    assert frame.loc[0, "label"] == 1
