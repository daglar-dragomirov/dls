from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import RelevanceAgent
from nlp_llm_agents.data import generate_dataset
from nlp_llm_agents.env_utils import first_openrouter_key
from nlp_llm_agents.models import BaselineRelevanceModel


def main() -> None:
    os.environ["USE_INSTRUCTOR"] = "1"
    if not first_openrouter_key():
        raise RuntimeError("No OpenRouter key found")

    train = generate_dataset(1200, seed=55)
    sample = pd.DataFrame(
        [
            {
                "query": "restaurant with a terrace",
                "organization_name": "Atlas Garden",
                "category": "restaurant",
                "public_description": "restaurant good food summer terrace reservation",
                "public_tags": "restaurant terrace good_food reservation",
                "review_snippets": "Reviews mention a terrace, dinner and good service.",
                "hidden_tags": "restaurant terrace good_food",
                "label": 1,
            },
            {
                "query": "кафе куда можно с собакой",
                "organization_name": "Север Daily",
                "category": "cafe",
                "public_description": "кофе десерты но с животными нельзя",
                "public_tags": "cafe no_pets desserts",
                "review_snippets": "В отзывах пишут: собак не пускают, питомцам нельзя.",
                "hidden_tags": "no_pets cafe",
                "label": 0,
            },
            {
                "query": "тихое место поработать с ноутбуком",
                "organization_name": "Мята Парк",
                "category": "cafe",
                "public_description": "кофейня интернет розетки тихо можно сидеть долго",
                "public_tags": "cafe wifi quiet power_outlets long_stay",
                "review_snippets": "В отзывах часто пишут: быстрый wifi, много розеток, спокойно.",
                "hidden_tags": "wifi quiet power_outlets cafe long_stay",
                "label": 1,
            },
        ]
    )
    baseline = BaselineRelevanceModel.fit(train)
    agent = RelevanceAgent(baseline)
    scored = agent.score_frame(sample)
    out = scored[
        [
            "query",
            "organization_name",
            "label",
            "agent_score",
            "agent_decision",
            "backend",
            "agent_rationale",
            "llm_review",
        ]
    ]
    Path("artifacts").mkdir(exist_ok=True)
    out.to_json("artifacts/nlp_openrouter_smoke.json", orient="records", force_ascii=False, indent=2)
    print(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2))
    assert out["llm_review"].notna().any(), "OpenRouter backend did not return any structured reviews"
    assert (out["agent_decision"] == out["label"]).all(), "Guarded NLP smoke examples should keep expected decisions"


if __name__ == "__main__":
    main()
