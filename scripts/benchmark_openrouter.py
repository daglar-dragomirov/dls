from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import RelevanceAgent
from nlp_llm_agents.data import generate_dataset
from nlp_llm_agents.env_utils import load_env_file
from nlp_llm_agents.metrics import binary_metrics
from nlp_llm_agents.models import BaselineRelevanceModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    os.environ["USE_INSTRUCTOR"] = "1"

    train = generate_dataset(20_000, seed=2026)
    eval_frame = generate_dataset(args.size, seed=2027)
    baseline = BaselineRelevanceModel.fit(train)
    baseline_scores = baseline.predict_proba(eval_frame)
    agent = RelevanceAgent(baseline)

    def score(item):
        index, row = item
        result = agent.score_one(row, float(baseline_scores[index]))
        return {"id": int(index), "label": int(row["label"]), "query": row["query"], **result}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(score, eval_frame.iterrows()))
    output = pd.DataFrame(results)
    metrics = {
        "n": len(output),
        "structured_llm_review_rate": float(output["llm_review"].notna().mean()),
        "backend_counts": output["backend"].value_counts().to_dict(),
        "metrics": binary_metrics(output["label"], output["score"]),
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    output.to_json(artifacts / "nlp_llm_benchmark_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    (artifacts / "nlp_llm_benchmark_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
