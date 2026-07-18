from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import CardEvidenceSearchTool, SimilarTrainExamplesTool
from nlp_llm_agents.data_io import LABEL_VALUES, load_official_train_eval
from nlp_llm_agents.env_utils import load_env_file
from nlp_llm_agents.instructor_backend import try_instructor_decision
from nlp_llm_agents.metrics import multiclass_metrics
from nlp_llm_agents.models import BaselineRelevanceModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the structured LLM agent on official eval")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    os.environ["USE_INSTRUCTOR"] = "1"

    train, evaluation = load_official_train_eval(args.train, args.eval)
    baseline = BaselineRelevanceModel.load(str(ROOT / "artifacts" / "baseline.joblib"))
    baseline_probabilities = baseline.predict_proba(evaluation)
    retriever = SimilarTrainExamplesTool(train, k=7)
    _, retrieved = retriever.search_many(evaluation)
    evidence_tool = CardEvidenceSearchTool()

    def score(index: int) -> dict:
        row = evaluation.iloc[index]
        evidence = evidence_tool.search(row)
        decision = try_instructor_decision(
            {
                "query": row["query"],
                "organization_card": {
                    "name": row["organization_name"],
                    "rubric": row["category"],
                    "address": row["address"],
                    "prices": row["prices_summarized"][:1_500],
                    "reviews": row["review_snippets"][:2_200],
                },
                "search_tool_result": evidence,
                "similar_labeled_train_examples": retrieved[index][:5],
                "baseline_probabilities": dict(zip(map(str, LABEL_VALUES), baseline_probabilities[index].round(4).tolist())),
            }
        )
        if decision is None:
            predicted = float(LABEL_VALUES[int(baseline_probabilities[index].argmax())])
            confidence = float(baseline_probabilities[index].max())
            backend = "baseline_fallback"
            rationale = "Structured LLM request failed; used the frozen baseline."
        else:
            predicted = float(decision.relevance)
            confidence = float(decision.confidence)
            backend = "instructor_openrouter"
            rationale = decision.rationale
        return {
            "index": index,
            "query": row["query"],
            "organization_name": row["organization_name"],
            "label": float(row["label"]),
            "predicted_relevance": predicted,
            "confidence": confidence,
            "backend": backend,
            "rationale": rationale,
            "search_evidence": evidence,
            "similar_train_examples": retrieved[index][:5],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(tqdm(pool.map(score, range(len(evaluation))), total=len(evaluation), desc="Official LLM agent"))
    predictions = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    probabilities = np.full((len(predictions), len(LABEL_VALUES)), 1e-6, dtype=float)
    for index, row in predictions.iterrows():
        class_index = LABEL_VALUES.index(float(row["predicted_relevance"]))
        confidence = float(np.clip(row["confidence"], 1 / 3, 0.999998))
        probabilities[index] = (1.0 - confidence) / 2
        probabilities[index, class_index] = confidence
    metrics = {
        "n": len(predictions),
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        "structured_output_rate": float((predictions["backend"] == "instructor_openrouter").mean()),
        "metrics": multiclass_metrics(predictions["label"], probabilities),
        "protocol": "Fixed prompt; official eval; no eval-driven prompt or threshold tuning",
    }
    output_dir = Path(args.output_dir)
    predictions.to_json(output_dir / "official_llm_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    (output_dir / "official_llm_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
