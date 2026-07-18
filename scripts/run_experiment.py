from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.experiment import run_official_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official DLS map-relevance experiment")
    parser.add_argument("--train", required=True, help="Path to data_for_train.jsonl")
    parser.add_argument("--eval", required=True, help="Path to data_for_eval.jsonl")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    metrics = run_official_experiment(args.train, args.eval, output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
