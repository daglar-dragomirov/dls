from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    metrics = run_experiment(
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

