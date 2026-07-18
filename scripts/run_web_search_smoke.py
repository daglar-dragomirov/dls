from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_llm_agents.agent import PublicWebSearchTool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and save one real public-search tool call")
    parser.add_argument("--query", default="Москва ресторан с летней верандой")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "web_search_audit.json"))
    args = parser.parse_args()
    result = PublicWebSearchTool().search(args.query, limit=5)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
