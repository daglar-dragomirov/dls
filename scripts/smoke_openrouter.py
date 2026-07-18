from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_toolformer.env_utils import first_openrouter_key
from speech_toolformer.pipeline import text_pipeline


def main() -> None:
    os.environ["USE_INSTRUCTOR"] = "1"
    if not first_openrouter_key():
        raise RuntimeError("No OpenRouter key found")

    prompts = [
        "Split a USD 120 bill between 4 people with 15% tip",
        "Convert 10 kilometers to miles",
        "Tell me a short fun fact about machine learning",
    ]
    results = [{"prompt": prompt, "result": text_pipeline(prompt)} for prompt in prompts]
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/speech_openrouter_smoke.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    assert all(item["result"]["valid"] for item in results)


if __name__ == "__main__":
    main()

