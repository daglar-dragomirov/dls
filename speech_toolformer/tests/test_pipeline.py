from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_toolformer.dataset import generate_dataset
from speech_toolformer.metrics import tool_metrics
from speech_toolformer.pipeline import text_pipeline


def test_text_pipeline_parses_and_executes_tool():
    result = text_pipeline("Split a USD 120 bill between 4 people with 15% tip")
    assert result["valid"]
    assert result["call"]["tool_name"] == "split_bill"
    assert result["tool_result"]["per_person"] == 34.5


def test_generated_dataset_text_metrics_are_strong():
    frame = generate_dataset(120, seed=3)
    records = [
        {"expected_call": row["expected_call"], "prediction": text_pipeline(row["text"])}
        for row in frame.to_dict(orient="records")
    ]
    metrics = tool_metrics(records)
    assert metrics["parsable_rate"] == 1.0
    assert metrics["f1"] > 0.95
    assert metrics["argument_accuracy"] > 0.9


def test_manual_russian_and_english_prompt_variants():
    ru_split = text_pipeline("Раздели счет 2400 рублей на 4 человек и добавь чаевые 10 процентов")
    assert ru_split["call"]["tool_name"] == "split_bill"
    assert ru_split["call"]["arguments"]["currency"] == "RUB"

    ru_convert = text_pipeline("Переведи 12 километров в мили")
    assert ru_convert["call"]["tool_name"] == "convert_units"
    assert ru_convert["call"]["arguments"]["from_unit"] == "kilometers"
    assert ru_convert["call"]["arguments"]["to_unit"] == "miles"

    en_how_many = text_pipeline("How many meters are in 12 feet")
    assert en_how_many["call"]["tool_name"] == "convert_units"
    assert en_how_many["call"]["arguments"]["from_unit"] == "feet"
    assert en_how_many["call"]["arguments"]["to_unit"] == "meters"
