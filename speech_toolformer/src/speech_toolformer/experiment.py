from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .asr import TranscriptASR, word_error_rate
from .dataset import generate_dataset
from .metrics import mean, tool_metrics
from .pipeline import cascaded_audio_pipeline, text_pipeline


def run_experiment(n: int = 300, seed: int = 2026, output_dir: str | Path = "artifacts") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    data = generate_dataset(n=n, seed=seed)
    data.to_json(output_dir / "speech_synthetic_dataset.jsonl", orient="records", lines=True, force_ascii=False)

    text_records = []
    audio_records = []
    asr = TranscriptASR(word_drop_prob=0.025, seed=seed)
    wers = []

    for row in data.to_dict(orient="records"):
        text_pred = text_pipeline(row["text"])
        text_records.append({"id": row["id"], "expected_call": row["expected_call"], "prediction": text_pred})

        audio_pred = cascaded_audio_pipeline(row["text"], asr=asr)
        wers.append(word_error_rate(row["text"], audio_pred["recognized_text"]))
        audio_records.append({"id": row["id"], "expected_call": row["expected_call"], "prediction": audio_pred})

    text_metrics = tool_metrics(text_records)
    audio_metrics = tool_metrics(audio_records)
    metrics = {
        "n": n,
        "seed": seed,
        "text_pipeline": text_metrics,
        "audio_cascaded_pipeline": audio_metrics,
        "modality_gap": {
            key: text_metrics[key] - audio_metrics[key]
            for key in ["f1", "parsable_rate", "tool_name_accuracy", "argument_accuracy"]
        },
        "wer": mean(wers),
    }

    rows = []
    for source, records in [("text", text_records), ("audio_cascaded", audio_records)]:
        for record in records:
            pred = record["prediction"]
            rows.append(
                {
                    "pipeline": source,
                    "id": record["id"],
                    "expected_tool": record["expected_call"]["tool_name"],
                    "predicted_tool": (pred.get("call") or {}).get("tool_name"),
                    "valid": pred.get("valid"),
                    "answer": pred.get("answer"),
                    "recognized_text": pred.get("recognized_text", ""),
                }
            )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / "speech_predictions.csv", index=False)
    (output_dir / "speech_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics, predictions, Path("reports") / "final_report.md")
    return metrics


def write_report(metrics: dict, predictions: pd.DataFrame, path: Path) -> None:
    sample = predictions.sample(min(8, len(predictions)), random_state=8)
    lines = [
        "# Final Report: Speech-Toolformer",
        "",
        "## Selected Topic",
        "",
        "Speech final project: **Speech-Toolformer**. The assistant accepts voice/text requests, emits JSON tool calls, executes tools, and returns a readable answer.",
        "",
        "## Tool Choice",
        "",
        "The project intentionally does not use the weather tool from the example. Implemented tools:",
        "",
        "- `split_bill(amount, people, tip_percent, currency)`",
        "- `convert_units(value, from_unit, to_unit)`",
        "",
        "## Metrics",
        "",
        "| Metric | Text pipeline | Audio cascaded pipeline |",
        "| --- | ---: | ---: |",
    ]
    for key in ["parsable_rate", "precision", "recall", "false_alarm_rate", "tool_name_accuracy", "argument_accuracy", "f1"]:
        lines.append(f"| {key} | {metrics['text_pipeline'][key]:.4f} | {metrics['audio_cascaded_pipeline'][key]:.4f} |")
    lines.extend(
        [
            "",
            f"Mean WER for ASR simulator: `{metrics['wer']:.4f}`.",
            "",
            "## Modality Gap",
            "",
        ]
    )
    for key, value in metrics["modality_gap"].items():
        lines.append(f"- `{key}` gap: `{value:+.4f}`")
    lines.extend(["", "## Qualitative Examples", ""])
    for _, row in sample.iterrows():
        lines.extend(
            [
                f"- Pipeline `{row['pipeline']}`, expected `{row['expected_tool']}`, predicted `{row['predicted_tool']}`, valid `{row['valid']}`.",
                f"  Answer: {row['answer']}",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")

