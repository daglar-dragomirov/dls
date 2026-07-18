from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_toolformer.asr import word_error_rate
from speech_toolformer.local_lfm import LocalLFM2Audio, MODEL_ID
from speech_toolformer.metrics import mean, tool_metrics
from speech_toolformer.parser import extract_json, validate_tool_call


def _prediction(text: str) -> dict:
    call = extract_json(text)
    valid, reason = validate_tool_call(call)
    return {"raw": text, "call": call, "valid": valid, "reason": reason}


def _records(rows: list[dict], column: str) -> list[dict]:
    return [{"expected_call": row["expected_call"], "prediction": row[column]} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local LFM2.5-Audio on real English TTS files")
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--manifest", default=str(ROOT / "artifacts" / "audio_manifest.csv"))
    parser.add_argument("--dataset", default=str(ROOT / "artifacts" / "speech_synthetic_dataset.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    labels = pd.read_json(args.dataset, lines=True)[["id", "expected_call"]]
    data = manifest.merge(labels, on="id", how="inner")
    data = data[data["text"].map(lambda value: str(value).isascii())].copy()
    data["tool_name"] = data["expected_call"].map(lambda value: value["tool_name"])
    pieces = []
    per_group = max(1, args.size // max(data["tool_name"].nunique(), 1))
    for _, group in data.groupby("tool_name"):
        pieces.append(group.sample(min(per_group, len(group)), random_state=2026))
    selected = pd.concat(pieces).sample(frac=1, random_state=2026).head(args.size).reset_index(drop=True)

    model = LocalLFM2Audio()
    rows: list[dict] = []
    zero_shot_limit = min(30, len(selected))
    for index, sample in tqdm(selected.iterrows(), total=len(selected), desc="Local LFM2.5-Audio"):
        audio_path = ROOT / str(sample["audio_path"]).replace("\\", "/")
        asr = model.transcribe(audio_path)
        direct = model.tool_call_from_audio(audio_path, tuned=True)
        cascaded = model.tool_call_from_text(asr.text)
        zero_shot = model.tool_call_from_audio(audio_path, tuned=False) if index < zero_shot_limit else None
        rows.append(
            {
                "id": int(sample["id"]),
                "text": sample["text"],
                "audio_path": str(sample["audio_path"]),
                "expected_call": sample["expected_call"],
                "recognized_text": asr.text,
                "wer": word_error_rate(sample["text"], asr.text),
                "pipeline_C_direct": _prediction(direct.text),
                "pipeline_D_cascaded": _prediction(cascaded.text),
                "zero_shot": _prediction(zero_shot.text) if zero_shot else None,
                "latency_asr": asr.latency_seconds,
                "latency_direct": direct.latency_seconds,
                "latency_cascaded_router": cascaded.latency_seconds,
            }
        )

    zero_rows = [row for row in rows if row["zero_shot"] is not None]
    metrics = {
        "model": MODEL_ID,
        "device": "NVIDIA A40",
        "n": len(rows),
        "zero_shot_n": len(zero_rows),
        "protocol": "English real TTS audio; local open-weight inference; fixed seed",
        "pipeline_A_zero_shot": tool_metrics(_records(zero_rows, "zero_shot")),
        "pipeline_B_asr": {"wer": mean([row["wer"] for row in rows]), "mean_latency_seconds": mean([row["latency_asr"] for row in rows])},
        "pipeline_C_direct_audio": {**tool_metrics(_records(rows, "pipeline_C_direct")), "mean_latency_seconds": mean([row["latency_direct"] for row in rows])},
        "pipeline_D_cascaded": {**tool_metrics(_records(rows, "pipeline_D_cascaded")), "mean_latency_seconds": mean([row["latency_asr"] + row["latency_cascaded_router"] for row in rows])},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "local_lfm_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "local_lfm_predictions.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
