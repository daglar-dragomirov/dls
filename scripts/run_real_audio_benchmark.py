from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_toolformer.audio_backend import direct_audio_tool_call, text_llm_tool_call, transcribe_audio
from speech_toolformer.asr import word_error_rate
from speech_toolformer.dataset import generate_dataset
from speech_toolformer.env_utils import load_env_file
from speech_toolformer.metrics import mean, tool_metrics


def _is_russian(text: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)


async def _synthesize_one(row: dict, output_dir: Path, semaphore: asyncio.Semaphore) -> dict:
    import edge_tts

    path = output_dir / f"{int(row['id']):04d}.mp3"
    if not path.exists():
        voice = "ru-RU-SvetlanaNeural" if _is_russian(row["text"]) else "en-US-JennyNeural"
        async with semaphore:
            await edge_tts.Communicate(row["text"], voice=voice).save(str(path))
    return {"id": row["id"], "text": row["text"], "audio_path": str(path.relative_to(ROOT)), "format": "mp3"}


async def synthesize(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(10)
    rows = await asyncio.gather(*[_synthesize_one(row, output_dir, semaphore) for row in frame.to_dict("records")])
    return pd.DataFrame(rows)


def _stratified_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["tool"] = frame["expected_call"].map(lambda call: call["tool_name"])
    parts = []
    per_group = max(1, size // frame["tool"].nunique())
    for _, group in frame.groupby("tool"):
        parts.append(group.sample(min(per_group, len(group)), random_state=seed))
    sampled = pd.concat(parts).drop_duplicates("id")
    if len(sampled) < size:
        remainder = frame.loc[~frame["id"].isin(sampled["id"])]
        sampled = pd.concat([sampled, remainder.sample(min(size - len(sampled), len(remainder)), random_state=seed + 1)])
    return sampled.head(size).sort_values("id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)

    artifacts = ROOT / "artifacts"
    audio_dir = artifacts / "audio_samples"
    data = generate_dataset(args.n, args.seed)
    manifest = asyncio.run(synthesize(data, audio_dir))
    manifest.to_csv(artifacts / "audio_manifest.csv", index=False)
    eval_frame = _stratified_sample(data, args.eval_size, args.seed)

    prompt_records = {"zero_shot": [], "schema_tuned": []}
    prompt_jobs = [(row, variant) for row in eval_frame.to_dict("records") for variant in prompt_records]
    def prompt_job(job):
        row, variant = job
        pred = text_llm_tool_call(row["text"], prompt_variant=variant)
        return variant, {"id": row["id"], "expected_call": row["expected_call"], "prediction": pred}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for variant, record in pool.map(prompt_job, prompt_jobs):
            prompt_records[variant].append(record)
    prompt_metrics = {name: tool_metrics(records) for name, records in prompt_records.items()}

    direct_records, cascaded_records, wers, output_rows = [], [], [], []
    def audio_job(row):
        audio_path = audio_dir / f"{int(row['id']):04d}.mp3"
        audio = audio_path.read_bytes()
        started = time.perf_counter()
        direct = direct_audio_tool_call(audio, "mp3", prompt_variant="schema_tuned")
        direct_latency = time.perf_counter() - started

        started = time.perf_counter()
        stt = transcribe_audio(audio, "mp3")
        cascaded = text_llm_tool_call(stt["text"], prompt_variant="schema_tuned")
        cascaded_latency = time.perf_counter() - started
        return (
            {"id": row["id"], "expected_call": row["expected_call"], "prediction": direct},
            {"id": row["id"], "expected_call": row["expected_call"], "prediction": cascaded},
            word_error_rate(row["text"], stt["text"]),
            [
                {"id": row["id"], "pipeline": "C_native_audio", "reference": row["text"], "recognized_text": "", "expected_call": row["expected_call"], "prediction": direct.get("call"), "valid": direct.get("valid"), "latency_s": direct_latency},
                {"id": row["id"], "pipeline": "D_cascaded", "reference": row["text"], "recognized_text": stt["text"], "expected_call": row["expected_call"], "prediction": cascaded.get("call"), "valid": cascaded.get("valid"), "latency_s": cascaded_latency},
            ],
        )
    with ThreadPoolExecutor(max_workers=8) as pool:
        for direct, cascaded, wer, rows in pool.map(audio_job, eval_frame.to_dict("records")):
            direct_records.append(direct); cascaded_records.append(cascaded); wers.append(wer); output_rows.extend(rows)

    metrics = {
        "dataset_size": args.n,
        "audio_files": len(manifest),
        "real_audio_eval_size": len(eval_frame),
        "prompt_tuning_A": prompt_metrics,
        "pipeline_B_asr": {"wer": mean(wers)},
        "pipeline_C_native_audio": tool_metrics(direct_records),
        "pipeline_D_cascaded": tool_metrics(cascaded_records),
        "latency_s": pd.DataFrame(output_rows).groupby("pipeline")["latency_s"].mean().to_dict(),
    }
    (artifacts / "real_audio_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(output_rows).to_csv(artifacts / "real_audio_predictions.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
