# Speech Final Project: Speech-Toolformer

## Goal

The project follows the selected DLS Speech topic **"Speech-Toolformer"**. It builds an assistant that accepts Russian or English requests, decides whether a tool is needed, emits a structured JSON tool call, optionally executes it, and returns a human-readable answer.

The selected tool is **not weather**. This project implements:

- `split_bill`: split a bill between people with optional tip.
- `convert_units`: convert simple distances and weights.

## What Is Implemented

- Synthetic bilingual dataset with 200-300 tool/no-tool examples.
- Robust JSON parser and validator.
- Text pipeline: text -> tool call -> tool result -> answer.
- Real audio pipeline C: uploaded audio -> native audio model -> JSON tool call.
- Real cascaded pipeline D: uploaded audio -> OpenRouter STT -> text model -> JSON tool call.
- Deterministic ASR simulator retained only for fast offline regression tests.
- Metrics:
  - Parsable Tool Invocation Rate.
  - Precision, Recall, False Alarm Rate.
  - Tool name accuracy.
  - Argument exact-match / numeric tolerance accuracy.
  - Text vs audio modality gap.
  - WER for ASR outputs.
- FastAPI service for Railway.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_experiment.py --n 300
python scripts/run_real_audio_benchmark.py --n 300 --eval-size 30 --env-file /path/to/.env
uvicorn app:app --host 0.0.0.0 --port 8000
```

Experiment artifacts:

- `artifacts/speech_metrics.json`
- `artifacts/speech_predictions.csv`
- `reports/final_report.md`

## API

- `GET /health`
- `GET /demo`
- `POST /assistant/text`
- `POST /assistant/audio_transcript`
- `POST /assistant/audio` (`multipart/form-data`, real file, `pipeline=direct|cascaded`)
