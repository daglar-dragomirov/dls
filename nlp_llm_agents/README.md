# NLP Final Project: LLM Agent for Map Organization Relevance

## Goal

The project follows the selected DLS topic **"LLM агенты"**: estimate whether an organization is relevant to a broad map query such as "романтичный джаз-бар" or "ресторан с верандой".

The grading sheet gives three blocks:

1. Strong baseline - 4 points.
2. Agent that can use search - 8 points.
3. Improve over baseline or show a serious negative result - 8 points.

This implementation targets all three blocks.

## What Is Implemented

- Synthetic but controlled Yandex.Maps-like benchmark:
  - `20_000` train pairs by default.
  - `500` eval pairs by default.
  - Each pair contains a broad query, public organization card, hidden reviews/evidence, and a relevance label.
- Baseline:
  - TF-IDF + Logistic Regression over public card fields only.
- Tool-using agent:
  - Analyzes query requirements.
  - Decides when evidence is missing.
  - Calls local search/profile tools over review snippets and structured tags.
  - Produces a score, decision, rationale, and trace of tool calls.
- Metrics:
  - ROC-AUC, PR-AUC, F1, accuracy, precision, recall.
  - Query-level NDCG@10.
  - Tool usage rate and parseable structured outputs.
- HTTP service:
  - `POST /score`
  - `POST /batch_score`
  - `GET /demo`
  - `GET /health`

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_experiment.py --train-size 20000 --eval-size 500
python scripts/benchmark_openrouter.py --size 50 --workers 10 --env-file /path/to/.env
uvicorn app:app --host 0.0.0.0 --port 8000
```

The experiment writes:

- `artifacts/nlp_metrics.json`
- `artifacts/nlp_eval_predictions.csv`
- `artifacts/nlp_llm_benchmark_metrics.json`
- `artifacts/nlp_llm_benchmark_predictions.jsonl`
- `reports/final_report.md`

## Railway

Use this folder as the service root, or deploy the branch where this project is placed at repository root.
