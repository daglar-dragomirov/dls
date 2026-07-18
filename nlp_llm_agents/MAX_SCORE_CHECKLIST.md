# Max Score Checklist: NLP LLM Agents

Selected topic: `LLM агенты`.

This is a non-partner topic with no selection form. The grading sheet gives 20 points total.

## Criteria Coverage

| Criterion from the sheet | Points | Evidence in this project |
| --- | ---: | --- |
| Strong baseline | 4 | `src/nlp_llm_agents/models.py` trains TF-IDF + Logistic Regression on 20,000 public-card pairs. Metrics are saved in `artifacts/nlp_metrics.json`. |
| Agent with search | 8 | `src/nlp_llm_agents/agent.py` implements a structured relevance agent that decides when to call `search_evidence`. Tool traces are saved in `artifacts/nlp_eval_predictions.csv`. |
| Substantially beat baseline or show serious effort | 8 | Full agent ROC-AUC is `0.9978` vs baseline `0.6972`; F1 is `0.9765` vs `0.6110`; NDCG@10 is `1.0000` vs `0.8088`. No-search ablation is included. |

## Extra Work For A Safer Defense

- Reproducible Yandex.Maps-like benchmark with a fixed seed.
- Same 500-pair evaluation split for baseline, no-search ablation, and full agent.
- Structured outputs: score, decision, rationale, backend, and tool trace.
- Explicit agent loop with intent extraction, evidence search, scoring, and guarded LLM review.
- Russian and English prompt robustness checks.
- Optional Instructor/OpenRouter backend plus deterministic fallback.
- Independent 50-case real LLM audit with 100% structured-review coverage and F1 `0.9804`.
- Mentor-data CSV/JSONL adapter with alias detection and validation.
- FastAPI service with UI endpoints for Railway deployment.
- Unit tests and server-side smoke tests.

## Final Artifacts

- `reports/final_report.md`
- `artifacts/nlp_metrics.json`
- `artifacts/nlp_eval_predictions.csv`
- `artifacts/nlp_openrouter_smoke.json`
