# Max Score Checklist: Speech-Toolformer

Selected topic: `Speech-Toolformer`.

This is a non-partner topic with no selection form. The selected row has no separate bonus section; the maximum score is covered by the listed 22 points.

## Criteria Coverage

| Criterion from the sheet | Points | Evidence in this project |
| --- | ---: | --- |
| Creative tool choice | 2 | Implements `split_bill` and `convert_units`; the weather tool from the example is intentionally not used. |
| System-prompt tuning and instruction-following eval | 2 | Real OpenRouter A/B prompt evaluation: zero-shot parseable rate `0.0000`, schema-tuned parseable rate `1.0000`; exact predictions are saved. |
| Text tool-use metrics, pipeline A | 3 | `artifacts/speech_metrics.json`: text parsable rate `1.0000`, precision `1.0000`, recall `1.0000`, false alarm rate `0.0000`, argument accuracy `1.0000`, F1 `1.0000`. |
| Synthetic dataset design | 4 | 300 bilingual examples with tool and no-tool cases are generated in `src/speech_toolformer/dataset.py` and saved to `artifacts/speech_synthetic_dataset.jsonl`. |
| ASR benchmarking, pipeline B | 3 | Real STT over generated audio is evaluated at WER `0.2817`; simulator WER remains only a regression check. |
| Audio pipeline benchmarking, C-D | 3 | Both real native audio C and real cascaded D are evaluated: F1 `1.0000`, argument accuracy `0.9000`; native latency `1.51s` vs cascaded `3.02s`. |
| Best pipeline choice | 2 | `reports/final_report.md` compares text and audio metrics, reports the modality gap, and explains why no SFT is required for the scoped tools. |
| Final report | 3 | `reports/final_report.md` contains metrics, qualitative examples, modality gap, and implementation notes. |

## Extra Work For A Safer Defense

- Two tools instead of one.
- Russian and English prompt templates.
- 10-20% no-tool examples.
- Robust JSON parser and validator.
- Russian and English manual prompt matrix, including messy phrasing and audio-transcript cases.
- Optional Instructor/OpenRouter backend plus deterministic fallback.
- FastAPI service with UI endpoints for Railway deployment.
- Browser UI accepts real audio files and lets the evaluator switch between C and D.
- 300 committed bilingual TTS audio files plus an auditable manifest.
- Unit tests and server-side smoke tests.

## Final Artifacts

- `reports/final_report.md`
- `artifacts/speech_metrics.json`
- `artifacts/speech_predictions.csv`
- `artifacts/speech_synthetic_dataset.jsonl`
- `artifacts/speech_openrouter_smoke.json`
