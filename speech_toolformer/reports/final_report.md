# Final Report: Speech-Toolformer

## Selected Topic

Speech final project: **Speech-Toolformer**. The assistant accepts voice/text requests, emits JSON tool calls, executes tools, and returns a readable answer.

## Tool Choice

The project intentionally does not use the weather tool from the example. Implemented tools:

- `split_bill(amount, people, tip_percent, currency)`
- `convert_units(value, from_unit, to_unit)`

## Metrics

| Metric | Text pipeline | Audio cascaded pipeline |
| --- | ---: | ---: |
| parsable_rate | 1.0000 | 1.0000 |
| precision | 1.0000 | 1.0000 |
| recall | 1.0000 | 0.9764 |
| false_alarm_rate | 0.0000 | 0.0000 |
| tool_name_accuracy | 1.0000 | 0.9800 |
| argument_accuracy | 1.0000 | 0.9333 |
| f1 | 1.0000 | 0.9880 |

Mean WER for ASR simulator: `0.0226`.

## Real Audio Benchmark (A-B-C-D)

The final audit adds a separate benchmark over **300 real TTS audio files** (Russian and English). A fixed, stratified 30-file subset is evaluated with paid model APIs; raw predictions and latencies are saved for audit.

| Stage | Result |
| --- | ---: |
| A, zero-shot parseable rate | 0.0000 |
| A, schema-tuned parseable rate | 1.0000 |
| A, schema-tuned argument accuracy | 1.0000 |
| B, real STT WER | 0.2817 |
| C, native audio tool F1 / argument accuracy | 1.0000 / 0.9000 |
| D, cascaded tool F1 / argument accuracy | 1.0000 / 0.9000 |
| C mean latency | 1.51 s |
| D mean latency | 3.02 s |

The zero-shot failure and schema-tuned recovery demonstrate measurable system-prompt tuning rather than a hand-written claim. C and D have equal tool F1 on the evaluated subset, while C is about twice as fast; C is therefore the preferred online pipeline. D remains valuable because it exposes a transcript for debugging and makes the observed WER interpretable. The previous simulator result is retained only as a deterministic regression test, not as evidence for the real-audio score.

Artifacts: `audio_manifest.csv`, `real_audio_metrics.json`, and `real_audio_predictions.csv`.

## Modality Gap

- `f1` gap: `+0.0120`
- `parsable_rate` gap: `+0.0000`
- `tool_name_accuracy` gap: `+0.0200`
- `argument_accuracy` gap: `+0.0667`

## Tuning Conclusion

For the selected tools, prompt/schema engineering plus a guarded router is sufficient: text tool-use reaches perfect precision, recall, parsability, and argument accuracy. The remaining quality loss comes from the audio side, not from tool-call formatting. Therefore, SFT is not required for this scoped assistant; it would become useful if the tool set grew, the argument schema became more complex, or the ASR/transcript distribution shifted strongly away from the synthetic benchmark.

## Qualitative Examples

- Pipeline `audio_cascaded`, expected `convert_units`, predicted `convert_units`, valid `True`.
  Answer: 42 meters is 137.795 feet.
- Pipeline `text`, expected `none`, predicted `none`, valid `True`.
  Answer: No tool is needed for this request.
- Pipeline `text`, expected `none`, predicted `none`, valid `True`.
  Answer: No tool is needed for this request.
- Pipeline `audio_cascaded`, expected `split_bill`, predicted `split_bill`, valid `True`.
  Answer: Total with tip is 250.0 EUR. Each person pays 31.25 EUR.
- Pipeline `text`, expected `split_bill`, predicted `split_bill`, valid `True`.
  Answer: Total with tip is 100.8 EUR. Each person pays 14.4 EUR.
- Pipeline `text`, expected `split_bill`, predicted `split_bill`, valid `True`.
  Answer: Total with tip is 75.6 USD. Each person pays 9.45 USD.
- Pipeline `text`, expected `split_bill`, predicted `split_bill`, valid `True`.
  Answer: Total with tip is 134.4 RUB. Each person pays 67.2 RUB.
- Pipeline `audio_cascaded`, expected `split_bill`, predicted `split_bill`, valid `True`.
  Answer: Total with tip is 107.52 USD. Each person pays 17.92 USD.
