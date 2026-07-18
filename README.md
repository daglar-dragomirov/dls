# DLS Speech Final: Speech-Toolformer

**Выбранная тема DLS: «Speech-Toolformer».**

Голосовой ассистент принимает английскую или русскую речь, определяет необходимость инструмента, формирует JSON-вызов, выполняет функцию и возвращает понятный ответ.

## Инструменты

- `split_bill(amount, people, tip_percent, currency)` — разделение счёта;
- `convert_units(value, from_unit, to_unit)` — перевод единиц;
- `none()` — запрос не требует инструмента.

Погода из исходного примера намеренно не используется.

## Два контура

**Исследовательский GPU-контур** использует локальную открытую модель `LiquidAI/LFM2.5-Audio-1.5B`:

- B: audio → ASR;
- C: audio → JSON tool-call за один проход;
- D: audio → transcript → text → JSON tool-call;
- отдельное сравнение zero-shot и schema-tuned prompt.

**Railway-демо** не загружает модель весом несколько гигабайт в CPU-контейнер. Оно использует тот же контракт инструментов через OpenRouter и позволяет загрузить аудио, выбрать C/D и проверить результат онлайн.

## Воспроизведение

Обычные тесты и веб-сервис:

```bash
pip install -r requirements.txt
pytest -q
uvicorn app:app --host 0.0.0.0 --port 8000
```

Локальный GPU-бенчмарк:

```bash
pip install -r requirements-gpu.txt
python scripts/run_local_lfm_benchmark.py --size 90
```

Артефакты:

- `artifacts/speech_synthetic_dataset.jsonl` — 300 размеченных запросов;
- `artifacts/audio_manifest.csv` и `audio_samples/` — 300 TTS-файлов;
- `artifacts/local_lfm_metrics.json` — локальные LFM2.5-метрики;
- `artifacts/local_lfm_predictions.jsonl` — сырые ответы и latency;
- `reports/final_report.md` — итоговый анализ и fail cases.

Финальный A40-прогон охватывает все 82 доступных англоязычных аудио: direct audio-to-tool получил parseability `1.000`, F1 `0.929`, tool accuracy `0.805` и среднюю latency `0.309` с. Каскад имеет ту же F1, но меньшую argument accuracy (`0.659`) и примерно вдвое большую latency (`0.632` с). ASR WER равен `0.250`.

Интерфейс русифицирован; английскими оставлены тестовые команды и JSON-поля. Готовый пример можно скачать по `GET /sample-audio`.

Живое демо: <https://dls-final-speech-production.up.railway.app/>
