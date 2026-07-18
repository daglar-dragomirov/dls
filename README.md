# DLS NLP Final: агент релевантности организаций

**Выбранная тема DLS: «LLM-агенты: оценка релевантности организаций запросам на Яндекс.Картах».**

Проект решает официальную задачу DLS: по широкому запросу и карточке организации предсказывает один из трёх классов релевантности:

- `1.0` — `RELEVANT_PLUS`, организация полностью соответствует запросу;
- `0.1` — `RELEVANT_MINUS`, соответствие частичное;
- `0.0` — `IRRELEVANT`.

## Что реализовано

- сильный baseline: word/char TF-IDF + трёхклассовая Logistic Regression;
- агент с поиском подтверждающих фрагментов в карточке и настоящим внешним web-search tool;
- retrieval похожих размеченных примеров только из train;
- структурированный LLM-review через Instructor/OpenRouter;
- честная абляция baseline → retrieval-agent;
- accuracy, macro/weighted F1, per-class metrics, confusion matrix и анализ ошибок;
- русскоязычный FastAPI-интерфейс и пакетный API.

## Защита от утечки

Все параметры выбираются на фиксированном holdout из `data_for_train.jsonl`. После выбора модель переобучается на полном train, а `data_for_eval.jsonl` используется один раз для итогового отчёта. Eval не используется для настройки промпта, весов или порогов.

## Воспроизведение

Скачайте официальный архив: <https://disk.yandex.ru/d/FK8B0ONM67VHCA>.

```bash
pip install -r requirements.txt
python scripts/run_experiment.py \
  --train /path/to/data_for_train.jsonl \
  --eval /path/to/data_for_eval.jsonl
pytest -q
uvicorn app:app --host 0.0.0.0 --port 8000
```

Основные результаты:

- `artifacts/official_metrics.json`;
- `artifacts/official_eval_predictions.jsonl`;
- `reports/final_report.md`;
- `artifacts/baseline.joblib`;
- `artifacts/retrieval_reference.jsonl`.
- `artifacts/official_llm_metrics.json` и построчные LLM-предсказания;
- `artifacts/web_search_audit.json` с фактической трассой внешнего поиска.

На official eval baseline получил accuracy `0.5404` и macro F1 `0.4355`. Structured LLM-agent дал 100% schema-valid ответов и macro F1 `0.4630`, но accuracy `0.4825`. В отчете подробно разобран этот отрицательный результат по основной метрике и рост recall класса `0.1` с `0.1143` до `0.5857`.

## API

- `GET /` — интерактивная демонстрация;
- `GET /metrics` — сохранённые метрики;
- `POST /score` — один запрос и одна организация;
- `POST /batch_score` — пакетная оценка;
- `GET /health` — проверка сервиса.

В форме можно отдельно включить реальный поиск в интернете. Он имеет ограниченный таймаут и возвращает в трассе статус, запрос, ссылки или явную сетевую ошибку.

Живое демо: <https://dls-final-nlp-production.up.railway.app/>
