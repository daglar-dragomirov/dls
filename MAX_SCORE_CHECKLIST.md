# Проверка критериев Speech — 20 баллов

| Критерий | Баллы | Где проверять |
| --- | ---: | --- |
| Креативность инструмента | 2 | Два аргументированных инструмента, не погода: `tools.py` |
| System-prompt tuning | 2 | zero-shot/schema-tuned LFM2.5 ablation в `local_lfm_metrics.json` |
| Метрики текстового tool-use A | 3 | precision, recall, FAR, parsability, tool/argument accuracy |
| Дизайн синтетического датасета | 4 | 300 text+audio, RU/EN, `none`-примеры, фиксированный seed |
| ASR benchmark B | 3 | реальная локальная LFM2.5 ASR, WER и примеры ошибок |
| Audio pipelines C–D | 3 | прямой и каскадный локальные прогоны на одинаковых MP3 |
| Выбор лучшего пайплайна | 2 | сравнение качества и latency в итоговом отчёте |
| Финальный отчёт | 3 | выводы, абляция и fail cases в `reports/final_report.md` |

Дополнительно: русскоязычное Railway-демо, загрузка аудио, готовый MP3, API, unit/integration tests и сохранённые сырые predictions.
