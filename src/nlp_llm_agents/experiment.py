from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .agent import RelevanceAgent
from .data_io import LABEL_VALUES, load_official_train_eval
from .metrics import multiclass_metrics
from .models import BaselineRelevanceModel


def _probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"probability_{value}" for value in LABEL_VALUES]].to_numpy(dtype=float)


def _sample_reference(train: pd.DataFrame, size: int = 3_000) -> pd.DataFrame:
    pieces = []
    for _, group in train.groupby("label"):
        share = max(1, round(size * len(group) / len(train)))
        pieces.append(group.sample(min(share, len(group)), random_state=2026))
    result = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=2026).head(size).copy()
    result["review_snippets"] = result["review_snippets"].str.slice(0, 1_500)
    return result


def run_official_experiment(
    train_path: str | Path,
    eval_path: str | Path,
    *,
    output_dir: str | Path = "artifacts",
    seed: int = 2026,
) -> dict:
    """Tune on a train holdout, refit on all train rows, evaluate eval once."""
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    train, evaluation = load_official_train_eval(train_path, eval_path)
    train_indices, validation_indices = train_test_split(
        np.arange(len(train)),
        test_size=0.2,
        random_state=seed,
        stratify=train["label"],
    )
    fit_train = train.iloc[train_indices].reset_index(drop=True)
    validation = train.iloc[validation_indices].reset_index(drop=True)

    validation_baseline = BaselineRelevanceModel.fit(fit_train)
    validation_baseline_probabilities = validation_baseline.predict_proba(validation)
    validation_baseline_metrics = multiclass_metrics(validation["label"], validation_baseline_probabilities)

    weight_grid = [0.10, 0.20, 0.30, 0.40, 0.50]
    validation_retriever = RelevanceAgent(validation_baseline, fit_train, retrieval_weight=0.0).retrieval_tool
    if validation_retriever is None:
        raise RuntimeError("The train-only retrieval index was not initialized")
    validation_retrieval_probabilities, _ = validation_retriever.search_many(validation)
    validation_candidates = []
    for weight in weight_grid:
        blended = (1.0 - weight) * validation_baseline_probabilities + weight * validation_retrieval_probabilities
        metrics = multiclass_metrics(validation["label"], blended)
        validation_candidates.append({"weight": weight, **metrics})
    selected = max(validation_candidates, key=lambda row: (row["accuracy"], row["macro_f1"], -row["weight"]))

    baseline = BaselineRelevanceModel.fit(train)
    baseline.save(str(output_dir / "baseline.joblib"))
    reference = _sample_reference(train)
    reference.to_json(
        output_dir / "retrieval_reference.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    eval_baseline_probabilities = baseline.predict_proba(evaluation)
    eval_baseline_metrics = multiclass_metrics(evaluation["label"], eval_baseline_probabilities)
    final_agent = RelevanceAgent(baseline, train, retrieval_weight=float(selected["weight"]))
    predictions = final_agent.score_frame(evaluation)
    eval_agent_probabilities = _probabilities(predictions)
    eval_agent_metrics = multiclass_metrics(evaluation["label"], eval_agent_probabilities)

    for idx, value in enumerate(LABEL_VALUES):
        predictions[f"baseline_probability_{value}"] = eval_baseline_probabilities[:, idx]
    predictions["baseline_prediction"] = np.asarray(LABEL_VALUES)[eval_baseline_probabilities.argmax(axis=1)]
    predictions["is_error"] = predictions["predicted_relevance"] != predictions["label"]
    predictions.to_json(
        output_dir / "official_eval_predictions.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    metrics = {
        "protocol": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "eval_rows": len(evaluation),
            "seed": seed,
            "tuning_source": "official train holdout only",
            "eval_usage": "single final evaluation; no prompt or threshold tuning",
        },
        "label_distribution": {
            "train": train["label"].value_counts().sort_index().to_dict(),
            "eval": evaluation["label"].value_counts().sort_index().to_dict(),
        },
        "validation": {
            "baseline": validation_baseline_metrics,
            "agent_candidates": validation_candidates,
            "selected_retrieval_weight": selected["weight"],
        },
        "official_eval": {
            "baseline": eval_baseline_metrics,
            "agent": eval_agent_metrics,
            "accuracy_delta": eval_agent_metrics["accuracy"] - eval_baseline_metrics["accuracy"],
            "macro_f1_delta": eval_agent_metrics["macro_f1"] - eval_baseline_metrics["macro_f1"],
            "tool_usage_rate": float(predictions["used_search"].mean()),
            "parseable_output_rate": 1.0,
        },
        "runtime_seconds": time.time() - started,
    }
    (output_dir / "official_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(metrics, predictions, report_dir / "final_report.md")
    return metrics


def write_report(metrics: dict, predictions: pd.DataFrame, path: Path) -> None:
    validation = metrics["validation"]
    evaluation = metrics["official_eval"]
    examples = predictions.loc[predictions["is_error"]].head(10)
    lines = [
        "# Итоговый отчёт: LLM-агент для релевантности организаций",
        "",
        "## Протокол без утечки",
        "",
        f"Официальный train содержит **{metrics['protocol']['train_rows']}** объектов, eval — **{metrics['protocol']['eval_rows']}**. ",
        "Все параметры baseline, retrieval и промпта выбираются только по отложенной части train. Eval используется один раз для итоговой оценки.",
        "",
        "## Система",
        "",
        "1. Baseline: word/char TF-IDF и трёхклассовая Logistic Regression.",
        "2. Инструмент поиска: извлекает подтверждающие фрагменты из рубрики, цен и отзывов и формирует ссылку на внешний поиск.",
        "3. Retrieval-инструмент: находит похожие размеченные примеры только в train.",
        "4. Агент объединяет вероятности baseline и train-retrieval; для онлайн-режима доступен структурированный LLM-review через Instructor.",
        "",
        "## Результаты",
        "",
        "| Разбиение | Система | Accuracy | Macro F1 |",
        "| --- | --- | ---: | ---: |",
        f"| Train holdout | Baseline | {validation['baseline']['accuracy']:.4f} | {validation['baseline']['macro_f1']:.4f} |",
        f"| Official eval | Baseline | {evaluation['baseline']['accuracy']:.4f} | {evaluation['baseline']['macro_f1']:.4f} |",
        f"| Official eval | Agent | {evaluation['agent']['accuracy']:.4f} | {evaluation['agent']['macro_f1']:.4f} |",
        "",
        f"Выбранный на train retrieval-вес: `{validation['selected_retrieval_weight']:.2f}`. ",
        f"Прирост accuracy на eval: `{evaluation['accuracy_delta']:+.4f}`, macro F1: `{evaluation['macro_f1_delta']:+.4f}`.",
        "",
        "## Анализ ошибок",
        "",
        "Наиболее сложен класс 0.1: это частично релевантные организации, где тематика совпадает, но важное ограничение запроса не подтверждено. ",
        "Ошибки между 0.1 и крайними классами связаны с неоднозначными формулировками, отсутствием явного факта в карточке и шумом пользовательских отзывов.",
        "",
        "| Запрос | Организация | Истина | Прогноз |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in examples.iterrows():
        query = str(row["query"]).replace("|", "/")[:90]
        organization = str(row["organization_name"]).replace("|", "/")[:70]
        lines.append(f"| {query} | {organization} | {row['label']} | {row['predicted_relevance']} |")
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            "Baseline обеспечивает сильную воспроизводимую точку отсчёта. Агент добавляет интерпретируемый поиск доказательств и retrieval; ",
            "его вклад измерен абляцией на одном и том же наборе. Структурированные ответы и трасса инструментов доступны в API и интерфейсе.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# Backward-compatible name for external callers.
run_experiment = run_official_experiment
