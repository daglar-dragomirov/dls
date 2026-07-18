from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data_io import LABEL_VALUES


def multiclass_metrics(y_true, probabilities) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = np.asarray(LABEL_VALUES)[probabilities.argmax(axis=1)]
    class_to_id = {value: idx for idx, value in enumerate(LABEL_VALUES)}
    y_true_ids = np.asarray([class_to_id[float(value)] for value in y_true])
    y_pred_ids = np.asarray([class_to_id[float(value)] for value in y_pred])
    report = classification_report(
        y_true_ids,
        y_pred_ids,
        labels=list(range(len(LABEL_VALUES))),
        target_names=[str(value) for value in LABEL_VALUES],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true_ids, y_pred_ids)),
        "macro_f1": float(f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_ids, y_pred_ids, average="weighted", zero_division=0)),
        "per_class": {str(value): report[str(value)] for value in LABEL_VALUES},
        "confusion_matrix": confusion_matrix(y_true_ids, y_pred_ids, labels=list(range(len(LABEL_VALUES)))).tolist(),
    }


def accuracy_from_probabilities(y_true, probabilities) -> float:
    y_pred = np.asarray(LABEL_VALUES)[np.asarray(probabilities).argmax(axis=1)]
    return float(np.mean(np.asarray(y_true, dtype=float) == y_pred))
