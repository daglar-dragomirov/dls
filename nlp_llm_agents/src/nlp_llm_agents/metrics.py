from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def query_ndcg(frame: pd.DataFrame, score_col: str, k: int = 10) -> float:
    scores = []
    for _, group in frame.groupby("query_id"):
        if len(group) < 2 or group["label"].nunique() < 2:
            continue
        y_true = group["label"].to_numpy(dtype=float).reshape(1, -1)
        y_score = group[score_col].to_numpy(dtype=float).reshape(1, -1)
        scores.append(float(ndcg_score(y_true, y_score, k=min(k, len(group)))))
    return float(np.mean(scores)) if scores else 0.0

