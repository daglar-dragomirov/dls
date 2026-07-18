from __future__ import annotations

from typing import Any

import numpy as np


def _same_tool(pred: dict[str, Any] | None, gold: dict[str, Any]) -> bool:
    return bool(pred) and pred.get("tool_name") == gold.get("tool_name")


def _arguments_match(pred: dict[str, Any] | None, gold: dict[str, Any], tol: float = 1e-4) -> bool:
    if not _same_tool(pred, gold):
        return False
    if gold.get("tool_name") == "none":
        return True
    pred_args = pred.get("arguments", {})
    gold_args = gold.get("arguments", {})
    for key, gold_value in gold_args.items():
        if key not in pred_args:
            return False
        pred_value = pred_args[key]
        if isinstance(gold_value, (int, float)):
            if abs(float(pred_value) - float(gold_value)) > tol:
                return False
        elif str(pred_value).lower() != str(gold_value).lower():
            return False
    return True


def tool_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    tp = fp = fn = tn = 0
    parsable = 0
    tool_correct = 0
    args_correct = 0
    for row in records:
        gold = row["expected_call"]
        pred = row["prediction"].get("call")
        valid = row["prediction"].get("valid", False)
        parsable += int(valid)
        gold_needs = gold["tool_name"] != "none"
        pred_needs = bool(pred) and pred.get("tool_name") != "none"
        if gold_needs and pred_needs:
            tp += 1
        elif not gold_needs and pred_needs:
            fp += 1
        elif gold_needs and not pred_needs:
            fn += 1
        else:
            tn += 1
        tool_correct += int(_same_tool(pred, gold))
        args_correct += int(_arguments_match(pred, gold))
    n = max(len(records), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    false_alarm_rate = fp / max(fp + tn, 1)
    return {
        "parsable_rate": parsable / n,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "tool_name_accuracy": tool_correct / n,
        "argument_accuracy": args_correct / n,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
    }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

