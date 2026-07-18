from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_VALUES = (0.0, 0.1, 1.0)

CANONICAL_COLUMNS = {
    "query": ("query", "request", "search_query", "Text"),
    "organization_name": ("organization_name", "org_name", "name", "organization"),
    "category": ("category", "rubric", "org_category", "normalized_main_rubric_name_ru"),
    "address": ("address",),
    "prices_summarized": ("prices_summarized", "prices", "price_summary"),
    "review_snippets": ("review_snippets", "reviews", "review_text", "evidence", "reviews_summarized"),
    "permalink": ("permalink", "organization_id", "org_id"),
    "label": ("label", "relevance", "target", "is_relevant", "relevance_new"),
    "query_id": ("query_id", "request_id", "qid"),
}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path)


def load_relevance_data(
    path: str | Path,
    column_map: dict[str, str] | None = None,
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Load the official DLS JSONL or an equivalent CSV into one data contract."""
    path = Path(path)
    frame = _read_table(path)
    column_map = column_map or {}
    renamed: dict[str, str] = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        explicit = target_column if canonical == "label" and target_column else column_map.get(canonical)
        source = explicit if explicit in frame.columns else next((name for name in aliases if name in frame.columns), None)
        if source:
            renamed[source] = canonical
    frame = frame.rename(columns=renamed)

    required = {"query", "organization_name", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {sorted(missing)}")

    for column in ["category", "address", "prices_summarized", "review_snippets", "permalink"]:
        if column not in frame:
            frame[column] = ""
    if "query_id" not in frame:
        frame["query_id"] = pd.factorize(frame["query"].fillna("").astype(str))[0]

    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(float)
    nearest = np.asarray(LABEL_VALUES)[np.abs(frame["label"].to_numpy()[:, None] - np.asarray(LABEL_VALUES)).argmin(axis=1)]
    if not np.allclose(frame["label"].to_numpy(), nearest, atol=1e-8):
        raise ValueError(f"Unexpected relevance values: {sorted(frame['label'].unique().tolist())}")
    frame["label"] = nearest
    frame["label_id"] = frame["label"].map({value: idx for idx, value in enumerate(LABEL_VALUES)}).astype(int)

    text_columns = ["query", "organization_name", "category", "address", "prices_summarized", "review_snippets"]
    frame[text_columns] = frame[text_columns].fillna("").astype(str)
    return frame.reset_index(drop=True)


def load_official_train_eval(train_path: str | Path, eval_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_relevance_data(train_path, target_column="relevance")
    evaluation = load_relevance_data(eval_path, target_column="relevance_new")
    return train, evaluation


def load_column_map(value: str | None) -> dict[str, str]:
    return json.loads(value) if value else {}
