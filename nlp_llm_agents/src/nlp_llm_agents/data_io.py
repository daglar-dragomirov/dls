from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CANONICAL_COLUMNS = {
    "query": ("query", "request", "search_query"),
    "organization_name": ("organization_name", "org_name", "name", "organization"),
    "category": ("category", "rubric", "org_category"),
    "public_description": ("public_description", "description", "org_description"),
    "public_tags": ("public_tags", "tags", "features"),
    "review_snippets": ("review_snippets", "reviews", "review_text", "evidence"),
    "hidden_tags": ("hidden_tags", "evidence_tags", "review_tags"),
    "label": ("label", "relevance", "target", "is_relevant"),
    "query_id": ("query_id", "request_id", "qid"),
}


def load_relevance_data(path: str | Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Load mentor CSV/JSONL and normalize it to the project data contract."""
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        frame = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        frame = pd.read_csv(path)
    column_map = column_map or {}
    renamed = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        explicit = column_map.get(canonical)
        source = explicit if explicit in frame.columns else next((name for name in aliases if name in frame.columns), None)
        if source:
            renamed[source] = canonical
    frame = frame.rename(columns=renamed)
    required = {"query", "organization_name", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {sorted(missing)}")
    for column in ["category", "public_description", "public_tags", "review_snippets", "hidden_tags"]:
        if column not in frame:
            frame[column] = ""
    if "query_id" not in frame:
        frame["query_id"] = pd.factorize(frame["query"].astype(str))[0]
    frame["label"] = frame["label"].astype(int)
    return frame


def load_column_map(value: str | None) -> dict[str, str]:
    return json.loads(value) if value else {}

