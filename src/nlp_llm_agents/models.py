from __future__ import annotations

from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data_io import LABEL_VALUES


TEXT_COLUMNS = ["query", "organization_name", "category", "address", "prices_summarized", "review_snippets"]


def relevance_text(frame: pd.DataFrame) -> pd.Series:
    data = frame.reindex(columns=TEXT_COLUMNS, fill_value="").fillna("").astype(str).copy()
    data["review_snippets"] = data["review_snippets"].str.slice(0, 4_000)
    return (
        "запрос: " + data["query"]
        + " [ORG] название: " + data["organization_name"]
        + " [RUBRIC] " + data["category"]
        + " [ADDRESS] " + data["address"]
        + " [PRICES] " + data["prices_summarized"]
        + " [REVIEWS] " + data["review_snippets"]
    )


@dataclass
class BaselineRelevanceModel:
    pipeline: Pipeline

    @classmethod
    def fit(cls, train: pd.DataFrame, *, c: float = 2.5) -> "BaselineRelevanceModel":
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.997,
                        max_features=120_000,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        pipeline = Pipeline(
            [
                ("features", features),
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        max_iter=600,
                        solver="liblinear",
                        random_state=2026,
                    ),
                ),
            ]
        )
        pipeline.fit(relevance_text(train), train["label"].astype(str))
        return cls(pipeline=pipeline)

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].classes_.astype(float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(relevance_text(frame))
        aligned = np.zeros((len(frame), len(LABEL_VALUES)), dtype=np.float64)
        for source_idx, value in enumerate(self.classes_):
            target_idx = LABEL_VALUES.index(float(value))
            aligned[:, target_idx] = raw[:, source_idx]
        return aligned

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(LABEL_VALUES)[self.predict_proba(frame).argmax(axis=1)]

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump(self.pipeline, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "BaselineRelevanceModel":
        with open(path, "rb") as file:
            return cls(pipeline=pickle.load(file))
