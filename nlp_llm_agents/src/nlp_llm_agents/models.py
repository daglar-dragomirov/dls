from __future__ import annotations

from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PUBLIC_COLUMNS = ["query", "organization_name", "category", "public_description", "public_tags"]


def public_text(frame: pd.DataFrame) -> pd.Series:
    return frame[PUBLIC_COLUMNS].fillna("").agg(" ".join, axis=1)


@dataclass
class BaselineRelevanceModel:
    pipeline: Pipeline

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "BaselineRelevanceModel":
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=80_000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=3.0,
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        )
        pipeline.fit(public_text(train), train["label"].astype(int))
        return cls(pipeline=pipeline)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(public_text(frame))[:, 1]

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump(self.pipeline, file)

    @classmethod
    def load(cls, path: str) -> "BaselineRelevanceModel":
        with open(path, "rb") as file:
            return cls(pipeline=pickle.load(file))
