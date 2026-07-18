from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .data_io import LABEL_VALUES
from .instructor_backend import try_instructor_decision
from .models import BaselineRelevanceModel, relevance_text


RUSSIAN_STOPWORDS = {
    "а", "без", "в", "во", "для", "до", "и", "или", "из", "к", "как", "на", "не", "но",
    "о", "от", "по", "под", "при", "с", "со", "у", "что", "это", "где", "рядом", "найти",
}


@dataclass
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class CardEvidenceSearchTool:
    """Search query terms in the supplied public organization card."""

    @staticmethod
    def _terms(query: str) -> list[str]:
        words = re.findall(r"[a-zа-яё0-9-]+", query.lower())
        return [word for word in words if len(word) >= 3 and word not in RUSSIAN_STOPWORDS]

    def search(self, row: pd.Series) -> dict[str, Any]:
        terms = self._terms(str(row.get("query", "")))
        sources = {
            "rubric": str(row.get("category", "")),
            "prices": str(row.get("prices_summarized", "")),
            "reviews": str(row.get("review_snippets", "")),
        }
        hits: dict[str, list[str]] = {}
        excerpts: list[str] = []
        for source, text in sources.items():
            text_lower = text.lower()
            matched = [term for term in terms if term in text_lower]
            hits[source] = matched
            if matched and text:
                excerpts.append(f"{source}: {text[:700]}")
        organization = quote_plus(str(row.get("organization_name", "")))
        query = quote_plus(str(row.get("query", "")))
        return {
            "query_terms": terms,
            "matched_terms": hits,
            "evidence_excerpts": excerpts[:3],
            "search_url": f"https://yandex.ru/search/?text={organization}+{query}",
            "maps_permalink": str(row.get("permalink", "")),
        }


class PublicWebSearchTool:
    """Run a small public web search and return an auditable result trace."""

    endpoint = "https://www.bing.com/search"

    def search(self, query: str, *, limit: int = 5, timeout: float = 7.0) -> dict[str, Any]:
        search_url = f"{self.endpoint}?q={quote_plus(query)}"
        request = Request(
            search_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DLS-Relevance-Agent/1.0)"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read(1_000_000).decode("utf-8", errors="ignore")
            results = []
            matches = re.findall(
                r'<li[^>]+class="[^"]*b_algo[^"]*".*?<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for url, title in matches:
                clean_title = unescape(re.sub(r"<[^>]+>", "", title)).strip()
                results.append({"title": clean_title, "url": unescape(url)})
                if len(results) >= limit:
                    break
            return {"status": "ok", "query": query, "search_url": search_url, "results": results}
        except Exception as exc:
            return {
                "status": "unavailable",
                "query": query,
                "search_url": search_url,
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
            }


class SimilarTrainExamplesTool:
    """Retrieve labeled examples exclusively from the official train split."""

    def __init__(self, train: pd.DataFrame, *, max_features: int = 70_000, k: int = 9):
        self.train = train.reset_index(drop=True).copy()
        self.k = min(k, len(self.train))
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(relevance_text(self.train).str.slice(0, 2_500))
        self.index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        self.index.fit(self.matrix)

    def search_many(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        query_matrix = self.vectorizer.transform(relevance_text(frame).str.slice(0, 2_500))
        distances, indices = self.index.kneighbors(query_matrix, n_neighbors=self.k)
        probabilities = np.zeros((len(frame), len(LABEL_VALUES)), dtype=np.float64)
        traces: list[list[dict[str, Any]]] = []
        for row_idx, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            weights = np.maximum(1.0 - row_distances, 1e-4) ** 3
            examples = []
            for distance, index, weight in zip(row_distances, row_indices, weights):
                label = float(self.train.iloc[index]["label"])
                probabilities[row_idx, LABEL_VALUES.index(label)] += float(weight)
                examples.append(
                    {
                        "similarity": round(float(1.0 - distance), 4),
                        "query": str(self.train.iloc[index]["query"]),
                        "organization": str(self.train.iloc[index]["organization_name"]),
                        "rubric": str(self.train.iloc[index]["category"]),
                        "relevance": label,
                    }
                )
            total = probabilities[row_idx].sum()
            probabilities[row_idx] = probabilities[row_idx] / total if total else np.full(len(LABEL_VALUES), 1 / 3)
            traces.append(examples)
        return probabilities, traces


class RelevanceAgent:
    def __init__(
        self,
        baseline: BaselineRelevanceModel,
        reference_train: pd.DataFrame | None = None,
        *,
        retrieval_weight: float = 0.25,
    ):
        self.baseline = baseline
        self.evidence_tool = CardEvidenceSearchTool()
        self.web_search_tool = PublicWebSearchTool()
        self.retrieval_weight = float(retrieval_weight)
        self.retrieval_tool = SimilarTrainExamplesTool(reference_train) if reference_train is not None and len(reference_train) else None

    def score_frame(
        self,
        frame: pd.DataFrame,
        *,
        use_llm: bool = False,
        use_web_search: bool = False,
    ) -> pd.DataFrame:
        frame = frame.reset_index(drop=True)
        baseline_probabilities = self.baseline.predict_proba(frame)
        if self.retrieval_tool is not None:
            retrieval_probabilities, retrieved = self.retrieval_tool.search_many(frame)
        else:
            retrieval_probabilities = baseline_probabilities.copy()
            retrieved = [[] for _ in range(len(frame))]
        combined = (1.0 - self.retrieval_weight) * baseline_probabilities + self.retrieval_weight * retrieval_probabilities

        rows: list[dict[str, Any]] = []
        for idx, row in frame.iterrows():
            evidence = self.evidence_tool.search(row)
            web_evidence = None
            if use_web_search:
                web_query = f"{row.get('organization_name', '')} {row.get('query', '')}".strip()
                web_evidence = self.web_search_tool.search(web_query)
            prediction_idx = int(combined[idx].argmax())
            predicted = float(LABEL_VALUES[prediction_idx])
            tool_calls = [
                ToolCall("search_card_evidence", {"query": row["query"]}, evidence).__dict__,
                ToolCall("retrieve_similar_train_examples", {"k": len(retrieved[idx])}, {"examples": retrieved[idx]}).__dict__,
            ]
            if use_web_search:
                tool_calls.append(
                    ToolCall("public_web_search", {"query": web_query}, web_evidence or {}).__dict__
                )
            backend = "baseline_retrieval_agent"
            llm_review = None
            if use_llm:
                decision = try_instructor_decision(
                    {
                        "query": row.get("query"),
                        "organization_card": {
                            "name": row.get("organization_name"),
                            "rubric": row.get("category"),
                            "address": row.get("address"),
                            "prices": str(row.get("prices_summarized", ""))[:1_500],
                        },
                        "search_evidence": evidence,
                        "public_web_search": web_evidence,
                        "similar_train_examples": retrieved[idx][:5],
                        "baseline_probabilities": dict(zip(map(str, LABEL_VALUES), baseline_probabilities[idx].round(4))),
                    }
                )
                if decision is not None:
                    llm_review = decision.model_dump()
                    predicted = float(decision.relevance)
                    prediction_idx = LABEL_VALUES.index(predicted)
                    combined[idx] = np.full(len(LABEL_VALUES), (1.0 - decision.confidence) / 2)
                    combined[idx, prediction_idx] = decision.confidence
                    combined[idx] /= combined[idx].sum()
                    backend = "structured_llm_agent"
            rows.append(
                {
                    "predicted_relevance": predicted,
                    "confidence": float(combined[idx, prediction_idx]),
                    "probability_0.0": float(combined[idx, 0]),
                    "probability_0.1": float(combined[idx, 1]),
                    "probability_1.0": float(combined[idx, 2]),
                    "used_search": True,
                    "used_web_search": use_web_search,
                    "tool_calls": tool_calls,
                    "backend": backend,
                    "llm_review": llm_review,
                    "rationale": (
                        f"Baseline={baseline_probabilities[idx].round(3).tolist()}, "
                        f"retrieval={retrieval_probabilities[idx].round(3).tolist()}, "
                        f"weight={self.retrieval_weight:.2f}; class={predicted}."
                    ),
                }
            )
        return pd.concat([frame, pd.DataFrame(rows)], axis=1)

    def score_one(
        self,
        row: pd.Series,
        *,
        use_llm: bool = False,
        use_web_search: bool = False,
    ) -> dict[str, Any]:
        result = self.score_frame(
            pd.DataFrame([row]),
            use_llm=use_llm,
            use_web_search=use_web_search,
        ).iloc[0]
        response = {key: result[key] for key in [
            "predicted_relevance", "confidence", "probability_0.0", "probability_0.1", "probability_1.0",
            "used_search", "used_web_search", "tool_calls", "backend", "llm_review", "rationale",
        ]}
        response["used_search"] = bool(response["used_search"])
        response["used_web_search"] = bool(response["used_web_search"])
        return response
