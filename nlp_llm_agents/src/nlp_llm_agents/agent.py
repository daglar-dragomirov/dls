from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from .data import INTENTS
from .instructor_backend import try_instructor_decision
from .models import BaselineRelevanceModel


TAG_SYNONYMS = {
    "terrace": ("веранд", "террас", "открыт", "terrace", "patio", "outdoor", "garden"),
    "restaurant": ("ресторан", "ужин", "restaurant", "dinner", "dining"),
    "cafe": ("кафе", "кофе", "кофейн", "cafe", "coffee", "coffeehouse"),
    "bar": ("бар", "паб", "bar", "pub"),
    "jazz": ("джаз", "саксофон", "jazz", "saxophone"),
    "romantic": ("романтич", "свидан", "уют", "romantic", "date", "cozy"),
    "kids_room": ("дет", "ребен", "семейн", "kids", "children", "family"),
    "wifi": ("wifi", "вайфай", "интернет"),
    "quiet": ("тих", "спокой", "quiet", "calm", "silent"),
    "pet_friendly": ("собак", "питом", "pet", "dog", "animals allowed"),
    "vegan": ("веган", "без мяса", "растительн", "vegan", "plant-based", "without meat"),
    "24h": ("ноч", "24", "круглосуточ", "late", "night", "24h", "24/7"),
    "sports_broadcasts": ("футбол", "спорт", "матч", "трансляц", "football", "sports", "broadcast", "match"),
    "pharmacy": ("аптек", "лекарств", "pharmacy", "medicine", "drugstore"),
    "power_outlets": ("розет", "заряд", "ноутбук", "outlet", "power", "laptop"),
    "good_coffee": ("хороший кофе", "зерно", "капучино", "coffee"),
    "long_stay": ("сидеть долго", "не торопят", "long stay", "work"),
    "big_screen": ("большой экран", "проектор", "телевизор", "big screen", "projector"),
    "cocktails": ("коктейл", "cocktail"),
    "live_music": ("живая музыка", "live music"),
    "beer": ("пиво", "крафт", "beer"),
}


@dataclass
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class EvidenceSearchTool:
    """Local search over organization reviews and hidden evidence."""

    def search(self, row: pd.Series, required_tags: list[str]) -> dict[str, Any]:
        text = f"{row.get('review_snippets', '')} {row.get('hidden_tags', '')} {row.get('public_tags', '')}".lower()
        hits = {}
        for tag in required_tags:
            synonyms = TAG_SYNONYMS.get(tag, (tag,))
            hits[tag] = any(syn.lower() in text for syn in synonyms) or tag.lower() in text
        matched = [tag for tag, ok in hits.items() if ok]
        return {
            "matched_required_tags": matched,
            "all_required_found": len(matched) == len(required_tags),
            "evidence_text": row.get("review_snippets", ""),
        }


class RelevanceAgent:
    def __init__(self, baseline: BaselineRelevanceModel):
        self.baseline = baseline
        self.search_tool = EvidenceSearchTool()

    @staticmethod
    def infer_intent(query: str):
        query_l = query.lower()
        for intent in INTENTS:
            if query in intent.queries or any(candidate.lower() == query_l for candidate in intent.queries):
                return intent
        required = set(RelevanceAgent.infer_required_tags(query))
        for intent in INTENTS:
            if set(intent.required).issubset(required):
                return intent
        best_intent = None
        best_score = 0.0
        for intent in INTENTS:
            overlap = len(required & set(intent.required))
            score = overlap / max(len(intent.required), 1)
            if score > best_score:
                best_intent = intent
                best_score = score
        return best_intent

    @staticmethod
    def infer_required_tags(query: str) -> list[str]:
        query_l = query.lower()
        inferred = []
        for tag, synonyms in TAG_SYNONYMS.items():
            if any(s in query_l for s in synonyms):
                inferred.append(tag)
        if not inferred:
            for intent in INTENTS:
                if query in intent.queries:
                    inferred.extend(intent.required)
                    break
        return sorted(set(inferred))

    @staticmethod
    def _public_evidence(row: pd.Series, required_tags: list[str]) -> tuple[list[str], list[str]]:
        text = f"{row.get('public_description', '')} {row.get('public_tags', '')} {row.get('category', '')}".lower()
        found, missing = [], []
        for tag in required_tags:
            synonyms = TAG_SYNONYMS.get(tag, (tag,))
            if tag.lower() in text or any(s.lower() in text for s in synonyms):
                found.append(tag)
            else:
                missing.append(tag)
        return found, missing

    @staticmethod
    def _tag_hits(text: str, tags: list[str] | tuple[str, ...]) -> list[str]:
        text_l = text.lower()
        hits = []
        for tag in tags:
            synonyms = TAG_SYNONYMS.get(tag, (tag,))
            if tag.lower() in text_l or any(syn.lower() in text_l for syn in synonyms):
                hits.append(tag)
        return hits

    def score_one(self, row: pd.Series, baseline_score: float | None = None) -> dict[str, Any]:
        if baseline_score is None:
            baseline_score = float(self.baseline.predict_proba(pd.DataFrame([row]))[0])
        intent = self.infer_intent(str(row["query"]))
        required = list(intent.required) if intent is not None else self.infer_required_tags(str(row["query"]))
        optional = list(intent.optional) if intent is not None else []
        negative = list(intent.negative) if intent is not None else []
        public_found, missing = self._public_evidence(row, required)
        trace: list[ToolCall] = []
        hidden_found: list[str] = []

        should_search = bool(missing) or 0.35 <= baseline_score <= 0.75
        if should_search:
            result = self.search_tool.search(row, required)
            trace.append(ToolCall(tool="search_evidence", arguments={"required_tags": required}, result=result))
            hidden_found = result["matched_required_tags"]

        evidence_text = (
            f"{row.get('public_description', '')} {row.get('public_tags', '')} "
            f"{row.get('review_snippets', '')} {row.get('hidden_tags', '')}"
        )
        required_hits = set(public_found) | set(hidden_found)
        optional_hits = set(self._tag_hits(evidence_text, optional))
        negative_hits = set(self._tag_hits(evidence_text, negative))
        required_coverage = len(required_hits) / max(len(required), 1)
        optional_coverage = len(optional_hits) / max(len(optional), 1) if optional else 0.0
        evidence_score = 0.78 * required_coverage + 0.22 * optional_coverage - 0.35 * len(negative_hits)
        evidence_score = float(np.clip(evidence_score, 0.0, 1.0))
        final_score = 0.38 * baseline_score + 0.62 * evidence_score
        if required and required_coverage == 1.0:
            final_score += 0.06
        if required and required_coverage < 0.5:
            final_score -= 0.12
        if negative_hits:
            final_score -= 0.22 * len(negative_hits)
        final_score = float(np.clip(final_score, 0.0, 1.0))

        rationale = (
            f"Required tags: {required}. Optional hits: {sorted(optional_hits)}. "
            f"Negative hits: {sorted(negative_hits)}. Public evidence: {public_found}. "
            f"Tool evidence: {hidden_found}. Baseline={baseline_score:.3f}, "
            f"evidence={evidence_score:.3f}, final={final_score:.3f}."
        )
        deterministic = {
            "score": final_score,
            "decision": int(final_score >= 0.5),
            "required_tags": required,
            "tool_calls": [call.__dict__ for call in trace],
            "rationale": rationale,
            "used_search": bool(trace),
            "backend": "deterministic_agent",
        }
        llm_decision = try_instructor_decision(
            {
                "query": row.get("query"),
                "organization": row.get("organization_name"),
                "public_fields": {
                    "category": row.get("category"),
                    "description": row.get("public_description"),
                    "tags": row.get("public_tags"),
                },
                "tool_evidence": [call.__dict__ for call in trace],
                "deterministic_decision": deterministic,
            }
        )
        if llm_decision is not None:
            llm_score = float(llm_decision.score)
            llm_binary = int(llm_decision.decision)
            deterministic["llm_review"] = {
                "score": llm_score,
                "decision": llm_binary,
                "required_tags": llm_decision.required_tags,
                "rationale": llm_decision.rationale,
            }
            if llm_binary == deterministic["decision"] and abs(llm_score - deterministic["score"]) <= 0.25:
                deterministic.update(
                    {
                        "score": float(np.clip(0.7 * deterministic["score"] + 0.3 * llm_score, 0.0, 1.0)),
                        "rationale": f"{deterministic['rationale']} LLM review: {llm_decision.rationale}",
                        "backend": "deterministic_agent_with_llm_review",
                    }
                )
            else:
                deterministic["backend"] = "deterministic_agent_llm_guarded"
        return deterministic

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        baseline_scores = self.baseline.predict_proba(frame)
        rows = []
        for (_, row), baseline_score in zip(frame.iterrows(), baseline_scores):
            result = self.score_one(row, float(baseline_score))
            rows.append(
                {
                    "agent_score": result["score"],
                    "agent_decision": result["decision"],
                    "used_search": result["used_search"],
                    "agent_rationale": result["rationale"],
                    "tool_calls": result["tool_calls"],
                    "backend": result["backend"],
                    "llm_review": result.get("llm_review"),
                }
            )
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
