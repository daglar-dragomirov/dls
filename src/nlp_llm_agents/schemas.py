from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class RelevanceDecision(BaseModel):
    relevance: Literal[0.0, 0.1, 1.0]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_summary: str
    rationale: str


class ToolTrace(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class SearchPlan(BaseModel):
    needs_search: bool
    query: str = Field(max_length=400, description="Concrete web search query; empty only when needs_search is false")
    reason: str

    @model_validator(mode="after")
    def require_search_query(self):
        if self.needs_search and not self.query.strip():
            raise ValueError("A search plan must include a nonempty query")
        return self


class SemanticDecision(BaseModel):
    evidence_summary: str = Field(min_length=1, max_length=3000)
    rationale: str = Field(min_length=1, max_length=3000)
    verdict: Literal["IRRELEVANT", "PARTIAL", "RELEVANT"] = Field(
        description="IRRELEVANT: explicit mismatch; PARTIAL: same topic but unconfirmed constraint; RELEVANT: all requirements supported"
    )
    confidence: float = Field(ge=0, le=1)
