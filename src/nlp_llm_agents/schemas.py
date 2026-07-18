from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RelevanceDecision(BaseModel):
    relevance: Literal[0.0, 0.1, 1.0]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_summary: str
    rationale: str


class ToolTrace(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
