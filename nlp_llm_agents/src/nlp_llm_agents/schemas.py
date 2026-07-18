from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RelevanceDecision(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    decision: Literal[0, 1]
    required_tags: list[str]
    evidence_summary: str
    rationale: str


class ToolTrace(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]

