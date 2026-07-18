from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SplitBillArgs(BaseModel):
    amount: float = Field(..., ge=0)
    people: int = Field(..., ge=1)
    tip_percent: float = Field(0.0, ge=0)
    currency: str = "USD"


class ConvertUnitsArgs(BaseModel):
    value: float
    from_unit: str
    to_unit: str


class ToolCall(BaseModel):
    tool_name: Literal["split_bill", "convert_units", "none"]
    arguments: dict = Field(default_factory=dict)

