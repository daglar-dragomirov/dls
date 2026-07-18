from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    answer: str


def split_bill(amount: float, people: int, tip_percent: float = 0.0, currency: str = "USD") -> ToolResult:
    people = max(int(people), 1)
    total = float(amount) * (1.0 + float(tip_percent) / 100.0)
    per_person = total / people
    result = {
        "total": round(total, 2),
        "per_person": round(per_person, 2),
        "currency": currency.upper(),
    }
    answer = (
        f"Total with tip is {result['total']} {result['currency']}. "
        f"Each person pays {result['per_person']} {result['currency']}."
    )
    return ToolResult(
        tool_name="split_bill",
        arguments={
            "amount": float(amount),
            "people": people,
            "tip_percent": float(tip_percent),
            "currency": currency.upper(),
        },
        result=result,
        answer=answer,
    )


UNIT_FACTORS = {
    ("kilometers", "miles"): 0.621371,
    ("miles", "kilometers"): 1.60934,
    ("meters", "feet"): 3.28084,
    ("feet", "meters"): 0.3048,
    ("kilograms", "pounds"): 2.20462,
    ("pounds", "kilograms"): 0.453592,
}


def convert_units(value: float, from_unit: str, to_unit: str) -> ToolResult:
    from_unit = normalize_unit(from_unit)
    to_unit = normalize_unit(to_unit)
    if (from_unit, to_unit) == ("fahrenheit", "celsius"):
        converted = (float(value) - 32.0) * 5.0 / 9.0
    elif (from_unit, to_unit) == ("celsius", "fahrenheit"):
        converted = float(value) * 9.0 / 5.0 + 32.0
    else:
        factor = UNIT_FACTORS[(from_unit, to_unit)]
        converted = float(value) * factor
    result = {"value": round(converted, 4), "unit": to_unit}
    answer = f"{value:g} {from_unit} is {result['value']:g} {to_unit}."
    return ToolResult(
        tool_name="convert_units",
        arguments={"value": float(value), "from_unit": from_unit, "to_unit": to_unit},
        result=result,
        answer=answer,
    )


def normalize_unit(unit: str) -> str:
    unit = unit.lower().strip()
    aliases = {
        "km": "kilometers",
        "kilometer": "kilometers",
        "kilometers": "kilometers",
        "километр": "kilometers",
        "километра": "kilometers",
        "километры": "kilometers",
        "километров": "kilometers",
        "mile": "miles",
        "miles": "miles",
        "миля": "miles",
        "милю": "miles",
        "мили": "miles",
        "миль": "miles",
        "mi": "miles",
        "meter": "meters",
        "meters": "meters",
        "m": "meters",
        "метр": "meters",
        "метра": "meters",
        "метров": "meters",
        "feet": "feet",
        "foot": "feet",
        "ft": "feet",
        "фут": "feet",
        "фута": "feet",
        "футов": "feet",
        "футах": "feet",
        "kg": "kilograms",
        "kilogram": "kilograms",
        "kilograms": "kilograms",
        "килограмм": "kilograms",
        "килограмма": "kilograms",
        "килограммов": "kilograms",
        "pound": "pounds",
        "pounds": "pounds",
        "lb": "pounds",
        "lbs": "pounds",
        "фунт": "pounds",
        "фунта": "pounds",
        "фунтов": "pounds",
        "фунтах": "pounds",
        "celsius": "celsius",
        "centigrade": "celsius",
        "цельсий": "celsius",
        "цельсия": "celsius",
        "fahrenheit": "fahrenheit",
        "фаренгейт": "fahrenheit",
        "фаренгейта": "fahrenheit",
        "фаренгейты": "fahrenheit",
        "фаренгейтов": "fahrenheit",
    }
    if unit not in aliases:
        raise ValueError(f"Unsupported unit: {unit}")
    return aliases[unit]


def execute_tool(call: dict[str, Any]) -> ToolResult:
    name = call.get("tool_name")
    args = call.get("arguments", {})
    if name == "split_bill":
        return split_bill(**args)
    if name == "convert_units":
        return convert_units(**args)
    raise ValueError(f"Unknown tool: {name}")
