from __future__ import annotations

import re
from typing import Any

from .instructor_backend import try_instructor_tool_call
from .parser import to_json_call
from .tools import normalize_unit


NUMBER_RE = r"(\d+(?:[.,]\d+)?)"


def _number(value: str) -> float:
    return float(value.replace(",", "."))


def _extract_number_before(text: str, unit_span_start: int) -> float | None:
    prefix = text[:unit_span_start]
    matches = list(re.finditer(NUMBER_RE, prefix))
    if not matches:
        return None
    return _number(matches[-1].group(1))


def _extract_units_with_positions(text: str) -> list[tuple[str, int]]:
    units = []
    for match in re.finditer(r"[^\W\d_]+", text, flags=re.UNICODE):
        raw = match.group(0)
        try:
            normalized = normalize_unit(raw)
        except ValueError:
            continue
        if not any(unit == normalized for unit, _ in units):
            units.append((normalized, match.start()))
    return units


def _detect_currency(text: str) -> str:
    aliases = {
        "USD": ("usd", "$", "dollar", "dollars", "доллар"),
        "EUR": ("eur", "€", "euro", "евро"),
        "RUB": ("rub", "₽", "руб", "рубл", "ruble", "rubles"),
    }
    for currency, markers in aliases.items():
        if any(marker in text for marker in markers):
            return currency
    return "USD"


def predict_tool_call(text: str) -> dict[str, Any]:
    """Rule-based tool caller used as a transparent baseline/final router."""
    instructor_call = try_instructor_tool_call(text)
    if instructor_call is not None:
        return instructor_call.model_dump()

    text_l = text.lower()
    numbers = [_number(x) for x in re.findall(NUMBER_RE, text_l)]
    split_markers = (
        "split",
        "divide",
        "each person",
        "friends",
        "people",
        "bill",
        "paid",
        "pay",
        "каждый",
        "раздели",
        "счет",
        "счёт",
        "чаев",
        "людей",
        "человек",
    )
    convert_markers = ("convert", "how many", "translate", "into", "переведи", "сколько будет", " в ")

    if any(marker in text_l for marker in split_markers) and len(numbers) >= 2:
        amount = numbers[0] if numbers else 0.0
        people = int(numbers[1]) if len(numbers) > 1 else 2
        tip = numbers[2] if len(numbers) > 2 else 0.0
        if "tip" in text_l or "чаев" in text_l:
            tip_matches = re.findall(rf"{NUMBER_RE}\s*(?:percent|процент|%)", text_l)
            if tip_matches:
                tip = _number(tip_matches[-1])
        currency = _detect_currency(text_l)
        return {
            "tool_name": "split_bill",
            "arguments": {"amount": amount, "people": people, "tip_percent": tip, "currency": currency},
        }

    units_with_positions = _extract_units_with_positions(text_l)
    has_unit_pair = len(units_with_positions) >= 2
    if any(marker in text_l for marker in convert_markers) or has_unit_pair:
        if has_unit_pair:
            first_unit, first_pos = units_with_positions[0]
            second_unit, second_pos = units_with_positions[1]
            value_before_first = _extract_number_before(text_l, first_pos)
            value_before_second = _extract_number_before(text_l, second_pos)
            if value_before_first is None and value_before_second is not None:
                value = value_before_second
                from_unit = second_unit
                to_unit = first_unit
            else:
                value = value_before_first
                if value is None:
                    value = numbers[0] if numbers else 1.0
                from_unit = first_unit
                to_unit = second_unit
            return {
                "tool_name": "convert_units",
                "arguments": {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            }

    return {"tool_name": "none", "arguments": {}}


def model_like_response(text: str) -> str:
    """Return the same JSON shape an instruction-tuned model would be prompted to emit."""
    call = predict_tool_call(text)
    return to_json_call(call["tool_name"], call.get("arguments", {}))
