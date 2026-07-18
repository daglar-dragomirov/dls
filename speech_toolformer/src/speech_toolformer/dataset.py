from __future__ import annotations

import random
from typing import Any

import pandas as pd


EN_SPLIT_TEMPLATES = (
    "Split a {currency} {amount} bill between {people} people with {tip}% tip",
    "We paid {amount} {currency}; divide it for {people} friends and add a {tip} percent tip",
    "How much should each person pay if dinner is {amount} {currency}, {people} people, tip {tip} percent",
)

RU_SPLIT_TEMPLATES = (
    "Раздели счет {amount} {currency} на {people} человек с чаевыми {tip} процентов",
    "Сколько платит каждый если счет {amount} {currency}, людей {people}, чаевые {tip} процентов",
)

EN_CONVERT_TEMPLATES = (
    "Convert {value} {from_unit} to {to_unit}",
    "How many {to_unit} are in {value} {from_unit}",
    "Please translate {value} {from_unit} into {to_unit}",
)

RU_CONVERT_TEMPLATES = (
    "Переведи {value} {from_unit} в {to_unit}",
    "Сколько будет {value} {from_unit} в {to_unit}",
)

NO_TOOL_TEMPLATES = (
    "Tell me a short fun fact about machine learning",
    "Скажи коротко, что такое градиентный спуск",
    "What can you do as a voice assistant",
    "Просто поздоровайся и ничего не вызывай",
)

UNIT_PAIRS = (
    ("kilometers", "miles"),
    ("miles", "kilometers"),
    ("meters", "feet"),
    ("feet", "meters"),
    ("kilograms", "pounds"),
    ("pounds", "kilograms"),
)


def generate_dataset(n: int = 300, seed: int = 2026) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        p = rng.random()
        if p < 0.45:
            amount = rng.choice([35, 48, 72, 96, 120, 180, 250])
            people = rng.randint(2, 8)
            tip = rng.choice([0, 5, 10, 12, 15, 18])
            currency = rng.choice(["USD", "EUR", "RUB"])
            template = rng.choice(EN_SPLIT_TEMPLATES + RU_SPLIT_TEMPLATES)
            text = template.format(amount=amount, people=people, tip=tip, currency=currency)
            call = {
                "tool_name": "split_bill",
                "arguments": {"amount": amount, "people": people, "tip_percent": tip, "currency": currency},
            }
        elif p < 0.82:
            value = rng.choice([3, 5, 7.5, 10, 12, 21, 42, 100])
            from_unit, to_unit = rng.choice(UNIT_PAIRS)
            template = rng.choice(EN_CONVERT_TEMPLATES + RU_CONVERT_TEMPLATES)
            text = template.format(value=value, from_unit=from_unit, to_unit=to_unit)
            call = {
                "tool_name": "convert_units",
                "arguments": {"value": value, "from_unit": from_unit, "to_unit": to_unit},
            }
        else:
            text = rng.choice(NO_TOOL_TEMPLATES)
            call = {"tool_name": "none", "arguments": {}}
        rows.append({"id": i, "text": text, "expected_call": call, "needs_tool": int(call["tool_name"] != "none")})
    return pd.DataFrame(rows)

