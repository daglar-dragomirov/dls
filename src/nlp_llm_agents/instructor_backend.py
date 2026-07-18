from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .env_utils import first_openrouter_key
from .schemas import RelevanceDecision


def instructor_available() -> bool:
    return os.getenv("USE_INSTRUCTOR", "0") == "1" and bool(first_openrouter_key())


def try_instructor_decision(payload: dict[str, Any]) -> RelevanceDecision | None:
    """Request a schema-validated three-class judgement from OpenRouter."""
    if not instructor_available():
        return None
    api_key = first_openrouter_key()
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    system = (
        "Ты оцениваешь релевантность организации запросу на картах. "
        "Верни 1.0, если организация полностью удовлетворяет запросу; "
        "0.1, если она относится к теме, но важная часть запроса не подтверждена; "
        "0.0, если организация не соответствует запросу. "
        "Опирайся на карточку, найденные доказательства и похожие размеченные TRAIN-примеры."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": str(payload)},
    ]
    try:
        import instructor

        client = instructor.from_openai(OpenAI(api_key=api_key, base_url=base_url))
        return client.chat.completions.create(
            model=model,
            response_model=RelevanceDecision,
            messages=messages,
            max_retries=2,
            temperature=0,
        )
    except Exception:
        return None
