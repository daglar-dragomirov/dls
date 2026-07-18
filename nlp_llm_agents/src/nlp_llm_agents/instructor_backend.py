from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .env_utils import first_openrouter_key
from .schemas import RelevanceDecision


def instructor_available() -> bool:
    return bool(os.getenv("USE_INSTRUCTOR")) and bool(first_openrouter_key())


def try_instructor_decision(payload: dict[str, Any]) -> RelevanceDecision | None:
    """Optional structured LLM scorer.

    The project remains fully reproducible without paid APIs. When `USE_INSTRUCTOR=1`
    and `OPENAI_API_KEY` are present, this backend demonstrates schema-first LLM
    output using the Instructor library.
    """

    if not instructor_available():
        return None
    api_key = first_openrouter_key()
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    messages = [
        {
            "role": "system",
            "content": (
                "Score whether the organization is relevant to the map query. "
                "Use the evidence and return a calibrated score from 0 to 1."
            ),
        },
        {"role": "user", "content": str(payload)},
    ]
    try:
        import instructor

        raw_client = OpenAI(api_key=api_key, base_url=base_url)
        client = instructor.from_openai(raw_client)
        return client.chat.completions.create(
            model=model,
            response_model=RelevanceDecision,
            messages=messages,
            max_retries=2,
            temperature=0,
        )
    except Exception:
        pass

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Return only JSON with keys: score, decision, required_tags, "
                        "evidence_summary, rationale."
                    ),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return RelevanceDecision.model_validate_json(response.choices[0].message.content)
    except Exception:
        return None

