from __future__ import annotations

import os

from openai import OpenAI

from .env_utils import first_openrouter_key
from .schemas import ToolCall


def try_instructor_tool_call(text: str) -> ToolCall | None:
    """Optional structured-output LLM router.

    The deterministic router is the reproducible default. Instructor is used
    only when `USE_INSTRUCTOR=1` and an LLM provider key is available.
    """
    api_key = first_openrouter_key()
    if not (os.getenv("USE_INSTRUCTOR") and api_key):
        return None
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    messages = [
        {
            "role": "system",
            "content": (
                "Emit a JSON tool call for a voice assistant. Available tools: "
                "split_bill(amount, people, tip_percent, currency), "
                "convert_units(value, from_unit, to_unit), or none."
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        import instructor

        raw_client = OpenAI(api_key=api_key, base_url=base_url)
        client = instructor.from_openai(raw_client)
        return client.chat.completions.create(
            model=model,
            response_model=ToolCall,
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
                    "content": "Return only JSON with keys `tool_name` and `arguments`.",
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return ToolCall.model_validate_json(response.choices[0].message.content)
    except Exception:
        return None
