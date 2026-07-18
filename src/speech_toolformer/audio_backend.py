from __future__ import annotations

import base64
import json
import os
from typing import Any

import requests

from .env_utils import first_openrouter_key
from .parser import extract_json, validate_tool_call


TOOL_SCHEMA = """Available calls:
1) split_bill(amount:number, people:integer, tip_percent:number, currency:USD|EUR|RUB)
2) convert_units(value:number, from_unit:string, to_unit:string)
3) none() when no tool is needed.
Return exactly one JSON object: {\"tool_name\": string, \"arguments\": object}."""

PROMPTS = {
    "zero_shot": "Listen carefully and return the required tool call as JSON.",
    "schema_tuned": (
        "You are a bilingual Russian-English voice tool router. " + TOOL_SCHEMA + "\n"
        "Do not invent missing numbers. A greeting, explanation request, or general question must use none."
    ),
}


def _headers() -> dict[str, str]:
    key = first_openrouter_key()
    if not key:
        raise RuntimeError("OpenRouter API key is not configured")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _validated_prediction(raw: str) -> dict[str, Any]:
    call = extract_json(raw)
    valid, reason = validate_tool_call(call)
    return {"raw": raw, "call": call, "valid": valid, "reason": reason}


def transcribe_audio(audio: bytes, audio_format: str, model: str | None = None) -> dict[str, Any]:
    payload = {
        "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": audio_format},
        "model": model or os.getenv("OPENROUTER_STT_MODEL", "openai/gpt-4o-mini-transcribe"),
        "temperature": 0,
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/audio/transcriptions", headers=_headers(), json=payload, timeout=180
    )
    response.raise_for_status()
    body = response.json()
    return {"text": body["text"], "usage": body.get("usage", {})}


def direct_audio_tool_call(
    audio: bytes,
    audio_format: str,
    prompt_variant: str = "schema_tuned",
    model: str | None = None,
) -> dict[str, Any]:
    prompt = PROMPTS[prompt_variant]
    payload = {
        "model": model or os.getenv("OPENROUTER_AUDIO_MODEL", "openai/gpt-audio-mini"),
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Route this spoken request."},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(audio).decode("ascii"),
                            "format": audio_format,
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", headers=_headers(), json=payload, timeout=180
    )
    response.raise_for_status()
    body = response.json()
    result = _validated_prediction(body["choices"][0]["message"]["content"])
    result["usage"] = body.get("usage", {})
    result["backend"] = payload["model"]
    return result


def text_llm_tool_call(text: str, prompt_variant: str = "schema_tuned", model: str | None = None) -> dict[str, Any]:
    payload = {
        "model": model or os.getenv("OPENROUTER_TEXT_MODEL", "openai/gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": PROMPTS[prompt_variant]},
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", headers=_headers(), json=payload, timeout=120
    )
    response.raise_for_status()
    body = response.json()
    result = _validated_prediction(body["choices"][0]["message"]["content"])
    result["usage"] = body.get("usage", {})
    result["backend"] = payload["model"]
    return result
