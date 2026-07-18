from __future__ import annotations

from typing import Any

from .asr import TranscriptASR
from .parser import extract_json, validate_tool_call
from .router import model_like_response
from .tools import execute_tool


def text_pipeline(text: str) -> dict[str, Any]:
    raw = model_like_response(text)
    call = extract_json(raw)
    valid, reason = validate_tool_call(call)
    if not valid:
        return {"raw": raw, "call": call, "valid": False, "reason": reason, "answer": "I could not produce a valid tool call."}
    if call["tool_name"] == "none":
        return {"raw": raw, "call": call, "valid": True, "reason": reason, "answer": "No tool is needed for this request."}
    result = execute_tool(call)
    return {"raw": raw, "call": call, "valid": True, "reason": reason, "tool_result": result.result, "answer": result.answer}


def cascaded_audio_pipeline(transcript: str, asr: TranscriptASR | None = None) -> dict[str, Any]:
    asr = asr or TranscriptASR(word_drop_prob=0.0)
    recognized = asr.transcribe(transcript)
    result = text_pipeline(recognized)
    result["transcript"] = transcript
    result["recognized_text"] = recognized
    return result

