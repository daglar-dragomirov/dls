from __future__ import annotations

import json
import re
from typing import Any


TOOL_SCHEMA = {
    "split_bill": {"required": {"amount", "people"}, "optional": {"tip_percent", "currency"}},
    "convert_units": {"required": {"value", "from_unit", "to_unit"}, "optional": set()},
}


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        repaired = match.group(0).replace("'", '"')
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def validate_tool_call(call: dict[str, Any] | None) -> tuple[bool, str]:
    if call is None:
        return False, "no_json"
    name = call.get("tool_name")
    if name == "none":
        return True, "no_tool"
    if name not in TOOL_SCHEMA:
        return False, "unknown_tool"
    args = call.get("arguments")
    if not isinstance(args, dict):
        return False, "bad_arguments"
    missing = TOOL_SCHEMA[name]["required"] - set(args)
    if missing:
        return False, f"missing:{sorted(missing)}"
    return True, "ok"


def to_json_call(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    return json.dumps({"tool_name": tool_name, "arguments": arguments or {}}, ensure_ascii=False)

