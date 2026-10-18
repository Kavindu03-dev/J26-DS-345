"""Convert xLAM rows into chat-format training examples.

xLAM row (Salesforce/xlam-function-calling-60k):
    query   : str                       the user's request
    tools   : JSON string, list of      {"name", "description", "parameters":
                                          {param: {"type": "str", "description", "default"?}}}
    answers : JSON string, list of      {"name", "arguments": {...}}

Output example (what the trainer reads):
    tools    : OpenAI-style tool list   [{"type": "function", "function": {name, description,
                                          parameters: JSON Schema}}]
    messages : [{"role": "user", ...}, {"role": "assistant", "tool_calls": [...]}]

Qwen2.5 and Llama-3.2 chat templates both accept this format through
tokenizer.apply_chat_template(messages, tools=tools), so the same files train
every model in the study.
"""

from __future__ import annotations

import json
import re
from typing import Any

# xLAM uses Python-style type names. Map them to JSON Schema types.
_SIMPLE_TYPES = {
    "str": "string", "string": "string", "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "double": "number", "bool": "boolean",
    "boolean": "boolean", "dict": "object", "object": "object", "list": "array",
    "tuple": "array", "set": "array", "array": "array",
}


class ConversionError(ValueError):
    """Raised when a row is malformed; the caller skips that row."""


def map_type(xlam_type: str) -> tuple[dict[str, Any], bool]:
    """Return (JSON Schema fragment, optional?) for an xLAM type string.

    Examples: "str" -> string; "int, optional" -> integer + optional;
    "List[int]" -> array of integer; "Dict[str, Any]" -> object.
    """
    raw = (xlam_type or "").strip()
    optional = bool(re.search(r"\boptional\b", raw, re.IGNORECASE))
    t = re.sub(r",?\s*optional\b", "", raw, flags=re.IGNORECASE).strip()
    t = re.sub(r"^Optional\[(.*)\]$", r"\1", t).strip()

    m = re.match(r"^(List|list|Tuple|tuple|Set|set|Sequence)\[(.*)\]$", t)
    if m:
        inner, _ = map_type(m.group(2).split(",")[0])
        return {"type": "array", "items": inner}, optional
    if re.match(r"^(Dict|dict|Mapping)\b", t):
        return {"type": "object"}, optional
    if t.startswith("Union["):
        return {"type": "string"}, optional
    base = t.split("[")[0].strip().lower()
    return {"type": _SIMPLE_TYPES.get(base, "string")}, optional


def convert_tool(tool: dict) -> dict:
    """xLAM tool -> OpenAI-style function tool with a JSON Schema."""
    if not isinstance(tool, dict) or not tool.get("name"):
        raise ConversionError("tool without a name")
    properties, required = {}, []
    for pname, spec in (tool.get("parameters") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        schema, optional = map_type(str(spec.get("type", "str")))
        if spec.get("description"):
            schema["description"] = str(spec["description"])
        has_default = "default" in spec and spec["default"] not in (None, "")
        if has_default:
            schema["default"] = spec["default"]
        properties[pname] = schema
        if not optional and not has_default:
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": str(tool.get("description", "")),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ConversionError(f"invalid JSON: {e}") from e
    return value


def convert_row(row: dict) -> dict:
    """Convert one xLAM row. Raises ConversionError if the row is not usable.

    Quality filter: every call must name a provided tool and use only that
    tool's parameters, and every required parameter must be present. Rows that
    fail are dropped rather than repaired.
    """
    query = (row.get("query") or "").strip()
    if not query:
        raise ConversionError("empty query")
    raw_tools, raw_answers = _loads(row.get("tools")), _loads(row.get("answers"))
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ConversionError("no tools")
    if not isinstance(raw_answers, list) or not raw_answers:
        raise ConversionError("no answers")

    tools = [convert_tool(t) for t in raw_tools]
    by_name = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    if len(by_name) != len(tools):
        raise ConversionError("duplicate tool names")

    calls = []
    for ans in raw_answers:
        if not isinstance(ans, dict) or ans.get("name") not in by_name:
            raise ConversionError(f"call to unknown tool {ans!r:.60}")
        args = ans.get("arguments") or {}
        if not isinstance(args, dict):
            raise ConversionError("arguments is not an object")
        params = by_name[ans["name"]]
        unknown = set(args) - set(params["properties"])
        if unknown:
            raise ConversionError(f"unknown argument(s) {sorted(unknown)}")
        missing = set(params["required"]) - set(args)
        if missing:
            raise ConversionError(f"missing required argument(s) {sorted(missing)}")
        calls.append({"type": "function", "function": {"name": ans["name"], "arguments": args}})

    return {
        "source_id": str(row.get("id", "")),
        "kind": "tool_call" if len(calls) == 1 else "parallel_tool_call",
        "tools": tools,
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ],
    }
