"""Load the versioned tool schemas and validate arguments/results. Nothing is coerced."""

import json
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator

from c4.common.config import PROJECT_ROOT

SCHEMA_VERSION = "v1"
# README 6.1: the research experiment uses Tools 1-4 only; Tool 5 is agent-only.
EXPERIMENT_TOOLS = ("get_review_insights", "recommend_hotels", "plan_itinerary", "web_search")
AGENT_TOOLS = (*EXPERIMENT_TOOLS, "search_knowledge_base")


class SchemaError(ValueError):
    def __init__(self, tool: str, errors: list[str]):
        super().__init__(f"{tool}: " + "; ".join(errors))
        self.tool = tool
        self.errors = errors


@lru_cache
def _load_all(version: str) -> dict[str, dict[str, Any]]:
    folder = PROJECT_ROOT / "tool_schemas" / version
    if not folder.is_dir():
        raise FileNotFoundError(f"No tool schemas for version {version!r}: {folder}")
    tools: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        tool = json.loads(path.read_text(encoding="utf-8"))
        if tool["name"] != path.stem:
            raise ValueError(f"{path.name}: tool name {tool['name']!r} must match the file name")
        Draft202012Validator.check_schema(tool["parameters"])
        if "returns" in tool:
            Draft202012Validator.check_schema(tool["returns"])
        tools[tool["name"]] = tool
    return tools


def load_tool(name: str, version: str = SCHEMA_VERSION) -> dict[str, Any]:
    tools = _load_all(version)
    if name not in tools:
        raise KeyError(f"Unknown tool {name!r}")
    return tools[name]


def load_tools(
    names: tuple[str, ...] = AGENT_TOOLS, version: str = SCHEMA_VERSION
) -> list[dict[str, Any]]:
    return [load_tool(name, version) for name in names]


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The shape chat templates expect. `returns` is not shown to the model."""
    return [
        {
            "type": "function",
            "function": {k: tool[k] for k in ("name", "description", "parameters")},
        }
        for tool in tools
    ]


def _errors(schema: dict[str, Any], value: Any) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    return [
        f"{e.json_path}: {e.message}"
        for e in sorted(validator.iter_errors(value), key=lambda e: e.json_path)
    ]


def validate_arguments(tool: str, arguments: Any, version: str = SCHEMA_VERSION) -> None:
    if errors := _errors(load_tool(tool, version)["parameters"], arguments):
        raise SchemaError(tool, errors)


def validate_result(tool: str, result: Any, version: str = SCHEMA_VERSION) -> None:
    schema = load_tool(tool, version).get("returns")
    if schema is None:
        raise KeyError(f"Tool {tool!r} has no 'returns' schema")
    if errors := _errors(schema, result):
        raise SchemaError(tool, errors)
