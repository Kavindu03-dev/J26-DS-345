import copy
import json

import pytest
from jsonschema import Draft202012Validator

from c4.common.config import PROJECT_ROOT
from c4.common.schemas import (
    AGENT_TOOLS,
    EXPERIMENT_TOOLS,
    SchemaError,
    load_tool,
    load_tools,
    to_openai_tools,
    validate_arguments,
    validate_result,
)

VALID = {
    "get_review_insights": {"hotel_id": "NE-001", "aspects": ["cleanliness", "wifi"]},
    "recommend_hotels": {
        "destination": "Nuwara Eliya",
        "check_in": "2026-12-20",
        "nights": 3,
        "guests": 4,
        "budget_min_lkr": 15000,
        "budget_max_lkr": 30000,
        "interests": ["tea plantations"],
        "travel_purpose": "family",
        "accessibility_needs": ["wheelchair"],
        "top_k": 3,
    },
    "plan_itinerary": {
        "hotel_id": "NE-001",
        "days": 3,
        "daily_hours": 8,
        "interests": ["nature"],
        "include_restaurants": True,
        "cuisine": "sri lankan",
    },
    "web_search": {"query": "weather in Ella tomorrow", "max_results": 3},
    "search_knowledge_base": {"query": "ETA rules", "location": "Colombo", "top_k": 4},
}
OPTIONAL = {
    "get_review_insights": {"aspects"},
    "recommend_hotels": {"accessibility_needs"},
    "plan_itinerary": {"cuisine"},
    "web_search": {"max_results"},
    "search_knowledge_base": {"location", "top_k"},
}
REQUIRED_CASES = [(t, f) for t, args in VALID.items() for f in args if f not in OPTIONAL[t]]
ALL_FIELDS = [(t, f) for t, args in VALID.items() for f in args]


def test_schema_folder_matches_tool_list():
    files = {p.stem for p in (PROJECT_ROOT / "tool_schemas" / "v1").glob("*.json")}
    assert files == set(AGENT_TOOLS) == set(VALID)


def test_experiment_uses_tools_1_to_4_only():
    assert len(EXPERIMENT_TOOLS) == 4
    assert "search_knowledge_base" not in EXPERIMENT_TOOLS


@pytest.mark.parametrize("name", AGENT_TOOLS)
def test_is_valid_json_schema(name):
    path = PROJECT_ROOT / "tool_schemas" / "v1" / f"{name}.json"
    tool = json.loads(path.read_text(encoding="utf-8"))
    assert tool["name"] == name
    assert tool["description"]
    Draft202012Validator.check_schema(tool["parameters"])
    assert tool["parameters"]["additionalProperties"] is False
    if name != "web_search":
        Draft202012Validator.check_schema(tool["returns"])


@pytest.mark.parametrize("name", AGENT_TOOLS)
def test_valid_arguments_pass(name):
    validate_arguments(name, VALID[name])
    required_only = {k: v for k, v in VALID[name].items() if k not in OPTIONAL[name]}
    validate_arguments(name, required_only)


@pytest.mark.parametrize(("name", "field"), REQUIRED_CASES)
def test_missing_required_argument_fails(name, field):
    args = {k: v for k, v in VALID[name].items() if k != field}
    with pytest.raises(SchemaError, match=field):
        validate_arguments(name, args)


@pytest.mark.parametrize(("name", "field"), ALL_FIELDS)
def test_wrong_type_fails(name, field):
    args = copy.deepcopy(VALID[name])
    args[field] = {"not": "this type"}
    with pytest.raises(SchemaError):
        validate_arguments(name, args)


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("recommend_hotels", {"nights": "3"}),  # a numeric string is not coerced
        ("recommend_hotels", {"nights": True}),
        ("recommend_hotels", {"nights": 0}),
        ("recommend_hotels", {"check_in": "20-12-2026"}),
        ("recommend_hotels", {"check_in": "2026-02-30"}),
        ("recommend_hotels", {"destination": ""}),
        ("get_review_insights", {"aspects": ["pool"]}),
        ("get_review_insights", {"aspects": []}),
        ("plan_itinerary", {"daily_hours": 0}),
        ("plan_itinerary", {"include_restaurants": "yes"}),
        ("web_search", {"unknown_argument": 1}),
    ],
)
def test_bad_values_fail(name, change):
    with pytest.raises(SchemaError):
        validate_arguments(name, {**VALID[name], **change})


@pytest.mark.parametrize("value", [None, [], "text", 3])
def test_arguments_must_be_an_object(value):
    with pytest.raises(SchemaError):
        validate_arguments("web_search", value)


def test_unknown_tool_and_version():
    with pytest.raises(KeyError):
        load_tool("book_flight")
    with pytest.raises(FileNotFoundError):
        load_tool("web_search", version="v999")


def test_knowledge_base_result_schema():
    chunk = {
        "text": "ETA is required.",
        "source_url": "https://example.org",
        "title": "ETA",
        "last_verified": "2026-11-01",
        "score": 0.9,
    }
    validate_result("search_knowledge_base", {"chunks": [chunk]})
    with pytest.raises(SchemaError):
        validate_result("search_knowledge_base", {"chunks": [{**chunk, "last_verified": "soon"}]})


def test_openai_shape_hides_returns():
    tools = to_openai_tools(load_tools(EXPERIMENT_TOOLS))
    assert [t["function"]["name"] for t in tools] == list(EXPERIMENT_TOOLS)
    for tool in tools:
        assert tool["type"] == "function"
        assert set(tool["function"]) == {"name", "description", "parameters"}
