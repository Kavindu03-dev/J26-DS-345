"""Tourism test set and agent tool definitions."""

import json

import jsonschema
from fastapi.testclient import TestClient

from evaluation import build_testset
from src.mocks.c2_mock import app as c2_app
from src.mocks.c3_mock import app as c3_app
from src.tools.definitions import TOOLS, to_c2_request, to_c3_request, validate_call


def test_testset_builds_with_planned_category_sizes():
    records = build_testset.build()
    counts = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    assert len(records) == 150
    assert counts == {"single_tool": 33, "multi_tool": 30, "web_search": 21,
                      "direct_answer": 18, "decline": 18, "tricky_arguments": 22, "clarify": 8}


def test_committed_testset_file_is_up_to_date():
    """Fails if testset_items.py changed without running `python -m evaluation.build_testset`."""
    committed = [json.loads(l) for l in build_testset.JSONL.read_text(encoding="utf-8").splitlines()]
    assert committed == build_testset.build()


def test_multi_step_items_carry_the_tool_result_and_the_top_hotel():
    for r in build_testset.build():
        if len(r["steps"]) == 2:
            result = json.loads(r["context_after_step"][0][1]["content"])
            top = result["ranked_hotels"][0]
            hotel_spec = r["steps"][1]["calls"][0]["arguments"]["hotel"]
            assert hotel_spec["accept"] == [top["name"], top["hotel_id"]]


def test_tools_are_valid_openai_style_definitions():
    assert [t["function"]["name"] for t in TOOLS] == ["get_review_insights", "recommend_hotels", "plan_itinerary", "web_search"]
    for t in TOOLS:
        jsonschema.Draft202012Validator.check_schema(t["function"]["parameters"])


def test_validate_call_is_strict():
    assert validate_call("get_review_insights", {"hotel": "Kandy Hills"}) == []
    assert validate_call("get_review_insights", {"hotel": "Kandy Hills", "extra": 1})  # unknown argument
    assert validate_call("get_review_insights", {})  # missing required
    assert validate_call("recommend_hotels", {"destination": "Ella", "nearby": ["airport"]})  # not in enum
    assert validate_call("book_flight", {})  # unknown tool


def test_recommend_hotels_maps_onto_the_c2_contract(contract):
    args = {"destination": "Ella", "budget_min_lkr": 10000, "budget_max_lkr": 25000, "adults": 2,
            "children": 2, "nearby": ["hospital"], "travel_purpose": "family", "top_k": 3}
    req = to_c2_request(args)
    jsonschema.validate(req, contract("c2_hotel_ranking.json")["request"])
    assert req["budget_lkr"] == {"min": 10000, "max": 25000}
    assert TestClient(c2_app).post("/recommendations", json=req).status_code == 200


def test_plan_itinerary_maps_onto_c3_real_api(c4_schema):
    hotel = {"hotel_id": "HTL_ELL_WINWOODE_0001", "name": "Winwood Ella", "latitude": 6.87, "longitude": 81.05, "district": "Badulla"}
    req = to_c3_request({"hotel": "Winwood Ella", "days": 3, "start_date": "2026-12-01", "mobility_needs": True}, hotel, "T1")
    jsonschema.validate(req, c4_schema("c3_plan_itinerary.json")["request"])
    assert req["trip"]["check_in"] == "2026-12-01" and req["trip"]["check_out"] == "2026-12-04"
    assert req["health"] == ["mobility"]
    assert TestClient(c3_app).post("/plan-itinerary", json=req).status_code == 200
