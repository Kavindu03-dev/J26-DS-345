import httpx
import pytest
from fastapi.testclient import TestClient

from src.mocks.c1_mock import app as c1_app
from src.mocks.c2_mock import app as c2_app
from src.mocks.c3_mock import app as c3_app
from src.tools.clients import ErrorCode, ToolClients, ToolError


@pytest.fixture
def tools():
    return ToolClients(c1=TestClient(c1_app), c2=TestClient(c2_app), c3=TestClient(c3_app))


def test_full_chain_c2_then_c3_then_c1(tools, ella):
    """The agent's main flow: rank hotels, plan around the top one, quote its reviews."""
    top = tools.recommend_hotels({"destination": "Ella", "top_k": 1})["ranked_hotels"][0]
    plan = tools.plan_itinerary({**ella, "selected_hotel": {**ella["selected_hotel"], "hotel_id": top["hotel_id"]}})
    reviews = tools.get_aspect_scores(top["hotel_id"])
    assert plan["status"] == "success"
    assert reviews["hotel_id"] == top["hotel_id"]


def test_not_found_becomes_tool_error(tools):
    with pytest.raises(ToolError) as e:
        tools.get_aspect_scores("HTL_NOPE")
    assert e.value.code is ErrorCode.NOT_FOUND
    assert e.value.to_dict()["error"]["code"] == "NOT_FOUND"


def test_validation_error_becomes_invalid_argument(tools):
    with pytest.raises(ToolError) as e:
        tools.plan_itinerary({"trip_id": "x"})
    assert e.value.code is ErrorCode.INVALID_ARGUMENT


def test_unreachable_service_becomes_unavailable():
    down = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.5)
    tools = ToolClients(c1=down, c2=down, c3=down)
    with pytest.raises(ToolError) as e:
        tools.get_aspect_scores("HTL_GAL_ORCHARDH_2FE6")
    assert e.value.code in (ErrorCode.UNAVAILABLE, ErrorCode.TIMEOUT)
