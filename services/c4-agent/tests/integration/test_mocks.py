"""In-process tests of every mock endpoint, plus one run over real HTTP."""

import socket
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from c4.common.config import PROJECT_ROOT, get_settings
from c4.common.schemas import validate_result
from c4.mocks.app import DESTINATIONS, HOTELS, app

HOTEL_ARGS = {
    "destination": "Nuwara Eliya",
    "check_in": "2026-12-20",
    "nights": 3,
    "guests": 4,
    "budget_min_lkr": 0,
    "budget_max_lkr": 100000,
    "interests": ["nature"],
    "travel_purpose": "family",
    "top_k": 5,
}
TRIP_ARGS = {
    "hotel_id": "NE-001",
    "days": 2,
    "daily_hours": 8,
    "interests": ["waterfalls"],
    "include_restaurants": True,
}
REVIEW_ARGS = {"hotel_id": "NE-001"}
ENDPOINTS = [
    ("/review-insights", "get_review_insights", REVIEW_ARGS),
    ("/recommend-hotels", "recommend_hotels", HOTEL_ARGS),
    ("/plan-itinerary", "plan_itinerary", TRIP_ARGS),
]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def post(client, path, args, **params):
    return client.post(path, json=args, params=params)


def test_health(client):
    assert client.get("/health").json() == {"service": "c4-mocks", "status": "ok"}


def test_fixture_coverage():
    assert set(DESTINATIONS) == {"Nuwara Eliya", "Kandy", "Ella"}
    for data in DESTINATIONS.values():
        assert len(data["hotels"]) >= 5
    assert len(HOTELS) == sum(len(d["hotels"]) for d in DESTINATIONS.values())


@pytest.mark.parametrize(("path", "tool", "args"), ENDPOINTS)
def test_output_is_schema_valid_and_deterministic(client, path, tool, args):
    first, second = post(client, path, args), post(client, path, args)
    assert first.status_code == 200
    assert first.headers["X-C4-Mock"] == "true"
    validate_result(tool, first.json())
    assert first.content == second.content


@pytest.mark.parametrize("hotel_id", sorted(HOTELS))
def test_every_hotel_has_valid_reviews_and_itinerary(client, hotel_id):
    reviews = post(client, "/review-insights", {"hotel_id": hotel_id}).json()
    validate_result("get_review_insights", reviews)
    trip = post(client, "/plan-itinerary", {**TRIP_ARGS, "hotel_id": hotel_id}).json()
    validate_result("plan_itinerary", trip)


@pytest.mark.parametrize("destination", ["Nuwara Eliya", "Kandy", "Ella", " kandy "])
def test_every_destination_returns_hotels(client, destination):
    reply = post(client, "/recommend-hotels", {**HOTEL_ARGS, "destination": destination}).json()
    validate_result("recommend_hotels", reply)
    assert len(reply["hotels"]) == 5


@pytest.mark.parametrize("path_args", [ENDPOINTS[0], ENDPOINTS[2]])
def test_unknown_hotel_is_404(client, path_args):
    path, _, args = path_args
    assert post(client, path, {**args, "hotel_id": "NO-999"}).status_code == 404


@pytest.mark.parametrize(("path", "tool", "args"), ENDPOINTS)
def test_bad_arguments_are_422(client, path, tool, args):
    assert post(client, path, {}).status_code == 422
    assert post(client, path, {**args, "extra": 1}).status_code == 422
    assert post(client, path, []).status_code == 422
    assert client.post(path, content=b"{bad json").status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"nights": "3"},
        {"check_in": "tomorrow"},
        {"top_k": 0},
        {"budget_min_lkr": 5000, "budget_max_lkr": 1000},
    ],
)
def test_bad_hotel_values_are_422(client, change):
    assert post(client, "/recommend-hotels", {**HOTEL_ARGS, **change}).status_code == 422


def test_review_aspect_filter_and_weak_alert(client):
    reply = post(client, "/review-insights", {"hotel_id": "NE-001", "aspects": ["food"]}).json()
    assert set(reply["aspect_scores"]) == {"food"}
    assert len(reply["verified_aspect_facts"]) == 1
    full = post(client, "/review-insights", REVIEW_ARGS).json()
    weak = [a for a, score in full["aspect_scores"].items() if score < 3.0]
    assert weak  # the fixture must exercise this path
    assert full["weak_aspect_alert"] == weak


def test_hotel_filters(client):
    def hotels(**change):
        return post(client, "/recommend-hotels", {**HOTEL_ARGS, **change}).json()["hotels"]

    everything = hotels()
    scores = [h["rank_score"] for h in everything]
    assert scores == sorted(scores, reverse=True)
    assert {h["overpriced"] for h in everything} == {True, False}
    assert all(
        h["overpriced"] == (h["listed_price_lkr"] > 1.10 * h["expected_price_lkr"])
        for h in everything
    )
    assert hotels(top_k=2) == everything[:2]
    in_budget = hotels(budget_min_lkr=15000, budget_max_lkr=20000)
    assert in_budget and all(15000 <= h["listed_price_lkr"] <= 20000 for h in in_budget)
    accessible = hotels(accessibility_needs=["wheelchair"])
    assert {h["hotel_id"] for h in accessible} == {"NE-001", "NE-002"}
    assert hotels(destination="Atlantis") == []


def test_hotel_id_passes_from_tool_2_to_tool_3(client):
    hotel_id = post(client, "/recommend-hotels", HOTEL_ARGS).json()["hotels"][0]["hotel_id"]
    trip = post(client, "/plan-itinerary", {**TRIP_ARGS, "hotel_id": hotel_id})
    assert trip.status_code == 200


def test_itinerary_content(client):
    trip = post(client, "/plan-itinerary", TRIP_ARGS).json()
    assert len(trip["days"]) == 2
    assert trip["days"][0]["stops"][0]["name"] == "Lover's Leap Waterfall"  # interest first
    visited = [s["attraction_id"] for day in trip["days"] for s in day["stops"]]
    assert len(visited) == len(set(visited)) > 0
    assert trip["days"][0]["route"]["distance_km"] > 0


def test_itinerary_respects_daily_hours(client):
    for hours in (1, 2.5, 5, 24):
        trip = post(client, "/plan-itinerary", {**TRIP_ARGS, "daily_hours": hours}).json()
        assert all(len(day["stops"]) * 2.5 <= hours for day in trip["days"])


def test_itinerary_restaurants(client):
    def first_day(**change):
        reply = post(client, "/plan-itinerary", {**TRIP_ARGS, **change})
        return reply.json()["days"][0]["restaurants"]

    assert len(first_day()) == 3
    assert first_day(include_restaurants=False) == []
    assert {r["cuisine"] for r in first_day(cuisine="Indian")} == {"indian"}


def test_long_trip_is_still_valid(client):
    trip = post(client, "/plan-itinerary", {**TRIP_ARGS, "days": 10}).json()
    validate_result("plan_itinerary", trip)
    assert len(trip["days"]) == 10


@pytest.mark.parametrize(("path", "tool", "args"), ENDPOINTS)
def test_delay_by_query(client, path, tool, args):
    start = time.perf_counter()
    assert post(client, path, args, delay_ms=300).status_code == 200
    assert time.perf_counter() - start >= 0.29


def test_delay_by_environment(client, monkeypatch):
    monkeypatch.setenv("MOCK_DELAY_MS", "300")
    get_settings.cache_clear()
    start = time.perf_counter()
    post(client, "/review-insights", REVIEW_ARGS)
    assert time.perf_counter() - start >= 0.29
    start = time.perf_counter()
    post(client, "/review-insights", REVIEW_ARGS, delay_ms=0)  # the query wins
    assert time.perf_counter() - start < 0.29


def test_bad_delay_is_422(client):
    assert post(client, "/review-insights", REVIEW_ARGS, delay_ms=-1).status_code == 422


def test_real_http_server_and_client_timeout():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "c4.mocks.app:app", "--port", str(port)],
        cwd=PROJECT_ROOT / "src",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                httpx.get(f"{base}/health", timeout=2).raise_for_status()
                break
            except httpx.HTTPError:
                assert server.poll() is None, "mock server stopped"
                assert time.monotonic() < deadline, "mock server did not start"
                time.sleep(0.2)
        for path, tool, args in ENDPOINTS:
            reply = httpx.post(base + path, json=args, timeout=10)
            assert reply.status_code == 200
            validate_result(tool, reply.json())
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(f"{base}/review-insights?delay_ms=3000", json=REVIEW_ARGS, timeout=0.5)
    finally:
        server.terminate()
        server.wait(timeout=15)
