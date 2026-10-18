import jsonschema
from fastapi.testclient import TestClient

from src.mocks.c2_mock import app

client = TestClient(app)


def test_contract_example_request_returns_contract_valid_response(contract):
    c2 = contract("c2_hotel_ranking.json")
    response = client.post("/recommendations", json=c2["example_request"])
    assert response.status_code == 200
    body = response.json()
    jsonschema.validate(body, c2["response"])
    assert 1 <= len(body["ranked_hotels"]) <= c2["example_request"]["top_k"]


def test_ranks_are_ordered_and_respect_budget():
    response = client.post(
        "/recommendations",
        json={"destination": "Ella", "budget_lkr": {"max": 20000}, "travel_purpose": "family", "top_k": 5},
    )
    hotels = response.json()["ranked_hotels"]
    assert [h["rank"] for h in hotels] == list(range(1, len(hotels) + 1))
    assert all(min(o["price_lkr"] for o in h["price"]["offers"]) <= 20000 for h in hotels)
    scores = [h["score"] for h in hotels]
    assert scores == sorted(scores, reverse=True)


def test_same_request_gives_same_answer():
    req = {"destination": "Kandy", "top_k": 3}
    assert client.post("/recommendations", json=req).json() == client.post("/recommendations", json=req).json()


def test_unknown_destination_is_404():
    assert client.post("/recommendations", json={"destination": "Atlantis"}).status_code == 404


def test_bad_budget_is_422():
    r = client.post("/recommendations", json={"destination": "Ella", "budget_lkr": {"min": 30000, "max": 10000}})
    assert r.status_code == 422


def test_explanation_keeps_currency_and_platform_capitalised():
    hotel = client.post("/recommendations", json={"destination": "Ella", "top_k": 1}).json()["ranked_hotels"][0]
    assert hotel["explanation"][0].isupper()
    assert "LKR" in hotel["explanation"]
    assert hotel["price"]["best_platform"] in hotel["explanation"]


def test_contract_example_response_is_valid(contract):
    c2 = contract("c2_hotel_ranking.json")
    jsonschema.validate(c2["example_response"], c2["response"])
