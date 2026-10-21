"""The API answers in the team contract format (contracts/*.json)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.c2.c1_client import DummyC1
from src.main import app

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
client = TestClient(app)


def contract(name):
    path = CONTRACTS / name
    if not path.exists():
        pytest.skip(f"{path} not found (run inside the team repo)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_recommendations_match_contract(data):
    jsonschema = pytest.importorskip("jsonschema")
    c2 = contract("c2_hotel_ranking.json")
    body = {"destination": "Kandy", "check_in": "2027-01-20", "check_out": "2027-01-23", "adults": 2,
            "children": 1, "budget_lkr": {"min": 0, "max": 20000}, "interests": ["local food"], "top_k": 3}
    jsonschema.validate(body, c2["request"])
    r = client.post("/recommendations", json=body)
    assert r.status_code == 200, r.text
    answer = r.json()
    jsonschema.validate(answer, c2["response"])
    assert 1 <= len(answer["ranked_hotels"]) <= 3
    assert [h["rank"] for h in answer["ranked_hotels"]] == list(range(1, len(answer["ranked_hotels"]) + 1))
    assert "for_c3" in answer


def test_contract_example_request_runs(data):
    c2 = contract("c2_hotel_ranking.json")
    r = client.post("/recommendations", json=c2["example_request"])     # Nuwara Eliya: not in the test data
    assert r.status_code == 200
    assert isinstance(r.json()["ranked_hotels"], list)


def test_dummy_c1_matches_c1_contract(data):
    jsonschema = pytest.importorskip("jsonschema")
    c1 = contract("c1_aspect_scores.json")
    dummy = DummyC1(data.hotels)
    for hid in data.hotels.index:
        jsonschema.validate(dummy.get(hid), c1["response"])


def test_hotel_details(data):
    r = client.get("/hotels/k-pool")
    assert r.status_code == 200
    assert r.json()["feature_checks"]["pool"]["status"] in ("yes", "no", "not sure")
    assert client.get("/hotels/unknown-id").status_code == 404


def test_from_text_without_ollama_gives_clear_error(data, monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")      # nothing listens there
    r = client.post("/recommendations/from-text", json={"text": "Hotel in Kandy for 2 nights from 5 December"})
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]


def test_from_text_flow_with_a_fake_nlp(data, monkeypatch):
    """The sentence path without Ollama: the NLP step is replaced by a fixed answer."""
    from src.nlp import understand as nlp
    from tests.helpers import make_request

    fake = make_request(bmax=10000, health=["heart"], must=["near_hospital"])
    fake["not_needed"] = []
    monkeypatch.setattr(nlp, "understand", lambda *a, **k: (fake, ["check_out calculated"], 1.0, "fixed"))
    r = client.post("/recommendations/from-text", json={"text": "anything", "health_consent": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["understood_request"]["hotel"]["budget_max_lkr"] == 10000
    assert body["ranked_hotels"] and body["towns"][0]["town"] == "Kandy"
    assert body["for_c3"]["health"] == {"consent_given": False}
    for h in body["ranked_hotels"]:
        assert h["price"]["price_per_night_lkr"] <= 10000


def test_bad_requests_get_a_hint(data):
    r = client.post("/recommendations", json={"text": "Hotel in Kandy for 2 nights"})
    assert r.status_code == 422
    assert "from-text" in r.json()["hint"]
    r = client.post("/recommendations", json={"destination": "Kandy", "check_in": "string"})
    assert r.status_code == 422
    r = client.post("/recommendations", json={"destination": "Kandy", "check_in": "2026-12-07",
                                               "check_out": "2026-12-05"})
    assert r.status_code == 422


def test_docs_examples_work(data):
    """The example shown on the /docs page is a request that really works."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    example = schema["RecommendationRequest"]["examples"][0]
    r = client.post("/recommendations", json=example)
    assert r.status_code == 200, r.text
    assert schema["TextRequest"]["examples"][0]["text"]
