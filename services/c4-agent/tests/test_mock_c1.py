import jsonschema
from fastapi.testclient import TestClient

from src.mocks.c1_mock import app

client = TestClient(app)


def test_response_matches_c1_contract(contract):
    schema = contract("c1_aspect_scores.json")["response"]
    response = client.get("/hotels/HTL_GAL_ORCHARDH_2FE6/aspect-scores")
    assert response.status_code == 200
    jsonschema.validate(response.json(), schema)


def test_unknown_hotel_is_404_like_real_c1():
    assert client.get("/hotels/HTL_DOES_NOT_EXIST_9999/aspect-scores").status_code == 404



def test_contract_example_is_valid(contract):
    c1 = contract("c1_aspect_scores.json")
    jsonschema.validate(c1["example"], c1["response"])
