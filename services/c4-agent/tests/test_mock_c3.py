import jsonschema
from fastapi.testclient import TestClient

from src.mocks.c3_mock import app

client = TestClient(app)

STOP_FIELDS = {"day", "stop_order", "poi_id", "name", "latitude", "longitude", "arrival_time"}


def test_ella_plan_has_real_c3_shape(ella):
    response = client.post("/plan-itinerary", json=ella)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert {"trip_id", "hotel", "trip", "medical_support", "days", "pipeline"} <= body.keys()
    assert len(body["days"]) == 3  # 20 -> 23 Dec
    for day in body["days"]:
        assert {"day", "stops", "recommended_restaurants"} <= day.keys()
        for stop in day["stops"]:
            assert STOP_FIELDS <= stop.keys()


def test_no_attraction_is_visited_twice_and_times_fit_the_day(ella):
    days = client.post("/plan-itinerary", json=ella).json()["days"]
    ids = [s["poi_id"] for d in days for s in d["stops"]]
    assert ids, "expected at least one stop near Ella"
    assert len(ids) == len(set(ids))
    for d in days:
        for s in d["stops"]:
            assert "09:00" <= s["arrival_time"] < "18:00"


def test_check_out_before_check_in_is_422(ella):
    bad = {**ella, "trip": {**ella["trip"], "check_out": "2026-12-19"}}
    assert client.post("/plan-itinerary", json=bad).status_code == 422


def test_missing_selected_hotel_is_422(ella):
    bad = {k: v for k, v in ella.items() if k != "selected_hotel"}
    assert client.post("/plan-itinerary", json=bad).status_code == 422



def test_response_matches_c3_real_api(c4_schema, ella):
    schema = c4_schema("c3_plan_itinerary.json")["response"]
    jsonschema.validate(client.post("/plan-itinerary", json=ella).json(), schema)


def test_schema_example_request_is_valid_and_accepted(c4_schema):
    c3 = c4_schema("c3_plan_itinerary.json")
    jsonschema.validate(c3["example_request"], c3["request"])
    assert client.post(c3["endpoint"]["path"], json=c3["example_request"]).status_code == 200
