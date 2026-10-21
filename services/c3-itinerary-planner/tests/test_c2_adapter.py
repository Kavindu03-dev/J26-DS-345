"""Adapter and API boundary checks; pipeline execution is isolated."""

from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from src.adapters.c2_adapter import (
    adapt_c2_handoff,
    DEFAULT_RESTAURANT_PREFERENCES,
    adapt_c2_to_c3,
    derive_num_days,
    map_interests_to_preferences,
)
from src import main


@pytest.fixture
def latest_handoff():
    return {
        "for_c3": {
            "trip": {"stay_in": ["Ella", "Kandy"], "want_to_visit": ["Yala"],
                     "check_in": "2026-12-20", "check_out": "2026-12-23",
                     "nights": 5, "dates_fixed": False, "adults": 2, "children": 1,
                     "travel_style": "family", "budget_total_lkr": None},
            "interests": ["nature", "wildlife", "hiking"],
            "health": {"consent_given": False},
            "recommended_hotels": [
                {"hotel_id": "ella", "name": "Ella example", "town": "Ella",
                 "stay_in": "Ella", "latitude": 6.86, "longitude": 81.04,
                 "nearest_hospital_km": 0.1},
                {"hotel_id": "kandy", "name": "Kandy example", "town": "Kandy",
                 "stay_in": "Kandy", "latitude": 7.29, "longitude": 80.63},
            ],
        },
        "selected_hotel_id": "ella",
    }


def test_latest_handoff_selection_and_defaults(latest_handoff):
    original = deepcopy(latest_handoff)
    request = main.PlanRequest.model_validate(latest_handoff)
    normalized = request.model_dump()
    assert normalized["hotel"] == {"hotel_id": "ella", "name": "Ella example",
                                   "latitude": 6.86, "longitude": 81.04, "district": "Badulla"}
    assert normalized["trip"] == {"num_days": 3, "day_start_hour": 9,
                                  "day_end_hour": 18, "travel_mode": "driving", "want_to_visit": ["Yala"]}
    assert normalized["restaurant_preferences"] == DEFAULT_RESTAURANT_PREFERENCES
    assert normalized["medical_support"] == {"required": False, "priority": "normal"}
    assert normalized["attraction_preferences"]["nature_wildlife"] == 5
    assert normalized["attraction_preferences"]["adventure_recreation"] == 5
    for key in ("stay_in", "adults", "children", "travel_style"):
        assert normalized["context"][key] == original["for_c3"]["trip"][key]
    assert normalized["context"]["num_days_source"] == "dates"
    assert "nearest_hospital_km" not in normalized["context"]
    assert request.trip_id.startswith("C3_") and len(request.trip_id) == 35
    assert adapt_c2_handoff(latest_handoff)["trip_id"] != request.trip_id
    assert latest_handoff == original
    latest_handoff["selected_hotel_id"] = "kandy"
    assert main.PlanRequest.model_validate(latest_handoff).hotel.district == "Kandy"


@pytest.mark.parametrize("selection", [None, "", "missing"])
def test_latest_selection_is_explicit(latest_handoff, selection):
    latest_handoff["selected_hotel_id"] = selection
    with pytest.raises(ValueError, match="selected_hotel_id"):
        main.PlanRequest.model_validate(latest_handoff)


@pytest.mark.parametrize("hotels", [[], None, [{"hotel_id": "ella"}, {"hotel_id": "ella"}]])
def test_latest_empty_or_ambiguous_recommendations(latest_handoff, hotels):
    latest_handoff["for_c3"]["recommended_hotels"] = hotels
    with pytest.raises(ValueError):
        main.PlanRequest.model_validate(latest_handoff)


@pytest.mark.parametrize("field", ["latitude", "longitude"])
@pytest.mark.parametrize("value", [None, "6.8", True, float("nan"), float("inf"), 999])
def test_latest_invalid_coordinates(latest_handoff, field, value):
    latest_handoff["for_c3"]["recommended_hotels"][0][field] = value
    with pytest.raises(ValueError, match=field):
        main.PlanRequest.model_validate(latest_handoff)


@pytest.mark.parametrize("field", ["latitude", "longitude"])
def test_latest_absent_coordinate(latest_handoff, field):
    del latest_handoff["for_c3"]["recommended_hotels"][0][field]
    with pytest.raises(ValueError, match=field):
        main.PlanRequest.model_validate(latest_handoff)


@pytest.mark.parametrize("town,district", [("Ella", "Badulla"), (" KANDY ", "Kandy"),
                                          ("Colombo", "Colombo"), ("Galle", "Galle"), ("Galle Fort", "Galle")])
def test_latest_district_mapping(latest_handoff, town, district):
    latest_handoff["for_c3"]["recommended_hotels"][0]["town"] = town
    assert main.PlanRequest.model_validate(latest_handoff).hotel.district == district


def test_latest_unknown_town_not_assumed_district(latest_handoff):
    latest_handoff["for_c3"]["recommended_hotels"][0]["town"] = "Unsupported town"
    with pytest.raises(ValueError, match="Cannot resolve district"):
        main.PlanRequest.model_validate(latest_handoff)


@pytest.mark.parametrize("start,end", [(None, None), ("invalid", None), ("2026-12-23", "2026-12-20")])
def test_latest_nights_fallback(latest_handoff, start, end):
    trip = latest_handoff["for_c3"]["trip"]
    trip.update(check_in=start, check_out=end, nights=2, adults=None, children=None, travel_style=None)
    request = main.PlanRequest.model_validate(latest_handoff)
    assert request.trip.num_days == 2
    assert request.context["num_days_source"] == "nights"
    assert "adults" not in request.context


@pytest.mark.parametrize("nights", [None, 0, -1, True, "3", 3.0, 15])
def test_latest_no_usable_days(latest_handoff, nights):
    latest_handoff["for_c3"]["trip"].update(check_in=None, check_out=None, nights=nights)
    with pytest.raises(ValueError):
        main.PlanRequest.model_validate(latest_handoff)


def test_latest_dates_over_limit_not_truncated(latest_handoff):
    latest_handoff["for_c3"]["trip"].update(check_out="2027-01-20", nights=3)
    with pytest.raises(ValueError, match="1 to 14"):
        main.PlanRequest.model_validate(latest_handoff)


@pytest.mark.parametrize("consent", [False, True])
def test_latest_health_consent_is_not_medical_mode(latest_handoff, consent):
    latest_handoff["for_c3"]["health"] = {"consent_given": consent, "needs": ["mobility", "heart", "unknown"]}
    request = main.PlanRequest.model_validate(latest_handoff)
    assert request.context["health"] == (["mobility", "heart"] if consent else [])
    assert request.accessibility["reduced_mobility"] is consent
    assert request.medical_support.required is False
    latest_handoff["medical_support"] = {"required": True, "priority": "high"}
    assert main.PlanRequest.model_validate(latest_handoff).medical_support.required is True


def test_latest_no_consent_ignores_even_invalid_needs(latest_handoff):
    latest_handoff["for_c3"]["health"]["needs"] = "not a list"
    assert main.PlanRequest.model_validate(latest_handoff).context["health"] == []


def test_latest_api_normalizes_before_pipeline(latest_handoff, isolated_pipeline):
    client, calls = isolated_pipeline
    response = client.post("/plan-itinerary", json=latest_handoff)
    assert response.status_code == 200
    saved = json.loads(main.INPUT_FILE.read_text(encoding="utf-8"))
    assert saved["hotel"]["hotel_id"] == latest_handoff["selected_hotel_id"]
    assert response.json()["trip_id"] == saved["trip_id"]
    assert saved["context"]["health_consent_given"] is False
    assert calls == main.PIPELINE_SCRIPTS


def test_latest_api_rejects_before_writing(latest_handoff, isolated_pipeline):
    client, calls = isolated_pipeline
    latest_handoff["selected_hotel_id"] = "absent"
    assert client.post("/plan-itinerary", json=latest_handoff).status_code == 422
    assert not main.INPUT_FILE.exists()
    assert calls == []


@pytest.fixture
def internal_request():
    """Legacy C3 contract, independent of runtime inputs and adapter defaults."""
    return {
        "trip_id": "TEST_TRIP_001",
        "hotel": {
            "hotel_id": "TEST_HOTEL_001", "name": "Kandy Test Depot",
            "latitude": 7.294373, "longitude": 80.641486, "district": "Kandy",
        },
        "trip": {
            "num_days": 3, "day_start_hour": 9, "day_end_hour": 18,
            "travel_mode": "driving",
        },
        "attraction_preferences": {
            "history_culture": 1.0, "nature_wildlife": 5.0,
            "beach_coastal": 1.0, "religious": 1.0,
            "adventure_recreation": 5.0, "urban_leisure": 2.0,
        },
        "restaurant_preferences": {
            "aspect_importance": {
                "food": 5.0, "service": 4.0, "ambience": 3.0,
                "value": 5.0, "location": 3.0,
            },
            "cuisine_preferences": {
                "sri_lankan": 5.0, "local": 5.0, "indian": 4.0,
                "south_indian": 4.0, "asian": 4.0, "thai": 3.0,
                "chinese": 3.0, "international": 3.0, "pizza": 2.0,
                "italian": 2.0,
            },
        },
        "medical_support": {"required": False, "priority": "normal"},
    }


@pytest.fixture
def ella():
    return {
        "trip_id": "TRIP_001",
        "trip": {
            "stay_in": ["Ella"], "want_to_visit": ["Yala"],
            "check_in": "2026-12-20", "check_out": "2026-12-23",
            "dates_fixed": False, "adults": 2, "children": 1,
            "travel_style": "family",
        },
        "selected_hotel": {
            "hotel_id": "HOTEL_001", "name": "Selected Ella Hotel",
            "latitude": 6.8667, "longitude": 81.0466, "district": "Badulla",
        },
        "hotel_preferences": {
            "budget_max_lkr": 15000, "must": ["budget", "low_steps"],
            "wishes": {"quiet": "high", "pool": "low"},
        },
        "health": ["mobility"], "interests": ["nature", "wildlife"],
    }


def test_ella_family(ella):
    original = deepcopy(ella)
    adapted = adapt_c2_to_c3(ella)
    assert adapted["hotel"] == ella["selected_hotel"]
    assert adapted["trip"] == {
        "num_days": 3, "day_start_hour": 9, "day_end_hour": 18,
        "travel_mode": "driving", "want_to_visit": ["Yala"],
    }
    assert adapted["context"] == {
        **{key: value for key, value in ella["trip"].items() if key != "want_to_visit"},
        "interests": ["nature", "wildlife"], "health": ["mobility"],
    }
    assert adapted["attraction_preferences"]["nature_wildlife"] == 5.0
    assert sum(adapted["attraction_preferences"].values()) == 10.0
    assert ella == original
    assert main.PlanRequest.model_validate(ella).model_dump(exclude_none=True) == adapted


@pytest.mark.parametrize("interest,category", [
    ("nature", "nature_wildlife"), ("wildlife", "nature_wildlife"),
    ("beach", "beach_coastal"), ("culture", "history_culture"),
    ("religious", "religious"), ("surfing", "adventure_recreation"),
    ("hiking", "adventure_recreation"), ("tea_country", "nature_wildlife"),
    ("nightlife", "urban_leisure"), ("food", "urban_leisure"),
])
def test_interest_mapping(interest, category):
    preferences = map_interests_to_preferences([interest])
    assert preferences[category] == 5.0
    assert all(score == 1.0 for key, score in preferences.items() if key != category)


def test_duplicate_interests():
    assert map_interests_to_preferences(["nature", "nature", "wildlife"]) == (
        map_interests_to_preferences(["nature"])
    )


def test_unknown_interest_preserved(ella):
    ella["interests"] = ["unmapped_interest"]
    adapted = adapt_c2_to_c3(ella)
    assert set(adapted["attraction_preferences"].values()) == {1.0}
    assert adapted["context"]["interests"] == ["unmapped_interest"]


def test_mobility_is_accessibility_only(ella):
    adapted = adapt_c2_to_c3(ella)
    assert adapted["accessibility"] == {
        "reduced_mobility": True, "prefer_low_walking": True,
    }
    assert adapted["medical_support"] == {"required": False, "priority": "normal"}


def test_absent_health_and_interests(ella):
    del ella["health"]
    del ella["interests"]
    adapted = adapt_c2_to_c3(ella)
    assert not any(adapted["accessibility"].values())
    assert set(adapted["attraction_preferences"].values()) == {1.0}


def test_restaurant_defaults_match_internal_input(ella, internal_request):
    adapted = adapt_c2_to_c3(ella)
    assert adapted["restaurant_preferences"] == internal_request["restaurant_preferences"]
    adapted["restaurant_preferences"]["aspect_importance"]["food"] = 1
    assert adapt_c2_to_c3(ella)["restaurant_preferences"] == DEFAULT_RESTAURANT_PREFERENCES


def test_explicit_preferences_and_schedule_preserved(ella):
    ella["restaurant_preferences"] = deepcopy(DEFAULT_RESTAURANT_PREFERENCES)
    ella["restaurant_preferences"]["cuisine_preferences"] = {"thai": 5.0}
    ella["medical_support"] = {"required": True, "priority": "high"}
    ella["trip"].update(day_start_hour=8, day_end_hour=17, travel_mode="driving")
    adapted = adapt_c2_to_c3(ella)
    assert adapted["restaurant_preferences"] == ella["restaurant_preferences"]
    assert adapted["medical_support"] == ella["medical_support"]
    assert adapted["trip"]["day_start_hour"] == 8
    assert adapted["trip"]["day_end_hour"] == 17


@pytest.mark.parametrize("start,end", [
    ("2026-12-23", "2026-12-20"), ("2026-12-20", "2026-12-20"),
    ("2026-02-30", "2026-03-02"), (None, "2026-12-23"),
    ("2026-12-20", None), ("20261220", "2026-12-23"),
    ("2026-12-20T09:00:00", "2026-12-23"),
])
def test_invalid_dates(start, end):
    with pytest.raises(ValueError):
        derive_num_days(start, end)


@pytest.mark.parametrize("field,value", [
    ("latitude", 91), ("longitude", -181), ("latitude", float("nan")),
    ("longitude", float("inf")), ("latitude", True), ("latitude", "6.8"),
    ("longitude", None), ("name", 123),
])
def test_invalid_hotel(ella, field, value):
    ella["selected_hotel"][field] = value
    with pytest.raises(ValueError):
        adapt_c2_to_c3(ella)


def test_missing_selected_hotel(ella):
    del ella["selected_hotel"]
    with pytest.raises(ValueError, match="selected_hotel"):
        adapt_c2_to_c3(ella)


@pytest.mark.parametrize("section,field,value", [
    ("trip", "adults", "2"), ("trip", "children", -1),
    ("trip", "dates_fixed", "false"), ("trip", "stay_in", "Ella"),
    ("trip", "want_to_visit", [123]), ("trip", "day_start_hour", True),
    ("trip", "day_end_hour", 9), ("trip", "travel_style", []),
    (None, "interests", "nature"), (None, "health", [1]),
])
def test_invalid_types(ella, section, field, value):
    target = ella[section] if section else ella
    target[field] = value
    with pytest.raises(ValueError):
        adapt_c2_to_c3(ella)


@pytest.fixture
def isolated_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "INPUT_FILE", tmp_path / "input.json")
    monkeypatch.setattr(main, "ROUTE_FOLDER", tmp_path / "routes")
    itinerary = tmp_path / "itinerary.csv"
    restaurants = tmp_path / "restaurants.csv"
    itinerary.write_text("day,stop_order\n", encoding="utf-8")
    restaurants.write_text("day,final_day_rank\n", encoding="utf-8")
    monkeypatch.setattr(main, "ITINERARY_FILE", itinerary)
    monkeypatch.setattr(main, "RESTAURANT_FILE", restaurants)
    calls = []
    monkeypatch.setattr(main, "run_script", lambda script: calls.append(script))
    return TestClient(main.app), calls


def test_api_external_input_persists_metadata_and_skips_medical(ella, isolated_pipeline):
    client, calls = isolated_pipeline
    response = client.post("/plan-itinerary", json=ella)
    assert response.status_code == 200
    saved = json.loads(main.INPUT_FILE.read_text(encoding="utf-8"))
    assert saved == adapt_c2_to_c3(ella)
    assert calls == main.PIPELINE_SCRIPTS
    assert main.MEDICAL_ACCESS_SCRIPT not in calls


def test_api_internal_contract_unchanged(isolated_pipeline, internal_request):
    client, calls = isolated_pipeline
    response = client.post("/plan-itinerary", json=internal_request)
    assert response.status_code == 200
    assert json.loads(main.INPUT_FILE.read_text(encoding="utf-8")) == internal_request
    assert response.json()["trip"] == internal_request["trip"]
    assert calls == main.PIPELINE_SCRIPTS


@pytest.mark.parametrize("invalid", ["dates", "hotel", "restaurants", "medical", "duration"])
def test_api_rejects_invalid_input_before_pipeline(ella, isolated_pipeline, invalid):
    client, calls = isolated_pipeline
    if invalid == "dates":
        ella["trip"]["check_out"] = ella["trip"]["check_in"]
    elif invalid == "hotel":
        del ella["selected_hotel"]
    elif invalid == "restaurants":
        ella["restaurant_preferences"] = {"aspect_importance": {}, "cuisine_preferences": {}}
    elif invalid == "medical":
        ella["medical_support"] = {"required": True, "priority": "invalid"}
    else:
        ella["trip"]["check_out"] = "2027-01-20"
    assert client.post("/plan-itinerary", json=ella).status_code == 422
    assert calls == []
    assert not main.INPUT_FILE.exists()


def test_api_explicit_medical_support_runs_existing_stage(ella, isolated_pipeline):
    client, calls = isolated_pipeline
    ella["medical_support"] = {"required": True, "priority": "high"}
    assert client.post("/plan-itinerary", json=ella).status_code == 200
    assert calls == [main.PIPELINE_SCRIPTS[0], main.MEDICAL_ACCESS_SCRIPT,
                     *main.PIPELINE_SCRIPTS[1:]]


def write_route(day, *, feature_day=None):
    coordinates = [[80.641486, 7.294373], [round(80.65 + day / 100, 6), 7.30]]
    main.ROUTE_FOLDER.mkdir(parents=True, exist_ok=True)
    path = main.ROUTE_FOLDER / f"day_{day}_route.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "day": day if feature_day is None else feature_day,
                "distance_km": 10 + day, "duration_min": 20.5 + day,
            },
            "geometry": {"type": "LineString", "coordinates": coordinates},
        }],
    }), encoding="utf-8")
    return coordinates


@pytest.fixture
def route_pipeline(isolated_pipeline):
    client, calls = isolated_pipeline
    main.ITINERARY_FILE.write_text(
        "day,stop_order,poi_id,name,latitude,longitude,arrival_time,visit_duration_min\n"
        "2,1,HOTEL,Kandy Test Depot,7.294373,80.641486,09:00,0\n"
        "1,2,POI_1,Kandy Attraction,7.30,80.66,09:20,60\n"
        "1,1,HOTEL,Kandy Test Depot,7.294373,80.641486,09:00,0\n",
        encoding="utf-8",
    )
    main.RESTAURANT_FILE.write_text(
        "day,final_day_rank,name,latitude,longitude,final_restaurant_score\n"
        "1,1,Kandy Restaurant,7.30,80.66,0.9\n",
        encoding="utf-8",
    )
    return client, calls


def test_routes_match_days_and_preserve_existing_response(route_pipeline, internal_request, monkeypatch):
    client, calls = route_pipeline
    expected = {}
    # A previous route may be replaced successfully, rather than simply absent.
    write_route(1, feature_day=99)

    def run(script):
        calls.append(script)
        if script.name == "03_build_route_geometry.py":
            # Deliberately generate in reverse order.
            for day in (2, 1):
                expected[day] = write_route(day)

    monkeypatch.setattr(main, "run_script", run)
    original_loader = main.load_day_route

    def load_under_lock(*args):
        assert main.PIPELINE_LOCK.locked()
        return original_loader(*args)

    monkeypatch.setattr(main, "load_day_route", load_under_lock)
    response = client.post("/plan-itinerary", json=internal_request)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"status", "trip_id", "hotel", "trip", "medical_support", "days", "pipeline"}
    assert data["status"] == "success"
    assert data["trip_id"] == internal_request["trip_id"]
    assert data["hotel"] == internal_request["hotel"]
    assert data["trip"] == internal_request["trip"]
    assert data["medical_support"] == {
        "required": False, "priority": "normal", "average_hospital_access_min": None,
        "best_hospital_access_min": None, "worst_hospital_access_min": None,
    }
    assert data["pipeline"] == [{"stage": script.name, "status": "success"} for script in calls]
    for day in data["days"]:
        number = day["day"]
        assert day["route"] == {
            "geometry": expected[number], "distance_km": float(10 + number),
            "duration_min": 20.5 + number,
        }
        assert isinstance(day["route"]["distance_km"], float)
        assert isinstance(day["route"]["duration_min"], float)
        assert day["route"]["geometry"][0] == [80.641486, 7.294373]
        assert day["stops"] == main.dataframe_to_records(
            main.pd.read_csv(main.ITINERARY_FILE).query("day == @number").sort_values("stop_order")
        )
        assert day["recommended_restaurants"] == main.dataframe_to_records(
            main.pd.read_csv(main.RESTAURANT_FILE).query("day == @number")
        )


@pytest.mark.parametrize("route_state", ["missing", "stale", "mismatched", "malformed", "invalid_metrics", "skipped"])
def test_unavailable_routes_are_null(route_pipeline, internal_request, monkeypatch, caplog, route_state):
    client, calls = route_pipeline
    if route_state in {"stale", "skipped"}:
        write_route(1)
    if route_state == "skipped":
        monkeypatch.setattr(main, "PIPELINE_SCRIPTS", [
            script for script in main.PIPELINE_SCRIPTS if script.name != "03_build_route_geometry.py"
        ])

    def run(script):
        calls.append(script)
        if script.name == "03_build_route_geometry.py":
            if route_state == "mismatched":
                write_route(1, feature_day=2)
            elif route_state == "malformed":
                write_route(1)
                (main.ROUTE_FOLDER / "day_1_route.geojson").write_text("{", encoding="utf-8")
            elif route_state == "invalid_metrics":
                write_route(1)
                path = main.ROUTE_FOLDER / "day_1_route.geojson"
                route = json.loads(path.read_text(encoding="utf-8"))
                route["features"][0]["properties"]["distance_km"] = None
                path.write_text(json.dumps(route), encoding="utf-8")
            write_route(2)

    monkeypatch.setattr(main, "run_script", run)
    response = client.post("/plan-itinerary", json=internal_request)
    assert response.status_code == 200
    days = response.json()["days"]
    assert days[0]["route"] is None
    if route_state == "skipped":
        assert days[1]["route"] is None
    else:
        assert days[1]["route"] is not None
        assert "Route unavailable for day 1" in caplog.text


def test_second_request_does_not_reuse_first_request_route(route_pipeline, internal_request, monkeypatch):
    client, calls = route_pipeline

    def run(script):
        if script.name == "03_build_route_geometry.py":
            write_route(1)

    monkeypatch.setattr(main, "run_script", run)
    assert client.post("/plan-itinerary", json=internal_request).json()["days"][0]["route"] is not None
    monkeypatch.setattr(main, "run_script", lambda script: calls.append(script))
    response = client.post("/plan-itinerary", json=internal_request)
    assert response.status_code == 200
    assert all(day["route"] is None for day in response.json()["days"])
