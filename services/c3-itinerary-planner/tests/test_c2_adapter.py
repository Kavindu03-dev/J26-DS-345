"""Adapter and API boundary checks; pipeline execution is isolated."""

from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.adapters.c2_adapter import (
    DEFAULT_RESTAURANT_PREFERENCES,
    adapt_c2_to_c3,
    derive_num_days,
    map_interests_to_preferences,
)
from src import main


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


def test_restaurant_defaults_match_internal_input(ella):
    internal = json.loads(main.INPUT_FILE.read_text(encoding="utf-8"))
    adapted = adapt_c2_to_c3(ella)
    assert adapted["restaurant_preferences"] == internal["restaurant_preferences"]
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


def test_api_internal_contract_unchanged(isolated_pipeline):
    internal_path = Path(main.PROJECT_ROOT) / "Data/input/c2_c3_input.json"
    internal = json.loads(internal_path.read_text(encoding="utf-8"))
    client, calls = isolated_pipeline
    response = client.post("/plan-itinerary", json=internal)
    assert response.status_code == 200
    assert json.loads(main.INPUT_FILE.read_text(encoding="utf-8")) == internal
    assert response.json()["trip"] == internal["trip"]
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
