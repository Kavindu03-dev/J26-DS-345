import json

import pytest

from src import config


@pytest.fixture(scope="session")
def contract():
    """Load a shared contract from the repo's contracts/ folder by file name."""

    def _load(name: str) -> dict:
        with open(config.CONTRACTS_DIR / name, encoding="utf-8") as f:
            return json.load(f)

    return _load


@pytest.fixture(scope="session")
def c4_schema():
    """Load one of C4's own schema files from services/c4-agent/schemas/."""

    def _load(name: str) -> dict:
        with open(config.SERVICE_ROOT / "schemas" / name, encoding="utf-8") as f:
            return json.load(f)

    return _load


@pytest.fixture
def ella():
    """Same payload as C3's own test (services/c3-itinerary-planner/tests/test_c2_adapter.py)."""
    return {
        "trip_id": "TRIP_001",
        "trip": {
            "stay_in": ["Ella"], "want_to_visit": ["Yala"],
            "check_in": "2026-12-20", "check_out": "2026-12-23",
            "dates_fixed": False, "adults": 2, "children": 1, "travel_style": "family",
        },
        "selected_hotel": {
            "hotel_id": "HOTEL_001", "name": "Selected Ella Hotel",
            "latitude": 6.8667, "longitude": 81.0466, "district": "Badulla",
        },
        "health": ["mobility"], "interests": ["nature", "wildlife"],
    }
