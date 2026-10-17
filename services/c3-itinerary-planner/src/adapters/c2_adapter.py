"""Translate external C2 hotel selections into the existing C3 contract.

Destination and accessibility metadata are retained for future consumers;
they do not change the current optimizer or activate medical support.
"""

from copy import deepcopy
from datetime import date
import math
import re


INTEREST_CATEGORIES = {
    "nature": "nature_wildlife",
    "wildlife": "nature_wildlife",
    "beach": "beach_coastal",
    "culture": "history_culture",
    "religious": "religious",
    "surfing": "adventure_recreation",
    "hiking": "adventure_recreation",
    "tea_country": "nature_wildlife",
    "nightlife": "urban_leisure",
    "food": "urban_leisure",
}

DEFAULT_RESTAURANT_PREFERENCES = {
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
}


def _string_list(value, field):
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{field} must be a list of strings.")
    return value


def derive_num_days(check_in: str, check_out: str) -> int:
    """Return stay length in days, requiring ISO calendar dates."""
    dates = []
    for field, value in (("check_in", check_in), ("check_out", check_out)):
        if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", value
        ):
            raise ValueError(f"{field} must be a date in YYYY-MM-DD format.")
        try:
            dates.append(date.fromisoformat(value))
        except ValueError as error:
            raise ValueError(f"{field} must be a valid calendar date.") from error
    num_days = (dates[1] - dates[0]).days
    if num_days < 1:
        raise ValueError("check_out must be later than check_in (at least one trip day).")
    return num_days


def map_interests_to_preferences(interests: list[str]) -> dict[str, float]:
    """Map known interests to 5.0, leaving other categories at 1.0."""
    _string_list(interests, "interests")
    preferences = dict.fromkeys([
        "history_culture", "nature_wildlife", "beach_coastal",
        "religious", "adventure_recreation", "urban_leisure",
    ], 1.0)
    for interest in interests:
        category = INTEREST_CATEGORIES.get(interest.strip().lower())
        if category:
            preferences[category] = 5.0
    return preferences


def adapt_c2_to_c3(payload: dict) -> dict:
    """Adapt an external C2 payload without mutating it.

    Missing restaurant preferences use a fresh copy of the C3 test defaults.
    Explicit restaurant and medical preferences pass through for validation
    by the existing API models. This function accepts external C2 only;
    the API retains its existing validation path for internal requests.
    """
    if not isinstance(payload, dict):
        raise ValueError("C2 input must be an object.")
    trip_id = payload.get("trip_id")
    if not isinstance(trip_id, str) or not trip_id.strip():
        raise ValueError("trip_id must be a non-empty string.")
    hotel = payload.get("selected_hotel")
    if not isinstance(hotel, dict):
        raise ValueError("selected_hotel is required and must be an object.")
    for field in ("hotel_id", "name", "district"):
        if not isinstance(hotel.get(field), str) or not hotel[field].strip():
            raise ValueError(f"selected_hotel.{field} must be a non-empty string.")
    for field, limit in (("latitude", 90), ("longitude", 180)):
        value = hotel.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not -limit <= value <= limit):
            raise ValueError(f"selected_hotel.{field} must be a valid coordinate.")

    trip = payload.get("trip")
    if not isinstance(trip, dict):
        raise ValueError("trip is required and must be an object.")
    num_days = derive_num_days(trip.get("check_in"), trip.get("check_out"))
    schedule = {
        "num_days": num_days,
        "day_start_hour": trip.get("day_start_hour", 9),
        "day_end_hour": trip.get("day_end_hour", 18),
        "travel_mode": trip.get("travel_mode", "driving"),
    }
    for field in ("day_start_hour", "day_end_hour"):
        if type(schedule[field]) is not int or not 0 <= schedule[field] <= 23:
            raise ValueError(f"trip.{field} must be an integer between 0 and 23.")
    if schedule["day_end_hour"] <= schedule["day_start_hour"]:
        raise ValueError("day_end_hour must be later than day_start_hour.")
    if not isinstance(schedule["travel_mode"], str):
        raise ValueError("trip.travel_mode must be a string.")
    if "want_to_visit" in trip:
        schedule["want_to_visit"] = deepcopy(
            _string_list(trip["want_to_visit"], "trip.want_to_visit")
        )

    context = {}
    for field in ("adults", "children", "travel_style", "dates_fixed",
                  "check_in", "check_out", "stay_in"):
        if field not in trip:
            continue
        value = trip[field]
        if field in ("adults", "children"):
            if type(value) is not int or value < 0:
                raise ValueError(f"trip.{field} must be a non-negative integer.")
        elif field == "dates_fixed" and type(value) is not bool:
            raise ValueError("trip.dates_fixed must be a boolean.")
        elif field == "travel_style" and not isinstance(value, str):
            raise ValueError("trip.travel_style must be a string.")
        elif field == "stay_in":
            _string_list(value, "trip.stay_in")
        context[field] = deepcopy(value)
    interests = _string_list(payload.get("interests", []), "interests")
    health = _string_list(payload.get("health", []), "health")
    context["interests"] = deepcopy(interests)
    context["health"] = deepcopy(health)
    reduced_mobility = any(item.strip().lower() == "mobility" for item in health)

    return {
        "trip_id": trip_id,
        "hotel": {field: hotel[field] for field in (
            "hotel_id", "name", "latitude", "longitude", "district"
        )},
        "trip": schedule,
        "attraction_preferences": map_interests_to_preferences(interests),
        "restaurant_preferences": deepcopy(payload.get(
            "restaurant_preferences", DEFAULT_RESTAURANT_PREFERENCES
        )),
        "medical_support": deepcopy(payload.get(
            "medical_support", {"required": False, "priority": "normal"}
        )),
        "accessibility": {
            "reduced_mobility": reduced_mobility,
            "prefer_low_walking": reduced_mobility,
        },
        "context": context,
    }
