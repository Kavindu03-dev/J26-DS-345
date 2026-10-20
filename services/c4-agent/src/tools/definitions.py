"""The four tools as the language model sees them, plus the mapping to C1-C3.

The model never sees the components' raw contracts. It sees these simpler,
user-level tools (hotel NAMES, "days" instead of check-out dates, ...) and the
agent's executor translates each call into the component's real request
(to_c2_request / to_c3_request). This keeps the model's job small and keeps
the shared contracts untouched.

Version every change: the tourism test set's gold labels are written against
TOOLS_VERSION.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

TOOLS_VERSION = "1.0.0"
TODAY = date(2026, 10, 18)  # fixed "today" used in the system prompt for reproducible evaluation

ASPECTS = ["food", "staff", "room", "cleanliness", "location", "wifi"]
NEARBY = ["hospital", "beach", "town_centre", "attractions"]
PURPOSES = ["leisure", "family", "business", "honeymoon"]
# C3's interest names (services/c3-itinerary-planner/src/adapters/c2_adapter.py)
C3_INTERESTS = ["nature", "wildlife", "beach", "culture", "religious", "surfing", "hiking", "tea_country", "nightlife", "food"]

DATE = {"type": "string", "description": "Date as YYYY-MM-DD."}

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_review_insights",
            "description": (
                "Trust-weighted guest-review scores (food, staff, room, cleanliness, location, wifi; 0-5), "
                "an overall trust score and verified facts from genuine reviews for ONE hotel in Sri Lanka. "
                "Use for questions about a specific hotel's quality or reviews."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel": {"type": "string", "description": "Hotel name as the user wrote it, or a hotel_id returned by recommend_hotels."},
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string", "enum": ASPECTS},
                        "description": "Only these aspects. Omit to get all six.",
                    },
                },
                "required": ["hotel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_hotels",
            "description": (
                "Rank hotels in a Sri Lankan town or district for this traveller, with a fair-price check and the "
                "cheapest booking platform (Booking.com, Agoda, Trip.com). Use when the user wants hotel suggestions, "
                "or needs a hotel before an itinerary can be planned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Town or district, e.g. Ella, Nuwara Eliya, Galle Fort."},
                    "check_in": DATE,
                    "check_out": DATE,
                    "adults": {"type": "integer", "minimum": 1},
                    "children": {"type": "integer", "minimum": 0},
                    "budget_min_lkr": {"type": "number", "description": "Minimum price per room per night in LKR."},
                    "budget_max_lkr": {"type": "number", "description": "Maximum price per room per night in LKR."},
                    "travel_purpose": {"type": "string", "enum": PURPOSES},
                    "nearby": {
                        "type": "array",
                        "items": {"type": "string", "enum": NEARBY},
                        "description": "Places the hotel should be close to.",
                    },
                    "interests": {"type": "array", "items": {"type": "string"}, "description": "Free-text interests, e.g. tea plantations."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3, "description": "How many hotels to return."},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_itinerary",
            "description": (
                "Build a day-by-day sightseeing route that starts and ends at ONE chosen hotel, with restaurants "
                "along the route. Needs a specific hotel: if the user has not chosen one, call recommend_hotels first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel": {"type": "string", "description": "Hotel name, or a hotel_id returned by recommend_hotels."},
                    "days": {"type": "integer", "minimum": 1, "maximum": 14},
                    "start_date": DATE,
                    "interests": {"type": "array", "items": {"type": "string", "enum": C3_INTERESTS}},
                    "mobility_needs": {"type": "boolean", "description": "True if a traveller has limited mobility (less walking)."},
                    "day_start_hour": {"type": "integer", "minimum": 0, "maximum": 23, "default": 9},
                    "day_end_hour": {"type": "integer", "minimum": 0, "maximum": 23, "default": 18},
                },
                "required": ["hotel", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for travel information the other tools do not cover: weather, trains and transport, "
                "entry fees, visas, opening hours, events, safety, currency. Do not use it for hotel quality, hotel "
                "recommendations or itineraries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

SYSTEM_PROMPT = (
    "You are a travel assistant for Sri Lanka. Use the tools to answer questions about hotels, reviews, "
    "itineraries and travel information. If a request needs no tool, or none of the tools can help, answer "
    f"directly without calling a tool. Today's date is {TODAY.isoformat()}."
)


def validate_call(name: str, arguments: dict) -> list[str]:
    """Schema errors for one call ([] = valid). Strict: unknown arguments are errors."""
    import jsonschema

    if name not in SCHEMAS:
        return [f"unknown tool '{name}'"]
    schema = dict(SCHEMAS[name], additionalProperties=False)
    return [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(arguments)]


def to_c2_request(args: dict) -> dict:
    """recommend_hotels arguments -> C2 POST /recommendations body (contracts/c2_hotel_ranking.json)."""
    req: dict[str, Any] = {"destination": args["destination"]}
    for k in ("check_in", "check_out", "adults", "children", "travel_purpose", "top_k"):
        if k in args:
            req[k] = args[k]
    if "budget_min_lkr" in args or "budget_max_lkr" in args:
        req["budget_lkr"] = {
            k: args[f"budget_{k}_lkr"] for k in ("min", "max") if f"budget_{k}_lkr" in args
        }
    if args.get("interests"):
        req["interests"] = list(args["interests"])
    if args.get("nearby"):  # C2's contract has no proximity field yet; carried as accessibility needs
        req["accessibility_needs"] = [f"near_{n}" for n in args["nearby"]]
    return req


def to_c3_request(args: dict, hotel: dict, trip_id: str, today: date = TODAY) -> dict:
    """plan_itinerary arguments + resolved hotel record -> C3 POST /plan-itinerary body (external C2 format).

    hotel: {"hotel_id", "name", "latitude", "longitude", "district"} (from the hotel directory).
    """
    start = date.fromisoformat(args["start_date"]) if args.get("start_date") else today + timedelta(days=1)
    trip: dict[str, Any] = {
        "check_in": start.isoformat(),
        "check_out": (start + timedelta(days=int(args["days"]))).isoformat(),
    }
    for k in ("day_start_hour", "day_end_hour"):
        if k in args:
            trip[k] = args[k]
    return {
        "trip_id": trip_id,
        "selected_hotel": hotel,
        "trip": trip,
        "interests": list(args.get("interests", [])),
        "health": ["mobility"] if args.get("mobility_needs") else [],
    }
