"""Mock of Component 3 (itinerary optimisation and route-based restaurants).

Schema: services/c4-agent/schemas/c3_plan_itinerary.json (C4's record of C3's real API).
The shared contracts/c3_itinerary.json is an older draft (POST /itineraries).
Endpoint: POST /plan-itinerary  (C3's real service runs on port 8003)

The request is the "external C2" payload that C3's adapter
(services/c3-itinerary-planner/src/adapters/c2_adapter.py) accepts.

What is real and what is simulated:
- REAL: attraction names, categories and coordinates (C1's destinations_seed.csv).
- SIMULATED: the route (a simple greedy nearest-next heuristic, not C3's
  TOPTW/OR-Tools optimiser) and all restaurants. Responses carry
  "source": "c4-mock".
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from ..data_access import Destination, load_destinations
from ..geo import drive_minutes, road_km

VISIT_MINUTES = 90
MAX_STOPS_PER_DAY = 3
MAX_RADIUS_KM = 120.0

# Same interest names and categories as C3's adapter (INTEREST_CATEGORIES).
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

# Keywords that map an attraction's free-text category to a C3 category.
CATEGORY_KEYWORDS = {
    "history_culture": ("heritage", "historical", "history", "fort", "colonial", "cultural", "museum", "ancient"),
    "religious": ("religious", "temple", "kovil", "church", "mosque", "stupa", "cave temple"),
    "nature_wildlife": ("nature", "national park", "wildlife", "waterfall", "mountain", "tea", "forest", "garden", "lake"),
    "beach_coastal": ("beach", "coastal", "bay", "lagoon"),
    "adventure_recreation": ("adventure", "hike", "hiking", "trek", "surf", "rock"),
    "urban_leisure": ("city", "urban", "market", "shopping", "leisure"),
}

MOCK_RESTAURANTS = (
    ("Rice & Curry House", "sri_lankan"),
    ("Spice Garden Cafe", "local"),
    ("Hoppers Corner", "sri_lankan"),
    ("Curry Leaf Kitchen", "south_indian"),
    ("Lotus Asian Bistro", "asian"),
)


class SelectedHotel(BaseModel):
    hotel_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    district: str = Field(min_length=1)


class TripDetails(BaseModel):
    check_in: date
    check_out: date
    stay_in: list[str] | None = None
    want_to_visit: list[str] | None = None
    adults: int | None = Field(default=None, ge=0)
    children: int | None = Field(default=None, ge=0)
    travel_style: str | None = None
    dates_fixed: bool | None = None
    day_start_hour: int = Field(default=9, ge=0, le=23)
    day_end_hour: int = Field(default=18, ge=0, le=23)
    travel_mode: str = "driving"

    @model_validator(mode="after")
    def _check(self):
        days = (self.check_out - self.check_in).days
        if days < 1:
            raise ValueError("check_out must be later than check_in (at least one trip day).")
        if days > 14:
            raise ValueError("Trips longer than 14 days are not supported.")
        if self.day_end_hour <= self.day_start_hour:
            raise ValueError("day_end_hour must be later than day_start_hour.")
        if self.travel_mode.lower().strip() != "driving":
            raise ValueError("Currently supported travel_mode is 'driving'.")
        return self


class MedicalSupport(BaseModel):
    required: bool = False
    priority: str = "normal"


class PlanItineraryRequest(BaseModel):
    """The "external C2" payload accepted by C3's /plan-itinerary."""

    trip_id: str = Field(min_length=1)
    selected_hotel: SelectedHotel
    trip: TripDetails
    interests: list[str] = Field(default_factory=list)
    health: list[str] = Field(default_factory=list)
    restaurant_preferences: dict | None = None
    medical_support: MedicalSupport = Field(default_factory=MedicalSupport)


def categorise(destination: Destination) -> str:
    text = destination.category.lower() + " " + destination.name.lower()
    for category, words in CATEGORY_KEYWORDS.items():
        if any(w in text for w in words):
            return category
    return "history_culture"


def _hhmm(start_hour: int, minutes_from_start: int) -> str:
    total = start_hour * 60 + minutes_from_start
    return f"{total // 60:02d}:{total % 60:02d}"


def plan(req: PlanItineraryRequest) -> dict:
    hotel = req.selected_hotel
    trip = req.trip
    num_days = (trip.check_out - trip.check_in).days
    wanted = {INTEREST_CATEGORIES[i.strip().lower()] for i in req.interests if i.strip().lower() in INTEREST_CATEGORIES}

    candidates = []
    for d in load_destinations():
        km = road_km(hotel.latitude, hotel.longitude, d.latitude, d.longitude)
        if km <= MAX_RADIUS_KM:
            match = 1.0 if categorise(d) in wanted else 0.4
            candidates.append((d, km, match))
    # Prefer attractions that match the interests, then closer ones.
    candidates.sort(key=lambda c: (-c[2], c[1]))

    window = (trip.day_end_hour - trip.day_start_hour) * 60
    used: set[str] = set()
    days = []
    for day in range(1, num_days + 1):
        stops, t = [], 0
        lat, lon = hotel.latitude, hotel.longitude
        while len(stops) < MAX_STOPS_PER_DAY:
            options = [c for c in candidates if c[0].destination_id not in used]
            if not options:
                break
            # Greedy: best interest match first, then nearest to the current position.
            dest, _, match = min(
                options, key=lambda c: (-c[2], road_km(lat, lon, c[0].latitude, c[0].longitude))
            )
            leg = drive_minutes(road_km(lat, lon, dest.latitude, dest.longitude))
            back = drive_minutes(road_km(dest.latitude, dest.longitude, hotel.latitude, hotel.longitude))
            if t + leg + VISIT_MINUTES + back > window:
                break
            t += leg
            stops.append(
                {
                    "day": day,
                    "stop_order": len(stops) + 1,
                    "poi_id": dest.destination_id,
                    "name": dest.name,
                    "latitude": dest.latitude,
                    "longitude": dest.longitude,
                    "arrival_time": _hhmm(trip.day_start_hour, t),
                    "arrival_min_from_start": t,
                    "preference_score": round(match * (dest.rating or 4.0), 2),
                }
            )
            used.add(dest.destination_id)
            t += VISIT_MINUTES
            lat, lon = dest.latitude, dest.longitude

        # Same fields as C3's final_ranked_restaurants.csv (06_rank_restaurants_with_absa.py).
        restaurants = []
        for rank, (name, cuisine) in enumerate(MOCK_RESTAURANTS[(day - 1) % 2 :: 2][:2], start=1):
            detour = 4 * rank + day
            restaurants.append(
                {
                    "day": day,
                    "final_day_rank": rank,
                    "name": f"{name} (mock)",
                    "cuisine": cuisine,
                    "detour_time_min": detour,
                    "detour_score": round(1 - detour / 30, 4),
                    "cuisine_preference_score": 1.0 if cuisine in ("sri_lankan", "local") else 0.8,
                    "weighted_absa_score": round(0.86 - 0.04 * rank, 4),
                    "total_review_count": 120 - 30 * rank,
                    "final_restaurant_score": round(0.84 - 0.05 * rank, 4),
                    "scoring_mode": "mock",
                }
            )
        days.append({"day": day, "stops": stops, "recommended_restaurants": restaurants})

    return {
        "status": "success",
        "trip_id": req.trip_id,
        "hotel": hotel.model_dump(),
        "trip": {
            "num_days": num_days,
            "day_start_hour": trip.day_start_hour,
            "day_end_hour": trip.day_end_hour,
            "travel_mode": trip.travel_mode,
        },
        "medical_support": {
            "required": req.medical_support.required,
            "priority": req.medical_support.priority,
            "average_hospital_access_min": None,
            "best_hospital_access_min": None,
            "worst_hospital_access_min": None,
        },
        "days": days,
        "pipeline": [{"stage": "c4-mock-greedy", "status": "success"}],
        "source": "c4-mock",
    }


app = FastAPI(
    title="C3 mock - itinerary planner",
    description="Greedy routes over real attractions; simulated restaurants. For C4 development only.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"service": "c3-mock", "status": "ok"}


@app.post("/plan-itinerary")
def plan_itinerary(req: PlanItineraryRequest):
    try:
        return plan(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
