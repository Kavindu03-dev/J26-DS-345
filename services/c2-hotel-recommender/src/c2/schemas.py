"""Request formats for the API (team contract + a few optional extras)."""
from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# free words other components may send -> our fixed words
ACCESS_WORDS = {"wheelchair": "wheelchair_access", "wheelchair_access": "wheelchair_access",
                "wheelchair_accessible": "wheelchair_access", "low_steps": "low_steps", "no_stairs": "low_steps",
                "lift": "low_steps", "elevator": "low_steps", "ground_floor": "low_steps",
                "near_hospital": "near_hospital", "hospital": "near_hospital"}
INTEREST_WORDS_MAP = {"beach": "beach", "nature": "nature", "waterfall": "nature", "wildlife": "wildlife",
                      "safari": "wildlife", "culture": "culture", "history": "culture", "heritage": "culture",
                      "temple": "religious", "religious": "religious", "surf": "surfing", "hiking": "hiking",
                      "hike": "hiking", "trek": "hiking", "tea": "tea_country", "nightlife": "nightlife",
                      "food": "food", "restaurant": "food"}
STYLE_WORDS = {"solo": "solo", "couple": "couple", "honeymoon": "couple", "family": "family", "friends": "friends",
               "business": "business", "work": "business"}
HEALTH_TO_MUST = {"heart": "near_hospital", "breathing": "near_hospital", "mobility": "low_steps",
                  "elderly": "low_steps", "wheelchair": "wheelchair_access"}

Importance = Literal["high", "medium", "low"]


def _check_date(v):
    if v is None:
        return v
    try:
        date.fromisoformat(v)
    except (TypeError, ValueError):
        raise ValueError("write the date as YYYY-MM-DD, for example 2026-12-20")
    return v


class Budget(BaseModel):
    min: Optional[float] = Field(None, ge=0)
    max: Optional[float] = Field(None, ge=0)


class RecommendationRequest(BaseModel):
    """contracts/c2_hotel_ranking.json request. The fields after top_k are optional extras (proposal v0.2)."""
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "destination": "Kandy", "check_in": "2026-12-05", "check_out": "2026-12-07", "adults": 2, "children": 0,
        "budget_lkr": {"max": 15000}, "interests": ["culture"], "top_k": 3,
        "wishes": {"pool": "medium"}, "not_needed": [], "health": [], "health_consent": False}]})
    destination: str
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    adults: Optional[int] = Field(None, ge=1)
    children: Optional[int] = Field(None, ge=0)
    budget_lkr: Optional[Budget] = None
    travel_purpose: Optional[str] = None
    interests: List[str] = []
    accessibility_needs: List[str] = []
    top_k: int = Field(5, ge=1, le=50)
    # ---- optional extras ----
    nights: Optional[int] = Field(None, ge=1, le=90)
    dates_fixed: Optional[bool] = None
    budget_total_lkr: Optional[float] = Field(None, ge=0)
    must: List[str] = []
    wishes: Dict[str, Importance] = {}
    not_needed: List[Literal["pool", "bar", "breakfast", "beachfront"]] = []
    health: List[Literal["heart", "mobility", "wheelchair", "breathing", "elderly"]] = []
    health_consent: bool = False
    want_to_visit: List[str] = []
    today: Optional[str] = None

    _dates = field_validator("check_in", "check_out", "today")(_check_date)

    @model_validator(mode="after")
    def _order(self):
        if self.check_in and self.check_out and self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        return self


class TextRequest(BaseModel):
    """POST /recommendations/from-text: the tourist's own sentence."""
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "text": "Hotel in Kandy for 2 nights from 5 December, under Rs 15,000, pool please",
        "top_k": 3, "health_consent": False}]})
    text: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    health_consent: bool = False
    today: Optional[str] = None
    live_rates: bool = True

    _today = field_validator("today")(_check_date)


class LiveRequest(BaseModel):
    """POST /recommendations/live: search Booking.com for one town of an understood request (Output 1)."""
    request: dict
    town: str
    top_k: int = Field(5, ge=1, le=50)
    health_consent: bool = False
    today: Optional[str] = None

    _today = field_validator("today")(_check_date)
