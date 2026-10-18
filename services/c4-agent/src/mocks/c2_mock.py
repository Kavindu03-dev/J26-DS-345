"""Mock of Component 2 (personalised hotel recommendation and price intelligence).

Contract: contracts/c2_hotel_ranking.json
Endpoint: POST /recommendations

What is real and what is simulated:
- REAL: hotel names, locations and the C1 aspect / trust scores used for ranking.
- SIMULATED: all prices and platform offers. The seed data has no real prices
  yet, so prices are generated deterministically from the hotel ID (the same
  hotel always gets the same price). Every response carries "source": "c4-mock".

The ranking formula is deliberately simple. It only has to behave sensibly
(respect the budget, prefer better-rated hotels) so that C4's routing and
argument-passing can be tested. The real ranking model belongs to C2.
"""

from __future__ import annotations

import hashlib
from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from ..data_access import Hotel, load_aspect_profiles
from ..hotel_directory import HotelDirectory

PLATFORMS = ("Booking.com", "Agoda", "Trip.com")
OVERPRICED_THRESHOLD = 1.10  # listed price more than 10% above expected

# Which C1 aspects matter most for each travel purpose (weights sum to 1).
PURPOSE_WEIGHTS: dict[str, dict[str, float]] = {
    "family": {"cleanliness": 0.3, "room": 0.25, "food": 0.2, "staff": 0.15, "location": 0.1, "wifi": 0.0},
    "business": {"wifi": 0.3, "location": 0.25, "staff": 0.2, "cleanliness": 0.15, "room": 0.1, "food": 0.0},
    "honeymoon": {"room": 0.3, "cleanliness": 0.25, "staff": 0.2, "food": 0.15, "location": 0.1, "wifi": 0.0},
    "leisure": {"location": 0.25, "food": 0.2, "staff": 0.2, "cleanliness": 0.2, "room": 0.15, "wifi": 0.0},
}
DEFAULT_WEIGHTS = {a: 1 / 6 for a in ("food", "staff", "room", "cleanliness", "location", "wifi")}

# Simulated base price per room per night (LKR) by star rating.
BASE_PRICE_LKR = {1: 9000, 2: 12000, 3: 18000, 4: 32000, 5: 55000}


class BudgetLKR(BaseModel):
    min: float | None = Field(default=None, ge=0)
    max: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _order(self):
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("budget_lkr.min must be <= budget_lkr.max")
        return self


class RecommendationRequest(BaseModel):
    """Mirrors the request schema in contracts/c2_hotel_ranking.json."""

    destination: str = Field(min_length=1)
    check_in: date | None = None
    check_out: date | None = None
    adults: int | None = Field(default=None, ge=1)
    children: int | None = Field(default=None, ge=0)
    budget_lkr: BudgetLKR | None = None
    travel_purpose: str | None = None
    interests: list[str] = Field(default_factory=list)
    accessibility_needs: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _dates(self):
        if self.check_in and self.check_out and self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        return self


def _unit(hotel_id: str, salt: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from a hotel ID."""
    digest = hashlib.sha256(f"{hotel_id}:{salt}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def simulated_price(hotel: Hotel, quality: float) -> dict:
    """Simulated expected price and platform offers, per room per night (LKR)."""
    stars = round(hotel.star_rating) if hotel.star_rating else 3
    base = BASE_PRICE_LKR.get(stars, BASE_PRICE_LKR[3])
    # Better-reviewed hotels can charge a little more; +/-25% spread between hotels.
    expected = base * (0.85 + 0.3 * quality) * (0.75 + 0.5 * _unit(hotel.hotel_id, "base"))
    offers = []
    for p in PLATFORMS:
        # Some hotels are listed above their fair value on some platforms.
        markup = 0.92 + 0.30 * _unit(hotel.hotel_id, p)
        offers.append({"platform": p, "price_lkr": round(expected * markup, -2)})
    best = min(offers, key=lambda o: o["price_lkr"])
    status = (
        "Potentially Overpriced"
        if best["price_lkr"] > expected * OVERPRICED_THRESHOLD
        else "Within Expected Price Range"
    )
    return {
        "expected_price_lkr": round(expected, -2),
        "price_status": status,
        "best_platform": best["platform"],
        "offers": offers,
    }


def _explain(aspects: dict, weights: dict, price: dict, budget: BudgetLKR | None) -> str:
    top = sorted(weights, key=lambda a: weights[a] * (aspects.get(a) or 0), reverse=True)[:2]
    parts = [f"strong {top[0]} ({aspects[top[0]]}/5) and {top[1]} ({aspects[top[1]]}/5) scores from verified reviews"]
    best = min(o["price_lkr"] for o in price["offers"])
    if budget and budget.max:
        parts.append(f"LKR {best:,.0f} per night on {price['best_platform']}, within your LKR {budget.max:,.0f} budget")
    else:
        parts.append(f"best price LKR {best:,.0f} per night on {price['best_platform']}")
    if price["price_status"] == "Potentially Overpriced":
        parts.append("note: currently listed above its expected fair price")
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


def rank_hotels(req: RecommendationRequest, directory: HotelDirectory | None = None) -> dict:
    directory = directory or HotelDirectory()
    candidates = directory.in_destination(req.destination)
    if not candidates:
        raise LookupError(f"Unknown destination '{req.destination}'.")

    profiles = load_aspect_profiles()
    weights = PURPOSE_WEIGHTS.get((req.travel_purpose or "").lower(), DEFAULT_WEIGHTS)
    scored = []
    for hotel in candidates:
        profile = profiles.get(hotel.hotel_id)
        if not profile:
            continue  # no trusted review data: C2 would not rank it either
        aspects = profile["aspect_scores"]
        quality = sum(w * (aspects.get(a) or 0) for a, w in weights.items()) / 5
        price = simulated_price(hotel, quality)
        best_price = min(o["price_lkr"] for o in price["offers"])
        b = req.budget_lkr
        if b and ((b.max is not None and best_price > b.max) or (b.min is not None and best_price < b.min)):
            continue
        value = 1.0 if price["price_status"] == "Within Expected Price Range" else 0.6
        score = 0.7 * quality + 0.2 * profile["overall_trust_score"] / 10 + 0.1 * value
        scored.append((round(score, 3), hotel, aspects, price))

    scored.sort(key=lambda s: (-s[0], s[1].hotel_id))
    ranked = [
        {
            "hotel_id": hotel.hotel_id,
            "name": hotel.name,
            "rank": i,
            "score": score,
            "price": price,
            "explanation": _explain(aspects, weights, price, req.budget_lkr),
        }
        for i, (score, hotel, aspects, price) in enumerate(scored[: req.top_k], start=1)
    ]
    return {"ranked_hotels": ranked, "source": "c4-mock"}


app = FastAPI(
    title="C2 mock - hotel recommendations",
    description="Simulated prices on top of real hotels and real C1 scores. For C4 development only.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"service": "c2-mock", "status": "ok"}


@app.post("/recommendations")
def recommendations(req: RecommendationRequest):
    try:
        return rank_hotels(req)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
