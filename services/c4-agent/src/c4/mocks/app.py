"""Mock Tools 1-3 on port 8090. Dummy data, fully deterministic (no clock, no randomness).

POST /review-insights, /recommend-hotels, /plan-itinerary take the tool arguments as JSON.
Bad arguments -> 422, unknown hotel_id -> 404. `?delay_ms=` (or MOCK_DELAY_MS) delays the
reply for timeout tests. Every successful reply carries `X-C4-Mock: true`.
"""

import asyncio
import json
import math
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Response

from c4.common.config import get_settings
from c4.common.schemas import SchemaError, validate_arguments

DATA = json.loads(Path(__file__).with_name("fixtures.json").read_text(encoding="utf-8"))
DESTINATIONS: dict[str, dict[str, Any]] = DATA["destinations"]
HOTELS = {h["hotel_id"]: (d, h) for d, v in DESTINATIONS.items() for h in v["hotels"]}

OVERPRICED_RATIO = 1.10
WEAK_BELOW = 3.0
VISIT_MIN = 120
TRANSFER_MIN = 30
MAX_STOPS_PER_DAY = 4
SPEED_KMH = 30


async def _delay(
    response: Response, delay_ms: Annotated[int | None, Query(ge=0, le=60_000)] = None
) -> None:
    response.headers["X-C4-Mock"] = "true"
    ms = get_settings().mock_delay_ms if delay_ms is None else delay_ms
    if ms:
        await asyncio.sleep(ms / 1000)


app = FastAPI(title="C4 mock Tools 1-3")
slow = [Depends(_delay)]
Arguments = Annotated[Any, Body()]


def _validate(tool: str, args: Any) -> None:
    try:
        validate_arguments(tool, args)
    except SchemaError as e:
        raise HTTPException(422, detail=e.errors) from e


def _hotel(hotel_id: str) -> tuple[str, dict[str, Any]]:
    if hotel_id not in HOTELS:
        raise HTTPException(404, detail=f"Unknown hotel_id: {hotel_id}")
    return HOTELS[hotel_id]


def _overlap(wanted: list[str], offered: list[str]) -> int:
    return len({w.lower() for w in wanted} & {o.lower() for o in offered})


def _km(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = (
        math.radians(v) for v in (a["latitude"], a["longitude"], b["latitude"], b["longitude"])
    )
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(h))


def _clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": "c4-mocks", "status": "ok"}


@app.post("/review-insights", dependencies=slow)
async def review_insights(args: Arguments) -> dict[str, Any]:
    _validate("get_review_insights", args)
    _, hotel = _hotel(args["hotel_id"])
    aspects = args.get("aspects", list(hotel["aspect_scores"]))
    scores = {a: hotel["aspect_scores"][a] for a in aspects}
    return {
        "aspect_scores": scores,
        "overall_trust_score": hotel["overall_trust_score"],
        "time_decay_rating": hotel["time_decay_rating"],
        "verified_aspect_facts": [hotel["facts"][a] for a in aspects],
        "weak_aspect_alert": [a for a, score in scores.items() if score < WEAK_BELOW],
    }


@app.post("/recommend-hotels", dependencies=slow)
async def recommend_hotels(args: Arguments) -> dict[str, Any]:
    """Prices are treated as per night. An unknown destination returns an empty list."""
    _validate("recommend_hotels", args)
    if args["budget_min_lkr"] > args["budget_max_lkr"]:
        raise HTTPException(422, detail=["budget_min_lkr must not exceed budget_max_lkr"])
    wanted = args["destination"].strip().lower()
    hotels = next((v["hotels"] for d, v in DESTINATIONS.items() if d.lower() == wanted), [])
    needs = {n.lower() for n in args.get("accessibility_needs", [])}

    ranked = []
    for h in hotels:
        listed, expected = h["listed_price_lkr"], h["expected_price_lkr"]
        if not args["budget_min_lkr"] <= listed <= args["budget_max_lkr"]:
            continue
        if not needs <= set(h["accessibility"]):
            continue
        matches = _overlap(args["interests"], h["interests"])
        fits_purpose = args["travel_purpose"].lower() in h["purposes"]
        overpriced = listed > expected * OVERPRICED_RATIO
        score = h["base_score"] + 0.1 * matches + 0.05 * fits_purpose - 0.1 * overpriced
        ranked.append(
            {
                "hotel_id": h["hotel_id"],
                "name": h["name"],
                "rank_score": round(score, 3),
                "expected_price_lkr": expected,
                "listed_price_lkr": listed,
                "overpriced": overpriced,
                "best_platform": h["best_platform"],
                "explanation": f"Within budget; matches {matches} of your interests.",
            }
        )
    ranked.sort(key=lambda r: (-r["rank_score"], r["hotel_id"]))
    return {"hotels": ranked[: int(args["top_k"])]}


@app.post("/plan-itinerary", dependencies=slow)
async def plan_itinerary(args: Arguments) -> dict[str, Any]:
    """Each attraction is visited once, so a long trip ends with empty days in this mock."""
    _validate("plan_itinerary", args)
    destination, hotel = _hotel(args["hotel_id"])
    data = DESTINATIONS[destination]
    queue = sorted(
        data["attractions"],
        key=lambda a: (-_overlap(args["interests"], a["interests"]), a["attraction_id"]),
    )
    slot = VISIT_MIN + TRANSFER_MIN
    per_day = min(MAX_STOPS_PER_DAY, int(args["daily_hours"] * 60 // slot))
    restaurants = data["restaurants"] if args["include_restaurants"] else []
    if cuisine := args.get("cuisine"):
        restaurants = [r for r in restaurants if r["cuisine"] == cuisine.lower()]

    days = []
    for _ in range(int(args["days"])):
        todays, queue = queue[:per_day], queue[per_day:]
        path = [hotel, *todays, hotel]
        km = sum(_km(a, b) for a, b in zip(path, path[1:], strict=False))
        stops = [
            {
                **{k: a[k] for k in ("attraction_id", "name", "latitude", "longitude")},
                "arrival_time": _clock(8 * 60 + i * slot),
                "departure_time": _clock(8 * 60 + i * slot + VISIT_MIN),
            }
            for i, a in enumerate(todays)
        ]
        days.append(
            {
                "stops": stops,
                "route": {
                    "distance_km": round(km, 2),
                    "travel_minutes": round(km / SPEED_KMH * 60, 1),
                },
                "restaurants": restaurants if todays else [],
            }
        )
    lat, lon = hotel["latitude"], hotel["longitude"]
    return {"days": days, "map_url": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}"}
