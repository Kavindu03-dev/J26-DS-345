"""
FastAPI Microservice for Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine.
Implements the shared contract defined in contracts/c1_aspect_scores.json.
Port: 8001
"""

import csv
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException

import json
from src.config import (
    HOTELS_SEED_FILE,
    REVIEWS_BOOKING_FILE,
    REVIEWS_GOOGLEMAPS_FILE,
    HOTEL_PROFILES_JSON
)
from contextlib import asynccontextmanager
from src.models.aggregator import AspectScoreAggregator


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dataset_indexes()
    yield


app = FastAPI(
    title="C1 Trust-Aware Review & Aspect Sentiment Engine",
    description="SLIIT Research Project J26-DS-345 - Component 1 API",
    version="1.0.0",
    lifespan=lifespan
)


from src.models.alert_engine import WeakAspectAlertEngine

# Initialize models
aggregator = AspectScoreAggregator()
alert_engine = WeakAspectAlertEngine()

# In-memory indexes for fast serving
HOTELS_INDEX: Dict[str, Dict[str, Any]] = {}
REVIEWS_BY_HOTEL: Dict[str, List[Dict[str, Any]]] = {}
PRECOMPUTED_PROFILES: Dict[str, Dict[str, Any]] = {}


def load_dataset_indexes():
    """Builds fast lookup tables and loads pre-computed profiles if available."""
    global HOTELS_INDEX, REVIEWS_BY_HOTEL, PRECOMPUTED_PROFILES
    HOTELS_INDEX.clear()
    REVIEWS_BY_HOTEL.clear()
    PRECOMPUTED_PROFILES.clear()

    # 1. Load Pre-computed Profiles (if batch_enricher has run)
    if HOTEL_PROFILES_JSON.exists():
        try:
            with open(HOTEL_PROFILES_JSON, "r", encoding="utf-8") as f:
                PRECOMPUTED_PROFILES = json.load(f)
            print(f"[C1 Service] Loaded {len(PRECOMPUTED_PROFILES)} pre-computed hotel profiles into fast cache.")
        except Exception as e:
            print(f"[!] Warning reading pre-computed profiles: {e}")

    # 2. Index Hotels
    if HOTELS_SEED_FILE.exists():
        with open(HOTELS_SEED_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                h_id = row.get("hotel_id", "")
                if h_id:
                    HOTELS_INDEX[h_id] = row
                    REVIEWS_BY_HOTEL[h_id] = []

    # 2. Index Booking Reviews
    if REVIEWS_BOOKING_FILE.exists():
        with open(REVIEWS_BOOKING_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                h_id = row.get("hotel_id", "")
                if h_id in REVIEWS_BY_HOTEL:
                    row["source_platform"] = "Booking.com"
                    REVIEWS_BY_HOTEL[h_id].append(row)

    # 3. Index Google Maps Reviews
    if REVIEWS_GOOGLEMAPS_FILE.exists():
        with open(REVIEWS_GOOGLEMAPS_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                h_id = row.get("hotel_id", "")
                if h_id in REVIEWS_BY_HOTEL:
                    row["source_platform"] = "Google Maps"
                    REVIEWS_BY_HOTEL[h_id].append(row)

    print(f"[C1 Service] Indexed {len(HOTELS_INDEX)} hotels and {sum(len(v) for v in REVIEWS_BY_HOTEL.values())} reviews.")

@app.get("/health")
def health():
    return {
        "service": "c1-review-engine",
        "status": "ok",
        "indexed_hotels": len(HOTELS_INDEX),
        "total_reviews": sum(len(v) for v in REVIEWS_BY_HOTEL.values())
    }


@app.get("/hotels/{hotel_id}/aspect-scores")
@app.get("/api/v1/aspects/{hotel_id}")
def get_hotel_aspect_scores(hotel_id: str):
    """
    Returns trust-weighted aspect scores, overall credibility, time-decay rating,
    and verified aspect facts for a specific hotel adhering to contracts/c1_aspect_scores.json.
    """
    if not HOTELS_INDEX and not PRECOMPUTED_PROFILES:
        load_dataset_indexes()

    # Fast path: Instant sub-millisecond cached profile
    if hotel_id in PRECOMPUTED_PROFILES:
        return PRECOMPUTED_PROFILES[hotel_id]

    # Dynamic fallback: Compute on-the-fly
    hotel = HOTELS_INDEX.get(hotel_id)
    if not hotel:
        raise HTTPException(
            status_code=404,
            detail=f"Hotel '{hotel_id}' not found in seed dataset."
        )

    reviews = REVIEWS_BY_HOTEL.get(hotel_id, [])
    response_payload = aggregator.aggregate_hotel_aspect_scores(hotel, reviews)
    return response_payload


@app.get("/hotels/{hotel_id}/alerts")
@app.get("/api/v1/hotels/{hotel_id}/alerts")
def get_hotel_alerts(hotel_id: str):
    """
    Returns active operational alerts (CRITICAL, WARNING, INFO) for a specific hotel (Requirement FR9).
    Includes actionable management recommendations and extracted negative review quotes.
    """
    if not HOTELS_INDEX:
        load_dataset_indexes()

    hotel = HOTELS_INDEX.get(hotel_id)
    if not hotel:
        raise HTTPException(
            status_code=404,
            detail=f"Hotel '{hotel_id}' not found in seed dataset."
        )

    reviews = REVIEWS_BY_HOTEL.get(hotel_id, [])
    hotel_name = hotel.get("hotel_name", "Unknown Hotel")
    try:
        baseline_rating = float(hotel.get("platform_overall_rating", 8.0) or 8.0)
    except (ValueError, TypeError):
        baseline_rating = 8.0

    alerts = alert_engine.evaluate_hotel_alerts(hotel_id, hotel_name, reviews, baseline_rating)
    return {
        "hotel_id": hotel_id,
        "hotel_name": hotel_name,
        "total_reviews_analyzed": len(reviews),
        "active_alerts_count": len(alerts),
        "critical_count": sum(1 for a in alerts if a["severity"] == "CRITICAL"),
        "warning_count": sum(1 for a in alerts if a["severity"] == "WARNING"),
        "info_count": sum(1 for a in alerts if a["severity"] == "INFO"),
        "alerts": alerts
    }


@app.get("/api/v1/alerts/critical")
def get_critical_alerts(district: str = None, limit: int = 25):
    """
    Scans hotels and returns properties experiencing CRITICAL operational quality deficits.
    Useful for district tourism boards and hotel management groups.
    """
    if not HOTELS_INDEX:
        load_dataset_indexes()

    critical_hotels = []
    count = 0

    for h_id, hotel in HOTELS_INDEX.items():
        if district and hotel.get("district", "").lower() != district.lower():
            continue

        reviews = REVIEWS_BY_HOTEL.get(h_id, [])
        if len(reviews) < 3:
            continue

        try:
            baseline_rating = float(hotel.get("platform_overall_rating", 8.0) or 8.0)
        except (ValueError, TypeError):
            baseline_rating = 8.0

        alerts = alert_engine.evaluate_hotel_alerts(h_id, hotel.get("hotel_name", ""), reviews, baseline_rating)
        crit_alerts = [a for a in alerts if a["severity"] == "CRITICAL"]

        if crit_alerts:
            critical_hotels.append({
                "hotel_id": h_id,
                "hotel_name": hotel.get("hotel_name", ""),
                "city": hotel.get("city", ""),
                "district": hotel.get("district", ""),
                "review_count": len(reviews),
                "critical_alerts": crit_alerts
            })
            count += 1
            if count >= limit:
                break

    return {
        "filter_district": district,
        "critical_hotels_count": len(critical_hotels),
        "critical_hotels": critical_hotels
    }

