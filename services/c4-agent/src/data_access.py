"""Read-only access to the datasets published by Component 1.

Files are loaded once and cached, because they are small (about 1 MB) and
never change while the service runs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Hotel:
    hotel_id: str
    name: str
    city: str
    district: str
    province: str
    latitude: float
    longitude: float
    property_type: str
    star_rating: float | None
    platform_rating: float | None
    review_count: int


@dataclass(frozen=True)
class Destination:
    destination_id: str
    name: str
    category: str
    city: str
    district: str
    latitude: float
    longitude: float
    rating: float | None


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Pull the latest main branch or set C1_DATA_DIR."
        )
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def load_hotels() -> dict[str, Hotel]:
    hotels: dict[str, Hotel] = {}
    for row in _read_csv(config.HOTELS_SEED_FILE):
        lat, lon = _to_float(row.get("latitude")), _to_float(row.get("longitude"))
        if not row.get("hotel_id") or lat is None or lon is None:
            continue
        stars = _to_float(row.get("star_rating"))
        hotels[row["hotel_id"]] = Hotel(
            hotel_id=row["hotel_id"],
            name=row["hotel_name"].strip(),
            city=row.get("city", "").strip(),
            district=row.get("district", "").strip(),
            province=row.get("province", "").strip(),
            latitude=lat,
            longitude=lon,
            property_type=row.get("property_type", "").strip(),
            # The seed file contains some placeholder values above 5; treat them as unknown.
            star_rating=stars if stars is not None and 0 < stars <= 5 else None,
            platform_rating=_to_float(row.get("platform_overall_rating")),
            review_count=int(_to_float(row.get("platform_total_reviews_count")) or 0),
        )
    return hotels


@lru_cache(maxsize=1)
def load_aspect_profiles() -> dict[str, dict]:
    path = config.HOTEL_PROFILES_FILE
    if not path.exists():
        raise FileNotFoundError(f"C1 profiles not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_destinations() -> list[Destination]:
    out = []
    for row in _read_csv(config.DESTINATIONS_FILE):
        lat, lon = _to_float(row.get("latitude")), _to_float(row.get("longitude"))
        if lat is None or lon is None:
            continue
        out.append(
            Destination(
                destination_id=row["destination_id"],
                name=row["destination_name"].strip(),
                category=row.get("category", "").strip(),
                city=row.get("city", "").strip(),
                district=row.get("district", "").strip(),
                latitude=lat,
                longitude=lon,
                rating=_to_float(row.get("platform_overall_rating")),
            )
        )
    return out
