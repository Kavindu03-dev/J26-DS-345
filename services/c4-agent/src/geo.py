"""Small geographic helpers shared by the mocks and the agent."""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0
# Sri Lankan roads are winding: road distance is roughly 1.4x the straight line.
ROAD_FACTOR = 1.4
AVG_SPEED_KMH = 35.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def road_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(lat1, lon1, lat2, lon2) * ROAD_FACTOR


def drive_minutes(km: float) -> int:
    return max(1, round(km / AVG_SPEED_KMH * 60))
