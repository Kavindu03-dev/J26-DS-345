"""Isolated nearest-feasible-POI baseline; run directly from any directory."""

import json
import math
from pathlib import Path
import sys

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "Data/input/c2_c3_input.json"
POI_FILE = PROJECT_ROOT / "Data/processed/poi_preference_scores.csv"
MEDICAL_ACCESS_FILE = PROJECT_ROOT / "Data/processed/poi_medical_access.csv"
OUTPUT_FILE = PROJECT_ROOT / "Data/processed/baseline_greedy_itinerary.csv"
OSRM_URL = "http://localhost:5000"
CANDIDATE_RADIUS_KM = 40
MAX_CANDIDATES = 25
VISIT_DURATION = {
    "history_culture": 60,
    "nature_wildlife": 90,
    "beach_coastal": 90,
    "religious": 60,
    "adventure_recreation": 120,
    "urban_leisure": 60,
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Match production's radius filter, never use this for routing."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def estimate_visit_duration(row):
    return max(
        (duration for category, duration in VISIT_DURATION.items()
         if row.get(category, 0) == 1),
        default=60,
    )


def road_legs(session, anchor, others, travel_mode, returning=False):
    """Get directed OSRM legs from anchor, or from each POI back to anchor.

    Small table batches keep requests below OSRM's usual 100-location limit.
    Unreachable legs stay infinite; there is no straight-line fallback.
    """
    legs = []
    for start in range(0, len(others), 90):
        batch = others[start:start + 90]
        coordinates = ";".join(
            f"{place['longitude']},{place['latitude']}"
            for place in [anchor, *batch]
        )
        indices = ";".join(str(i) for i in range(1, len(batch) + 1))
        try:
            response = session.get(
                f"{OSRM_URL}/table/v1/{travel_mode}/{coordinates}",
                params={
                    "annotations": "duration,distance",
                    "sources": indices if returning else "0",
                    "destinations": "0" if returning else indices,
                },
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"OSRM request failed at {OSRM_URL}. "
                "Ensure the production OSRM driving service is running."
            ) from exc
        response.raise_for_status()
        data = response.json()
        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM error: {data}")
        for i in range(len(batch)):
            row, col = (i, 0) if returning else (0, i)
            seconds = data["durations"][row][col]
            meters = data["distances"][row][col]
            if seconds is None or meters is None:
                legs.append((math.inf, math.inf))
            else:
                legs.append((seconds / 60.0, meters))
    return legs


def itinerary_row(day, order, place, elapsed, start_hour):
    # Display full minutes, retaining precise elapsed time for evaluation.
    clock_min = start_hour * 60 + math.floor(elapsed)
    return {
        "day": day,
        "stop_order": order,
        "poi_id": place["poi_id"],
        "name": place["name"],
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "arrival_time": f"{clock_min // 60:02d}:{clock_min % 60:02d}",
        "preference_score": place["preference_score"],
        "visit_duration_min": place["visit_duration_min"],
        "arrival_min_from_start": elapsed,
    }


def main():
    with INPUT_FILE.open(encoding="utf-8") as handle:
        config = json.load(handle)
    trip = config["trip"]
    num_days = int(trip["num_days"])
    start_hour = int(trip["day_start_hour"])
    end_hour = int(trip["day_end_hour"])
    travel_mode = trip.get("travel_mode", "driving")
    if num_days <= 0 or not 0 <= start_hour < end_hour <= 23:
        raise ValueError("Require positive num_days and 0 <= start < end <= 23.")
    if travel_mode != "driving":
        raise ValueError("This driving baseline requires travel_mode='driving'.")
    daily_limit = (end_hour - start_hour) * 60
    hotel = {
        "poi_id": "HOTEL",
        "name": str(config["hotel"]["name"]),
        "latitude": float(config["hotel"]["latitude"]),
        "longitude": float(config["hotel"]["longitude"]),
        "preference_score": 0.0,
        "visit_duration_min": 0,
    }
    if not (-90 <= hotel["latitude"] <= 90
            and -180 <= hotel["longitude"] <= 180):
        raise ValueError("Invalid hotel coordinates.")

    pois = pd.read_csv(POI_FILE)
    original_count = len(pois)
    pois["_normalized_name"] = (
        pois["name"].astype(str).str.lower().str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    # Match production exactly: retain the first normalized-name occurrence.
    pois = pois.drop_duplicates("_normalized_name", keep="first").copy()
    pois = pois.drop(columns=["_normalized_name"])
    print(f"Original POIs           : {original_count}")
    print(f"Duplicate POIs removed  : {original_count - len(pois)}")
    print(f"POIs after dedup        : {len(pois)}")
    pois["hotel_straight_line_km"] = pois.apply(
        lambda row: haversine_km(
            hotel["latitude"], hotel["longitude"],
            row["latitude"], row["longitude"],
        ), axis=1,
    )
    candidates = pois[
        (pois["hotel_straight_line_km"] <= CANDIDATE_RADIUS_KM)
        & (pois["preference_score"] > 0)
    ].copy()

    # Reproduce production's shortlist scoring, including medical-aware weights.
    # These scores prepare candidates only; greedy routing uses road time alone.
    medical_support = config.get("medical_support", {})
    if bool(medical_support.get("required", False)):
        priority = str(medical_support.get("priority", "normal")).strip().lower()
        if priority == "high":
            preference_weight, proximity_weight, medical_weight = 0.55, 0.15, 0.30
        else:
            preference_weight, proximity_weight, medical_weight = 0.60, 0.20, 0.20
        medical_access = pd.read_csv(MEDICAL_ACCESS_FILE)
        medical_columns = [
            "poi_id", "medical_support_score",
            "nearest_emergency_hospital_name", "emergency_hospital_travel_time_min",
        ]
        missing = set(medical_columns) - set(medical_access.columns)
        if missing:
            raise ValueError(f"Medical access file is missing columns: {sorted(missing)}")
        candidates["poi_id"] = candidates["poi_id"].astype(str)
        medical_access["poi_id"] = medical_access["poi_id"].astype(str)
        candidates = candidates.merge(
            medical_access[medical_columns], on="poi_id", how="left",
        )
        candidates["medical_support_score"] = (
            pd.to_numeric(candidates["medical_support_score"], errors="coerce")
            .fillna(0.0).clip(lower=0.0, upper=1.0)
        )
    else:
        preference_weight, proximity_weight, medical_weight = 0.70, 0.30, 0.00
        candidates["medical_support_score"] = 0.0

    candidates["preference_normalized"] = candidates["preference_score"] / 5.0
    candidates["proximity_score"] = (
        1 - candidates["hotel_straight_line_km"] / CANDIDATE_RADIUS_KM
    ).clip(lower=0, upper=1)
    candidates["candidate_score"] = (
        preference_weight * candidates["preference_normalized"]
        + proximity_weight * candidates["proximity_score"]
        + medical_weight * candidates["medical_support_score"]
    )
    candidates = (
        candidates.sort_values(
            by=["candidate_score", "preference_score"], ascending=[False, False],
        ).head(MAX_CANDIDATES).reset_index(drop=True)
    )
    candidates["visit_duration_min"] = [
        estimate_visit_duration(row) for _, row in candidates.iterrows()
    ]
    places = candidates.to_dict("records")
    print(f"Hotel/depot             : {hotel['name']}")
    print(f"Daily schedule          : {start_hour:02d}:00 - {end_hour:02d}:00")
    print(f"Travel mode             : {travel_mode}")
    print(f"Candidate radius        : {CANDIDATE_RADIUS_KM} km")
    print(f"Candidate POIs          : {len(places)}")
    print("Requesting OSRM road travel times and distances...")

    rows = []
    visited = set()
    selected_ids = []
    total_distance = total_driving = total_visit = total_utility = 0.0
    with requests.Session() as session:
        returns = road_legs(session, hotel, places, travel_mode, returning=True)
        for day in range(1, num_days + 1):
            current = hotel
            current_index = None
            elapsed = 0.0
            order = 1
            day_count = 0
            rows.append(itinerary_row(day, order, hotel, elapsed, start_hour))
            while True:
                remaining = [
                    i for i in range(len(places)) if places[i]["poi_id"] not in visited
                ]
                outbound = road_legs(
                    session, current, [places[i] for i in remaining], travel_mode,
                )
                feasible = [
                    (minutes, i, meters)
                    for i, (minutes, meters) in zip(remaining, outbound)
                    if elapsed + minutes + places[i]["visit_duration_min"]
                    + returns[i][0] <= daily_limit
                ]
                if not feasible:
                    break
                # Shortlist order breaks road-time ties; route choice uses time alone.
                minutes, current_index, meters = min(feasible)
                current = places[current_index]
                elapsed += minutes
                order += 1
                rows.append(itinerary_row(day, order, current, elapsed, start_hour))
                elapsed += current["visit_duration_min"]
                visited.add(current["poi_id"])
                selected_ids.append(current["poi_id"])
                day_count += 1
                total_distance += meters
                total_driving += minutes
                total_visit += current["visit_duration_min"]
                total_utility += current["preference_score"]

            if current_index is not None:
                minutes, meters = returns[current_index]
                elapsed += minutes
                total_driving += minutes
                total_distance += meters
            if elapsed > daily_limit:
                raise RuntimeError(f"Day {day} exceeds its time limit.")
            final_row = itinerary_row(day, order + 1, hotel, elapsed, start_hour)
            rows.append(final_row)
            print(f"Day {day}: {day_count} POIs, return {final_row['arrival_time']}, "
                  f"elapsed {elapsed:.2f} min")

    unique_count = len(set(selected_ids))
    repeated_count = len(selected_ids) - unique_count
    if repeated_count:
        raise RuntimeError("POI repetition check failed.")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    print("\nGREEDY BASELINE TRIP SUMMARY")
    print(f"Number of days          : {num_days}")
    print(f"Total POIs visited      : {len(selected_ids)}")
    print(f"Unique POIs visited     : {unique_count}")
    print(f"Repeated POIs           : {repeated_count}")
    print(f"Total road distance     : {total_distance / 1000:.2f} km")
    print(f"Total driving time      : {total_driving:.2f} min")
    print(f"Total POI visit time    : {total_visit:.2f} min")
    print(f"Total preference utility: {total_utility:.2f}")
    print(f"Daily time limit        : {daily_limit} min/day")
    print("POI repetition check    : PASS")
    print(f"Output file             : {OUTPUT_FILE}")
    print("Status: SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, RuntimeError) as exc:
        print(f"Status: FAILED - {exc}", file=sys.stderr)
        sys.exit(1)
