"""Controlled medical-access-aware itinerary evaluation using production scripts.

Run directly from any directory with the production Python dependencies installed
and OSRM running at http://localhost:5000. Run without concurrent planner jobs:
the production scripts share input/output paths, restored here in finally.
Distance uses production's two-decimal stdout total; driving time uses its
rounded-per-leg minutes. Neither metric is estimated from POI coordinates.
"""

import copy
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "Data/input/c2_c3_input.json"
ITINERARY_FILE = PROJECT_ROOT / "Data/processed/optimized_itinerary.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_FILE = RESULTS_DIR / "medical_support_evaluation.csv"
LOG_FILE = RESULTS_DIR / "medical_support_evaluation.log"
RUNTIME_FILES = (
    INPUT_FILE,
    PROJECT_ROOT / "Data/processed/poi_preference_scores.csv",
    PROJECT_ROOT / "Data/processed/poi_medical_access.csv",
    ITINERARY_FILE,
)

BASE_REQUEST = {
    "context": {
        "adults": 2, "children": 0, "travel_style": "couple",
        "dates_fixed": False, "check_in": "2026-12-20",
        "check_out": "2026-12-23", "stay_in": ["Kandy"],
        "interests": ["nature", "wildlife", "hiking"], "health": [],
    },
    "accessibility": {"reduced_mobility": False, "prefer_low_walking": False},
    "trip_id": "TEST_TRIP_001",
    "hotel": {
        "hotel_id": "TEST_HOTEL_001", "name": "Kandy Test Depot",
        "latitude": 7.294373, "longitude": 80.641486, "district": "Kandy",
    },
    "trip": {
        "want_to_visit": ["Kandy"], "num_days": 3,
        "day_start_hour": 9, "day_end_hour": 18, "travel_mode": "driving",
    },
    "attraction_preferences": {
        "history_culture": 1.0, "nature_wildlife": 5.0, "beach_coastal": 1.0,
        "religious": 1.0, "adventure_recreation": 5.0, "urban_leisure": 2.0,
    },
    "restaurant_preferences": {
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
    },
}


def run_production(relative_path, log):
    """Capture complete production output for inspection; never import scripts."""
    print(f"Running {relative_path}", flush=True)
    result = subprocess.run(
        [sys.executable, "-u", str(PROJECT_ROOT / relative_path)],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    log.write(f"\n$ {relative_path}\n{result.stdout}\n")
    log.flush()
    if result.returncode:
        raise RuntimeError(
            f"{relative_path} exited with {result.returncode}.\n"
            + "\n".join(result.stdout.splitlines()[-15:])
        )
    return result.stdout


def summary_number(stdout, label, unit=""):
    """Require exactly one trip-summary value; reject missing/ambiguous data."""
    summary = stdout.partition("MULTI-DAY TRIP SUMMARY")[2]
    pattern = (
        rf"^{re.escape(label)}\s*:\s*([0-9]+(?:\.[0-9]+)?)"
        rf"\s*{re.escape(unit)}\s*$"
    )
    matches = re.findall(pattern, summary, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one production trip-summary value: {label}")
    return float(matches[0])


def number(row, column):
    value = float(row[column])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {column} for POI {row['poi_id']}: {value}")
    return value


def collect_metrics(scenario, required, priority, stdout):
    with ITINERARY_FILE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"poi_id", "preference_score", "visit_duration_min"}
        if required:
            expected.add("emergency_hospital_travel_time_min")
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Itinerary is missing columns: {sorted(expected)}")
        # Production explicitly identifies both departure/return depots as HOTEL.
        selected = [row for row in reader if row["poi_id"] != "HOTEL"]
    ids = [row["poi_id"] for row in selected]
    utility = sum(number(row, "preference_score") for row in selected)
    visit = sum(number(row, "visit_duration_min") for row in selected)
    metrics = {
        "region": "Kandy", "scenario": scenario,
        "medical_required": required, "medical_priority": priority,
        "total_pois_visited": len(ids), "unique_pois_visited": len(set(ids)),
        "repeated_pois": len(ids) - len(set(ids)),
        "total_preference_utility": round(utility, 6),
        "total_road_distance_km": summary_number(stdout, "Total road distance", "km"),
        "total_driving_time_min": summary_number(stdout, "Total driving time", "min"),
        "total_visit_time_min": visit,
        "average_hospital_access_min": None,
        "best_hospital_access_min": None,
        "worst_hospital_access_min": None,
    }
    for key, label in (
        ("total_pois_visited", "Total POIs visited"),
        ("unique_pois_visited", "Unique POIs visited"),
        ("repeated_pois", "Repeated POIs"),
        ("total_preference_utility", "Total preference utility"),
        ("total_visit_time_min", "Total POI visit time"),
    ):
        reported = summary_number(stdout, label, "min" if key == "total_visit_time_min" else "")
        if not math.isclose(metrics[key], reported, abs_tol=0.0051):
            raise ValueError(f"CSV/stdout mismatch for {key}: {metrics[key]} != {reported}")
    if required:
        available = [
            number(row, "emergency_hospital_travel_time_min")
            for row in selected
            if row["emergency_hospital_travel_time_min"].strip()
        ]
        print(f"Emergency hospital access coverage: {len(available)}/{len(selected)} selected POI rows")
        if available:
            metrics.update(
                average_hospital_access_min=sum(available) / len(available),
                best_hospital_access_min=min(available),
                worst_hospital_access_min=max(available),
            )
        # Missing access times remain unknown, never zero or general-hospital fallback.
    return metrics


def display_comparison(rows):
    disabled, enabled = rows
    print("\nmedical-access-aware itinerary evaluation")
    print(f"{'Metric':<34} {'medical_disabled':>18} {'medical_high':>18}")
    print("-" * 72)
    for key in disabled:
        if key in ("region", "scenario"):
            continue
        values = [row[key] for row in rows]
        formatted = [
            "N/A" if value is None else
            f"{value:.2f}" if isinstance(value, float) else str(value)
            for value in values
        ]
        print(f"{key:<34} {formatted[0]:>18} {formatted[1]:>18}")
    print("\nChanges (medical_high minus medical_disabled):")
    for key, label, unit in (
        ("total_pois_visited", "POIs visited", ""),
        ("total_preference_utility", "Preference utility", ""),
        ("total_road_distance_km", "Distance", " km"),
        ("total_driving_time_min", "Driving time", " min"),
    ):
        print(f"  {label}: {enabled[key] - disabled[key]:+.2f}{unit}")
    print("Hospital-access figures describe modeled road access; they do not establish")
    print("medical safety or guarantee emergency response.")
    print("Distance: production stdout (0.01 km precision), including hotel returns.")
    print("Driving time: production rounded-per-leg minutes, including hotel returns.")


def main():
    print("medical-access-aware itinerary evaluation", flush=True)
    # Byte snapshots preserve the user's encoding, whitespace and newlines exactly.
    original_input = INPUT_FILE.read_bytes()
    backups = {
        path: path.read_bytes() if path.exists() else None
        for path in RUNTIME_FILES if path != INPUT_FILE
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with LOG_FILE.open("w", encoding="utf-8") as log:
            for scenario, required, priority in (
                ("medical_disabled", False, "normal"),
                ("medical_high", True, "high"),
            ):
                print(f"\nKandy / {scenario}", flush=True)
                request = copy.deepcopy(BASE_REQUEST)
                request["medical_support"] = {"required": required, "priority": priority}
                INPUT_FILE.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
                run_production("src/optimization/02_score_pois.py", log)
                if required:
                    run_production("src/medical/02_compute_poi_medical_access.py", log)
                # Remove prior output so a missing new file cannot pass as fresh data.
                ITINERARY_FILE.unlink(missing_ok=True)
                stdout = run_production("src/optimization/04_optimize_multi_day.py", log)
                rows.append(collect_metrics(scenario, required, priority, stdout))
    finally:
        INPUT_FILE.write_bytes(original_input)
        if INPUT_FILE.read_bytes() != original_input:
            raise RuntimeError("Original input restoration verification failed")
        print("Original input restored: YES (byte-for-byte verified)", flush=True)
        for path, contents in backups.items():
            if contents is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(contents)
        print("Original generated production files restored: YES", flush=True)
    # Publish only a complete, validated comparison after restoring production data.
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    display_comparison(rows)
    print(f"Output CSV: {OUTPUT_FILE}")
    print(f"Full production output log: {LOG_FILE}")
    print("Status: SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Status: FAILED - {exc}", file=sys.stderr)
        print("No new comparison CSV published. Resolve the failure and rerun.", file=sys.stderr)
        sys.exit(1)
