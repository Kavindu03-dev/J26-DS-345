from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POI_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "poi_preference_scores.csv"
)

MEDICAL_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "medical_facilities.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "poi_medical_access.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

OSRM_BASE_URL = "http://127.0.0.1:5000"

NEAREST_GEOGRAPHIC_CANDIDATES = 15


# ============================================================
# DISTANCE UTILITY
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    radius_km = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return radius_km * c


# ============================================================
# MEDICAL SUPPORT SCORE
# ============================================================

def medical_support_score(
    travel_time_min,
):

    if pd.isna(travel_time_min):
        return 0.0

    if travel_time_min <= 10:
        return 1.0

    if travel_time_min <= 20:
        return 0.8

    if travel_time_min <= 30:
        return 0.5

    if travel_time_min <= 45:
        return 0.3

    return 0.1


# ============================================================
# GEOGRAPHIC PREFILTER
# ============================================================

def get_nearest_geographic_candidates(
    poi_lat,
    poi_lon,
    facilities,
    limit=NEAREST_GEOGRAPHIC_CANDIDATES,
):

    nearby = facilities.copy()

    nearby[
        "straight_line_distance_km"
    ] = nearby.apply(
        lambda row:
            haversine_km(
                poi_lat,
                poi_lon,
                float(row["latitude"]),
                float(row["longitude"]),
            ),
        axis=1,
    )

    return (
        nearby
        .sort_values(
            "straight_line_distance_km"
        )
        .head(limit)
        .copy()
    )


# ============================================================
# OSRM ROAD ACCESS
# ============================================================

def get_best_road_access(
    poi_lat,
    poi_lon,
    facilities,
):

    if facilities.empty:
        return None

    coordinates = [
        f"{poi_lon},{poi_lat}"
    ]

    for _, row in facilities.iterrows():

        coordinates.append(
            f"{row['longitude']},{row['latitude']}"
        )

    coordinate_string = ";".join(
        coordinates
    )

    destination_indexes = ";".join(
        str(index)
        for index in range(
            1,
            len(coordinates),
        )
    )

    url = (
        f"{OSRM_BASE_URL}/table/v1/driving/"
        f"{coordinate_string}"
    )

    params = {
        "sources": "0",
        "destinations": destination_indexes,
        "annotations": "duration,distance",
    }

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    result = response.json()

    if result.get("code") != "Ok":

        raise RuntimeError(
            f"OSRM table request failed: {result}"
        )

    durations = result[
        "durations"
    ][0]

    distances = result[
        "distances"
    ][0]

    best_index = None
    best_duration = None

    for index, duration_sec in enumerate(
        durations
    ):

        if duration_sec is None:
            continue

        if (
            best_duration is None
            or duration_sec < best_duration
        ):

            best_duration = duration_sec
            best_index = index

    if best_index is None:
        return None

    facility = facilities.iloc[
        best_index
    ]

    return {
        "facility_id":
            facility["facility_id"],

        "facility_name":
            facility["name"],

        "facility_type":
            facility["amenity"],

        "emergency":
            facility.get(
                "emergency"
            ),

        "latitude":
            facility["latitude"],

        "longitude":
            facility["longitude"],

        "road_travel_time_min":
            round(
                durations[
                    best_index
                ] / 60,
                2,
            ),

        "road_distance_km":
            round(
                distances[
                    best_index
                ] / 1000,
                2,
            ),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("COMPONENT 3 - POI MEDICAL ACCESS COMPUTATION")
print("=" * 75)

print("\nLoading POIs...")

pois = pd.read_csv(
    POI_FILE
)

print(
    "POIs loaded:",
    len(pois),
)


print("\nLoading medical facilities...")

facilities = pd.read_csv(
    MEDICAL_FILE
)

facilities = facilities[
    facilities["latitude"].notna()
    &
    facilities["longitude"].notna()
].copy()

print(
    "All medical facilities:",
    len(facilities),
)


# ============================================================
# CREATE MEDICAL TIERS
# ============================================================

hospitals = facilities[
    facilities["amenity"]
    .astype(str)
    .str.lower()
    == "hospital"
].copy()


# ============================================================
# CLEAN GENERAL EMERGENCY HOSPITALS
# ============================================================

emergency_hospitals = hospitals[
    (
        hospitals["emergency"]
        .astype(str)
        .str.lower()
        == "yes"
    )
    &
    hospitals["name"].notna()
].copy()


# Exclude facilities that are too specialized to represent
# general emergency medical support for itinerary planning.
EXCLUDED_EMERGENCY_TERMS = (
    r"dental|dentist|"
    r"ayur|aurvedic|"
    r"rehabilitation|"
    r"diabetic|"
    r"school ground|"
    r"ward|"
    r"specialist centre|"
    r"for women|"
    r"children|"
    r"maternity"
)


emergency_hospitals = emergency_hospitals[
    ~emergency_hospitals[
        "name"
    ]
    .astype(str)
    .str.lower()
    .str.contains(
        EXCLUDED_EMERGENCY_TERMS,
        regex=True,
        na=False,
    )
].copy()


print(
    "Hospitals:",
    len(hospitals),
)

print(
    "Emergency hospitals:",
    len(emergency_hospitals),
)


# ============================================================
# VALIDATE POI COLUMNS
# ============================================================

required_poi_columns = {
    "poi_id",
    "name",
    "latitude",
    "longitude",
}

missing = (
    required_poi_columns
    - set(pois.columns)
)

if missing:

    raise ValueError(
        "POI file missing required columns: "
        f"{sorted(missing)}"
    )


# ============================================================
# PROCESS POIS
# ============================================================

results = []

print(
    "\nComputing emergency-hospital and "
    "general-hospital road access..."
)


for index, poi in pois.iterrows():

    poi_lat = float(
        poi["latitude"]
    )

    poi_lon = float(
        poi["longitude"]
    )


    # --------------------------------------------------------
    # EMERGENCY HOSPITAL
    # --------------------------------------------------------

    emergency_candidates = (
        get_nearest_geographic_candidates(
            poi_lat,
            poi_lon,
            emergency_hospitals,
        )
    )

    emergency_access = (
        get_best_road_access(
            poi_lat,
            poi_lon,
            emergency_candidates,
        )
    )


    # --------------------------------------------------------
    # GENERAL HOSPITAL FALLBACK
    # --------------------------------------------------------

    hospital_candidates = (
        get_nearest_geographic_candidates(
            poi_lat,
            poi_lon,
            hospitals,
        )
    )

    hospital_access = (
        get_best_road_access(
            poi_lat,
            poi_lon,
            hospital_candidates,
        )
    )


    # --------------------------------------------------------
    # SELECT PRIMARY MEDICAL SUPPORT
    # --------------------------------------------------------

    if emergency_access is not None:

        primary_access = (
            emergency_access
        )

        support_basis = (
            "emergency_hospital"
        )

    elif hospital_access is not None:

        primary_access = (
            hospital_access
        )

        support_basis = (
            "general_hospital_fallback"
        )

    else:

        primary_access = None
        support_basis = "unavailable"


    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    if primary_access is None:

        support_time = None
        support_score = 0.0

    else:

        support_time = (
            primary_access[
                "road_travel_time_min"
            ]
        )

        support_score = (
            medical_support_score(
                support_time
            )
        )


    result = {
        "poi_id":
            poi["poi_id"],

        "poi_name":
            poi["name"],

        "poi_latitude":
            poi_lat,

        "poi_longitude":
            poi_lon,


        # Emergency hospital
        "nearest_emergency_hospital_id":
            (
                emergency_access[
                    "facility_id"
                ]
                if emergency_access
                else None
            ),

        "nearest_emergency_hospital_name":
            (
                emergency_access[
                    "facility_name"
                ]
                if emergency_access
                else None
            ),

        "emergency_hospital_distance_km":
            (
                emergency_access[
                    "road_distance_km"
                ]
                if emergency_access
                else None
            ),

        "emergency_hospital_travel_time_min":
            (
                emergency_access[
                    "road_travel_time_min"
                ]
                if emergency_access
                else None
            ),


        # General hospital fallback
        "nearest_hospital_id":
            (
                hospital_access[
                    "facility_id"
                ]
                if hospital_access
                else None
            ),

        "nearest_hospital_name":
            (
                hospital_access[
                    "facility_name"
                ]
                if hospital_access
                else None
            ),

        "hospital_distance_km":
            (
                hospital_access[
                    "road_distance_km"
                ]
                if hospital_access
                else None
            ),

        "hospital_travel_time_min":
            (
                hospital_access[
                    "road_travel_time_min"
                ]
                if hospital_access
                else None
            ),


        # Main medical-support metric
        "medical_support_basis":
            support_basis,

        "medical_support_travel_time_min":
            support_time,

        "medical_support_score":
            support_score,
    }


    results.append(
        result
    )


    print(
        f"[{index + 1}/{len(pois)}] "
        f"{poi['name']} -> "
        f"{support_basis} | "
        f"{primary_access['facility_name'] if primary_access else None} | "
        f"{support_time} min"
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

output = pd.DataFrame(
    results
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("POI MEDICAL ACCESS SUMMARY")
print("=" * 75)

print(
    "POIs processed:",
    len(output),
)

print(
    "Emergency hospital access available:",
    output[
        "emergency_hospital_travel_time_min"
    ]
    .notna()
    .sum(),
)

print(
    "General hospital access available:",
    output[
        "hospital_travel_time_min"
    ]
    .notna()
    .sum(),
)

print(
    "Average medical support time:",
    round(
        output[
            "medical_support_travel_time_min"
        ]
        .mean(),
        2,
    ),
    "min",
)

print(
    "Maximum medical support time:",
    round(
        output[
            "medical_support_travel_time_min"
        ]
        .max(),
        2,
    ),
    "min",
)


print(
    "\nMedical support basis:"
)

print(
    output[
        "medical_support_basis"
    ]
    .value_counts()
)


print(
    "\nMedical support score distribution:"
)

print(
    output[
        "medical_support_score"
    ]
    .value_counts()
    .sort_index(
        ascending=False
    )
)


print("\nOutput saved:")
print(OUTPUT_FILE)

print("\nStatus: SUCCESS")