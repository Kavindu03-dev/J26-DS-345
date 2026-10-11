from pathlib import Path
import json
import math

import pandas as pd
import requests

from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POI_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "poi_preference_scores.csv"
)

MEDICAL_ACCESS_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "poi_medical_access.csv"
)

ITINERARY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "optimized_itinerary.csv"
)

C2_C3_INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "input"
    / "c2_c3_input.json"
)


# ============================================================
# LOAD COMPONENT 2 -> COMPONENT 3 INPUT
# ============================================================

print("\nLoading Component 2 -> Component 3 input...")

if not C2_C3_INPUT_FILE.exists():
    raise FileNotFoundError(
        "Component 2 -> Component 3 input file not found:\n"
        f"{C2_C3_INPUT_FILE}"
    )

with open(
    C2_C3_INPUT_FILE,
    "r",
    encoding="utf-8",
) as f:
    C2_INPUT = json.load(f)


for section in [
    "hotel",
    "trip",
]:
    if section not in C2_INPUT:
        raise ValueError(
            f"Missing required input section: {section}"
        )


for field in [
    "name",
    "latitude",
    "longitude",
]:
    if field not in C2_INPUT["hotel"]:
        raise ValueError(
            f"Missing hotel field: {field}"
        )


for field in [
    "num_days",
    "day_start_hour",
    "day_end_hour",
]:
    if field not in C2_INPUT["trip"]:
        raise ValueError(
            f"Missing trip field: {field}"
        )


# ============================================================
# CONFIGURATION
# ============================================================

OSRM_URL = "http://localhost:5000"


# Hotel supplied by Component 2.
HOTEL = {
    "name": str(
        C2_INPUT[
            "hotel"
        ][
            "name"
        ]
    ),
    "latitude": float(
        C2_INPUT[
            "hotel"
        ][
            "latitude"
        ]
    ),
    "longitude": float(
        C2_INPUT[
            "hotel"
        ][
            "longitude"
        ]
    ),
}

HOTEL_ID = (
    C2_INPUT[
        "hotel"
    ].get(
        "hotel_id"
    )
)

HOTEL_DISTRICT = (
    C2_INPUT[
        "hotel"
    ].get(
        "district"
    )
)


# Trip settings.
NUM_DAYS = int(
    C2_INPUT[
        "trip"
    ][
        "num_days"
    ]
)

DAY_START_HOUR = int(
    C2_INPUT[
        "trip"
    ][
        "day_start_hour"
    ]
)

DAY_END_HOUR = int(
    C2_INPUT[
        "trip"
    ][
        "day_end_hour"
    ]
)

TRAVEL_MODE = (
    C2_INPUT[
        "trip"
    ].get(
        "travel_mode",
        "driving",
    )
)


# ============================================================
# OPTIONAL MEDICAL SUPPORT
# ============================================================

MEDICAL_SUPPORT = (
    C2_INPUT.get(
        "medical_support",
        {},
    )
)

MEDICAL_SUPPORT_REQUIRED = bool(
    MEDICAL_SUPPORT.get(
        "required",
        False,
    )
)

MEDICAL_SUPPORT_PRIORITY = str(
    MEDICAL_SUPPORT.get(
        "priority",
        "normal",
    )
).strip().lower()


# ============================================================
# VALIDATE INPUT
# ============================================================

if NUM_DAYS <= 0:
    raise ValueError(
        "num_days must be greater than 0."
    )

if not 0 <= DAY_START_HOUR <= 23:
    raise ValueError(
        "day_start_hour must be between 0 and 23."
    )

if not 0 <= DAY_END_HOUR <= 23:
    raise ValueError(
        "day_end_hour must be between 0 and 23."
    )

if DAY_END_HOUR <= DAY_START_HOUR:
    raise ValueError(
        "day_end_hour must be later than day_start_hour."
    )

if not -90 <= HOTEL["latitude"] <= 90:
    raise ValueError(
        "Hotel latitude is invalid."
    )

if not -180 <= HOTEL["longitude"] <= 180:
    raise ValueError(
        "Hotel longitude is invalid."
    )


# ============================================================
# ALGORITHM SETTINGS
# ============================================================

CANDIDATE_RADIUS_KM = 40
MAX_CANDIDATES = 25

DAILY_TIME_LIMIT_MIN = (
    DAY_END_HOUR
    - DAY_START_HOUR
) * 60


# ============================================================
# DYNAMIC WEIGHTS
# ============================================================

if MEDICAL_SUPPORT_REQUIRED:

    if MEDICAL_SUPPORT_PRIORITY == "high":

        PREFERENCE_WEIGHT = 0.55
        PROXIMITY_WEIGHT = 0.15
        MEDICAL_WEIGHT = 0.30

        OR_PREFERENCE_WEIGHT = 0.65
        OR_MEDICAL_WEIGHT = 0.35

    else:

        PREFERENCE_WEIGHT = 0.60
        PROXIMITY_WEIGHT = 0.20
        MEDICAL_WEIGHT = 0.20

        OR_PREFERENCE_WEIGHT = 0.75
        OR_MEDICAL_WEIGHT = 0.25

else:

    PREFERENCE_WEIGHT = 0.70
    PROXIMITY_WEIGHT = 0.30
    MEDICAL_WEIGHT = 0.00

    OR_PREFERENCE_WEIGHT = 1.00
    OR_MEDICAL_WEIGHT = 0.00


PREFERENCE_PENALTY_SCALE = 50


# ============================================================
# INPUT SUMMARY
# ============================================================

print(
    f"Trip ID              : "
    f"{C2_INPUT.get('trip_id', 'N/A')}"
)

print(
    f"Hotel                : "
    f"{HOTEL['name']}"
)

print(
    f"Hotel coordinates    : "
    f"{HOTEL['latitude']}, "
    f"{HOTEL['longitude']}"
)

print(
    f"Hotel district       : "
    f"{HOTEL_DISTRICT or 'N/A'}"
)

print(
    f"Number of days       : "
    f"{NUM_DAYS}"
)

print(
    f"Daily schedule       : "
    f"{DAY_START_HOUR:02d}:00 - "
    f"{DAY_END_HOUR:02d}:00"
)

print(
    f"Daily time limit     : "
    f"{DAILY_TIME_LIMIT_MIN} min"
)

print(
    f"Travel mode          : "
    f"{TRAVEL_MODE}"
)

print(
    f"Medical support      : "
    f"{'ENABLED' if MEDICAL_SUPPORT_REQUIRED else 'disabled'}"
)

if MEDICAL_SUPPORT_REQUIRED:
    print(
        f"Medical priority     : "
        f"{MEDICAL_SUPPORT_PRIORITY}"
    )

print(
    "C2 -> C3 input       : SUCCESS"
)


# ============================================================
# VISIT DURATION HEURISTIC
# ============================================================

VISIT_DURATION = {
    "history_culture": 60,
    "nature_wildlife": 90,
    "beach_coastal": 90,
    "religious": 60,
    "adventure_recreation": 120,
    "urban_leisure": 60,
}


def estimate_visit_duration(row):

    durations = []

    for category, duration in (
        VISIT_DURATION.items()
    ):

        if (
            category in row.index
            and row[
                category
            ] == 1
        ):
            durations.append(
                duration
            )

    if not durations:
        return 60

    return max(
        durations
    )


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    earth_radius = 6371.0

    lat1 = math.radians(
        lat1
    )
    lon1 = math.radians(
        lon1
    )
    lat2 = math.radians(
        lat2
    )
    lon2 = math.radians(
        lon2
    )

    dlat = (
        lat2
        - lat1
    )

    dlon = (
        lon2
        - lon1
    )

    a = (
        math.sin(
            dlat / 2
        ) ** 2
        +
        math.cos(
            lat1
        )
        *
        math.cos(
            lat2
        )
        *
        math.sin(
            dlon / 2
        ) ** 2
    )

    c = (
        2
        *
        math.atan2(
            math.sqrt(
                a
            ),
            math.sqrt(
                1 - a
            ),
        )
    )

    return (
        earth_radius
        * c
    )


# ============================================================
# LOAD POIS
# ============================================================

print(
    "\nLoading POIs..."
)

pois = pd.read_csv(
    POI_FILE
)

print(
    f"Original POIs         : "
    f"{len(pois)}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before_dedup = len(
    pois
)

pois[
    "_normalized_name"
] = (
    pois[
        "name"
    ]
    .astype(
        str
    )
    .str.lower()
    .str.strip()
    .str.replace(
        r"\s+",
        " ",
        regex=True,
    )
)

pois = (
    pois
    .drop_duplicates(
        subset=[
            "_normalized_name"
        ],
        keep="first",
    )
    .copy()
)

pois = pois.drop(
    columns=[
        "_normalized_name"
    ]
)

removed_duplicates = (
    before_dedup
    - len(
        pois
    )
)

print(
    f"Duplicate POIs removed: "
    f"{removed_duplicates}"
)

print(
    f"POIs after dedup       : "
    f"{len(pois)}"
)


# ============================================================
# DISTANCE FROM HOTEL
# ============================================================

pois[
    "hotel_straight_line_km"
] = pois.apply(
    lambda row:
        haversine_km(
            HOTEL[
                "latitude"
            ],
            HOTEL[
                "longitude"
            ],
            row[
                "latitude"
            ],
            row[
                "longitude"
            ],
        ),
    axis=1,
)


# ============================================================
# CANDIDATE FILTERING
# ============================================================

candidates = (
    pois[
        pois[
            "hotel_straight_line_km"
        ]
        <= CANDIDATE_RADIUS_KM
    ]
    .copy()
)

candidates = (
    candidates[
        candidates[
            "preference_score"
        ]
        > 0
    ]
    .copy()
)


# ============================================================
# OPTIONAL MEDICAL ACCESS DATA
# ============================================================

if MEDICAL_SUPPORT_REQUIRED:

    if not MEDICAL_ACCESS_FILE.exists():

        raise FileNotFoundError(
            "Medical support was requested, "
            "but poi_medical_access.csv was not found. "
            "Run "
            "src/medical/"
            "02_compute_poi_medical_access.py first."
        )

    medical_access = pd.read_csv(
        MEDICAL_ACCESS_FILE
    )

    required_medical_columns = {
        "poi_id",
        "medical_support_score",
        "nearest_emergency_hospital_name",
        "emergency_hospital_travel_time_min",
    }

    missing_medical_columns = (
        required_medical_columns
        - set(
            medical_access.columns
        )
    )

    if missing_medical_columns:

        raise ValueError(
            "Medical access file is missing required columns: "
            f"{sorted(missing_medical_columns)}"
        )

    candidates[
        "poi_id"
    ] = (
        candidates[
            "poi_id"
        ]
        .astype(
            str
        )
    )

    medical_access[
        "poi_id"
    ] = (
        medical_access[
            "poi_id"
        ]
        .astype(
            str
        )
    )

    candidates = (
        candidates.merge(
            medical_access[
                [
                    "poi_id",
                    "medical_support_score",
                    "nearest_emergency_hospital_name",
                    "emergency_hospital_travel_time_min",
                ]
            ],
            on="poi_id",
            how="left",
        )
    )

    candidates[
        "medical_support_score"
    ] = (
        pd.to_numeric(
            candidates[
                "medical_support_score"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

else:

    candidates[
        "medical_support_score"
    ] = 0.0

    candidates[
        "nearest_emergency_hospital_name"
    ] = None

    candidates[
        "emergency_hospital_travel_time_min"
    ] = None


# ============================================================
# CANDIDATE PRE-SCORING
# ============================================================

candidates[
    "preference_normalized"
] = (
    candidates[
        "preference_score"
    ]
    / 5.0
)

candidates[
    "proximity_score"
] = (
    1
    -
    (
        candidates[
            "hotel_straight_line_km"
        ]
        / CANDIDATE_RADIUS_KM
    )
).clip(
    lower=0,
    upper=1,
)

candidates[
    "candidate_score"
] = (
    PREFERENCE_WEIGHT
    *
    candidates[
        "preference_normalized"
    ]
    +
    PROXIMITY_WEIGHT
    *
    candidates[
        "proximity_score"
    ]
    +
    MEDICAL_WEIGHT
    *
    candidates[
        "medical_support_score"
    ]
)

candidates = (
    candidates
    .sort_values(
        by=[
            "candidate_score",
            "preference_score",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(
        MAX_CANDIDATES
    )
)

candidates = (
    candidates
    .reset_index(
        drop=True
    )
)

candidates[
    "visit_duration_min"
] = candidates.apply(
    estimate_visit_duration,
    axis=1,
)


print(
    f"Hotel/depot          : "
    f"{HOTEL['name']}"
)

print(
    f"Number of days       : "
    f"{NUM_DAYS}"
)

print(
    f"Candidate radius     : "
    f"{CANDIDATE_RADIUS_KM} km"
)

print(
    f"Candidate POIs       : "
    f"{len(candidates)}"
)

if MEDICAL_SUPPORT_REQUIRED:

    print(
        "Medical-aware candidate scoring: "
        "ENABLED"
    )

if len(
    candidates
) == 0:

    raise RuntimeError(
        "No candidate POIs found."
    )


# ============================================================
# CREATE LOCATION TABLE
# ============================================================

locations = [
    {
        "id":
            "HOTEL",

        "name":
            HOTEL[
                "name"
            ],

        "latitude":
            HOTEL[
                "latitude"
            ],

        "longitude":
            HOTEL[
                "longitude"
            ],

        "preference_score":
            0.0,

        "medical_support_score":
            0.0,

        "optimization_utility":
            0.0,

        "nearest_emergency_hospital_name":
            None,

        "emergency_hospital_travel_time_min":
            None,

        "visit_duration_min":
            0,
    }
]


for _, row in candidates.iterrows():

    preference_score = float(
        row[
            "preference_score"
        ]
    )

    medical_score = float(
        row.get(
            "medical_support_score",
            0.0,
        )
    )

    optimization_utility = (
        OR_PREFERENCE_WEIGHT
        *
        (
            preference_score
            / 5.0
        )
        +
        OR_MEDICAL_WEIGHT
        *
        medical_score
    ) * 5.0

    locations.append(
        {
            "id":
                row[
                    "poi_id"
                ],

            "name":
                row[
                    "name"
                ],

            "latitude":
                float(
                    row[
                        "latitude"
                    ]
                ),

            "longitude":
                float(
                    row[
                        "longitude"
                    ]
                ),

            "preference_score":
                preference_score,

            "medical_support_score":
                medical_score,

            "optimization_utility":
                optimization_utility,

            "nearest_emergency_hospital_name":
                row.get(
                    "nearest_emergency_hospital_name"
                ),

            "emergency_hospital_travel_time_min":
                row.get(
                    "emergency_hospital_travel_time_min"
                ),

            "visit_duration_min":
                int(
                    row[
                        "visit_duration_min"
                    ]
                ),
        }
    )


# ============================================================
# GET OSRM ROAD MATRIX
# ============================================================

print(
    "\nRequesting road travel matrix from OSRM..."
)

coordinates = ";".join(
    f"{loc['longitude']},"
    f"{loc['latitude']}"
    for loc in locations
)

table_url = (
    f"{OSRM_URL}"
    f"/table/v1/driving/"
    f"{coordinates}"
)

response = requests.get(
    table_url,
    params={
        "annotations":
            "duration,distance"
    },
    timeout=60,
)

response.raise_for_status()

osrm_data = response.json()

if (
    osrm_data.get(
        "code"
    )
    != "Ok"
):

    raise RuntimeError(
        f"OSRM error: "
        f"{osrm_data}"
    )


durations = (
    osrm_data[
        "durations"
    ]
)

distances = (
    osrm_data[
        "distances"
    ]
)


# ============================================================
# CONVERT OSRM SECONDS TO MINUTES
# ============================================================

travel_time_matrix = []

for row_index, row in enumerate(
    durations
):

    converted_row = []

    for col_index, value in enumerate(
        row
    ):

        if value is None:

            converted_row.append(
                999999
            )

        elif (
            row_index
            == col_index
        ):

            converted_row.append(
                0
            )

        else:

            converted_row.append(
                max(
                    1,
                    round(
                        value
                        / 60
                    ),
                )
            )

    travel_time_matrix.append(
        converted_row
    )


print(
    f"OSRM matrix          : "
    f"{len(locations)} x "
    f"{len(locations)}"
)


# ============================================================
# OR-TOOLS MULTI-DAY MODEL
# ============================================================

manager = (
    pywrapcp
    .RoutingIndexManager(
        len(
            locations
        ),
        NUM_DAYS,
        0,
    )
)

routing = (
    pywrapcp
    .RoutingModel(
        manager
    )
)


# ============================================================
# TIME CALLBACK
# ============================================================

def time_callback(
    from_index,
    to_index,
):

    from_node = (
        manager
        .IndexToNode(
            from_index
        )
    )

    to_node = (
        manager
        .IndexToNode(
            to_index
        )
    )

    travel_time = (
        travel_time_matrix[
            from_node
        ][
            to_node
        ]
    )

    service_time = (
        locations[
            from_node
        ][
            "visit_duration_min"
        ]
    )

    return (
        travel_time
        + service_time
    )


time_callback_index = (
    routing
    .RegisterTransitCallback(
        time_callback
    )
)

routing.SetArcCostEvaluatorOfAllVehicles(
    time_callback_index
)


# ============================================================
# DAILY TIME CONSTRAINT
# ============================================================

routing.AddDimension(
    time_callback_index,
    0,
    DAILY_TIME_LIMIT_MIN,
    True,
    "Time",
)

time_dimension = (
    routing
    .GetDimensionOrDie(
        "Time"
    )
)

for day in range(
    NUM_DAYS
):

    end_index = (
        routing.End(
            day
        )
    )

    time_dimension.CumulVar(
        end_index
    ).SetMax(
        DAILY_TIME_LIMIT_MIN
    )


# ============================================================
# OPTIONAL POIS + UTILITY
# ============================================================

for node in range(
    1,
    len(
        locations
    ),
):

    routing_index = (
        manager.NodeToIndex(
            node
        )
    )

    optimization_utility = (
        locations[
            node
        ][
            "optimization_utility"
        ]
    )

    penalty = int(
        optimization_utility
        *
        PREFERENCE_PENALTY_SCALE
    )

    routing.AddDisjunction(
        [
            routing_index
        ],
        penalty,
    )


# ============================================================
# SEARCH PARAMETERS
# ============================================================

search_parameters = (
    pywrapcp
    .DefaultRoutingSearchParameters()
)

search_parameters.first_solution_strategy = (
    routing_enums_pb2
    .FirstSolutionStrategy
    .PARALLEL_CHEAPEST_INSERTION
)

search_parameters.local_search_metaheuristic = (
    routing_enums_pb2
    .LocalSearchMetaheuristic
    .GUIDED_LOCAL_SEARCH
)

search_parameters.time_limit.seconds = 20


# ============================================================
# SOLVE
# ============================================================

print(
    "\nRunning multi-day OR-Tools optimization..."
)

solution = (
    routing
    .SolveWithParameters(
        search_parameters
    )
)

if solution is None:

    raise RuntimeError(
        "No feasible multi-day itinerary found."
    )


# ============================================================
# FORMAT CLOCK
# ============================================================

def format_clock(
    minutes_after_start,
):

    total_minutes = (
        DAY_START_HOUR
        * 60
        +
        minutes_after_start
    )

    hour = (
        total_minutes
        // 60
    )

    minute = (
        total_minutes
        % 60
    )

    return (
        f"{hour:02d}:"
        f"{minute:02d}"
    )


# ============================================================
# EXTRACT MULTI-DAY RESULTS
# ============================================================

trip_total_distance_m = 0

trip_total_travel_min = 0

trip_total_visit_min = 0

trip_total_preference = 0

trip_total_pois = 0

# Store hospital access times for all selected POIs.
trip_medical_times = []

all_visited_poi_ids = []

itinerary_rows = []


print(
    "\nMULTI-DAY OPTIMIZED ITINERARY"
)

print(
    "=" * 75
)


for day in range(
    NUM_DAYS
):

    print(
        f"\nDAY {day + 1}"
    )

    print(
        "-" * 75
    )

    index = (
        routing.Start(
            day
        )
    )

    day_distance_m = 0

    day_travel_min = 0

    day_visit_min = 0

    day_preference = 0

    day_pois = 0

    position = 1


    while not routing.IsEnd(
        index
    ):

        node = (
            manager
            .IndexToNode(
                index
            )
        )

        location = (
            locations[
                node
            ]
        )

        arrival_min = (
            solution.Value(
                time_dimension
                .CumulVar(
                    index
                )
            )
        )


        itinerary_rows.append(
            {
                "day":
                    day
                    + 1,

                "stop_order":
                    position,

                "poi_id":
                    location[
                        "id"
                    ],

                "name":
                    location[
                        "name"
                    ],

                "latitude":
                    location[
                        "latitude"
                    ],

                "longitude":
                    location[
                        "longitude"
                    ],

                "arrival_time":
                    format_clock(
                        arrival_min
                    ),

                "arrival_min_from_start":
                    arrival_min,

                "preference_score":
                    location[
                        "preference_score"
                    ],

                "medical_support_score":
                    location.get(
                        "medical_support_score",
                        0.0,
                    ),

                "nearest_emergency_hospital_name":
                    location.get(
                        "nearest_emergency_hospital_name"
                    ),

                "emergency_hospital_travel_time_min":
                    location.get(
                        "emergency_hospital_travel_time_min"
                    ),

                "visit_duration_min":
                    location[
                        "visit_duration_min"
                    ],
            }
        )


        print(
            f"{position:>2}. "
            f"{format_clock(arrival_min)} | "
            f"{location['name']} "
            f"| preference="
            f"{location['preference_score']} "
            f"| visit="
            f"{location['visit_duration_min']} min"
        )


        if node != 0:

            day_pois += 1

            day_preference += (
                location[
                    "preference_score"
                ]
            )

            day_visit_min += (
                location[
                    "visit_duration_min"
                ]
            )

            all_visited_poi_ids.append(
                location[
                    "id"
                ]
            )

            # -----------------------------------------------
            # COLLECT MEDICAL ACCESS TIMES
            # -----------------------------------------------

            medical_time = (
                location.get(
                    "emergency_hospital_travel_time_min"
                )
            )

            if (
                MEDICAL_SUPPORT_REQUIRED
                and medical_time is not None
                and not pd.isna(
                    medical_time
                )
            ):

                trip_medical_times.append(
                    float(
                        medical_time
                    )
                )


        next_index = (
            solution.Value(
                routing.NextVar(
                    index
                )
            )
        )

        next_node = (
            manager
            .IndexToNode(
                next_index
            )
        )

        road_distance = (
            distances[
                node
            ][
                next_node
            ]
        )

        if road_distance is not None:

            day_distance_m += (
                road_distance
            )

        day_travel_min += (
            travel_time_matrix[
                node
            ][
                next_node
            ]
        )

        index = (
            next_index
        )

        position += 1


    # ========================================================
    # FINAL RETURN TO HOTEL
    # ========================================================

    final_node = (
        manager
        .IndexToNode(
            index
        )
    )

    final_location = (
        locations[
            final_node
        ]
    )

    final_arrival = (
        solution.Value(
            time_dimension
            .CumulVar(
                index
            )
        )
    )


    itinerary_rows.append(
        {
            "day":
                day
                + 1,

            "stop_order":
                position,

            "poi_id":
                final_location[
                    "id"
                ],

            "name":
                final_location[
                    "name"
                ],

            "latitude":
                final_location[
                    "latitude"
                ],

            "longitude":
                final_location[
                    "longitude"
                ],

            "arrival_time":
                format_clock(
                    final_arrival
                ),

            "arrival_min_from_start":
                final_arrival,

            "preference_score":
                0,

            "medical_support_score":
                0.0,

            "nearest_emergency_hospital_name":
                None,

            "emergency_hospital_travel_time_min":
                None,

            "visit_duration_min":
                0,
        }
    )


    print(
        f"{position:>2}. "
        f"{format_clock(final_arrival)} | "
        f"{final_location['name']} "
        f"| preference=0 "
        f"| visit=0 min"
    )


    # ========================================================
    # DAY SUMMARY
    # ========================================================

    print(
        "\nDay summary:"
    )

    print(
        f"  POIs visited     : "
        f"{day_pois}"
    )

    print(
        f"  Road distance    : "
        f"{day_distance_m / 1000:.2f} km"
    )

    print(
        f"  Driving time     : "
        f"{day_travel_min} min"
    )

    print(
        f"  Visit time       : "
        f"{day_visit_min} min"
    )

    print(
        f"  Preference score : "
        f"{day_preference:.2f}"
    )

    print(
        f"  Return to hotel  : "
        f"{format_clock(final_arrival)}"
    )


    trip_total_distance_m += (
        day_distance_m
    )

    trip_total_travel_min += (
        day_travel_min
    )

    trip_total_visit_min += (
        day_visit_min
    )

    trip_total_preference += (
        day_preference
    )

    trip_total_pois += (
        day_pois
    )


# ============================================================
# VALIDATION
# ============================================================

unique_visited = set(
    all_visited_poi_ids
)

repeated_count = (
    len(
        all_visited_poi_ids
    )
    -
    len(
        unique_visited
    )
)


# ============================================================
# TRIP SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "MULTI-DAY TRIP SUMMARY"
)

print(
    "-" * 75
)


print(
    f"Number of days          : "
    f"{NUM_DAYS}"
)

print(
    f"Total POIs visited      : "
    f"{trip_total_pois}"
)

print(
    f"Unique POIs visited     : "
    f"{len(unique_visited)}"
)

print(
    f"Repeated POIs           : "
    f"{repeated_count}"
)

print(
    f"Total road distance     : "
    f"{trip_total_distance_m / 1000:.2f} km"
)

print(
    f"Total driving time      : "
    f"{trip_total_travel_min} min"
)

print(
    f"Total POI visit time    : "
    f"{trip_total_visit_min} min"
)

print(
    f"Total preference utility: "
    f"{trip_total_preference:.2f}"
)

print(
    f"Daily time limit        : "
    f"{DAILY_TIME_LIMIT_MIN} min/day"
)


# ============================================================
# MEDICAL ACCESS SUMMARY
# ============================================================

if (
    MEDICAL_SUPPORT_REQUIRED
    and trip_medical_times
):

    average_medical_time = (
        sum(
            trip_medical_times
        )
        /
        len(
            trip_medical_times
        )
    )

    best_medical_time = min(
        trip_medical_times
    )

    worst_medical_time = max(
        trip_medical_times
    )

    print(
        f"Average hospital access : "
        f"{average_medical_time:.2f} min"
    )

    print(
        f"Best hospital access    : "
        f"{best_medical_time:.2f} min"
    )

    print(
        f"Worst hospital access   : "
        f"{worst_medical_time:.2f} min"
    )


if repeated_count == 0:

    print(
        "POI repetition check    : PASS"
    )

else:

    print(
        "POI repetition check    : FAIL"
    )


# ============================================================
# SAVE STRUCTURED ITINERARY
# ============================================================

itinerary_df = pd.DataFrame(
    itinerary_rows
)

itinerary_df = (
    itinerary_df
    .sort_values(
        by=[
            "day",
            "stop_order",
        ]
    )
    .reset_index(
        drop=True
    )
)

itinerary_df.to_csv(
    ITINERARY_OUTPUT_FILE,
    index=False,
)


print(
    "\nSTRUCTURED ITINERARY OUTPUT"
)

print(
    "-" * 75
)

print(
    f"Rows saved              : "
    f"{len(itinerary_df)}"
)

print(
    f"Output file             : "
    f"{ITINERARY_OUTPUT_FILE}"
)


# ============================================================
# DISPLAY SAVED ITINERARY
# ============================================================

print(
    "\nSAVED ITINERARY PREVIEW"
)

print(
    "-" * 75
)

print(
    itinerary_df[
        [
            "day",
            "stop_order",
            "poi_id",
            "name",
            "arrival_time",
            "preference_score",
            "medical_support_score",
            "nearest_emergency_hospital_name",
            "emergency_hospital_travel_time_min",
            "visit_duration_min",
        ]
    ].to_string(
        index=False
        
    )
)


print(
    "\nStatus: SUCCESS"
)