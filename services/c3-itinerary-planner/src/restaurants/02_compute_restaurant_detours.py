from pathlib import Path

import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ITINERARY_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "optimized_itinerary.csv"
)

RESTAURANT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "restaurants_near_routes.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "restaurant_detour_scores.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

OSRM_URL = "http://localhost:5000"


# Only named restaurants are useful for recommendations.
REQUIRE_NAME = True


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

itinerary = pd.read_csv(
    ITINERARY_FILE
)

restaurants = pd.read_csv(
    RESTAURANT_FILE
)


print(
    f"Itinerary rows       : "
    f"{len(itinerary)}"
)

print(
    f"Restaurant candidates: "
    f"{len(restaurants)}"
)


# ============================================================
# BASIC CLEANING
# ============================================================

if REQUIRE_NAME:

    restaurants = restaurants[
        restaurants["name"].notna()
    ].copy()

    restaurants["name"] = (
        restaurants["name"]
        .astype(str)
        .str.strip()
    )

    restaurants = restaurants[
        restaurants["name"] != ""
    ].copy()


print(
    f"Named candidates     : "
    f"{len(restaurants)}"
)


# ============================================================
# RESULT STORAGE
# ============================================================

result_rows = []


# ============================================================
# PROCESS EACH DAY
# ============================================================

for day in sorted(
    itinerary["day"].unique()
):

    print(
        f"\nProcessing Day {day}..."
    )


    # --------------------------------------------------------
    # GET ORDERED ITINERARY STOPS
    # --------------------------------------------------------

    day_stops = itinerary[
        itinerary["day"] == day
    ].copy()

    day_stops = day_stops.sort_values(
        "stop_order"
    ).reset_index(
        drop=True
    )


    day_restaurants = restaurants[
        restaurants["day"] == day
    ].copy()

    day_restaurants = (
        day_restaurants.reset_index(
            drop=True
        )
    )


    print(
        f"Itinerary stops      : "
        f"{len(day_stops)}"
    )

    print(
        f"Named restaurants    : "
        f"{len(day_restaurants)}"
    )


    if len(day_restaurants) == 0:
        continue


    # ========================================================
    # BUILD ONE OSRM TABLE
    # ========================================================
    #
    # First locations:
    #   itinerary stops
    #
    # Remaining locations:
    #   restaurant candidates
    #
    # This allows us to calculate insertion detours using
    # actual road-network travel times.
    # ========================================================

    coordinate_rows = []


    for row in day_stops.itertuples():

        coordinate_rows.append(
            {
                "type": "stop",
                "latitude": row.latitude,
                "longitude": row.longitude,
            }
        )


    for row in day_restaurants.itertuples():

        coordinate_rows.append(
            {
                "type": "restaurant",
                "latitude": row.latitude,
                "longitude": row.longitude,
            }
        )


    coordinates = ";".join(
        f"{row['longitude']},"
        f"{row['latitude']}"
        for row in coordinate_rows
    )


    table_url = (
        f"{OSRM_URL}"
        f"/table/v1/driving/"
        f"{coordinates}"
    )


    print(
        f"OSRM matrix locations: "
        f"{len(coordinate_rows)}"
    )


    response = requests.get(
        table_url,
        params={
            "annotations":
                "duration,distance"
        },
        timeout=120,
    )


    response.raise_for_status()

    data = response.json()


    if data.get("code") != "Ok":

        raise RuntimeError(
            f"OSRM table error "
            f"for Day {day}: "
            f"{data}"
        )


    durations = data[
        "durations"
    ]

    distances = data[
        "distances"
    ]


    stop_count = len(
        day_stops
    )


    # ========================================================
    # CALCULATE BEST INSERTION FOR EACH RESTAURANT
    # ========================================================

    for restaurant_index, restaurant in (
        day_restaurants.iterrows()
    ):

        matrix_restaurant_index = (
            stop_count
            + restaurant_index
        )


        best_detour_time_sec = None
        best_detour_distance_m = None

        best_segment_index = None


        # ----------------------------------------------------
        # Test restaurant insertion between every pair of
        # consecutive itinerary stops.
        # ----------------------------------------------------

        for segment_index in range(
            len(day_stops) - 1
        ):

            from_index = (
                segment_index
            )

            to_index = (
                segment_index + 1
            )


            original_time = (
                durations[
                    from_index
                ][
                    to_index
                ]
            )


            to_restaurant_time = (
                durations[
                    from_index
                ][
                    matrix_restaurant_index
                ]
            )


            from_restaurant_time = (
                durations[
                    matrix_restaurant_index
                ][
                    to_index
                ]
            )


            original_distance = (
                distances[
                    from_index
                ][
                    to_index
                ]
            )


            to_restaurant_distance = (
                distances[
                    from_index
                ][
                    matrix_restaurant_index
                ]
            )


            from_restaurant_distance = (
                distances[
                    matrix_restaurant_index
                ][
                    to_index
                ]
            )


            required_values = [
                original_time,
                to_restaurant_time,
                from_restaurant_time,
                original_distance,
                to_restaurant_distance,
                from_restaurant_distance,
            ]


            if any(
                value is None
                for value in required_values
            ):
                continue


            detour_time_sec = (
                to_restaurant_time
                + from_restaurant_time
                - original_time
            )


            detour_distance_m = (
                to_restaurant_distance
                + from_restaurant_distance
                - original_distance
            )


            # Negative values can occasionally occur because
            # of snapping or route-network approximation.
            detour_time_sec = max(
                0,
                detour_time_sec
            )

            detour_distance_m = max(
                0,
                detour_distance_m
            )


            if (
                best_detour_time_sec
                is None
                or detour_time_sec
                < best_detour_time_sec
            ):

                best_detour_time_sec = (
                    detour_time_sec
                )

                best_detour_distance_m = (
                    detour_distance_m
                )

                best_segment_index = (
                    segment_index
                )


        # ----------------------------------------------------
        # Skip restaurant if no valid insertion exists.
        # ----------------------------------------------------

        if best_segment_index is None:
            continue


        from_stop = (
            day_stops.iloc[
                best_segment_index
            ]
        )

        to_stop = (
            day_stops.iloc[
                best_segment_index + 1
            ]
        )


        detour_time_min = (
            best_detour_time_sec
            / 60
        )


        detour_distance_km = (
            best_detour_distance_m
            / 1000
        )


        # ====================================================
        # DETOUR CONVENIENCE SCORE
        # ====================================================
        #
        # 0 min detour = 1.0
        # 10+ min detour = 0.0
        #
        # This is a prototype normalization and can later be
        # evaluated/tuned experimentally.
        # ====================================================

        detour_score = (
            1
            - (
                detour_time_min
                / 10.0
            )
        )

        detour_score = max(
            0.0,
            min(
                1.0,
                detour_score
            )
        )


        result_rows.append(
            {
                "day":
                    int(day),

                "osm_id":
                    restaurant[
                        "osm_id"
                    ],

                "name":
                    restaurant[
                        "name"
                    ],

                "cuisine":
                    restaurant.get(
                        "cuisine"
                    ),

                "opening_hours":
                    restaurant.get(
                        "opening_hours"
                    ),

                "addr_city":
                    restaurant.get(
                        "addr_city"
                    ),

                "latitude":
                    restaurant[
                        "latitude"
                    ],

                "longitude":
                    restaurant[
                        "longitude"
                    ],

                "distance_to_route_m":
                    restaurant[
                        "distance_to_route_m"
                    ],

                "insert_after_stop_order":
                    int(
                        from_stop[
                            "stop_order"
                        ]
                    ),

                "insert_after":
                    from_stop[
                        "name"
                    ],

                "insert_before":
                    to_stop[
                        "name"
                    ],

                "detour_time_min":
                    round(
                        detour_time_min,
                        2,
                    ),

                "detour_distance_km":
                    round(
                        detour_distance_km,
                        2,
                    ),

                "detour_score":
                    round(
                        detour_score,
                        3,
                    ),
            }
        )


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

results = pd.DataFrame(
    result_rows
)


if results.empty:

    raise RuntimeError(
        "No valid restaurant "
        "detour results generated."
    )


# ============================================================
# SORT BY DAY + DETOUR
# ============================================================

results = results.sort_values(
    by=[
        "day",
        "detour_time_min",
        "distance_to_route_m",
    ],
    ascending=[
        True,
        True,
        True,
    ],
).reset_index(
    drop=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    "\nRESTAURANT DETOUR SUMMARY"
)

print("=" * 80)


for day in sorted(
    results["day"].unique()
):

    day_results = results[
        results["day"] == day
    ]


    print(
        f"\nDAY {day}"
    )

    print("-" * 80)

    print(
        f"Restaurants scored : "
        f"{len(day_results)}"
    )


    print(
        "\nLowest-detour restaurants:"
    )


    print(
        day_results[
            [
                "name",
                "cuisine",
                "insert_after",
                "insert_before",
                "detour_time_min",
                "detour_distance_km",
                "detour_score",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


print(
    f"\nTotal scored restaurants: "
    f"{len(results)}"
)


print(
    "\nOutput saved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nStatus: SUCCESS"
)
