from pathlib import Path
import json

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

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "route_geometries"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "route_geometry_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

OSRM_URL = "http://localhost:5000"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD OPTIMIZED ITINERARY
# ============================================================

print("\nLoading optimized itinerary...")

itinerary = pd.read_csv(
    ITINERARY_FILE
)


print(
    f"Itinerary rows loaded : "
    f"{len(itinerary)}"
)

print(
    f"Number of days        : "
    f"{itinerary['day'].nunique()}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "day",
    "stop_order",
    "name",
    "latitude",
    "longitude",
}


missing_columns = (
    required_columns
    - set(itinerary.columns)
)


if missing_columns:

    raise ValueError(
        f"Missing itinerary columns: "
        f"{sorted(missing_columns)}"
    )


# Sort itinerary explicitly.
itinerary = itinerary.sort_values(
    by=[
        "day",
        "stop_order",
    ]
).reset_index(
    drop=True
)


# ============================================================
# BUILD ROUTE FOR EACH DAY
# ============================================================

summary_rows = []

combined_features = []


print(
    "\nRequesting full road geometry "
    "from OSRM..."
)


for day in sorted(
    itinerary["day"].unique()
):

    day_data = itinerary[
        itinerary["day"] == day
    ].copy()


    day_data = day_data.sort_values(
        "stop_order"
    )


    print(
        f"\nProcessing Day {day}..."
    )

    print(
        f"Stops               : "
        f"{len(day_data)}"
    )


    # --------------------------------------------------------
    # CREATE OSRM COORDINATE STRING
    # --------------------------------------------------------

    coordinates = ";".join(
        f"{row.longitude},"
        f"{row.latitude}"
        for row
        in day_data.itertuples()
    )


    route_url = (
        f"{OSRM_URL}"
        f"/route/v1/driving/"
        f"{coordinates}"
    )


    # --------------------------------------------------------
    # REQUEST ROUTE
    # --------------------------------------------------------

    response = requests.get(
        route_url,
        params={
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        },
        timeout=60,
    )


    response.raise_for_status()

    route_data = response.json()


    if route_data.get("code") != "Ok":

        raise RuntimeError(
            f"OSRM route error "
            f"for Day {day}: "
            f"{route_data}"
        )


    if not route_data.get("routes"):

        raise RuntimeError(
            f"No route returned "
            f"for Day {day}."
        )


    route = route_data["routes"][0]


    distance_m = route["distance"]
    duration_sec = route["duration"]

    geometry = route["geometry"]


    distance_km = (
        distance_m / 1000
    )

    duration_min = (
        duration_sec / 60
    )


    # --------------------------------------------------------
    # CREATE GEOJSON FEATURE
    # --------------------------------------------------------

    feature = {
        "type": "Feature",
        "properties": {
            "day": int(day),
            "distance_km": round(
                distance_km,
                2,
            ),
            "duration_min": round(
                duration_min,
                2,
            ),
            "number_of_stops": int(
                len(day_data)
            ),
        },
        "geometry": geometry,
    }


    # --------------------------------------------------------
    # SAVE INDIVIDUAL DAY GEOJSON
    # --------------------------------------------------------

    day_geojson = {
        "type": "FeatureCollection",
        "features": [
            feature
        ],
    }


    day_output_file = (
        OUTPUT_FOLDER
        / f"day_{int(day)}_route.geojson"
    )


    with open(
        day_output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            day_geojson,
            file,
            indent=2,
        )


    combined_features.append(
        feature
    )


    # --------------------------------------------------------
    # SAVE SUMMARY INFORMATION
    # --------------------------------------------------------

    summary_rows.append(
        {
            "day": int(day),
            "number_of_stops": len(
                day_data
            ),
            "distance_km": round(
                distance_km,
                2,
            ),
            "duration_min": round(
                duration_min,
                2,
            ),
            "geometry_points": len(
                geometry[
                    "coordinates"
                ]
            ),
            "output_file":
                day_output_file.name,
        }
    )


    print(
        f"Road distance       : "
        f"{distance_km:.2f} km"
    )

    print(
        f"Driving duration    : "
        f"{duration_min:.2f} min"
    )

    print(
        f"Geometry points     : "
        f"{len(geometry['coordinates'])}"
    )

    print(
        f"Saved               : "
        f"{day_output_file.name}"
    )


# ============================================================
# SAVE COMBINED ROUTES
# ============================================================

combined_geojson = {
    "type": "FeatureCollection",
    "features": combined_features,
}


combined_output_file = (
    OUTPUT_FOLDER
    / "all_days_routes.geojson"
)


with open(
    combined_output_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        combined_geojson,
        file,
        indent=2,
    )


# ============================================================
# SAVE ROUTE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    "\nROUTE GEOMETRY SUMMARY"
)

print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nCombined route file:"
)

print(
    combined_output_file
)


print(
    "\nSummary file:"
)

print(
    SUMMARY_FILE
)


print(
    "\nStatus: SUCCESS"
)
