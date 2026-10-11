from pathlib import Path

import geopandas as gpd
import pandas as pd

from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESTAURANT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "restaurants_osm_with_districts.csv"
)

ROUTE_FOLDER = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "route_geometries"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "restaurants_near_routes.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Route corridor radius in metres.
# Restaurants within this distance from the actual road route
# become candidates for recommendation.
ROUTE_BUFFER_METERS = 1000


# ============================================================
# LOAD RESTAURANTS
# ============================================================

print("\nLoading OSM restaurants...")

restaurants = pd.read_csv(
    RESTAURANT_FILE
)

print(
    f"Restaurants loaded     : "
    f"{len(restaurants)}"
)


# ============================================================
# VALIDATE RESTAURANT COORDINATES
# ============================================================

restaurants = restaurants.dropna(
    subset=[
        "latitude",
        "longitude",
    ]
).copy()


restaurants = restaurants[
    restaurants["latitude"].between(
        5.5,
        10.2,
    )
    &
    restaurants["longitude"].between(
        79.0,
        82.2,
    )
].copy()


print(
    f"Valid coordinates      : "
    f"{len(restaurants)}"
)


# ============================================================
# CREATE RESTAURANT GEODATAFRAME
# ============================================================

restaurant_gdf = gpd.GeoDataFrame(
    restaurants,
    geometry=[
        Point(
            lon,
            lat,
        )
        for lon, lat in zip(
            restaurants["longitude"],
            restaurants["latitude"],
        )
    ],
    crs="EPSG:4326",
)


# ============================================================
# USE METRIC CRS
# ============================================================
#
# EPSG:5234 is Sri Lanka Grid 1999.
# We project both routes and restaurants to a metre-based CRS
# so distance and buffer calculations are meaningful.
# ============================================================

METRIC_CRS = "EPSG:5234"

restaurant_gdf_metric = (
    restaurant_gdf.to_crs(
        METRIC_CRS
    )
)


# ============================================================
# PROCESS EACH DAY
# ============================================================

all_candidates = []


route_files = sorted(
    ROUTE_FOLDER.glob(
        "day_*_route.geojson"
    )
)


if not route_files:

    raise FileNotFoundError(
        "No day route GeoJSON files found."
    )


print(
    f"Route files found      : "
    f"{len(route_files)}"
)


for route_file in route_files:

    # --------------------------------------------------------
    # EXTRACT DAY NUMBER
    # --------------------------------------------------------

    day = int(
        route_file.stem
        .replace(
            "day_",
            ""
        )
        .replace(
            "_route",
            ""
        )
    )


    print(
        f"\nProcessing Day {day}..."
    )


    # --------------------------------------------------------
    # LOAD ROUTE
    # --------------------------------------------------------

    route_gdf = gpd.read_file(
        route_file
    )


    if route_gdf.empty:

        print(
            "Route file is empty. "
            "Skipping."
        )

        continue


    route_gdf_metric = (
        route_gdf.to_crs(
            METRIC_CRS
        )
    )


    # Normally each file contains one route LineString.
    route_geometry = (
        route_gdf_metric
        .geometry
        .iloc[0]
    )


    # --------------------------------------------------------
    # CREATE ROUTE BUFFER
    # --------------------------------------------------------

    route_buffer = (
        route_geometry.buffer(
            ROUTE_BUFFER_METERS
        )
    )


    # --------------------------------------------------------
    # COMPUTE DISTANCE FROM EACH RESTAURANT TO ROUTE
    # --------------------------------------------------------

    distances_to_route = (
        restaurant_gdf_metric
        .geometry
        .distance(
            route_geometry
        )
    )


    day_candidates = (
        restaurant_gdf.copy()
    )


    day_candidates[
        "distance_to_route_m"
    ] = distances_to_route.values


    # Keep only restaurants inside the route corridor.
    day_candidates = day_candidates[
        day_candidates[
            "distance_to_route_m"
        ]
        <= ROUTE_BUFFER_METERS
    ].copy()


    # --------------------------------------------------------
    # ADD DAY INFORMATION
    # --------------------------------------------------------

    day_candidates["day"] = day


    day_candidates[
        "route_buffer_m"
    ] = ROUTE_BUFFER_METERS


    # --------------------------------------------------------
    # CLEAN DISTANCE
    # --------------------------------------------------------

    day_candidates[
        "distance_to_route_m"
    ] = (
        day_candidates[
            "distance_to_route_m"
        ]
        .round(1)
    )


    # --------------------------------------------------------
    # SORT BY DISTANCE TO ROUTE
    # --------------------------------------------------------

    day_candidates = (
        day_candidates.sort_values(
            by=[
                "distance_to_route_m",
                "name",
            ],
            ascending=[
                True,
                True,
            ],
        )
    )


    # --------------------------------------------------------
    # DROP GEOMETRY BEFORE EXPORT
    # --------------------------------------------------------

    day_candidates = (
        pd.DataFrame(
            day_candidates.drop(
                columns=[
                    "geometry"
                ]
            )
        )
    )


    print(
        f"Restaurants within "
        f"{ROUTE_BUFFER_METERS} m: "
        f"{len(day_candidates)}"
    )


    if len(day_candidates) > 0:

        print(
            "\nClosest 10:"
        )

        display_columns = [
            "osm_id",
            "name",
            "cuisine",
            "addr_city",
            "distance_to_route_m",
        ]

        print(
            day_candidates[
                display_columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )


    all_candidates.append(
        day_candidates
    )


# ============================================================
# COMBINE ALL DAYS
# ============================================================

if not all_candidates:

    raise RuntimeError(
        "No route candidate results created."
    )


combined = pd.concat(
    all_candidates,
    ignore_index=True,
)


# ============================================================
# HANDLE SAME RESTAURANT ON MULTIPLE DAY ROUTES
# ============================================================
#
# A restaurant can legitimately be close to more than one
# daily route.
#
# We therefore keep day-specific rows rather than deleting
# them globally.
# ============================================================

combined = combined.sort_values(
    by=[
        "day",
        "distance_to_route_m",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nROUTE-BASED RESTAURANT FILTER SUMMARY"
)

print("=" * 75)


summary = (
    combined.groupby("day")
    .agg(
        restaurant_candidates=(
            "osm_id",
            "count"
        ),
        named_restaurants=(
            "name",
            lambda x: x.notna().sum()
        ),
        cuisine_available=(
            "cuisine",
            lambda x: x.notna().sum()
        ),
        opening_hours_available=(
            "opening_hours",
            lambda x: x.notna().sum()
        ),
        average_distance_to_route_m=(
            "distance_to_route_m",
            "mean"
        ),
    )
    .reset_index()
)


summary[
    "average_distance_to_route_m"
] = (
    summary[
        "average_distance_to_route_m"
    ]
    .round(1)
)


print(
    summary.to_string(
        index=False
    )
)


print(
    f"\nTotal day-specific candidates: "
    f"{len(combined)}"
)


print(
    f"\nOutput saved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nStatus: SUCCESS"
)
