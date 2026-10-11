from pathlib import Path

import json
import subprocess
import sys
import threading

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Component 3 Itinerary Planner",
    version="1.0.0",
)


# ============================================================
# PROJECT PATHS
# ============================================================

# main.py is located at:
# services/c3-itinerary-planner/src/main.py
#
# Therefore parents[1] gives:
# services/c3-itinerary-planner
PROJECT_ROOT = Path(__file__).resolve().parents[1]


INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "input"
    / "c2_c3_input.json"
)


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
    / "final_ranked_restaurants.csv"
)


# ============================================================
# PIPELINE SCRIPTS
# ============================================================

PIPELINE_SCRIPTS = [
    PROJECT_ROOT
    / "src"
    / "optimization"
    / "02_score_pois.py",

    PROJECT_ROOT
    / "src"
    / "optimization"
    / "04_optimize_multi_day.py",

    PROJECT_ROOT
    / "src"
    / "routing"
    / "03_build_route_geometry.py",

    PROJECT_ROOT
    / "src"
    / "restaurants"
    / "01_filter_restaurants_near_route.py",

    PROJECT_ROOT
    / "src"
    / "restaurants"
    / "02_compute_restaurant_detours.py",

    PROJECT_ROOT
    / "src"
    / "restaurants"
    / "06_rank_restaurants_with_absa.py",
]


# Prevent two API requests from modifying the same
# pipeline files at the same time.
PIPELINE_LOCK = threading.Lock()


# ============================================================
# REQUEST MODELS
# ============================================================

class HotelInput(BaseModel):

    hotel_id: str

    name: str

    latitude: float = Field(
        ge=-90,
        le=90,
    )

    longitude: float = Field(
        ge=-180,
        le=180,
    )

    district: str


class TripInput(BaseModel):

    num_days: int = Field(
        ge=1,
        le=14,
    )

    day_start_hour: int = Field(
        ge=0,
        le=23,
    )

    day_end_hour: int = Field(
        ge=0,
        le=23,
    )

    travel_mode: str = "driving"

    @field_validator(
        "travel_mode"
    )
    @classmethod
    def validate_travel_mode(
        cls,
        value,
    ):

        allowed = {
            "driving",
        }

        value = value.lower().strip()

        if value not in allowed:
            raise ValueError(
                "Currently supported travel_mode "
                "is 'driving'."
            )

        return value


class AttractionPreferences(BaseModel):

    history_culture: float = Field(
        ge=1,
        le=5,
    )

    nature_wildlife: float = Field(
        ge=1,
        le=5,
    )

    beach_coastal: float = Field(
        ge=1,
        le=5,
    )

    religious: float = Field(
        ge=1,
        le=5,
    )

    adventure_recreation: float = Field(
        ge=1,
        le=5,
    )

    urban_leisure: float = Field(
        ge=1,
        le=5,
    )


class AspectImportance(BaseModel):

    food: float = Field(
        ge=1,
        le=5,
    )

    service: float = Field(
        ge=1,
        le=5,
    )

    ambience: float = Field(
        ge=1,
        le=5,
    )

    value: float = Field(
        ge=1,
        le=5,
    )

    location: float = Field(
        ge=1,
        le=5,
    )


class RestaurantPreferences(BaseModel):

    aspect_importance: AspectImportance

    cuisine_preferences: dict[
        str,
        float
    ]

    @field_validator(
        "cuisine_preferences"
    )
    @classmethod
    def validate_cuisine_preferences(
        cls,
        value,
    ):

        if not value:
            raise ValueError(
                "At least one cuisine preference "
                "is required."
            )

        for cuisine, score in value.items():

            if not 1 <= float(score) <= 5:
                raise ValueError(
                    f"Cuisine preference "
                    f"'{cuisine}' must be "
                    "between 1 and 5."
                )

        return value


class PlanRequest(BaseModel):

    trip_id: str

    hotel: HotelInput

    trip: TripInput

    attraction_preferences: (
        AttractionPreferences
    )

    restaurant_preferences: (
        RestaurantPreferences
    )

    @field_validator(
        "trip"
    )
    @classmethod
    def validate_trip_time(
        cls,
        value,
    ):

        if (
            value.day_end_hour
            <= value.day_start_hour
        ):
            raise ValueError(
                "day_end_hour must be later "
                "than day_start_hour."
            )

        return value


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def run_script(
    script_path: Path,
):

    if not script_path.exists():
        raise RuntimeError(
            f"Pipeline script not found: "
            f"{script_path}"
        )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"\nPipeline stage failed:\n"
            f"{script_path.name}\n\n"
            f"STDOUT:\n"
            f"{result.stdout}\n\n"
            f"STDERR:\n"
            f"{result.stderr}"
        )

    return result.stdout


def dataframe_to_records(
    dataframe: pd.DataFrame,
):

    clean_df = dataframe.copy()

    clean_df = clean_df.astype(
        object
    ).where(
        pd.notna(clean_df),
        None,
    )

    return clean_df.to_dict(
        orient="records"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "service":
            "component-3-itinerary-planner",

        "status":
            "ok",
    }


# ============================================================
# PLAN ITINERARY
# ============================================================

@app.post("/plan-itinerary")
def plan_itinerary(
    request: PlanRequest,
):

    with PIPELINE_LOCK:

        try:

            # --------------------------------------------
            # SAVE COMPONENT 2 INPUT
            # --------------------------------------------

            INPUT_FILE.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            request_data = (
                request.model_dump()
            )

            with open(
                INPUT_FILE,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    request_data,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )


            # --------------------------------------------
            # RUN COMPONENT 3 PIPELINE
            # --------------------------------------------

            pipeline_logs = []

            for script in PIPELINE_SCRIPTS:

                output = run_script(
                    script
                )

                pipeline_logs.append(
                    {
                        "stage":
                            script.name,

                        "status":
                            "success",
                    }
                )


            # --------------------------------------------
            # CHECK OUTPUT FILES
            # --------------------------------------------

            if not ITINERARY_FILE.exists():

                raise RuntimeError(
                    "optimized_itinerary.csv "
                    "was not generated."
                )


            if not RESTAURANT_FILE.exists():

                raise RuntimeError(
                    "final_ranked_restaurants.csv "
                    "was not generated."
                )


            # --------------------------------------------
            # LOAD ITINERARY
            # --------------------------------------------

            itinerary_df = pd.read_csv(
                ITINERARY_FILE
            )


            # --------------------------------------------
            # LOAD RESTAURANT RANKING
            # --------------------------------------------

            restaurants_df = pd.read_csv(
                RESTAURANT_FILE
            )


            # Top 5 restaurants for each day.
            top_restaurants_df = (
                restaurants_df[
                    restaurants_df[
                        "final_day_rank"
                    ]
                    <= 5
                ]
                .copy()
            )


            # --------------------------------------------
            # GROUP RESULTS BY DAY
            # --------------------------------------------

            days = []

            day_numbers = sorted(
                itinerary_df[
                    "day"
                ]
                .dropna()
                .unique()
            )


            for day_number in day_numbers:

                day_itinerary = (
                    itinerary_df[
                        itinerary_df[
                            "day"
                        ]
                        == day_number
                    ]
                    .sort_values(
                        "stop_order"
                    )
                )


                day_restaurants = (
                    top_restaurants_df[
                        top_restaurants_df[
                            "day"
                        ]
                        == day_number
                    ]
                    .sort_values(
                        "final_day_rank"
                    )
                )


                days.append(
                    {
                        "day":
                            int(day_number),

                        "stops":
                            dataframe_to_records(
                                day_itinerary
                            ),

                        "recommended_restaurants":
                            dataframe_to_records(
                                day_restaurants
                            ),
                    }
                )


            # --------------------------------------------
            # RESPONSE
            # --------------------------------------------

            return {
                "status":
                    "success",

                "trip_id":
                    request.trip_id,

                "hotel":
                    request.hotel.model_dump(),

                "trip":
                    request.trip.model_dump(),

                "days":
                    days,

                "pipeline":
                    pipeline_logs,
            }


        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=str(error),
            )