from pathlib import Path
import json

import pandas as pd

import numpy as np





# ============================================================

# PATHS

# ============================================================



PROJECT_ROOT = Path(__file__).resolve().parents[2]



DETOUR_FILE = (

    PROJECT_ROOT

    / "Data"

    / "processed"

    / "restaurant_detour_scores.csv"

)



ABSA_FILE = (

    PROJECT_ROOT

    / "ABSA"

    / "output"

    / "restaurant_aspect_scores.csv"

)



OUTPUT_FILE = (

    PROJECT_ROOT

    / "Data"

    / "processed"

    / "final_ranked_restaurants.csv"

)





# ============================================================

# COMPONENT 2 -> COMPONENT 3 INPUT

# ============================================================


C2_C3_INPUT_FILE = (

    PROJECT_ROOT

    / "Data"

    / "input"

    / "c2_c3_input.json"

)


print(

    "\nLoading Component 2 -> Component 3 restaurant preferences..."

)


if not C2_C3_INPUT_FILE.exists():

    raise FileNotFoundError(

        "Component 2 -> Component 3 input file "

        f"not found:\n{C2_C3_INPUT_FILE}"

    )


with open(

    C2_C3_INPUT_FILE,

    "r",

    encoding="utf-8",

) as f:

    C2_INPUT = json.load(f)


if "restaurant_preferences" not in C2_INPUT:

    raise ValueError(

        "Missing required input section: "

        "restaurant_preferences"

    )


restaurant_preferences = (

    C2_INPUT["restaurant_preferences"]

)


if "aspect_importance" not in restaurant_preferences:

    raise ValueError(

        "Missing restaurant preference field: "

        "aspect_importance"

    )


if "cuisine_preferences" not in restaurant_preferences:

    raise ValueError(

        "Missing restaurant preference field: "

        "cuisine_preferences"

    )


TOURIST_ASPECT_IMPORTANCE = {

    str(aspect).strip().lower():

        float(importance)

    for aspect, importance in (

        restaurant_preferences[

            "aspect_importance"

        ].items()

    )

}


TOURIST_CUISINE_PREFERENCES = {

    str(cuisine)

    .strip()

    .lower()

    .replace(" ", "_")

    .replace("-", "_"):

        float(score)

    for cuisine, score in (

        restaurant_preferences[

            "cuisine_preferences"

        ].items()

    )

}


if not TOURIST_ASPECT_IMPORTANCE:

    raise ValueError(

        "aspect_importance cannot be empty."

    )


if not TOURIST_CUISINE_PREFERENCES:

    raise ValueError(

        "cuisine_preferences cannot be empty."

    )


for aspect, importance in (

    TOURIST_ASPECT_IMPORTANCE.items()

):

    if importance < 0:

        raise ValueError(

            f"Aspect importance for '{aspect}' "

            "cannot be negative."

        )


for cuisine, score in (

    TOURIST_CUISINE_PREFERENCES.items()

):

    if score < 0:

        raise ValueError(

            f"Cuisine preference for '{cuisine}' "

            "cannot be negative."

        )


print(

    f"Trip ID                  : "

    f"{C2_INPUT.get('trip_id', 'N/A')}"

)


print(

    f"Aspect preferences       : "

    f"{TOURIST_ASPECT_IMPORTANCE}"

)


print(

    f"Cuisine preferences      : "

    f"{TOURIST_CUISINE_PREFERENCES}"

)


print(

    "C2 -> C3 preferences    : SUCCESS"

)




# ============================================================

# FINAL SCORE WEIGHTS

# ============================================================



WEIGHT_ABSA = 0.45

WEIGHT_CUISINE = 0.20

WEIGHT_DETOUR = 0.35





# ============================================================

# LOAD DATA

# ============================================================



print(

    "\nFINAL ROUTE-BASED RESTAURANT RANKING"

)



print(

    "=" * 70

)





restaurants = pd.read_csv(

    DETOUR_FILE

)



absa = pd.read_csv(

    ABSA_FILE

)





print(

    f"Route restaurant candidates : "

    f"{len(restaurants)}"

)



print(

    f"Restaurants with ABSA data  : "

    f"{len(absa)}"

)





# ============================================================

# BASIC CLEANING

# ============================================================



restaurants["name"] = (

    restaurants["name"]

    .astype("string")

    .str.strip()

)





restaurants["cuisine"] = (

    restaurants["cuisine"]

    .astype("string")

    .str.strip()

    .str.lower()

)





# ============================================================

# CUISINE PREFERENCE SCORE

# ============================================================



def calculate_cuisine_score(cuisine_value):



    if pd.isna(cuisine_value):

        return 0.5





    cuisine_text = str(

        cuisine_value

    ).lower()





    cuisine_tokens = [

        token.strip()

        for token in cuisine_text.split(";")

        if token.strip()

    ]





    matched_scores = []





    for token in cuisine_tokens:



        normalized = (

            token

            .replace(" ", "_")

            .replace("-", "_")

        )





        if normalized in (

            TOURIST_CUISINE_PREFERENCES

        ):



            matched_scores.append(

                TOURIST_CUISINE_PREFERENCES[

                    normalized

                ]

            )





    if not matched_scores:

        return 0.5





    best_score = max(

        matched_scores

    )





    return best_score / 5.0





restaurants[

    "cuisine_preference_score"

] = (

    restaurants["cuisine"]

    .apply(

        calculate_cuisine_score

    )

)





# ============================================================

# MERGE ABSA DATA

# ============================================================



absa_columns = [

    "osm_id",

    "food_score",

    "service_score",

    "ambience_score",

    "value_score",

    "location_score",

    "total_review_count",

    "available_aspect_count",

    "absa_quality_score",

]





absa_small = absa[

    [

        column

        for column in absa_columns

        if column in absa.columns

    ]

].copy()





restaurants = restaurants.merge(

    absa_small,

    on="osm_id",

    how="left"

)





# ============================================================

# PREFERENCE-WEIGHTED ABSA SCORE

# ============================================================



def calculate_weighted_absa(row):



    weighted_sum = 0.0

    total_weight = 0.0

    available_aspects = 0





    for aspect, importance in (

        TOURIST_ASPECT_IMPORTANCE.items()

    ):



        column = (

            f"{aspect}_score"

        )





        if column not in row.index:

            continue





        value = row[column]





        if pd.isna(value):

            continue





        weight = float(

            importance

        )





        weighted_sum += (

            float(value)

            * weight

        )





        total_weight += weight

        available_aspects += 1





    if total_weight == 0:

        return pd.Series(

            {

                "weighted_absa_score":

                    np.nan,



                "weighted_absa_aspects":

                    0,

            }

        )





    return pd.Series(

        {

            "weighted_absa_score":

                weighted_sum

                / total_weight,



            "weighted_absa_aspects":

                available_aspects,

        }

    )





weighted_results = (

    restaurants.apply(

        calculate_weighted_absa,

        axis=1

    )

)





restaurants[

    "weighted_absa_score"

] = (

    weighted_results[

        "weighted_absa_score"

    ]

)





restaurants[

    "weighted_absa_aspects"

] = (

    weighted_results[

        "weighted_absa_aspects"

    ]

)





# ============================================================

# REVIEW EVIDENCE

# ============================================================



restaurants[

    "has_absa_data"

] = (

    restaurants[

        "weighted_absa_score"

    ]

    .notna()

)





restaurants[

    "review_evidence_status"

] = np.where(

    restaurants[

        "has_absa_data"

    ],

    "available",

    "unavailable"

)





# ============================================================

# FINAL SCORE

# ============================================================



def calculate_final_score(row):



    detour_score = float(

        row["detour_score"]

    )





    cuisine_score = float(

        row[

            "cuisine_preference_score"

        ]

    )





    absa_score = row[

        "weighted_absa_score"

    ]





    # --------------------------------------------------------

    # CASE 1:

    # ABSA review evidence is available

    # --------------------------------------------------------



    if not pd.isna(

        absa_score

    ):



        score = (

            WEIGHT_ABSA

            * float(absa_score)



            + WEIGHT_CUISINE

            * cuisine_score



            + WEIGHT_DETOUR

            * detour_score

        )





        scoring_mode = (

            "absa+cuisine+detour"

        )





    # --------------------------------------------------------

    # CASE 2:

    # No review evidence

    #

    # Do NOT assign ABSA = 0.

    # Re-normalize available weights instead.

    # --------------------------------------------------------



    else:



        available_weight = (

            WEIGHT_CUISINE

            + WEIGHT_DETOUR

        )





        score = (



            WEIGHT_CUISINE

            * cuisine_score



            + WEIGHT_DETOUR

            * detour_score



        ) / available_weight





        scoring_mode = (

            "cuisine+detour"

        )





    return pd.Series(

        {

            "final_restaurant_score":

                score,



            "scoring_mode":

                scoring_mode,

        }

    )





final_results = (

    restaurants.apply(

        calculate_final_score,

        axis=1

    )

)





restaurants[

    "final_restaurant_score"

] = (

    final_results[

        "final_restaurant_score"

    ]

)





restaurants[

    "scoring_mode"

] = (

    final_results[

        "scoring_mode"

    ]

)





# ============================================================

# ROUND

# ============================================================



score_columns = [

    "cuisine_preference_score",

    "weighted_absa_score",

    "final_restaurant_score",

]





for column in score_columns:



    if column in restaurants.columns:



        restaurants[column] = (

            restaurants[column]

            .round(4)

        )





# ============================================================

# RANK WITHIN EACH ITINERARY DAY

# ============================================================



restaurants = restaurants.sort_values(

    by=[

        "day",

        "final_restaurant_score",

        "detour_time_min",

    ],

    ascending=[

        True,

        False,

        True,

    ]

)





restaurants[

    "final_day_rank"

] = (

    restaurants

    .groupby(

        "day"

    )

    .cumcount()

    + 1

)





# ============================================================

# SAVE

# ============================================================



OUTPUT_FILE.parent.mkdir(

    parents=True,

    exist_ok=True

)





restaurants.to_csv(

    OUTPUT_FILE,

    index=False,

    encoding="utf-8-sig"

)





# ============================================================

# SUMMARY

# ============================================================



print(

    "\nFINAL RANKING SUMMARY"

)



print(

    "-" * 70

)





print(

    f"Restaurants ranked : "

    f"{len(restaurants)}"

)





print(

    f"With ABSA data     : "

    f"{restaurants['has_absa_data'].sum()}"

)





print(

    f"Without ABSA data  : "

    f"{(~restaurants['has_absa_data']).sum()}"

)





print(

    "\nScoring modes:"

)





print(

    restaurants[

        "scoring_mode"

    ]

    .value_counts()

    .to_string()

)





# ============================================================

# TOP 5 PER DAY

# ============================================================



print(

    "\nTOP 5 RESTAURANTS PER DAY"

)



print(

    "-" * 70

)





top_restaurants = (

    restaurants[

        restaurants[

            "final_day_rank"

        ] <= 5

    ]

    [

        [

            "day",

            "final_day_rank",

            "name",

            "cuisine",

            "detour_time_min",

            "detour_score",

            "cuisine_preference_score",

            "weighted_absa_score",

            "total_review_count",

            "final_restaurant_score",

            "scoring_mode",

        ]

    ]

)





print(

    top_restaurants.to_string(

        index=False

    )

)





print(

    f"\nOutput file:\n"

    f"{OUTPUT_FILE}"

)





print(

    "\nStatus: SUCCESS"

)


