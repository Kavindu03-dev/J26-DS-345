from pathlib import Path
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "poi_interest_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "poi_preference_scores.csv"
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

print(
    "\nLoading Component 2 -> Component 3 "
    "attraction preferences..."
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


if "attraction_preferences" not in C2_INPUT:
    raise ValueError(
        "Missing 'attraction_preferences' "
        "in c2_c3_input.json"
    )


TOURIST_PREFERENCES = {
    category: float(weight)
    for category, weight in (
        C2_INPUT[
            "attraction_preferences"
        ].items()
    )
}


# ============================================================
# VALIDATE TOURIST PREFERENCES
# ============================================================

EXPECTED_CATEGORIES = {
    "history_culture",
    "nature_wildlife",
    "beach_coastal",
    "religious",
    "adventure_recreation",
    "urban_leisure",
}


missing_preferences = (
    EXPECTED_CATEGORIES
    - set(TOURIST_PREFERENCES.keys())
)

if missing_preferences:
    raise ValueError(
        "Missing attraction preferences: "
        f"{sorted(missing_preferences)}"
    )


unexpected_preferences = (
    set(TOURIST_PREFERENCES.keys())
    - EXPECTED_CATEGORIES
)

if unexpected_preferences:
    raise ValueError(
        "Unexpected attraction preferences: "
        f"{sorted(unexpected_preferences)}"
    )


for category, weight in (
    TOURIST_PREFERENCES.items()
):
    if not 1 <= weight <= 5:
        raise ValueError(
            f"Preference for '{category}' "
            "must be between 1 and 5."
        )


CATEGORY_COLUMNS = list(
    TOURIST_PREFERENCES.keys()
)


print(
    f"Trip ID                  : "
    f"{C2_INPUT.get('trip_id', 'N/A')}"
)

print(
    "C2 -> C3 attraction prefs: SUCCESS"
)


# ============================================================
# LOAD DATA
# ============================================================

pois = pd.read_csv(
    INPUT_FILE
)


print(
    "\nTOURIST PREFERENCE PROFILE"
)

print(
    "-----------------------------------------"
)


for category, weight in (
    TOURIST_PREFERENCES.items()
):
    print(
        f"{category:<24}: "
        f"{weight:g}/5"
    )


# ============================================================
# VALIDATE COLUMNS
# ============================================================

missing_columns = (
    set(CATEGORY_COLUMNS)
    - set(pois.columns)
)


if missing_columns:
    raise ValueError(
        f"Missing category columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# CALCULATE PREFERENCE SCORE
# ============================================================

def calculate_preference_score(row):

    matched_weights = []

    for category in CATEGORY_COLUMNS:

        if row[category] == 1:
            matched_weights.append(
                TOURIST_PREFERENCES[
                    category
                ]
            )

    # No recognised interest category.
    if not matched_weights:
        return 0.0

    # Average preference across all categories
    # associated with the POI.
    #
    # This avoids automatically giving
    # multi-label attractions larger scores
    # simply because they matched more
    # categories.

    return round(
        sum(matched_weights)
        / len(matched_weights),
        2,
    )


pois[
    "preference_score"
] = pois.apply(
    calculate_preference_score,
    axis=1,
)


# Normalize from 0-5 to 0-1.
pois[
    "normalized_preference_score"
] = (
    pois[
        "preference_score"
    ]
    / 5.0
).round(3)


# ============================================================
# RANK POIS
# ============================================================

pois = (
    pois.sort_values(
        by=[
            "preference_score",
            "interest_category_count",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


pois[
    "preference_rank"
] = range(
    1,
    len(pois) + 1,
)


# ============================================================
# SAVE
# ============================================================

pois.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\nTOP 20 POIs FOR THIS TOURIST"
)

print(
    "-----------------------------------------"
)


print(
    pois[
        [
            "preference_rank",
            "poi_id",
            "name",
            "matched_categories",
            "preference_score",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nSCORE DISTRIBUTION"
)

print(
    "-----------------------------------------"
)


print(
    pois[
        "preference_score"
    ]
    .value_counts()
    .sort_index(
        ascending=False
    )
    .to_string()
)


print(
    "\nSUMMARY"
)

print(
    "-----------------------------------------"
)

print(
    f"Total POIs      : "
    f"{len(pois)}"
)

print(
    f"Scored POIs     : "
    f"{(
        pois['preference_score']
        > 0
    ).sum()}"
)

print(
    f"Zero-score POIs : "
    f"{(
        pois['preference_score']
        == 0
    ).sum()}"
)


print(
    f"\nOutput saved:\n"
    f"{OUTPUT_FILE}"
)


print(
    "\nStatus: SUCCESS"
)
