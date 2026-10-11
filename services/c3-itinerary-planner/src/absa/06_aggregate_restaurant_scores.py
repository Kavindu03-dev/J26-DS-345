from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "output"
    / "restaurant_aspect_sentiments.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "output"
    / "restaurant_aspect_scores.csv"
)


# ============================================================
# SETTINGS
# ============================================================

ASPECTS = [
    "food",
    "service",
    "ambience",
    "value",
    "location",
]


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\nRESTAURANT-LEVEL ASPECT SCORE AGGREGATION"
)

print(
    "=" * 70
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"Aspect sentiment records loaded : "
    f"{len(df)}"
)


if df.empty:

    raise RuntimeError(
        "No aspect sentiment records found."
    )


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "osm_id",
    "restaurant_name",
    "district",
    "review_id",
    "aspect_category",
    "sentiment_score",
    "sentiment_confidence",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN
# ============================================================

df["sentiment_score"] = pd.to_numeric(
    df["sentiment_score"],
    errors="coerce"
)

df["sentiment_confidence"] = pd.to_numeric(
    df["sentiment_confidence"],
    errors="coerce"
)


df = df[
    df["sentiment_score"].notna()
].copy()


# ============================================================
# CONVERT SENTIMENT FROM [-1, +1] TO [0, 1]
# ============================================================

# -1.0 -> 0.0
#  0.0 -> 0.5
# +1.0 -> 1.0

df["quality_score"] = (
    df["sentiment_score"] + 1.0
) / 2.0


df["quality_score"] = (
    df["quality_score"]
    .clip(
        lower=0.0,
        upper=1.0
    )
)


# ============================================================
# AGGREGATE EACH RESTAURANT + ASPECT
# ============================================================

aspect_summary = (
    df.groupby(
        [
            "osm_id",
            "restaurant_name",
            "district",
            "aspect_category",
        ],
        dropna=False
    )
    .agg(
        aspect_score=(
            "quality_score",
            "mean"
        ),

        average_confidence=(
            "sentiment_confidence",
            "mean"
        ),

        aspect_mentions=(
            "aspect_category",
            "size"
        ),

        review_count=(
            "review_id",
            "nunique"
        ),
    )
    .reset_index()
)


# ============================================================
# CREATE WIDE RESTAURANT TABLE
# ============================================================

restaurant_base = (
    df[
        [
            "osm_id",
            "restaurant_name",
            "district"
        ]
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)


# Aspect scores
score_pivot = (
    aspect_summary
    .pivot(
        index=[
            "osm_id",
            "restaurant_name",
            "district"
        ],
        columns="aspect_category",
        values="aspect_score"
    )
    .reset_index()
)


# Aspect evidence counts
count_pivot = (
    aspect_summary
    .pivot(
        index=[
            "osm_id",
            "restaurant_name",
            "district"
        ],
        columns="aspect_category",
        values="review_count"
    )
    .reset_index()
)


# Rename count columns
count_rename = {
    aspect: f"{aspect}_review_count"
    for aspect in ASPECTS
}

count_pivot = (
    count_pivot.rename(
        columns=count_rename
    )
)


# ============================================================
# ENSURE ALL FIVE ASPECT COLUMNS EXIST
# ============================================================

for aspect in ASPECTS:

    if aspect not in score_pivot.columns:

        score_pivot[aspect] = np.nan


    count_column = (
        f"{aspect}_review_count"
    )

    if count_column not in count_pivot.columns:

        count_pivot[
            count_column
        ] = 0


# Rename score columns
score_pivot = (
    score_pivot.rename(
        columns={
            aspect:
                f"{aspect}_score"
            for aspect in ASPECTS
        }
    )
)


# ============================================================
# MERGE
# ============================================================

scores = score_pivot.merge(
    count_pivot,
    on=[
        "osm_id",
        "restaurant_name",
        "district"
    ],
    how="left"
)


# ============================================================
# TOTAL REVIEW EVIDENCE
# ============================================================

restaurant_review_counts = (
    df.groupby(
        "osm_id"
    )[
        "review_id"
    ]
    .nunique()
    .rename(
        "total_review_count"
    )
    .reset_index()
)


scores = scores.merge(
    restaurant_review_counts,
    on="osm_id",
    how="left"
)


# ============================================================
# AVAILABLE ASPECT COUNT
# ============================================================

score_columns = [
    f"{aspect}_score"
    for aspect in ASPECTS
]


scores["available_aspect_count"] = (
    scores[
        score_columns
    ]
    .notna()
    .sum(
        axis=1
    )
)


# ============================================================
# UNWEIGHTED ABSA QUALITY SCORE
# ============================================================

# Used only as a generic restaurant-quality signal.
#
# Later the tourist's preferences will dynamically weight
# Food / Service / Ambience / Value / Location differently.

scores["absa_quality_score"] = (
    scores[
        score_columns
    ]
    .mean(
        axis=1,
        skipna=True
    )
)


# ============================================================
# ROUND
# ============================================================

numeric_score_columns = (
    score_columns
    + [
        "absa_quality_score"
    ]
)


for column in numeric_score_columns:

    scores[column] = (
        scores[column]
        .round(4)
    )


# ============================================================
# ORDER COLUMNS
# ============================================================

final_columns = [
    "osm_id",
    "restaurant_name",
    "district",

    "food_score",
    "service_score",
    "ambience_score",
    "value_score",
    "location_score",

    "food_review_count",
    "service_review_count",
    "ambience_review_count",
    "value_review_count",
    "location_review_count",

    "total_review_count",
    "available_aspect_count",
    "absa_quality_score",
]


scores = scores[
    final_columns
]


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


scores.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nRESTAURANT ASPECT SCORE SUMMARY"
)

print(
    "-" * 70
)


print(
    f"Restaurants scored : "
    f"{len(scores)}"
)


print(
    "\nRestaurant scores:"
)


print(
    scores.to_string(
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
