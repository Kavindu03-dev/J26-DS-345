from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "reviews"
    / "restaurant_reviews_banana_chill_canonical.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "output"
    / "restaurant_review_aspects.csv"
)


# ============================================================
# ASPECT TAXONOMY
# ============================================================

ASPECT_KEYWORDS = {

    "food": [
        "food",
        "meal",
        "meals",
        "dish",
        "dishes",
        "taste",
        "tasty",
        "delicious",
        "flavour",
        "flavor",
        "curry",
        "rice",
        "kottu",
        "hoppers",
        "hopper",
        "noodles",
        "pizza",
        "burger",
        "seafood",
        "fish",
        "chicken",
        "meat",
        "dessert",
        "desserts",
        "drink",
        "drinks",
        "coffee",
        "tea",
        "menu",
        "portion",
        "portions",
        "breakfast",
        "lunch",
        "dinner",
    ],

    "service": [
        "service",
        "staff",
        "waiter",
        "waiters",
        "waitress",
        "waitresses",
        "server",
        "servers",
        "manager",
        "management",
        "customer service",
        "friendly",
        "helpful",
        "rude",
        "slow service",
        "quick service",
        "fast service",
    ],

    "ambience": [
        "ambience",
        "ambiance",
        "atmosphere",
        "environment",
        "decor",
        "decoration",
        "interior",
        "music",
        "view",
        "seating",
        "clean",
        "cleanliness",
        "hygiene",
        "comfortable",
        "cozy",
        "cosy",
        "crowded",
        "quiet",
        "noise",
        "noisy",
    ],

    "value": [
        "price",
        "prices",
        "pricing",
        "cost",
        "costly",
        "expensive",
        "cheap",
        "affordable",
        "value",
        "value for money",
        "worth",
        "overpriced",
        "reasonable price",
        "budget",
        "portion size",
    ],

    "location": [
        "location",
        "located",
        "area",
        "place",
        "nearby",
        "near",
        "distance",
        "parking",
        "access",
        "accessible",
        "road",
        "route",
        "view",
        "lake view",
        "sea view",
        "city centre",
        "city center",
        "town",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def split_into_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def find_aspects(sentence):

    sentence_lower = (
        sentence.lower()
    )

    detected = []


    for aspect_category, keywords in (
        ASPECT_KEYWORDS.items()
    ):

        matched_terms = []

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword.lower())
                + r"\b"
            )

            if re.search(
                pattern,
                sentence_lower
            ):

                matched_terms.append(
                    keyword
                )


        if matched_terms:

            detected.append(
                {
                    "aspect_category":
                        aspect_category,

                    "aspect_term":
                        matched_terms[0],

                    "all_matched_terms":
                        ";".join(
                            sorted(
                                set(
                                    matched_terms
                                )
                            )
                        ),
                }
            )


    return detected


# ============================================================
# LOAD REVIEW DATA
# ============================================================

print(
    "\nRESTAURANT ASPECT EXTRACTION"
)

print(
    "=" * 65
)


reviews = pd.read_csv(
    INPUT_FILE
)


print(
    f"Reviews loaded : "
    f"{len(reviews)}"
)


required_columns = [
    "review_id",
    "osm_id",
    "restaurant_name",
    "district",
    "source",
    "rating",
    "review_date",
    "review_title",
    "review_text",
    "language",
]


missing_columns = [
    column
    for column in required_columns
    if column not in reviews.columns
]


if missing_columns:

    raise RuntimeError(
        "Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# CLEAN REVIEW DATA
# ============================================================

reviews["review_text"] = (
    reviews["review_text"]
    .apply(
        clean_text
    )
)


reviews["review_title"] = (
    reviews["review_title"]
    .apply(
        clean_text
    )
)


reviews = reviews[
    reviews["review_text"] != ""
].copy()


reviews = (
    reviews
    .drop_duplicates(
        subset=[
            "review_id",
            "osm_id",
            "review_text",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    f"Usable reviews : "
    f"{len(reviews)}"
)


# ============================================================
# EXTRACT ASPECTS
# ============================================================

output_rows = []


for _, review in reviews.iterrows():

    review_id = (
        review["review_id"]
    )

    review_text = (
        review["review_text"]
    )


    sentences = (
        split_into_sentences(
            review_text
        )
    )


    for sentence_index, sentence in enumerate(
        sentences,
        start=1
    ):

        detected_aspects = (
            find_aspects(
                sentence
            )
        )


        for detected in detected_aspects:

            output_rows.append(
                {
                    "review_id":
                        review_id,

                    "osm_id":
                        review["osm_id"],

                    "restaurant_name":
                        review["restaurant_name"],

                    "district":
                        review["district"],

                    "source":
                        review["source"],

                    "rating":
                        review["rating"],

                    "review_date":
                        review["review_date"],

                    "language":
                        review["language"],

                    "sentence_index":
                        sentence_index,

                    "sentence":
                        sentence,

                    "aspect_category":
                        detected[
                            "aspect_category"
                        ],

                    "aspect_term":
                        detected[
                            "aspect_term"
                        ],

                    "all_matched_terms":
                        detected[
                            "all_matched_terms"
                        ],
                }
            )


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

aspect_df = pd.DataFrame(
    output_rows
)


# ============================================================
# HANDLE EMPTY RESULT
# ============================================================

expected_columns = [
    "review_id",
    "osm_id",
    "restaurant_name",
    "district",
    "source",
    "rating",
    "review_date",
    "language",
    "sentence_index",
    "sentence",
    "aspect_category",
    "aspect_term",
    "all_matched_terms",
]


if aspect_df.empty:

    aspect_df = pd.DataFrame(
        columns=expected_columns
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


aspect_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nASPECT EXTRACTION SUMMARY"
)

print(
    "-" * 65
)


print(
    f"Input reviews          : "
    f"{len(reviews)}"
)

print(
    f"Aspect records created : "
    f"{len(aspect_df)}"
)


if not aspect_df.empty:

    print(
        "\nAspect distribution:"
    )

    print(
        aspect_df[
            "aspect_category"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nUnique reviews with detected aspects:"
    )

    print(
        aspect_df[
            "review_id"
        ]
        .nunique()
    )


print(
    f"\nOutput file:\n"
    f"{OUTPUT_FILE}"
)


print(
    "\nStatus: SUCCESS"
)
