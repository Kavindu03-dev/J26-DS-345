from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "Data" / "processed" / "tourist_pois.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "Data" / "processed" / "poi_interest_features.csv"
)


# ============================================================
# INTEREST CATEGORIES
# ============================================================

CATEGORY_KEYWORDS = {

    "history_culture": [
        "ancient", "history", "historical", "historic",
        "heritage", "archaeological", "archaeology",
        "museum", "fort", "fortress", "palace",
        "king", "queen", "royal", "memorial",
        "monument", "ruins", "colonial",
        "traditional", "tradition",
        "culture", "cultural",
        "art", "theatre", "carving",
        "indigenous", "walawwa",
        "ambalama", "canal"
    ],

    "nature_wildlife": [
        "waterfall", "falls", "ella",
        "lake", "river", "reservoir",
        "forest", "mountain", "hill",
        "garden", "park", "plains",
        "grassland", "swamp",
        "wildlife", "wild", "horse",
        "elephant", "bird",
        "sanctuary", "reserve",
        "nature", "natural",
        "cave", "rock",
        "spring", "springs",
        "water", "aquatic"
    ],

    "beach_coastal": [
        "beach", "bay", "coast", "coastal",
        "shore", "sea", "ocean",
        "lagoon", "island", "marine",
        "lighthouse", "light house",
        "point"
    ],

    "religious": [
        "temple", "monastery",
        "buddha", "buddhist",
        "devalaya", "kovil",
        "viharaya", "viharava",
        "church", "mosque",
        "shrine", "sacred",
        "pilgrim", "religious",
        "bodhi", "bodhiya",
        "stupa", "pagoda",
        "meditation"
    ],

    "adventure_recreation": [
        "hiking", "hike",
        "trekking", "trek",
        "surfing", "surf",
        "climbing", "climb",
        "diving", "snorkeling",
        "rafting", "cycling",
        "adventure", "camping",
        "recreation"
    ],

    "urban_leisure": [
        "shopping", "shop",
        "market", "racecourse",
        "event", "events",
        "relax", "chill",
        "jewellery", "jewelry",
        "gem", "gems",
        "city", "square"
    ]
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Remove punctuation/special characters
    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    # Remove repeated spaces
    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# KEYWORD MATCHING
# ============================================================

def keyword_found(text, keyword):
    """
    Check whether a complete keyword/phrase occurs in the text.
    Word boundaries help prevent partial-word matches.
    """

    pattern = (
        r"\b"
        + re.escape(keyword)
        + r"\b"
    )

    return bool(re.search(pattern, text))


def category_score(text, keywords):
    """
    Multi-label binary classification.

    Returns 1 if at least one keyword belonging to the
    category occurs in the POI name/description.
    Otherwise returns 0.
    """

    for keyword in keywords:
        if keyword_found(text, keyword):
            return 1

    return 0


def get_matched_categories(row, category_columns):
    """
    Produce a human-readable list of categories assigned
    to each attraction.
    """

    categories = [
        category
        for category in category_columns
        if row[category] == 1
    ]

    if not categories:
        return "unclassified"

    return "|".join(categories)


# ============================================================
# LOAD DATA
# ============================================================

pois = pd.read_csv(INPUT_FILE)

print("\nPreparing POI interest features...")
print("-----------------------------------------")
print(f"POIs loaded: {len(pois)}")


# ============================================================
# PREPARE CLASSIFICATION TEXT
# ============================================================

# Both attraction name and description are used because
# the name itself often contains important information:
# e.g. "Alankuda Beach", "Bambarakanda Falls", etc.

pois["classification_text"] = (
    pois["name"].fillna("").astype(str)
    + " "
    + pois["description"].fillna("").astype(str)
).apply(normalize_text)


# ============================================================
# MULTI-LABEL CLASSIFICATION
# ============================================================

for category, keywords in CATEGORY_KEYWORDS.items():

    pois[category] = pois[
        "classification_text"
    ].apply(
        lambda text: category_score(
            text,
            keywords
        )
    )


category_columns = list(
    CATEGORY_KEYWORDS.keys()
)


# Number of interest categories assigned to each attraction
pois["interest_category_count"] = (
    pois[category_columns].sum(axis=1)
)


# Human-readable category field
pois["matched_categories"] = pois.apply(
    lambda row: get_matched_categories(
        row,
        category_columns
    ),
    axis=1
)


# ============================================================
# VALIDATION
# ============================================================

unclassified = pois[
    pois["interest_category_count"] == 0
].copy()

multi_label = pois[
    pois["interest_category_count"] > 1
].copy()


print("\nCategory coverage")
print("-----------------------------------------")

for category in category_columns:

    count = int(
        pois[category].sum()
    )

    percentage = (
        count / len(pois)
    ) * 100

    print(
        f"{category:<24}: "
        f"{count:>3} "
        f"({percentage:>5.1f}%)"
    )


classified_count = (
    len(pois) - len(unclassified)
)

classified_percentage = (
    classified_count / len(pois)
) * 100


print("\nClassification summary")
print("-----------------------------------------")

print(
    f"Total POIs          : {len(pois)}"
)

print(
    f"Classified POIs     : "
    f"{classified_count} "
    f"({classified_percentage:.1f}%)"
)

print(
    f"Unclassified POIs   : "
    f"{len(unclassified)}"
)

print(
    f"Multi-label POIs    : "
    f"{len(multi_label)}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_columns = [
    "poi_id",
    "name",
    "district",
    "latitude",
    "longitude",
    "description",
] + category_columns + [
    "interest_category_count",
    "matched_categories"
]


pois[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nOutput saved:")
print(OUTPUT_FILE)


# ============================================================
# SHOW SAMPLE CLASSIFICATIONS
# ============================================================

print("\nSAMPLE CLASSIFICATIONS")
print("-----------------------------------------")

print(
    pois[
        [
            "poi_id",
            "name",
            "matched_categories"
        ]
    ].head(15).to_string(
        index=False
    )
)


# ============================================================
# SHOW UNCLASSIFIED POIS
# ============================================================

if len(unclassified) > 0:

    print("\nUNCLASSIFIED POIs")
    print("-----------------------------------------")

    print(
        unclassified[
            [
                "poi_id",
                "name",
                "description"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print("\nNo unclassified POIs.")


print("\nStatus: SUCCESS")
