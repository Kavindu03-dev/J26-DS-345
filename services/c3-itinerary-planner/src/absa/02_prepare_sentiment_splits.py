from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "data"
    / "semeval_restaurants_train.csv"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "data"
    / "sentiment_train.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "data"
    / "sentiment_validation.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "data"
    / "sentiment_test.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

VALID_LABELS = [
    "positive",
    "neutral",
    "negative",
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print(
    "\nABSA SENTIMENT DATASET SPLIT"
)

print(
    "=" * 65
)

print(
    f"Input rows : {len(df)}"
)


# ============================================================
# CLEAN LABELS
# ============================================================

df["label"] = (
    df["label"]
    .astype("string")
    .str.strip()
    .str.lower()
)


df["text"] = (
    df["text"]
    .astype("string")
    .str.strip()
)


df["span"] = (
    df["span"]
    .astype("string")
    .str.strip()
)


# Remove rows without required information.
df = df[
    df["text"].notna()
    & df["span"].notna()
    & df["label"].notna()
].copy()


# ============================================================
# REMOVE CONFLICT CLASS
# ============================================================

conflict_count = (
    df["label"]
    .eq("conflict")
    .sum()
)


print(
    f"Conflict rows removed : "
    f"{conflict_count}"
)


df = df[
    df["label"].isin(
        VALID_LABELS
    )
].copy()


df = df.reset_index(
    drop=True
)


print(
    f"Usable labelled rows  : "
    f"{len(df)}"
)


# ============================================================
# LABEL MAPPING
# ============================================================

label_to_id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


df["label_id"] = (
    df["label"]
    .map(
        label_to_id
    )
)


# ============================================================
# SPLIT 80 / 10 / 10
# ============================================================

# First:
# 80% training
# 20% temporary validation/test pool

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)


# Split temporary 20% equally:
# 10% validation
# 10% test

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)


# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(
    drop=True
)

validation_df = (
    validation_df
    .reset_index(
        drop=True
    )
)

test_df = test_df.reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False,
    encoding="utf-8-sig"
)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    TEST_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def show_split_summary(
    name,
    split_df
):

    print(
        f"\n{name}"
    )

    print(
        "-" * 40
    )

    print(
        f"Rows: {len(split_df)}"
    )

    print(
        split_df[
            "label"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nPercentages:"
    )

    percentages = (
        split_df[
            "label"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print(
        percentages.to_string()
    )


# ============================================================
# REPORT
# ============================================================

show_split_summary(
    "TRAIN",
    train_df
)

show_split_summary(
    "VALIDATION",
    validation_df
)

show_split_summary(
    "TEST",
    test_df
)


print(
    "\nFINAL SPLIT SUMMARY"
)

print(
    "-" * 65
)

print(
    f"Train      : "
    f"{len(train_df)}"
)

print(
    f"Validation : "
    f"{len(validation_df)}"
)

print(
    f"Test       : "
    f"{len(test_df)}"
)

print(
    f"Total      : "
    f"{len(train_df) + len(validation_df) + len(test_df)}"
)


print(
    "\nOutput files:"
)

print(
    TRAIN_FILE
)

print(
    VALIDATION_FILE
)

print(
    TEST_FILE
)


print(
    "\nStatus: SUCCESS"
)
