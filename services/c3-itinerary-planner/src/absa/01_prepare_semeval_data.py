from pathlib import Path
import pandas as pd
from datasets import load_dataset


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ABSA_DATA_DIR = (
    PROJECT_ROOT
    / "ABSA"
    / "data"
)

TRAIN_OUTPUT = (
    ABSA_DATA_DIR
    / "semeval_restaurants_train.csv"
)

TEST_OUTPUT = (
    ABSA_DATA_DIR
    / "semeval_restaurants_test.csv"
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_NAME = (
    "tomaarsen/"
    "setfit-absa-semeval-restaurants"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nSEMEVAL RESTAURANT ABSA DATA PREPARATION")
print("=" * 65)

print(
    f"Loading dataset:\n"
    f"{DATASET_NAME}"
)


dataset = load_dataset(
    DATASET_NAME
)


print(
    "\nAvailable splits:"
)

print(
    dataset
)


# ============================================================
# CONVERT TO DATAFRAMES
# ============================================================

train_df = (
    dataset["train"]
    .to_pandas()
)

test_df = (
    dataset["test"]
    .to_pandas()
)


print(
    f"\nTraining rows : "
    f"{len(train_df)}"
)

print(
    f"Test rows     : "
    f"{len(test_df)}"
)


# ============================================================
# INSPECT COLUMNS
# ============================================================

print(
    "\nTraining columns:"
)

print(
    train_df.columns.tolist()
)


print(
    "\nFirst 5 raw rows:"
)

print(
    train_df.head().to_string(
        index=False
    )
)


# ============================================================
# BASIC CLEANING
# ============================================================

required_columns = [
    "text",
    "span",
    "label",
    "ordinal",
]


missing_columns = [
    column
    for column in required_columns
    if column not in train_df.columns
]


if missing_columns:

    raise RuntimeError(
        "Expected dataset columns missing: "
        f"{missing_columns}"
    )


def clean_dataframe(df):

    df = df.copy()

    # Convert text-like fields to strings.
    for column in [
        "text",
        "span",
        "label",
    ]:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )


    # Remove rows without review text.
    df = df[
        df["text"].notna()
    ].copy()


    df = df[
        df["text"] != ""
    ].copy()


    # Remove exact duplicate annotations.
    df = (
        df
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    return df


train_clean = clean_dataframe(
    train_df
)

test_clean = clean_dataframe(
    test_df
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\nCLEANING SUMMARY"
)

print(
    "-" * 65
)

print(
    f"Train rows before : "
    f"{len(train_df)}"
)

print(
    f"Train rows after  : "
    f"{len(train_clean)}"
)

print(
    f"Test rows before  : "
    f"{len(test_df)}"
)

print(
    f"Test rows after   : "
    f"{len(test_clean)}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\nSentiment label distribution - TRAIN"
)

print(
    train_clean[
        "label"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


print(
    "\nSentiment label distribution - TEST"
)

print(
    test_clean[
        "label"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# ASPECT TERM SUMMARY
# ============================================================

print(
    "\nMost common aspect spans - TRAIN"
)

print(
    train_clean[
        "span"
    ]
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

ABSA_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


train_clean.to_csv(
    TRAIN_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


test_clean.to_csv(
    TEST_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\nOUTPUT FILES"
)

print(
    "-" * 65
)

print(
    TRAIN_OUTPUT
)

print(
    TEST_OUTPUT
)


print(
    "\nStatus: SUCCESS"
)
