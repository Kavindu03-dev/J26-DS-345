from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "output"
    / "restaurant_review_aspects.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ABSA"
    / "output"
    / "restaurant_aspect_sentiments.csv"
)

MODEL_DIR = Path(
    os.getenv(
        "C3_ABSA_MODEL_DIR",
        str(PROJECT_ROOT / "models" / "restaurant_absa_sentiment_final"),
    )
)


# ============================================================
# SETTINGS
# ============================================================

MAX_LENGTH = 128

BATCH_SIZE = 16


# ============================================================
# START
# ============================================================

print(
    "\nRESTAURANT ABSA SENTIMENT INFERENCE"
)

print(
    "=" * 65
)


# ============================================================
# VALIDATE MODEL
# ============================================================

if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"Model directory not found:\n{MODEL_DIR}"
    )


print(
    f"Model:\n{MODEL_DIR}"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"\nDevice : {device}"
)


# ============================================================
# LOAD INPUT
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print(
    f"Aspect records loaded : "
    f"{len(df)}"
)


required_columns = [
    "review_id",
    "osm_id",
    "restaurant_name",
    "district",
    "sentence",
    "aspect_category",
    "aspect_term",
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


if df.empty:

    print(
        "\nNo aspect records available."
    )

    raise SystemExit(0)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print(
    "\nLoading trained tokenizer..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_DIR
    )
)


print(
    "Loading trained ABSA model..."
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR
    )
)


model.to(
    device
)

model.eval()


# ============================================================
# LABEL MAPPING
# ============================================================

id_to_label = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# STORAGE
# ============================================================

predicted_labels = []

confidence_scores = []

negative_probabilities = []

neutral_probabilities = []

positive_probabilities = []

sentiment_scores = []


# ============================================================
# RUN BATCH INFERENCE
# ============================================================

print(
    "\nRunning aspect sentiment inference..."
)


for start in range(
    0,
    len(df),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(df)
    )


    batch = df.iloc[
        start:end
    ]


    sentences = (
        batch["sentence"]
        .astype(str)
        .tolist()
    )


    aspect_terms = (
        batch["aspect_term"]
        .astype(str)
        .tolist()
    )


    # Same input structure used during training:
    #
    # sequence 1 = review sentence
    # sequence 2 = aspect term

    encoded = tokenizer(
        sentences,
        aspect_terms,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


    encoded = {
        key: value.to(device)
        for key, value
        in encoded.items()
    }


    with torch.no_grad():

        outputs = model(
            **encoded
        )


    probabilities = (
        torch.softmax(
            outputs.logits,
            dim=-1
        )
        .cpu()
        .numpy()
    )


    predictions = np.argmax(
        probabilities,
        axis=1
    )


    for prediction, probs in zip(
        predictions,
        probabilities
    ):

        label = (
            id_to_label[
                int(prediction)
            ]
        )


        confidence = float(
            probs[
                prediction
            ]
        )


        negative_prob = float(
            probs[0]
        )

        neutral_prob = float(
            probs[1]
        )

        positive_prob = float(
            probs[2]
        )


        # Continuous sentiment score:
        #
        # -1 = strongly negative
        #  0 = neutral
        # +1 = strongly positive

        sentiment_score = (
            positive_prob
            - negative_prob
        )


        predicted_labels.append(
            label
        )

        confidence_scores.append(
            confidence
        )

        negative_probabilities.append(
            negative_prob
        )

        neutral_probabilities.append(
            neutral_prob
        )

        positive_probabilities.append(
            positive_prob
        )

        sentiment_scores.append(
            sentiment_score
        )


# ============================================================
# ADD PREDICTIONS
# ============================================================

df["predicted_sentiment"] = (
    predicted_labels
)

df["sentiment_confidence"] = (
    confidence_scores
)

df["negative_probability"] = (
    negative_probabilities
)

df["neutral_probability"] = (
    neutral_probabilities
)

df["positive_probability"] = (
    positive_probabilities
)

df["sentiment_score"] = (
    sentiment_scores
)


# ============================================================
# ROUND NUMERIC OUTPUTS
# ============================================================

numeric_columns = [
    "sentiment_confidence",
    "negative_probability",
    "neutral_probability",
    "positive_probability",
    "sentiment_score",
]


for column in numeric_columns:

    df[column] = (
        df[column]
        .round(4)
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nABSA INFERENCE SUMMARY"
)

print(
    "-" * 65
)


print(
    f"Aspect records processed : "
    f"{len(df)}"
)


print(
    "\nPredicted sentiment distribution:"
)


print(
    df[
        "predicted_sentiment"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nAspect × sentiment:"
)


summary = pd.crosstab(
    df["aspect_category"],
    df["predicted_sentiment"]
)


print(
    summary.to_string()
)


print(
    "\nAverage sentiment score by aspect:"
)


aspect_scores = (
    df.groupby(
        "aspect_category"
    )[
        "sentiment_score"
    ]
    .mean()
    .round(4)
)


print(
    aspect_scores.to_string()
)


print(
    f"\nOutput file:\n"
    f"{OUTPUT_FILE}"
)


print(
    "\nStatus: SUCCESS"
)


