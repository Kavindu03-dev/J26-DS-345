from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# INPUT DATA
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# MODEL STORAGE
#
# IMPORTANT:
# Large model/checkpoint files are stored on D:
# to avoid filling the C: drive.
# ------------------------------------------------------------

MODEL_DIR = Path(
    os.getenv(
        "C3_ABSA_MODEL_DIR",
        str(PROJECT_ROOT / "models" / "restaurant_absa_sentiment_final"),
    )
)


# Existing successfully saved epoch-2 checkpoint
RESUME_CHECKPOINT = Path(
    os.getenv(
        "C3_ABSA_RESUME_CHECKPOINT",
        str(PROJECT_ROOT / "models" / "checkpoint-720"),
    )
)


# ------------------------------------------------------------
# SMALL EVALUATION OUTPUTS
#
# These remain inside the main project because they are small
# and useful for the report / evaluation / viva.
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ABSA"
    / "output"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "sentiment_model_metrics.json"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "sentiment_classification_report.txt"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "sentiment_confusion_matrix.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 16

NUM_EPOCHS = 3

LEARNING_RATE = 2e-5

RANDOM_SEED = 42


# ============================================================
# LABEL DEFINITIONS
# ============================================================

label_to_id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

id_to_label = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# STARTUP INFORMATION
# ============================================================

print(
    "\nBERT-BASED RESTAURANT ABSA TRAINING"
)

print(
    "=" * 70
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"PyTorch device : {device}"
)


if device == "cpu":

    print(
        "CPU training detected. "
        "Training will be slower than GPU training."
    )


print(
    f"\nFinal model directory:\n"
    f"{MODEL_DIR}"
)


print(
    f"\nResume checkpoint:\n"
    f"{RESUME_CHECKPOINT}"
)


# ============================================================
# VALIDATE CHECKPOINT
# ============================================================

if not RESUME_CHECKPOINT.exists():

    raise FileNotFoundError(
        "Resume checkpoint does not exist:\n"
        f"{RESUME_CHECKPOINT}"
    )


checkpoint_model_file = (
    RESUME_CHECKPOINT
    / "model.safetensors"
)


checkpoint_state_file = (
    RESUME_CHECKPOINT
    / "trainer_state.json"
)


if not checkpoint_model_file.exists():

    raise FileNotFoundError(
        "Checkpoint model file missing:\n"
        f"{checkpoint_model_file}"
    )


if not checkpoint_state_file.exists():

    raise FileNotFoundError(
        "Checkpoint trainer state missing:\n"
        f"{checkpoint_state_file}"
    )


print(
    "\nCheckpoint validation: OK"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"\nTrain rows      : "
    f"{len(train_df)}"
)

print(
    f"Validation rows : "
    f"{len(validation_df)}"
)

print(
    f"Test rows       : "
    f"{len(test_df)}"
)


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "text",
    "span",
    "label",
    "label_id",
]


for name, df in [
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df),
]:

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing:

        raise RuntimeError(
            f"{name} dataset is missing columns: "
            f"{missing}"
        )


print(
    "\nDataset validation: OK"
)


# ============================================================
# PREPARE HUGGING FACE DATASETS
# ============================================================

def dataframe_to_dataset(df):

    prepared = df[
        [
            "text",
            "span",
            "label_id",
        ]
    ].copy()


    prepared = prepared.rename(
        columns={
            "label_id": "labels"
        }
    )


    prepared["labels"] = (
        prepared["labels"]
        .astype(int)
    )


    return Dataset.from_pandas(
        prepared,
        preserve_index=False
    )


train_dataset = dataframe_to_dataset(
    train_df
)

validation_dataset = dataframe_to_dataset(
    validation_df
)

test_dataset = dataframe_to_dataset(
    test_df
)


# ============================================================
# TOKENIZER
# ============================================================

print(
    f"\nLoading tokenizer: "
    f"{MODEL_NAME}"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_batch(batch):
    """
    Aspect-Based Sentiment Analysis input:

        Sequence 1 = review sentence
        Sequence 2 = aspect term

    Example:

        Review:
        "The food was excellent but the service was slow."

        Aspect:
        "service"

        Expected sentiment:
        negative

    The model therefore predicts sentiment toward a
    particular aspect rather than overall review sentiment.
    """

    return tokenizer(
        batch["text"],
        batch["span"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


print(
    "Tokenizing datasets..."
)


train_dataset = train_dataset.map(
    tokenize_batch,
    batched=True
)


validation_dataset = (
    validation_dataset.map(
        tokenize_batch,
        batched=True
    )
)


test_dataset = test_dataset.map(
    tokenize_batch,
    batched=True
)


# ============================================================
# REMOVE ORIGINAL TEXT FIELDS FROM MODEL INPUT
# ============================================================

columns_to_remove = [
    "text",
    "span",
]


train_dataset = (
    train_dataset.remove_columns(
        columns_to_remove
    )
)


validation_dataset = (
    validation_dataset.remove_columns(
        columns_to_remove
    )
)


test_dataset = (
    test_dataset.remove_columns(
        columns_to_remove
    )
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print(
    f"\nLoading model architecture: "
    f"{MODEL_NAME}"
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id_to_label,
        label2id=label_to_id,
    )
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_prediction
):

    logits = (
        eval_prediction.predictions
    )

    labels = (
        eval_prediction.label_ids
    )


    predictions = np.argmax(
        logits,
        axis=-1
    )


    accuracy = accuracy_score(
        labels,
        predictions
    )


    (
        precision_macro,
        recall_macro,
        f1_macro,
        _
    ) = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )


    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _
    ) = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )


    return {

        "accuracy":
            accuracy,

        "precision_macro":
            precision_macro,

        "recall_macro":
            recall_macro,

        "f1_macro":
            f1_macro,

        "precision_weighted":
            precision_weighted,

        "recall_weighted":
            recall_weighted,

        "f1_weighted":
            f1_weighted,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    # All large checkpoints now go to D:
    output_dir=str(
        MODEL_DIR
        / "checkpoints"
    ),


    learning_rate=(
        LEARNING_RATE
    ),


    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),


    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),


    num_train_epochs=(
        NUM_EPOCHS
    ),


    weight_decay=0.01,


    # Evaluate once after each epoch
    eval_strategy="epoch",


    # Save once after each epoch
    save_strategy="epoch",


    logging_strategy="steps",


    logging_steps=50,


    # At the end, automatically restore
    # the best validation model
    load_best_model_at_end=True,


    metric_for_best_model=(
        "f1_macro"
    ),


    greater_is_better=True,


    # Keep only the most useful recent checkpoints
    # instead of accumulating many large checkpoints.
    save_total_limit=2,


    report_to="none",


    seed=RANDOM_SEED,


    # Prevent CPU pin-memory warning.
    dataloader_pin_memory=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=(
        train_dataset
    ),

    eval_dataset=(
        validation_dataset
    ),

    processing_class=(
        tokenizer
    ),

    compute_metrics=(
        compute_metrics
    ),
)


# ============================================================
# RESUME TRAINING
# ============================================================

print(
    "\nResuming model training..."
)


print(
    "Existing completed checkpoint:"
)

print(
    RESUME_CHECKPOINT
)


print(
    "\nTraining should resume from approximately "
    "epoch 2 instead of starting from epoch 0."
)


train_result = trainer.train(
    resume_from_checkpoint=str(
        RESUME_CHECKPOINT
    )
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print(
    "\nVALIDATION RESULTS"
)

print(
    "-" * 70
)


validation_metrics = (
    trainer.evaluate(
        validation_dataset
    )
)


for key, value in (
    validation_metrics.items()
):

    print(
        f"{key}: "
        f"{value}"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print(
    "\nFINAL TEST EVALUATION"
)

print(
    "-" * 70
)


test_output = trainer.predict(
    test_dataset
)


test_logits = (
    test_output.predictions
)


test_true = (
    test_output.label_ids
)


test_pred = np.argmax(
    test_logits,
    axis=-1
)


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_true,
    test_pred
)


(
    test_precision_macro,
    test_recall_macro,
    test_f1_macro,
    _
) = precision_recall_fscore_support(

    test_true,

    test_pred,

    average="macro",

    zero_division=0,
)


(
    test_precision_weighted,
    test_recall_weighted,
    test_f1_weighted,
    _
) = precision_recall_fscore_support(

    test_true,

    test_pred,

    average="weighted",

    zero_division=0,
)


print(
    f"Accuracy           : "
    f"{test_accuracy:.4f}"
)

print(
    f"Macro Precision    : "
    f"{test_precision_macro:.4f}"
)

print(
    f"Macro Recall       : "
    f"{test_recall_macro:.4f}"
)

print(
    f"Macro F1           : "
    f"{test_f1_macro:.4f}"
)

print(
    f"Weighted F1        : "
    f"{test_f1_weighted:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    test_true,

    test_pred,

    labels=[
        0,
        1,
        2
    ],

    target_names=[
        "negative",
        "neutral",
        "positive",
    ],

    digits=4,

    zero_division=0,
)


print(
    "\nCLASSIFICATION REPORT"
)

print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(

    test_true,

    test_pred,

    labels=[
        0,
        1,
        2
    ]
)


matrix_df = pd.DataFrame(

    matrix,

    index=[
        "actual_negative",
        "actual_neutral",
        "actual_positive",
    ],

    columns=[
        "pred_negative",
        "pred_neutral",
        "pred_positive",
    ]
)


print(
    "\nCONFUSION MATRIX"
)


print(
    matrix_df.to_string()
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print(
    "\nSaving final trained model..."
)


trainer.save_model(
    MODEL_DIR
)


tokenizer.save_pretrained(
    MODEL_DIR
)


print(
    "Final model saved successfully."
)


# ============================================================
# SAVE EVALUATION OUTPUTS
# ============================================================

metrics = {

    "base_model":
        MODEL_NAME,

    "final_model_directory":
        str(MODEL_DIR),

    "resume_checkpoint":
        str(RESUME_CHECKPOINT),

    "device":
        device,

    "train_examples":
        len(train_df),

    "validation_examples":
        len(validation_df),

    "test_examples":
        len(test_df),

    "labels":
        label_to_id,

    "max_length":
        MAX_LENGTH,

    "epochs":
        NUM_EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "test_accuracy":
        float(
            test_accuracy
        ),

    "test_precision_macro":
        float(
            test_precision_macro
        ),

    "test_recall_macro":
        float(
            test_recall_macro
        ),

    "test_f1_macro":
        float(
            test_f1_macro
        ),

    "test_precision_weighted":
        float(
            test_precision_weighted
        ),

    "test_recall_weighted":
        float(
            test_recall_weighted
        ),

    "test_f1_weighted":
        float(
            test_f1_weighted
        ),
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report
    )


matrix_df.to_csv(
    CONFUSION_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)


print(
    "FINAL ABSA TRAINING SUMMARY"
)


print(
    "=" * 70
)


print(
    "\nMODEL SAVED TO:"
)

print(
    MODEL_DIR
)


print(
    "\nEVALUATION OUTPUTS:"
)

print(
    METRICS_FILE
)

print(
    REPORT_FILE
)

print(
    CONFUSION_FILE
)


print(
    "\nTest Accuracy : "
    f"{test_accuracy:.4f}"
)


print(
    "Test Macro F1 : "
    f"{test_f1_macro:.4f}"
)


print(
    "Weighted F1   : "
    f"{test_f1_weighted:.4f}"
)


print(
    "\nStatus: SUCCESS"
)



