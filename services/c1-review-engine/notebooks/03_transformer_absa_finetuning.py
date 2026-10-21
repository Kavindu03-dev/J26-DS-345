"""
Google Colab Pro GPU Fine-Tuning Pipeline for DeBERTa-v3 and RoBERTa ABSA.
Research Component 1: Trust-Aware Review Analysis Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

This script fine-tunes Transformer models on the 400-sentence Sri Lankan
hospitality ground-truth dataset for 3-class aspect sentiment polarity classification:
  - 0: Negative
  - 1: Neutral
  - 2: Positive

Designed to run in Google Colab Pro (GPU acceleration - NVIDIA T4 / A100)
as budgeted in Proposal Table 11 (Expense: LKR 14,000).

Usage in Colab / Local GPU:
  pip install transformers datasets accelerate torch evaluate
  python 03_transformer_absa_finetuning.py --model_name "microsoft/deberta-v3-small"
  python 03_transformer_absa_finetuning.py --model_name "roberta-base"
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


def train_transformer_absa(model_name: str = "microsoft/deberta-v3-small",
                           dataset_path: str = "data/processed/ground_truth_annotated_hospitality.csv",
                           hard_challenges_path: str = "data/processed/hard_linguistic_challenges.csv",
                           output_dir: str = "models/deberta_absa_weights",
                           epochs: int = 5,
                           batch_size: int = 16,
                           learning_rate: float = 2e-5):
    """Fine-tunes a pre-trained Transformer on hospitality aspect reviews (450 unified sentences)."""
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer
        )
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("[!] PyTorch and Hugging Face Transformers are required to train.")
        print("    Run: pip install torch transformers datasets accelerate scikit-learn")
        return

    print("=" * 80)
    print(f"   STARTING TRANSFORMER FINE-TUNING: {model_name}")
    print(f"   Device Available: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    # 1. Load ground truth benchmark and hard challenges
    records = []
    if os.path.exists(dataset_path):
        df_std = pd.read_csv(dataset_path)
        for _, row in df_std.iterrows():
            sent = str(row.get("sentiment_label", "")).strip()
            if sent in LABEL2ID:
                records.append({
                    "aspect": str(row["aspect_category"]).lower().strip(),
                    "sentence": str(row["review_sentence"]).strip(),
                    "sentiment": sent,
                    "label": LABEL2ID[sent]
                })

    if os.path.exists(hard_challenges_path):
        df_hard = pd.read_csv(hard_challenges_path)
        for _, row in df_hard.iterrows():
            sent = str(row.get("ground_truth_sentiment", "")).strip()
            if sent in LABEL2ID:
                records.append({
                    "aspect": str(row["target_aspect"]).lower().strip(),
                    "sentence": str(row["review_sentence"]).strip(),
                    "sentiment": sent,
                    "label": LABEL2ID[sent]
                })

    df = pd.DataFrame(records)
    # Combine aspect category with sentence for aspect-conditioned classification
    df["input_text"] = "Aspect: " + df["aspect"] + " | Sentence: " + df["sentence"]

    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
    print(f"[+] Total Unified Samples: {len(df)} (400 standard + 50 hard challenges)")
    print(f"[+] Dataset Split: {len(train_df)} Training sentences, {len(val_df)} Validation sentences.")

    # 2. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_data(texts):
        return tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    train_encodings = tokenize_data(train_df["input_text"])
    val_encodings = tokenize_data(val_df["input_text"])

    class ABSADataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ABSADataset(train_encodings, train_df["label"].values)
    val_dataset = ABSADataset(val_encodings, val_df["label"].values)

    # 3. Model Architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 4. Training Arguments
    import inspect
    sig = inspect.signature(TrainingArguments.__init__).parameters
    desired_args = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "learning_rate": learning_rate,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "logging_steps": 10,
        "report_to": "none"
    }
    if "eval_strategy" in sig:
        desired_args["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        desired_args["evaluation_strategy"] = "epoch"

    filtered_args = {k: v for k, v in desired_args.items() if k in sig}
    training_args = TrainingArguments(**filtered_args)


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print(f"\n[+] Fine-tuning {model_name} on hospitality ground-truth...")
    trainer.train()

    # 6. Save Model & Tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("\n" + "=" * 80)
    print(f"[OK] TRANSFORMER FINE-TUNING COMPLETE!")
    print(f"   Model Weights Saved To : {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Transformer on Hospitality ABSA")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-small", help="Pretrained model identifier")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="models/deberta_absa_weights", help="Save directory")
    args = parser.parse_args()

    train_transformer_absa(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
