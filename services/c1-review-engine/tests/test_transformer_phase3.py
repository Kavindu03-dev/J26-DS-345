"""
Unit and Contract Tests for Phase 3: DeBERTa Training Pipeline & Scale Harmonization.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR
from src.train_deberta_absa import DeBERTaTrainer


def test_deberta_trainer_dataset_unification():
    """Verifies that the DeBERTa training pipeline unifies the 400 standard + 50 hard samples into 450 rows."""
    trainer = DeBERTaTrainer()
    df = trainer.load_and_prepare_dataset()

    assert len(df) == 450
    assert "formatted_input" in df.columns
    assert "label" in df.columns

    # Verify input format: "Aspect: {aspect} | Review: {sentence}"
    first_input = df.iloc[0]["formatted_input"]
    assert first_input.startswith("Aspect: ")
    assert " | Review: " in first_input

    # Verify label distribution (Negative=0, Neutral=1, Positive=2)
    labels = set(df["label"].unique())
    assert labels == {0, 1, 2}


def test_deberta_absa_config_manifest():
    """Verifies that the exported DeBERTa model manifest is valid and conforms to expected production schema."""
    config_path = MODELS_DIR / "deberta_absa_config.json"
    if not config_path.exists():
        trainer = DeBERTaTrainer()
        df = trainer.load_and_prepare_dataset()
        trainer.export_training_manifest(df)
    assert config_path.exists(), f"Missing config manifest at: {config_path}"


    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "model_metadata" in data
    assert "hyperparameters" in data
    assert "empirical_evaluation_metrics" in data

    meta = data["model_metadata"]
    assert meta["num_classes"] == 3
    assert meta["base_architecture"] == "microsoft/deberta-v3-small"
    assert meta["total_samples"] == 450

    metrics = data["empirical_evaluation_metrics"]
    assert metrics["overall_accuracy_pct"] >= 90.0
    assert metrics["macro_f1_pct"] >= 90.0


def test_hotel_aspect_and_rating_scales_bounded():
    """Verifies that hotel aspect scores and ratings in the database strictly adhere to 0.0 - 5.0 star bounds."""
    profiles_json = DATA_DIR / "hotel_aspect_profiles.json"
    if not profiles_json.exists():
        pytest.skip("Hotel aspect profiles JSON not generated yet.")

    with open(profiles_json, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    # Inspect first 20 hotels
    sample_hotels = list(profiles.items())[:20]
    for hotel_id, p in sample_hotels:
        # Check time decay rating (1.0 to 5.0)
        td_rating = p.get("time_decay_rating", 3.0)
        assert 1.0 <= td_rating <= 5.0, f"Hotel {hotel_id} time_decay_rating out of bounds: {td_rating}"

        # Check aspect scores (0.0 to 5.0)
        aspects = p.get("aspect_scores", {})
        for asp_name, score in aspects.items():
            assert 0.0 <= score <= 5.0, f"Hotel {hotel_id} aspect {asp_name} score out of bounds: {score}"
