"""
Unit Tests for the Numbered Stage-by-Stage Pipeline Architecture.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
"""

import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_numbered_pipeline_directories_exist():
    """Verifies that all 5 numbered pipeline stages exist in src/."""
    src_dir = ROOT_DIR / "src"
    expected_stages = [
        "01_data_preparation",
        "02_trust_classification",
        "03_aspect_sentiment",
        "04_aggregation_alerts",
        "05_scientific_evaluation"
    ]
    for stage in expected_stages:
        stage_dir = src_dir / stage
        assert stage_dir.exists(), f"Missing pipeline stage directory: {stage}"
        assert (stage_dir / "__init__.py").exists(), f"Missing __init__.py in: {stage}"


def test_stage_01_data_preparation_scripts_exist():
    """Verifies stage 01 data preparation scripts exist."""
    stage_dir = ROOT_DIR / "src" / "01_data_preparation"
    assert (stage_dir / "01_build_ground_truth.py").exists()
    assert (stage_dir / "02_generate_hard_challenges.py").exists()
    assert (stage_dir / "03_audit_annotation_agreement.py").exists()


def test_stage_02_trust_classification_scripts_exist():
    """Verifies stage 02 trust classification scripts exist."""
    stage_dir = ROOT_DIR / "src" / "02_trust_classification"
    assert (stage_dir / "01_calculate_asd_metrics.py").exists()
    assert (stage_dir / "02_train_trust_classifiers.py").exists()
    assert (stage_dir / "03_audit_feature_ablation.py").exists()
    assert (stage_dir / "04_evaluate_trust_engine.py").exists()


def test_stage_03_aspect_sentiment_scripts_exist():
    """Verifies stage 03 ABSA scripts exist."""
    stage_dir = ROOT_DIR / "src" / "03_aspect_sentiment"
    assert (stage_dir / "01_lexical_clause_engine.py").exists()
    assert (stage_dir / "02_train_deberta_transformer.py").exists()
    assert (stage_dir / "03_benchmark_absa_comparison.py").exists()


def test_stage_04_aggregation_alerts_scripts_exist():
    """Verifies stage 04 aggregation & alerts scripts exist."""
    stage_dir = ROOT_DIR / "src" / "04_aggregation_alerts"
    assert (stage_dir / "01_extract_verified_facts.py").exists()
    assert (stage_dir / "02_aggregate_contract_scores.py").exists()
    assert (stage_dir / "03_detect_weak_aspect_alerts.py").exists()
    assert (stage_dir / "04_batch_precompute_profiles.py").exists()


def test_stage_05_scientific_evaluation_scripts_exist():
    """Verifies stage 05 scientific evaluation scripts exist."""
    stage_dir = ROOT_DIR / "src" / "05_scientific_evaluation"
    assert (stage_dir / "01_evaluate_halflife_sensitivity.py").exists()
    assert (stage_dir / "02_evaluate_temporal_backtest.py").exists()
    assert (stage_dir / "03_run_master_ablation_study.py").exists()
