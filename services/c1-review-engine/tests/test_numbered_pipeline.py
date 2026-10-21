"""
Unit Tests for the Numbered Stage-by-Stage Pipeline Architecture.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
"""

import importlib.util
import inspect
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_stage_01_data_preparation_scripts_import_and_uniqueness():
    """Verifies stage 01 data preparation scripts import cleanly and contain 100% unique spam samples."""
    stage_dir = ROOT_DIR / "src" / "01_data_preparation"
    m1 = _load_module_from_path("s01_01", stage_dir / "01_build_ground_truth.py")
    m2 = _load_module_from_path("s01_02", stage_dir / "02_generate_hard_challenges.py")
    m3 = _load_module_from_path("s01_03", stage_dir / "03_audit_annotation_agreement.py")

    assert hasattr(m1, "GroundTruthBuilder")
    assert len(m1.AI_GENERATED_SPAM_CORPUS) == 40
    # Verify all 40 spam sentences are 100% unique (no duplicate 10*4 leakage)
    spam_texts = [t[0] for t in m1.AI_GENERATED_SPAM_CORPUS]
    assert len(set(spam_texts)) == 40

    assert hasattr(m2, "generate_benchmark_csv")
    assert hasattr(m3, "InterAnnotatorAgreementStudy")


def test_stage_02_trust_classification_scripts_import_cleanly():
    """Verifies stage 02 trust classification scripts import cleanly with valid classes and methods."""
    stage_dir = ROOT_DIR / "src" / "02_trust_classification"
    m1 = _load_module_from_path("s02_01", stage_dir / "01_calculate_asd_metrics.py")
    m2 = _load_module_from_path("s02_02", stage_dir / "02_train_trust_classifiers.py")
    m3 = _load_module_from_path("s02_03", stage_dir / "03_audit_feature_ablation.py")
    m4 = _load_module_from_path("s02_04", stage_dir / "04_evaluate_trust_engine.py")

    assert hasattr(m1, "demo_asd")
    assert hasattr(m2, "TrustClassifierTrainer")
    assert hasattr(m2.TrustClassifierTrainer, "train_and_export_models")
    assert hasattr(m3, "TrustClassifierAblationStudy")
    assert hasattr(m3.TrustClassifierAblationStudy, "run_ablation_suite")
    assert hasattr(m4, "demo_trust_scoring")


def test_stage_03_aspect_sentiment_scripts_no_ground_truth_cheating():
    """Verifies stage 03 ABSA scripts import cleanly and predict_deberta/predict_roberta never take ground_truth."""
    stage_dir = ROOT_DIR / "src" / "03_aspect_sentiment"
    m1 = _load_module_from_path("s03_01", stage_dir / "01_lexical_clause_engine.py")
    m2 = _load_module_from_path("s03_02", stage_dir / "02_train_deberta_transformer.py")
    m3 = _load_module_from_path("s03_03", stage_dir / "03_benchmark_absa_comparison.py")

    assert hasattr(m1, "demo_lexical_absa")
    assert hasattr(m2, "DeBERTaTrainer")
    assert hasattr(m3, "ABSAModelBenchmark")

    # Verify predict_deberta and predict_roberta signatures do NOT include ground_truth
    deb_sig = inspect.signature(m3.ABSAModelBenchmark.predict_deberta)
    rob_sig = inspect.signature(m3.ABSAModelBenchmark.predict_roberta)
    assert "ground_truth" not in deb_sig.parameters
    assert "ground_truth" not in rob_sig.parameters


def test_stage_04_aggregation_alerts_scripts_import_cleanly():
    """Verifies stage 04 aggregation & alerts scripts import cleanly."""
    stage_dir = ROOT_DIR / "src" / "04_aggregation_alerts"
    m1 = _load_module_from_path("s04_01", stage_dir / "01_extract_verified_facts.py")
    m2 = _load_module_from_path("s04_02", stage_dir / "02_aggregate_contract_scores.py")
    m3 = _load_module_from_path("s04_03", stage_dir / "03_detect_weak_aspect_alerts.py")
    m4 = _load_module_from_path("s04_04", stage_dir / "04_batch_precompute_profiles.py")

    assert hasattr(m1, "demo_fact_extraction")
    assert hasattr(m2, "demo_contract_aggregator")
    assert hasattr(m3, "demo_alerts")
    assert hasattr(m4, "BatchEnricher")
    assert hasattr(m4.BatchEnricher, "run")


def test_stage_05_scientific_evaluation_scripts_import_cleanly():
    """Verifies stage 05 scientific evaluation scripts import cleanly."""
    stage_dir = ROOT_DIR / "src" / "05_scientific_evaluation"
    m1 = _load_module_from_path("s05_01", stage_dir / "01_evaluate_halflife_sensitivity.py")
    m2 = _load_module_from_path("s05_02", stage_dir / "02_evaluate_temporal_backtest.py")
    m3 = _load_module_from_path("s05_03", stage_dir / "03_run_master_ablation_study.py")

    assert hasattr(m1, "RankingAblationStudy")
    assert hasattr(m2, "RankingAblationStudy")
    assert hasattr(m3, "RankingAblationStudy")
