"""
Unit and Contract Tests for Phase 2: NLP Rigor, Hard Linguistic Challenges & Annotation Agreement.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
"""

import pytest
import pandas as pd
from pathlib import Path

from src.config import PROCESSED_DATA_DIR
from src.annotation_agreement import InterAnnotatorAgreementStudy
from src.benchmark_absa_comparison import ABSAModelBenchmark


@pytest.fixture
def hard_challenges_df():
    path = PROCESSED_DATA_DIR / "hard_linguistic_challenges.csv"
    assert path.exists(), f"Missing hard challenges file at: {path}"
    return pd.read_csv(path)


@pytest.fixture
def ground_truth_df():
    path = PROCESSED_DATA_DIR / "ground_truth_annotated_hospitality.csv"
    assert path.exists(), f"Missing ground truth file at: {path}"
    return pd.read_csv(path)


def test_hard_challenges_dataset_schema_and_distribution(hard_challenges_df):
    """Verifies that the 50 hard linguistic challenges cover all 4 adversarial categories."""
    assert len(hard_challenges_df) == 50
    expected_cols = {
        "challenge_id",
        "phenomenon_type",
        "review_sentence",
        "target_aspect",
        "ground_truth_sentiment",
        "linguistic_failure_mode",
    }
    assert expected_cols.issubset(set(hard_challenges_df.columns))

    phenomena = set(hard_challenges_df["phenomenon_type"].unique())
    expected_phenomena = {
        "Implicit Sentiment",
        "Sarcasm & Irony",
        "Double Negation",
        "Sri Lankan Vernacular",
    }
    assert phenomena == expected_phenomena

    # Verify no nulls
    assert hard_challenges_df["review_sentence"].isnull().sum() == 0
    assert hard_challenges_df["ground_truth_sentiment"].isnull().sum() == 0


def test_annotation_agreement_metrics_contract():
    """Verifies that the inter-annotator agreement study executes and returns valid Cohen's Kappa."""
    study = InterAnnotatorAgreementStudy()
    results = study.compute_agreement_metrics()

    assert "sentiment_agreement" in results
    assert "aspect_agreement" in results

    sent_kappa = results["sentiment_agreement"]["cohens_kappa"]
    asp_kappa = results["aspect_agreement"]["cohens_kappa"]

    # Cohen's Kappa should be in realistic substantial-to-almost-perfect range
    assert 0.60 <= sent_kappa <= 1.00
    assert 0.70 <= asp_kappa <= 1.00

    # Ensure confusion matrix is 3x3 for sentiment
    cm = results["sentiment_agreement"]["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


def test_absa_benchmark_lexicon_inversion_proof():
    """
    Verifies that the ABSA benchmarking engine captures the empirical performance inversion:
    Lexicon scores high on standard keyword-matched sentences, but collapses on hard linguistic challenges.
    """
    benchmark = ABSAModelBenchmark()
    results = benchmark.run_full_benchmark()

    assert "standard_benchmark" in results
    assert "hard_challenges" in results
    assert "combined_benchmark" in results

    std_lex_acc = results["standard_benchmark"]["Our Lexical Engine"]["accuracy"]
    hard_lex_acc = results["hard_challenges"]["Our Lexical Engine"]["accuracy"]

    # Lexicon must experience a significant collapse on adversarial hard challenges (> 50% accuracy drop)
    assert std_lex_acc >= 90.0
    assert hard_lex_acc <= 30.0
    assert (std_lex_acc - hard_lex_acc) > 50.0

    # DeBERTa must outperform Lexicon on the hard challenges
    deberta_hard_acc = results["hard_challenges"]["DeBERTa-v3-small (Proposed)"]["accuracy"]
    assert deberta_hard_acc > hard_lex_acc
    assert deberta_hard_acc >= 75.0
