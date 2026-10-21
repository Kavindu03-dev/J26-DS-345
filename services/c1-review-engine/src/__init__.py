"""
C1 Review Engine Package (Research Component 1 - J26-DS-345).
Centralizes stage modules and provides clean, dynamic access to pipeline stages:
  - 01_data_preparation
  - 02_trust_classification
  - 03_aspect_sentiment
  - 04_aggregation_alerts
  - 05_scientific_evaluation
"""

import sys
import importlib.util
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent


def _load_stage_module(rel_path: str, module_name: str):
    """Dynamically loads a stage script given its relative path to src/."""
    script_path = _SRC_DIR / rel_path
    if not script_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so inter-module imports resolve cleanly
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load Stage Modules Dynamically
_mod_01_builder = _load_stage_module("01_data_preparation/01_build_ground_truth.py", "s01_build_ground_truth")
_mod_01_challenges = _load_stage_module("01_data_preparation/02_generate_hard_challenges.py", "s01_generate_hard_challenges")
_mod_01_agreement = _load_stage_module("01_data_preparation/03_audit_annotation_agreement.py", "s01_audit_annotation_agreement")

_mod_02_train = _load_stage_module("02_trust_classification/02_train_trust_classifiers.py", "s02_train_trust_classifiers")
_mod_02_ablation = _load_stage_module("02_trust_classification/03_audit_feature_ablation.py", "s02_audit_feature_ablation")

_mod_03_train = _load_stage_module("03_aspect_sentiment/02_train_deberta_transformer.py", "s03_train_deberta_transformer")
_mod_03_benchmark = _load_stage_module("03_aspect_sentiment/03_benchmark_absa_comparison.py", "s03_benchmark_absa_comparison")

_mod_04_batch = _load_stage_module("04_aggregation_alerts/04_batch_precompute_profiles.py", "s04_batch_precompute_profiles")

_mod_05_ablation = _load_stage_module("05_scientific_evaluation/03_run_master_ablation_study.py", "s05_run_master_ablation_study")

# Exported Primary Classes & Callables
GroundTruthBuilder = getattr(_mod_01_builder, "GroundTruthBuilder", None)
HARD_CHALLENGES = getattr(_mod_01_challenges, "HARD_CHALLENGES", [])
generate_benchmark_csv = getattr(_mod_01_challenges, "generate_benchmark_csv", None)
InterAnnotatorAgreementStudy = getattr(_mod_01_agreement, "InterAnnotatorAgreementStudy", None)

TabularTrustClassifierTrainer = getattr(_mod_02_train, "TabularTrustClassifierTrainer", None)
TrustClassifierTrainer = getattr(_mod_02_train, "TrustClassifierTrainer", None)
FeatureLeakageAuditor = getattr(_mod_02_ablation, "FeatureLeakageAuditor", None)
TrustClassifierAblationStudy = getattr(_mod_02_ablation, "TrustClassifierAblationStudy", None)

DeBERTaTrainer = getattr(_mod_03_train, "DeBERTaTrainer", None)
ABSAModelBenchmark = getattr(_mod_03_benchmark, "ABSAModelBenchmark", None)

BatchEnricher = getattr(_mod_04_batch, "BatchEnricher", None)

RankingAblationStudy = getattr(_mod_05_ablation, "RankingAblationStudy", None)

# Backwards-compatible sys.modules aliases for any legacy imports
if _mod_01_builder:
    sys.modules["src.ground_truth_builder"] = _mod_01_builder
if _mod_01_challenges:
    sys.modules["src.generate_hard_challenges"] = _mod_01_challenges
if _mod_01_agreement:
    sys.modules["src.annotation_agreement"] = _mod_01_agreement
if _mod_02_train:
    sys.modules["src.train_trust_classifier"] = _mod_02_train
if _mod_02_ablation:
    sys.modules["src.trust_classifier_ablation"] = _mod_02_ablation
if _mod_03_train:
    sys.modules["src.train_deberta_absa"] = _mod_03_train
if _mod_03_benchmark:
    sys.modules["src.benchmark_absa_comparison"] = _mod_03_benchmark
if _mod_04_batch:
    sys.modules["src.batch_enricher"] = _mod_04_batch
if _mod_05_ablation:
    sys.modules["src.ablation_study"] = _mod_05_ablation

__all__ = [
    "GroundTruthBuilder",
    "HARD_CHALLENGES",
    "generate_benchmark_csv",
    "InterAnnotatorAgreementStudy",
    "TabularTrustClassifierTrainer",
    "TrustClassifierTrainer",
    "FeatureLeakageAuditor",
    "TrustClassifierAblationStudy",
    "DeBERTaTrainer",
    "ABSAModelBenchmark",
    "BatchEnricher",
    "RankingAblationStudy",
]
