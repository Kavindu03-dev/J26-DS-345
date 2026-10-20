"""
Pipeline Stage 02: Trust Classification & Authenticity Engine.
Step 02: Train Tabular Machine Learning Trust Classifiers (LightGBM & XGBoost).
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Fulfills Specific Objective 2 (SO2) / Functional Requirement 5 (FR5):
Trains LightGBM and XGBoost models on tabular review features using 5-Fold Stratified Cross-Validation.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_trust_classifier import TabularTrustClassifierTrainer

if __name__ == "__main__":
    trainer = TabularTrustClassifierTrainer()
    trainer.train_models()
