"""
Pipeline Stage 03: Aspect-Based Sentiment Analysis (ABSA).
Step 02: Train & Fine-Tune DeBERTa-v3 Transformer Architecture.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Fulfills Specific Objective 3 (SO3) / Functional Requirement 6 (FR6):
Ingests the unified 450-sentence gold-standard pool, formats aspect-conditioned sequence pairs,
and executes fine-tuning (or exports the verified production training manifest).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_deberta_absa import DeBERTaTrainer

if __name__ == "__main__":
    trainer = DeBERTaTrainer()
    trainer.run_training_or_manifest()
