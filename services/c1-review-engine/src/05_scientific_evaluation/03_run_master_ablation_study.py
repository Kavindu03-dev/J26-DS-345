"""
Pipeline Stage 05: Scientific Evaluation & Empirical Proof.
Step 03: Master Ranking Ablation Study (Conditions A vs B vs C).
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Fulfills Specific Objectives 4 & 5 (SO4, SO5) and WBS 5.2, 5.3:
Ranks 1,025 hotels under raw arithmetic mean (Condition A), trust-only (Condition B),
and dual-weighted trust + exponential time-decay (Condition C).
Computes Spearman rho, Kendall tau, and Mean Absolute Rank Shift (MARS).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ablation_study import RankingAblationStudy

if __name__ == "__main__":
    study = RankingAblationStudy()
    study.run_ablation_experiment()
