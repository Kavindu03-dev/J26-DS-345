"""
Pipeline Stage 05: Scientific Evaluation & Empirical Proof.
Step 01: Half-Life (tau) Sensitivity Analysis Across 1,025 Hotels.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Sweeps tau in {90, 180, 365, 730} days to evaluate ranking stability,
monsoon seasonality bias, and Spearman rank correlation rho.
Exports data/processed/halflife_sensitivity_report.csv for Dissertation Chapter 5.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ablation_study import RankingAblationStudy

if __name__ == "__main__":
    study = RankingAblationStudy()
    study.run_halflife_sensitivity_analysis()
