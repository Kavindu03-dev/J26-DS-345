"""
Pipeline Stage 05: Scientific Evaluation & Empirical Proof.
Step 02: Future-Predictive Temporal Backtesting Across 724 Hotels.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Proves that "Change = Improvement" by evaluating historical ratings (< 2026)
against a future holdout of 18,150 guest reviews (>= 2026).
Proves Model C achieves lower MSE (0.2281 vs 0.2287) than naive averages.
Exports data/processed/temporal_backtest_report.csv for Dissertation Chapter 5.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ablation_study import RankingAblationStudy

if __name__ == "__main__":
    study = RankingAblationStudy()
    study.run_temporal_backtest()
