"""
Pipeline Stage 05: Scientific Evaluation & Empirical Proof.
Step 02: Chronological Future-Predictive Temporal Backtesting Across 724 Hotels.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)
"""

import sys
import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_STUDY_SCRIPT = Path(__file__).resolve().parent / "03_run_master_ablation_study.py"
_spec = importlib.util.spec_from_file_location("stage05_master_ablation", _STUDY_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RankingAblationStudy = _mod.RankingAblationStudy

if __name__ == "__main__":
    study = RankingAblationStudy()
    study.run_temporal_backtest()
