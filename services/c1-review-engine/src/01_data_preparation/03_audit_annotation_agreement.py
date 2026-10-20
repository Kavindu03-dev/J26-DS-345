"""
Pipeline Stage 01: Data Preparation & Benchmark Generation.
Step 03: Inter-Annotator Agreement Audit (Cohen's Kappa).
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.annotation_agreement import InterAnnotatorAgreementStudy

if __name__ == "__main__":
    study = InterAnnotatorAgreementStudy()
    study.compute_agreement_metrics()
