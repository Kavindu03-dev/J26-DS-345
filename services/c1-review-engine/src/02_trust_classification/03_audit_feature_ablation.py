"""
Pipeline Stage 02: Trust Classification & Authenticity Engine.
Step 03: Feature Leakage Audit & Multi-Configuration Feature Ablation Study.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Proves the true empirical value of Aspect Specificity Density (ASD):
- Audits LightGBM feature gain (identifying is_verified_stay leakage).
- Evaluates 4 feature configurations under 5-Fold Stratified Cross-Validation.
- Demonstrates +48.04% Macro-F1 lift of ASD + Content over naive word counts.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.trust_classifier_ablation import FeatureLeakageAuditor

if __name__ == "__main__":
    auditor = FeatureLeakageAuditor()
    auditor.run_audit_and_ablation()
