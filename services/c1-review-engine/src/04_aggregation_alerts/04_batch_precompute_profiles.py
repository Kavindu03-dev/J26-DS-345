"""
Pipeline Stage 04: Aggregation & Alerts Engine.
Step 04: Batch Precompute Hotel Aspect Profiles for 1,025 Hotels.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Fulfills Functional Requirement 10 (FR10):
Bulk processes all 39,399 verified reviews, applies ASD, Trust, Time-Decay,
and ABSA scoring, and saves data/hotel_aspect_profiles.json and .csv for Teams 2 & 4.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.batch_enricher import BatchEnricher

if __name__ == "__main__":
    enricher = BatchEnricher()
    enricher.run_batch_enrichment()
