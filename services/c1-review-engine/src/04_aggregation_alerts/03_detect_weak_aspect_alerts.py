"""
Pipeline Stage 04: Aggregation & Alerts Engine.
Step 03: Detect Weak-Aspect Degradation Alerts for Hotel Owners.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.alert_engine import WeakAspectAlertEngine

def demo_alerts():
    engine = WeakAspectAlertEngine()
    print("=" * 80)
    print("   STAGE 04: HOTEL OWNER WEAK-ASPECT ALERT ENGINE DEMO (FR9)")
    print("=" * 80)

    # Simulated reviews with recent degradation in Cleanliness and Wi-Fi
    sample_reviews = [
        # Recent reviews (past 60 days) with negative cleanliness and wifi
        {
            "review_text": "The bathroom was dirty and towels had moldy odor. WiFi was completely broken and useless.",
            "rating_score": 4.0,
            "review_date": "2026-10-01"
        },
        {
            "review_text": "Unclean bathroom with dirty shower drain and disconnected wifi.",
            "rating_score": 4.0,
            "review_date": "2026-10-05"
        },
        # Historical reviews (past 1-2 years) with high scores
        {
            "review_text": "Clean rooms and fast internet connection.",
            "rating_score": 9.0,
            "review_date": "2025-05-10"
        },
        {
            "review_text": "Spotless property with lovely swimming pool.",
            "rating_score": 9.0,
            "review_date": "2025-04-12"
        }
    ]

    report = engine.evaluate_hotel_alerts("HTL_DEMO_001", "Mirissa Ocean Boutique Villa", sample_reviews)
    print(json.dumps(report, indent=2))
    print("=" * 80)

if __name__ == "__main__":
    demo_alerts()
