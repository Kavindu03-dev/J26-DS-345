"""
Pipeline Stage 04: Aggregation & Alerts Engine.
Step 02: Aggregate Hotel Aspect Profile Adhering to C1 Contract Schema.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Integrates TrustEngine, ABSAEngine, and FactExtractor into the canonical
JSON output format defined by contracts/c1_aspect_scores.json.
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.aggregator import AspectScoreAggregator, HotelAspectAggregator

def demo_contract_aggregator():
    agg = AspectScoreAggregator()
    print("=" * 80)
    print("   STAGE 04: HOTEL ASPECT PROFILE CONTRACT AGGREGATOR DEMO")
    print("=" * 80)

    hotel_meta = {
        "hotel_id": "HTL_DEMO_001",
        "hotel_name": "Mirissa Ocean Boutique Villa",
        "city": "Mirissa",
        "district": "Matara",
        "rating": 8.8,
        "total_reviews": 45
    }

    sample_reviews = [
        {
            "review_text": "Spacious sea-view room with strong split AC. Breakfast hoppers were hot and fresh. Staff Amal was very kind.",
            "author_total_reviews": 15,
            "author_account_age": "Google Local Guide (Level 6)",
            "is_verified_stay": True,
            "rating_score": 9.0,
            "helpful_votes": 3,
            "review_date": "2026-09-20"
        },
        {
            "review_text": "Wi-Fi was weak in room 204, but the oceanfront location and infinity pool make up for it.",
            "author_total_reviews": 4,
            "author_account_age": "Verified Guest",
            "is_verified_stay": True,
            "rating_score": 8.0,
            "helpful_votes": 1,
            "review_date": "2026-08-10"
        }
    ]

    profile = agg.aggregate_hotel_aspect_scores(hotel_meta, sample_reviews)
    print(json.dumps(profile, indent=2))
    print("=" * 80)

if __name__ == "__main__":
    demo_contract_aggregator()
