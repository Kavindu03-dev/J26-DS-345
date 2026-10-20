"""
Pipeline Stage 04: Aggregation & Alerts Engine.
Step 01: Extract Grounded & Verified Aspect Facts for Component 4 RAG.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.fact_extractor import FactExtractor
from src.models.trust_engine import TrustEngine

def demo_fact_extraction():
    extractor = FactExtractor()
    trust_engine = TrustEngine()
    print("=" * 80)
    print("   STAGE 04: VERIFIED ASPECT FACT EXTRACTION DEMO (FOR COMPONENT 4 RAG)")
    print("=" * 80)

    sample_reviews = [
        {"review_text": "The swimming pool is open until 10 pm and they offer free airport shuttle every morning.", "rating_score": 9.0, "is_verified_stay": True},
        {"review_text": "Breakfast starts at 6:30 am with fresh egg hoppers and Ceylon tea.", "rating_score": 9.0, "is_verified_stay": True},
        {"review_text": "Had a truly amazing and wonderful magical time in paradise!", "rating_score": 10.0, "is_verified_stay": False}
    ]

    facts = extractor.extract_facts_from_reviews(sample_reviews, trust_engine=trust_engine)
    print(f"Total High-Trust Facts Extracted: {len(facts)}")
    for idx, f in enumerate(facts, 1):
        print(f"  [{idx}] \"{f}\"")
    print("=" * 80)

if __name__ == "__main__":
    demo_fact_extraction()
