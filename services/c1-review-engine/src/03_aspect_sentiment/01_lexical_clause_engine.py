"""
Pipeline Stage 03: Aspect-Based Sentiment Analysis (ABSA).
Step 01: Lexical Clause Splitting & Negation Windowing Engine.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.absa_engine import ABSAEngine

def demo_lexical_absa():
    engine = ABSAEngine()
    print("=" * 80)
    print("   STAGE 03: ASPECT-BASED SENTIMENT ANALYSIS (LEXICAL ENGINE DEMO)")
    print("=" * 80)

    sample_reviews = [
        "The room was spacious and clean, but the wifi in the evening was not working at all.",
        "Delicious breakfast buffet with fresh fruit, however the front desk staff was very rude and unhelpful."
    ]

    for s in sample_reviews:
        print(f"\nReview Text: \"{s}\"")
        clauses = engine.split_into_clauses_and_sentences(s)
        print(f"Clauses ({len(clauses)}): {clauses}")

        aspect_res = engine.analyze_review({"review_text": s, "rating_score": 8.0})
        for asp, data in aspect_res.items():
            if data["mentions_count"] > 0:
                print(f"  - Aspect [{asp:<11}] : Sentiment = {data['sentiment']:<8} | Score = {data['score_5']:.2f} / 5.0 | Mentions = {data['mentions_count']}")

    print("=" * 80)

if __name__ == "__main__":
    demo_lexical_absa()
