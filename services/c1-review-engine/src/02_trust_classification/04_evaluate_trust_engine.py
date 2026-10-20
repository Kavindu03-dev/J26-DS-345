"""
Pipeline Stage 02: Trust Classification & Authenticity Engine.
Step 04: Evaluate Heuristic & Tabular Trust Engine Inference.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.trust_engine import TrustEngine
from src.models.ml_trust_classifier import MLTrustClassifier

def demo_trust_scoring():
    engine = TrustEngine()
    ml_clf = MLTrustClassifier()

    print("=" * 80)
    print("   STAGE 02: REVIEW TRUST & CREDIBILITY SCORING DEMO")
    print("=" * 80)

    sample_reviews = [
        {
            "review_text": "We stayed at the beach chalet for 3 nights. Room 102 was clean, breakfast hopper spread was delicious, and receptionist Amal organized our safari jeep.",
            "author_total_reviews": 12,
            "author_account_age": "Google Local Guide (Level 5)",
            "is_verified_stay": True,
            "rating_score": 9.0,
            "helpful_votes": 4,
            "review_date": "2026-09-15"
        },
        {
            "review_text": "Best ever paradise! Incredible vibes and perfection! Truly magical paradise!",
            "author_total_reviews": 1,
            "author_account_age": "New",
            "is_verified_stay": False,
            "rating_score": 10.0,
            "helpful_votes": 0,
            "review_date": "2026-10-01"
        }
    ]

    for idx, r in enumerate(sample_reviews, 1):
        t_score = engine.calculate_review_trust(r)
        is_susp = engine.detect_suspicious_review(r, t_score)
        ml_pred = ml_clf.predict_review(r)

        print(f"\nReview #{idx} : \"{r['review_text']}\"")
        print(f"Verified Stay   : {r['is_verified_stay']} | Author History: {r['author_total_reviews']} reviews")
        print(f"Heuristic Trust : T(r) = {t_score:.4f} | Anomaly Flag: {'SUSPICIOUS / BOT' if is_susp else 'AUTHENTIC'}")
        print(f"ML Classifier   : Credibility = {ml_pred['ml_credibility_score']:.4f} (Spam Prob = {ml_pred['spam_probability']:.2%})")

    print("=" * 80)

if __name__ == "__main__":
    demo_trust_scoring()
