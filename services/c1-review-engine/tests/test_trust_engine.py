"""
Unit tests for C1 TrustEngine using standard library unittest.
Verifies trust score calculation, time decay, bounds, and contract adherence.
"""

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.trust_engine import TrustEngine


class TestTrustEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TrustEngine()

    def test_trust_score_range(self):
        sample_review = {
            "author_total_reviews": 12,
            "author_account_age": "Local Guide · 12 reviews · 5 photos",
            "is_verified_stay": False,
            "text_length_words": 85,
            "review_text": "The hotel was very nice, clean pool and great breakfast.",
            "rating_score": 5.0,
            "helpful_votes": 3
        }
        trust = self.engine.calculate_review_trust(sample_review, hotel_baseline_rating=8.5)
        self.assertTrue(0.05 <= trust <= 1.0, f"Trust {trust} out of range [0.05, 1.0]")
        self.assertGreater(trust, 0.5, "Trust should be > 0.5 for active local guide with photos")

    def test_suspicious_review_flag(self):
        # Short extreme rant from unverified user
        fake_review = {
            "author_total_reviews": 1,
            "author_account_age": "Google User",
            "is_verified_stay": False,
            "text_length_words": 3,
            "review_text": "WORST PLACE EVER",
            "rating_score": 1.0,
            "helpful_votes": 0
        }
        trust = self.engine.calculate_review_trust(fake_review, hotel_baseline_rating=9.0)
        is_suspicious = self.engine.detect_suspicious_review(fake_review, trust)
        self.assertEqual(is_suspicious, 1, "Suspicious review was not flagged")

    def test_time_decay_recent_vs_old(self):
        recent_weight = self.engine.calculate_time_decay_weight("September 2026")
        old_weight = self.engine.calculate_time_decay_weight("January 2024")
        self.assertGreater(recent_weight, old_weight, "Recent review should have higher weight than old")
        self.assertGreaterEqual(recent_weight, 0.85)
        self.assertLessEqual(old_weight, 0.40)

    def test_hotel_trust_metrics_contract_bounds(self):
        reviews = [
            {
                "author_total_reviews": 40,
                "author_account_age": "Local Guide · 40 reviews",
                "is_verified_stay": True,
                "text_length_words": 110,
                "review_text": "Delicious breakfast, friendly staff, and clean rooms. Beautiful view.",
                "rating_score": 9.0,
                "helpful_votes": 2,
                "review_date": "August 2026"
            },
            {
                "author_total_reviews": 2,
                "author_account_age": "Verified Guest",
                "is_verified_stay": True,
                "text_length_words": 35,
                "review_text": "Good stay overall. Close to the fort.",
                "rating_score": 8.0,
                "helpful_votes": 0,
                "review_date": "July 2026"
            }
        ]
        metrics = self.engine.compute_hotel_trust_metrics(reviews, hotel_rating_baseline=8.5)
        
        # Check compliance with contracts/c1_aspect_scores.json
        self.assertTrue(0.0 <= metrics["overall_trust_score"] <= 10.0, "overall_trust_score must be in [0, 10]")
        self.assertTrue(1.0 <= metrics["time_decay_rating"] <= 5.0, "time_decay_rating must be in [1, 5]")
        self.assertEqual(metrics["total_reviews_analyzed"], 2)


if __name__ == "__main__":
    unittest.main()
