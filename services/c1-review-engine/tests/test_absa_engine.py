"""
Unit tests for C1 ABSAEngine and AspectScoreAggregator.
Verifies aspect extraction, negation logic, fact extraction, and contract adherence.
"""

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.absa_engine import ABSAEngine
from src.models.fact_extractor import FactExtractor
from src.models.aggregator import AspectScoreAggregator


class TestABSAEngine(unittest.TestCase):
    def setUp(self):
        self.absa = ABSAEngine()
        self.aggregator = AspectScoreAggregator()

    def test_aspect_detection_and_negation(self):
        # Review mentioning food (negative due to negation) and staff (positive with intensifier)
        review = {
            "review_text": "The staff were extremely friendly and attentive, but the breakfast was not good at all.",
            "rating_score": 7.0
        }
        res = self.absa.analyze_review(review, baseline_rating_10=7.0)

        # Staff should be positive (> 4.0)
        self.assertEqual(res["staff"]["sentiment"], "Positive")
        self.assertGreaterEqual(res["staff"]["score_5"], 4.0)

        # Food should be negative (< 3.0)
        self.assertEqual(res["food"]["sentiment"], "Negative")
        self.assertLessEqual(res["food"]["score_5"], 3.0)

    def test_positive_idiom(self):
        review = {
            "review_text": "Nothing was too much trouble for the manager and staff.",
            "rating_score": 9.0
        }
        res = self.absa.analyze_review(review, baseline_rating_10=9.0)
        self.assertEqual(res["staff"]["sentiment"], "Positive")
        self.assertGreaterEqual(res["staff"]["score_5"], 4.5)

    def test_contract_compliance(self):
        hotel = {
            "hotel_id": "HTL_GAL_TEST_001",
            "hotel_name": "Test Boutique Villa",
            "city": "Galle",
            "platform_overall_rating": 8.8
        }
        reviews = [
            {
                "author_id": "USR_ABC123",
                "author_account_age": "Local Guide · 30 reviews",
                "author_total_reviews": 30,
                "is_verified_stay": True,
                "review_date": "September 2026",
                "rating_score": 9.0,
                "review_text": "Breakfast was fresh and served until 10 am with delicious hoppers. Very clean room and fast wifi.",
                "helpful_votes": 2
            },
            {
                "author_id": "USR_XYZ999",
                "author_account_age": "Verified Guest",
                "author_total_reviews": 3,
                "is_verified_stay": True,
                "review_date": "August 2026",
                "rating_score": 8.0,
                "review_text": "Great location in the heart of Galle fort, 5 mins walk to everything. Staff were lovely.",
                "helpful_votes": 0
            }
        ]

        result = self.aggregator.aggregate_hotel_aspect_scores(hotel, reviews)

        # Verify contract requirements from contracts/c1_aspect_scores.json
        self.assertEqual(result["hotel_id"], "HTL_GAL_TEST_001")
        self.assertIn("aspect_scores", result)
        self.assertIn("overall_trust_score", result)
        self.assertIn("time_decay_rating", result)
        self.assertIn("verified_aspect_facts", result)
        self.assertIn("last_updated", result)

        # Check required aspect keys
        required_aspects = ["food", "staff", "room", "cleanliness", "location", "wifi"]
        for asp in required_aspects:
            self.assertIn(asp, result["aspect_scores"])
            score = result["aspect_scores"][asp]
            self.assertTrue(0.0 <= score <= 5.0, f"Aspect {asp} score {score} out of bounds")

        # Check overall trust score (0.0 to 10.0)
        self.assertTrue(0.0 <= result["overall_trust_score"] <= 10.0)

        # Check time decay rating (1.0 to 5.0)
        self.assertTrue(1.0 <= result["time_decay_rating"] <= 5.0)

        # Check verified facts
        self.assertIsInstance(result["verified_aspect_facts"], list)
        self.assertGreater(len(result["verified_aspect_facts"]), 0)


if __name__ == "__main__":
    unittest.main()
