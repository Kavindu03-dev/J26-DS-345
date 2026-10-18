"""
Unit tests for ML Trust & Spam Classifier.
Verifies LightGBM model loading, feature extraction, and prediction outputs.
"""

import unittest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.ml_trust_classifier import MLTrustClassifier


class TestMLTrustClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = MLTrustClassifier()

    def test_authentic_booking_review(self):
        """Authentic verified booking review with high specificity."""
        review = {
            "source_platform": "Booking.com",
            "is_verified_stay": True,
            "author_total_reviews": 5,
            "rating_score": 8.5,
            "platform_overall_rating": 8.4,
            "review_text": (
                "Great stay in room 102. The egg hoppers at breakfast were very tasty "
                "and the swimming pool was clean. Walking distance to the beach."
            )
        }
        res = self.classifier.predict_review(review)
        self.assertFalse(res["is_spam"])
        self.assertGreater(res["ml_credibility_score"], 0.70)
        self.assertGreater(res["aspect_specificity_density"], 0.20)

    def test_ai_generated_generic_spam(self):
        """Unverified AI-generated text full of superlatives and zero entities."""
        review = {
            "source_platform": "AI-Generated",
            "is_verified_stay": False,
            "author_total_reviews": 1,
            "rating_score": 10.0,
            "platform_overall_rating": 7.0,
            "review_text": (
                "An absolutely breathtaking and magnificent experience. The ambiance was truly "
                "flawless, the hospitality was extraordinary, and the service was sublime and world-class. "
                "A heavenly oasis and an unforgettable paradise!"
            )
        }
        res = self.classifier.predict_review(review)
        self.assertTrue(res["is_spam"])
        self.assertGreater(res["spam_probability"], 0.80)
        self.assertLess(res["ml_credibility_score"], 0.20)


if __name__ == "__main__":
    unittest.main()
