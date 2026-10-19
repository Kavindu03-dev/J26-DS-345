"""
Unit Tests for Weak-Aspect Alert Engine (Requirement FR9 / WBS 6.3).
"""

import unittest
from datetime import datetime
from src.models.alert_engine import WeakAspectAlertEngine


class TestWeakAspectAlertEngine(unittest.TestCase):
    def setUp(self):
        self.engine = WeakAspectAlertEngine(rolling_days=60)
        self.hotel_id = "HTL_TEST_HOTEL_001"
        self.hotel_name = "Test Eco Lodge"

    def test_empty_reviews_returns_empty_alerts(self):
        alerts = self.engine.evaluate_hotel_alerts(self.hotel_id, self.hotel_name, [])
        self.assertEqual(len(alerts), 0)

    def test_cleanliness_critical_alert_generation(self):
        # Create reviews with severe cleanliness issues in recent 60-day window
        reviews = [
            {
                "review_id": "REV_001",
                "review_date": "October 10, 2026",  # 5 days before ref_date
                "rating_score": 2.0,
                "review_text": "The cleanliness was terrible. The bathroom was dirty and dusty with cockroaches on the floor.",
                "is_verified_stay": True,
                "author_total_reviews": 5
            },
            {
                "review_id": "REV_002",
                "review_date": "October 05, 2026",  # 10 days before ref_date
                "rating_score": 2.0,
                "review_text": "Horrible hygiene. The towels had stains and the room smelled like mold.",
                "is_verified_stay": True,
                "author_total_reviews": 3
            },
            {
                "review_id": "REV_003",
                "review_date": "September 20, 2026",  # 25 days before ref_date
                "rating_score": 3.0,
                "review_text": "Staff was friendly, but the room was dirty and bed linen unwashed.",
                "is_verified_stay": True,
                "author_total_reviews": 2
            }
        ]

        alerts = self.engine.evaluate_hotel_alerts(self.hotel_id, self.hotel_name, reviews, hotel_baseline_10=4.0)
        
        # Verify that an alert for cleanliness was triggered
        cleanliness_alerts = [a for a in alerts if a["aspect"] == "cleanliness"]
        self.assertTrue(len(cleanliness_alerts) > 0, "Expected cleanliness alert to be triggered")
        
        alert = cleanliness_alerts[0]
        self.assertIn(alert["severity"], ["CRITICAL", "WARNING"])
        self.assertTrue(alert["recent_score_60d"] <= 3.2)
        self.assertTrue(len(alert["sample_negative_quotes"]) > 0)
        self.assertIn("housekeeping", alert["actionable_recommendation"].lower())

    def test_wifi_critical_alert_generation(self):
        reviews = [
            {
                "review_id": "REV_WIFI_01",
                "review_date": "October 12, 2026",
                "rating_score": 2.0,
                "review_text": "The wifi was awful. Slow internet and no connection in the bedroom.",
                "is_verified_stay": True,
                "author_total_reviews": 8
            },
            {
                "review_id": "REV_WIFI_02",
                "review_date": "October 02, 2026",
                "rating_score": 2.0,
                "review_text": "Poor wifi speed and constant network drops.",
                "is_verified_stay": True,
                "author_total_reviews": 4
            },
            {
                "review_id": "REV_WIFI_03",
                "review_date": "September 15, 2026",
                "rating_score": 3.0,
                "review_text": "The hotel was nice but internet connection was completely broken.",
                "is_verified_stay": True,
                "author_total_reviews": 6
            }
        ]

        alerts = self.engine.evaluate_hotel_alerts(self.hotel_id, self.hotel_name, reviews, hotel_baseline_10=6.0)
        wifi_alerts = [a for a in alerts if a["aspect"] == "wifi"]
        self.assertTrue(len(wifi_alerts) > 0, "Expected wifi alert to be triggered")
        self.assertTrue(any(w in wifi_alerts[0]["actionable_recommendation"].lower() for w in ["wi-fi", "wifi", "internet"]))

    def test_positive_turnaround_info_alert(self):
        # Historical reviews were poor, but recent reviews are stellar
        reviews = [
            # 2 Recent stellar reviews
            {
                "review_id": "REV_NEW_01",
                "review_date": "October 10, 2026",
                "rating_score": 10.0,
                "review_text": "The food and breakfast was delicious and wonderful fresh tropical buffet.",
                "is_verified_stay": True,
                "author_total_reviews": 10
            },
            {
                "review_id": "REV_NEW_02",
                "review_date": "October 08, 2026",
                "rating_score": 10.0,
                "review_text": "Chef prepared amazing dinner curry and tasty seafood dishes.",
                "is_verified_stay": True,
                "author_total_reviews": 7
            },
            # 2 Old poor reviews
            {
                "review_id": "REV_OLD_01",
                "review_date": "August 10, 2025",
                "rating_score": 2.0,
                "review_text": "The food was terrible and cold breakfast.",
                "is_verified_stay": True,
                "author_total_reviews": 3
            },
            {
                "review_id": "REV_OLD_02",
                "review_date": "July 15, 2025",
                "rating_score": 2.0,
                "review_text": "Disappointing dinner and poor meal quality.",
                "is_verified_stay": True,
                "author_total_reviews": 2
            }
        ]

        alerts = self.engine.evaluate_hotel_alerts(self.hotel_id, self.hotel_name, reviews, hotel_baseline_10=6.0)
        food_alerts = [a for a in alerts if a["aspect"] == "food"]
        self.assertTrue(len(food_alerts) > 0)
        self.assertIn(food_alerts[0]["severity"], ["INFO", "WARNING", "CRITICAL"])


if __name__ == "__main__":
    unittest.main()
