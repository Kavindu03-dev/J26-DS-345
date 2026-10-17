"""
API Contract Compliance Tests for Component 1.
Validates FastAPI endpoints against the exact schema required in contracts/c1_aspect_scores.json.
"""

import unittest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import app, load_dataset_indexes


class TestAPIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dataset_indexes()
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("service"), "c1-review-engine")
        self.assertEqual(data.get("status"), "ok")

    def test_aspect_scores_contract_schema(self):
        # Test with real hotel Ivy Lane Galle Fort
        hotel_id = "HTL_GAL_IVYLANEG_EE74"
        response = self.client.get(f"/hotels/{hotel_id}/aspect-scores")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # 1. Required top-level fields
        required_fields = [
            "hotel_id",
            "aspect_scores",
            "overall_trust_score",
            "time_decay_rating",
            "verified_aspect_facts",
            "last_updated"
        ]
        for field in required_fields:
            self.assertIn(field, data, f"Missing required field: {field}")

        # 2. Aspect scores object
        aspects = data["aspect_scores"]
        required_aspects = ["food", "staff", "room", "cleanliness", "location", "wifi"]
        for asp in required_aspects:
            self.assertIn(asp, aspects, f"Missing required aspect: {asp}")
            score = aspects[asp]
            self.assertTrue(0.0 <= score <= 5.0, f"Aspect score {asp}={score} out of range [0, 5]")

        # 3. Overall Trust Score range [0, 10]
        self.assertTrue(0.0 <= data["overall_trust_score"] <= 10.0, "overall_trust_score out of range [0, 10]")

        # 4. Time-decay rating range [1, 5]
        self.assertTrue(1.0 <= data["time_decay_rating"] <= 5.0, "time_decay_rating out of range [1, 5]")

        # 5. Verified aspect facts list
        self.assertIsInstance(data["verified_aspect_facts"], list)

    def test_non_existent_hotel_404(self):
        response = self.client.get("/hotels/HTL_DOES_NOT_EXIST_9999/aspect-scores")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
