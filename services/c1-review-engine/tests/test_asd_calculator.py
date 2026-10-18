"""
Unit tests for Aspect Specificity Density (ASD) Engine.
Validates entity extraction, fluff detection, and AI spam classification.
"""

import unittest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.asd_calculator import AspectSpecificityDensityCalculator


class TestASDCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = AspectSpecificityDensityCalculator()

    def test_authentic_human_review_high_asd(self):
        """Authentic review mentioning specific Sri Lankan food, facilities, and landmarks."""
        text = (
            "We stayed in room 204 for 3 nights. Breakfast had delicious live egg hoppers, "
            "fresh papaya, and authentic Ceylon tea. The split AC was quiet and the bathroom "
            "had hot water. Just a 5 minutes walk to the Galle Fort lighthouse."
        )
        res = self.calculator.calculate_asd(text)
        self.assertGreater(res["asd_score"], 0.35)
        self.assertGreater(res["concrete_entities_count"], 3)
        self.assertFalse(res["is_likely_ai_spam"])
        self.assertIn("egg hoppers", res["detected_entities"])
        self.assertIn("galle fort", res["detected_entities"])


    def test_ai_generated_generic_fluff_low_asd(self):
        """Zero-shot ChatGPT style review filled with superlatives but zero tangible entities."""
        text = (
            "An absolutely breathtaking and magnificent experience. The ambiance was truly "
            "flawless, the hospitality was extraordinary, and the service was sublime and world-class. "
            "A heavenly oasis and an unforgettable paradise! Truly delightful and magical."
        )
        res = self.calculator.calculate_asd(text)
        self.assertLess(res["asd_score"], 0.25)
        self.assertEqual(res["concrete_entities_count"], 0)
        self.assertTrue(res["is_likely_ai_spam"])
        self.assertGreater(res["generic_fluff_count"], 4)

    def test_empty_and_short_reviews(self):
        """Handles empty or boundary review cases gracefully."""
        res_empty = self.calculator.calculate_asd("")
        self.assertEqual(res_empty["concrete_entities_count"], 0)
        self.assertFalse(res_empty["is_likely_ai_spam"])

        res_none = self.calculator.calculate_asd(None)
        self.assertEqual(res_none["concrete_entities_count"], 0)


if __name__ == "__main__":
    unittest.main()
