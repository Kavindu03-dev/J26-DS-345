"""
Unit Tests for Ranking Ablation Study Engine (SO4, SO5 / WBS 5.2, 5.3).
"""

import unittest
import pandas as pd
from src.ablation_study import RankingAblationStudy


class TestRankingAblationStudy(unittest.TestCase):
    def setUp(self):
        self.study = RankingAblationStudy()

    def test_statistical_metrics_calculation(self):
        # Create a synthetic dataframe with 5 hotels
        sample_hotels = pd.DataFrame([
            {
                "hotel_id": "H1", "hotel_name": "Hotel One", "district": "Colombo",
                "review_count": 20, "avg_trust_score": 0.85, "avg_recency_weight": 0.90,
                "score_cond_a": 4.5, "score_cond_b": 4.6, "score_cond_c": 4.7,
                "rank_cond_a": 1, "rank_cond_b": 1, "rank_cond_c": 1,
                "rank_shift_c_vs_a": 0, "rank_shift_c_vs_b": 0, "rank_shift_b_vs_a": 0,
                "score_delta_c_vs_a": 0.2, "score_delta_c_vs_b": 0.1
            },
            {
                "hotel_id": "H2", "hotel_name": "Hotel Two", "district": "Galle",
                "review_count": 15, "avg_trust_score": 0.80, "avg_recency_weight": 0.70,
                "score_cond_a": 4.2, "score_cond_b": 4.3, "score_cond_c": 4.4,
                "rank_cond_a": 2, "rank_cond_b": 2, "rank_cond_c": 2,
                "rank_shift_c_vs_a": 0, "rank_shift_c_vs_b": 0, "rank_shift_b_vs_a": 0,
                "score_delta_c_vs_a": 0.2, "score_delta_c_vs_b": 0.1
            },
            {
                "hotel_id": "H3", "hotel_name": "Hotel Three", "district": "Kandy",
                "review_count": 10, "avg_trust_score": 0.60, "avg_recency_weight": 0.40,
                "score_cond_a": 4.0, "score_cond_b": 3.8, "score_cond_c": 3.6,
                "rank_cond_a": 3, "rank_cond_b": 4, "rank_cond_c": 4,
                "rank_shift_c_vs_a": -1, "rank_shift_c_vs_b": 0, "rank_shift_b_vs_a": -1,
                "score_delta_c_vs_a": -0.4, "score_delta_c_vs_b": -0.2
            },
            {
                "hotel_id": "H4", "hotel_name": "Hotel Four", "district": "Colombo",
                "review_count": 12, "avg_trust_score": 0.75, "avg_recency_weight": 0.85,
                "score_cond_a": 3.9, "score_cond_b": 4.0, "score_cond_c": 4.1,
                "rank_cond_a": 4, "rank_cond_b": 3, "rank_cond_c": 3,
                "rank_shift_c_vs_a": 1, "rank_shift_c_vs_b": 0, "rank_shift_b_vs_a": 1,
                "score_delta_c_vs_a": 0.2, "score_delta_c_vs_b": 0.1
            },
            {
                "hotel_id": "H5", "hotel_name": "Hotel Five", "district": "Galle",
                "review_count": 5, "avg_trust_score": 0.50, "avg_recency_weight": 0.30,
                "score_cond_a": 3.5, "score_cond_b": 3.3, "score_cond_c": 3.0,
                "rank_cond_a": 5, "rank_cond_b": 5, "rank_cond_c": 5,
                "rank_shift_c_vs_a": 0, "rank_shift_c_vs_b": 0, "rank_shift_b_vs_a": 0,
                "score_delta_c_vs_a": -0.5, "score_delta_c_vs_b": -0.3
            }
        ])

        # Mock df_reviews for summary metadata
        self.study.df_reviews = pd.DataFrame([{"review_id": i} for i in range(50)])

        metrics = self.study.compute_statistical_metrics(sample_hotels)

        # Assert metrics structure
        self.assertIn("spearman_rank_correlation", metrics)
        self.assertIn("mean_absolute_rank_shift_mars", metrics)
        self.assertIn("rank_displacement_distribution", metrics)
        self.assertIn("case_study_recovering_sme_hotels", metrics)

        # Check Spearman rho
        rho_ac = metrics["spearman_rank_correlation"]["cond_a_vs_cond_c"]["rho"]
        self.assertTrue(0.0 <= rho_ac <= 1.0)

        # Check MARS
        mars_ac = metrics["mean_absolute_rank_shift_mars"]["mars_a_vs_c"]
        self.assertAlmostEqual(mars_ac, 0.4, places=2)


if __name__ == "__main__":
    unittest.main()
