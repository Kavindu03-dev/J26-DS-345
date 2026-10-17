"""
Trust-Weighted Aspect Aggregator for Research Component 1.
Merges Trust Scoring, Exponential Time-Decay, ABSA Aspect Sentiment, and Fact Extraction.
Generates compliant responses adhering to contracts/c1_aspect_scores.json.
"""

from datetime import datetime, timezone
from typing import Dict, List, Any

from src.config import ASPECT_CATEGORIES
from src.models.trust_engine import TrustEngine
from src.models.absa_engine import ABSAEngine
from src.models.fact_extractor import FactExtractor


class AspectScoreAggregator:
    def __init__(self, trust_engine: TrustEngine = None, absa_engine: ABSAEngine = None, fact_extractor: FactExtractor = None):
        self.trust_engine = trust_engine or TrustEngine()
        self.absa_engine = absa_engine or ABSAEngine()
        self.fact_extractor = fact_extractor or FactExtractor()

    def aggregate_hotel_aspect_scores(self, hotel_dict: Dict[str, Any], reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes all reviews for a given hotel and emits the finalized contract dictionary.
        """
        hotel_id = hotel_dict.get("hotel_id", "unknown_hotel")
        baseline_rating_10 = float(hotel_dict.get("platform_overall_rating", 8.0) or 8.0)
        baseline_5 = max(2.5, min(4.8, baseline_rating_10 / 2.0))

        # 1. Compute Hotel-level Trust Metrics
        trust_metrics = self.trust_engine.compute_hotel_trust_metrics(reviews, hotel_rating_baseline=baseline_rating_10)

        # 2. Compute Trust-Weighted Aspect Scores
        aspect_weighted_sums = {cat: 0.0 for cat in ASPECT_CATEGORIES}
        aspect_weight_totals = {cat: 0.0 for cat in ASPECT_CATEGORIES}
        aspect_mention_counts = {cat: 0 for cat in ASPECT_CATEGORIES}

        for r in reviews:
            t_score = self.trust_engine.calculate_review_trust(r, hotel_baseline_rating=baseline_rating_10)
            w_time = self.trust_engine.calculate_time_decay_weight(str(r.get("review_date", "")))
            eff_weight = t_score * w_time

            # Run ABSA on the review
            absa_res = self.absa_engine.analyze_review(r, baseline_rating_10=baseline_rating_10)

            for cat in ASPECT_CATEGORIES:
                cat_info = absa_res.get(cat, {})
                s_5 = float(cat_info.get("score_5", baseline_5))
                m_count = int(cat_info.get("mentions_count", 0))

                # If explicitly mentioned in review, give full weight; if not mentioned, mild baseline weight
                cat_weight = eff_weight * (1.0 if m_count > 0 else 0.20)

                aspect_weighted_sums[cat] += cat_weight * s_5
                aspect_weight_totals[cat] += cat_weight
                aspect_mention_counts[cat] += m_count

        final_aspect_scores = {}
        for cat in ASPECT_CATEGORIES:
            total_w = aspect_weight_totals[cat]
            if total_w > 0:
                raw_aspect = aspect_weighted_sums[cat] / total_w
            else:
                raw_aspect = baseline_5
            # Clamp cleanly between 0.0 and 5.0 as required by contract
            final_aspect_scores[cat] = round(max(0.0, min(5.0, raw_aspect)), 1)

        # 3. Extract Verified Facts for C4
        facts = self.fact_extractor.extract_facts_from_reviews(reviews, self.trust_engine, max_facts=5)

        # 4. Assemble final contract response matching contracts/c1_aspect_scores.json
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "hotel_id": hotel_id,
            "aspect_scores": final_aspect_scores,
            "overall_trust_score": trust_metrics["overall_trust_score"],
            "time_decay_rating": trust_metrics["time_decay_rating"],
            "verified_aspect_facts": facts,
            "last_updated": now_iso
        }
