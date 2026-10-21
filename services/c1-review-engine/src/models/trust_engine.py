"""
Trust Scoring & Behavioral Analysis Engine for Research Component 1.
Implements:
1. Review-level Credibility & Trust Scoring: T(r) in [0.05, 1.0]
2. Anomaly & Suspicious Review Detection (Astroturfing / Bot reviews)
3. Exponential Time-Decay Weighting: w(t) = exp(-lambda * delta_t)
4. Hotel-level Overall Trust Score in [0.0, 10.0] and Time-Decay Rating in [1.0, 5.0]
"""

import math
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any

from src.config import TRUST_PARAMS
from src.models.asd_calculator import AspectSpecificityDensityCalculator
from src.models.ml_trust_classifier import MLTrustClassifier


class TrustEngine:
    def __init__(self, params: Dict[str, Any] = None, use_ml_classifier: bool = True):
        self.params = params or TRUST_PARAMS
        self.half_life_days = float(self.params.get("half_life_days", 365.0))
        self.decay_lambda = math.log(2.0) / self.half_life_days
        self.ref_date = datetime(2026, 10, 15)  # Reference date matching research year
        self.asd_calculator = AspectSpecificityDensityCalculator()
        self.ml_classifier = MLTrustClassifier() if use_ml_classifier else None

    def parse_review_date(self, date_str: str) -> datetime:
        """Parses various date formats from Booking.com, Google Maps, and TripAdvisor."""
        if not date_str:
            return self.ref_date

        clean_d = re.sub(r'Reviewed:\s*|\s*\(1 night\)', '', str(date_str)).strip()

        # Format candidates
        formats = [
            "%B %d, %Y",       # September 17, 2026
            "%b %d, %Y",       # Sep 17, 2026
            "%B %Y",           # September 2026
            "%b %Y",           # Sep 2026
            "%Y-%m-%d",        # 2026-09-17
            "%d/%m/%Y",        # 17/09/2026
            "%Y",              # 2026
        ]

        for fmt in formats:
            try:
                return datetime.strptime(clean_d, fmt)
            except ValueError:
                continue

        # Relative date patterns (e.g., "3 weeks ago", "2 months ago", "a year ago")
        rel_match = re.search(r'([0-9]+)\s+(day|week|month|year)s?\s+ago', clean_d, re.IGNORECASE)
        if rel_match:
            num = int(rel_match.group(1))
            unit = rel_match.group(2).lower()
            days = num * (365 if unit == "year" else (30 if unit == "month" else (7 if unit == "week" else 1)))
            return self.ref_date.fromtimestamp(self.ref_date.timestamp() - (days * 86400))

        # Year fallback
        year_match = re.search(r'(202[0-9])', clean_d)
        if year_match:
            try:
                return datetime(int(year_match.group(1)), 6, 1)
            except Exception:
                pass

        return self.ref_date

    def calculate_time_decay_weight(self, date_str: str) -> float:
        """
        Computes the exponential time-decay weight w(t) = exp(-lambda * delta_t).
        Recent reviews have w(t) -> 1.0; 1-year-old reviews have w(t) -> 0.5.
        """
        rev_dt = self.parse_review_date(date_str)
        delta_days = max(0.0, (self.ref_date - rev_dt).total_seconds() / 86400.0)
        weight = math.exp(-self.decay_lambda * delta_days)
        # Bounded between 0.15 and 1.0 so older reviews still contribute historical baseline
        return max(0.15, min(1.0, weight))

    def calculate_reviewer_credibility(self, review: Dict[str, Any]) -> float:
        """
        Computes Author Credibility C_author in [0.25, 1.0]
        Based on review history, verified guest status, and Google Local Guide badges.
        """
        total_revs = int(review.get("author_total_reviews", 1) or 1)
        account_age_str = str(review.get("author_account_age", "")).lower()
        is_verified = bool(review.get("is_verified_stay", True))

        # Logarithmic experience curve
        cred = 0.45 + 0.12 * math.log10(max(1, total_revs))

        # Badge adjustments
        if "local guide" in account_age_str:
            cred += 0.15
            # Bonus for high photo contributors
            if "photo" in account_age_str:
                cred += 0.05
        elif "verified guest" in account_age_str or is_verified:
            cred += 0.10

        return max(0.25, min(1.0, cred))

    def calculate_text_richness(self, text: str, word_count: int = None) -> float:
        """
        Computes Informational Richness I_text in [0.25, 1.0]
        Saturates logarithmically at 120 words.
        """
        w_count = word_count if word_count is not None else len(text.split())
        if w_count <= 2:
            return 0.20
        # Logarithmic saturation
        richness = 0.30 + 0.70 * (math.log(1.0 + min(120, w_count)) / math.log(1.0 + 120))
        return max(0.25, min(1.0, richness))

    def calculate_rating_disparity(self, review_score: float, hotel_rating: float) -> Tuple[float, float]:
        """
        Computes disparity penalty P_disp.
        review_score: rating given by user (normalized to 1-5 or 1-10)
        hotel_rating: hotel baseline rating (normalized to 1-10)
        """
        # Standardize review_score to 1-10 scale for comparison
        score_10 = review_score * 2.0 if review_score <= 5.0 else review_score
        baseline_10 = hotel_rating if hotel_rating > 5.0 else hotel_rating * 2.0

        disparity = abs(score_10 - baseline_10)
        # Disparity penalty: higher if rating drastically differs
        penalty = min(0.40, (disparity / 10.0) * 0.50)
        return disparity, penalty

    def calculate_review_trust(self, review: Dict[str, Any], hotel_baseline_rating: float = 8.0) -> float:
        """
        Calculates individual review trust score T(r) in [0.05, 1.0].
        Combines behavioral credibility, stay verification, informational richness,
        Aspect Specificity Density (ASD), and LightGBM ML trust classification.
        """
        # 1. Author Credibility
        c_author = self.calculate_reviewer_credibility(review)

        # 2. Verification Multiplier
        is_verified = bool(review.get("is_verified_stay", True))
        v_stay = self.params.get("verified_stay_weight", 1.0) if is_verified else self.params.get("unverified_stay_weight", 0.80)

        # 3. Text Richness & Aspect Specificity Density (ASD)
        text = str(review.get("review_text", "") or "")
        w_count = int(review.get("text_length_words", len(text.split())) or len(text.split()))
        i_text = self.calculate_text_richness(text, w_count)

        # 4. Rating Disparity
        try:
            r_score = float(review.get("rating_score", 8.0) or 8.0)
        except (ValueError, TypeError):
            r_score = 8.0
        _, p_disp = self.calculate_rating_disparity(r_score, hotel_baseline_rating)
        # Disparity penalty is reduced if text is rich and detailed
        p_disp_adjusted = p_disp * (1.0 - 0.65 * i_text)

        # 5. Helpful Votes Endorsement
        try:
            h_votes = int(review.get("helpful_votes", 0) or 0)
        except (ValueError, TypeError):
            h_votes = 0
        h_boost = min(0.15, 0.05 * math.log10(1.0 + h_votes))

        # 6. ASD & ML Trust Classifier Integration (SO2 / FR4, FR5)
        ml_cred = 0.85
        is_ai_spam = False
        if self.ml_classifier is not None and text.strip():
            ml_res = self.ml_classifier.predict_review(review)
            ml_cred = float(ml_res.get("ml_credibility_score", 0.85))
            is_ai_spam = bool(ml_res.get("is_spam", False)) or bool(ml_res.get("is_ai_heuristic_flag", False))
        elif text.strip():
            asd_res = self.asd_calculator.calculate_asd(text)
            is_ai_spam = bool(asd_res.get("is_likely_ai_spam", False))
            ml_cred = 0.15 if is_ai_spam else min(1.0, 0.60 + 0.40 * float(asd_res.get("asd_score", 0.25)))

        # Calibrated convex trust formula: combines author credibility, informational richness, and ML/ASD credibility
        base_trust = (0.30 * c_author) + (0.45 * i_text) + (0.25 * ml_cred)
        trust = (base_trust * v_stay * (1.0 - p_disp_adjusted)) + h_boost

        # Apply deceptive AI spam penalty if flagged by ASD / LightGBM
        if is_ai_spam:
            trust = min(trust * 0.30, 0.20)

        return max(self.params.get("min_trust_score", 0.05), min(self.params.get("max_trust_score", 1.0), round(trust, 4)))

    def detect_suspicious_review(self, review: Dict[str, Any], trust_score: float) -> int:
        """
        Flags potential astroturfing, promotional, or AI-generated bot reviews.
        Returns 1 if suspicious, 0 if authentic.
        """
        text = str(review.get("review_text", "") or "")
        w_count = int(review.get("text_length_words", len(text.split())) or len(text.split()))
        try:
            score = float(review.get("rating_score", 8.0) or 8.0)
        except (ValueError, TypeError):
            score = 8.0
        score_5 = score / 2.0 if score > 5.0 else score
        is_verified = bool(review.get("is_verified_stay", True))

        # Condition 1: Extreme rating with almost no text from unverified account
        if (score_5 >= 4.8 or score_5 <= 1.5) and w_count < self.params.get("suspicious_short_word_len", 6) and not is_verified:
            return 1

        # Condition 2: Computed trust score is below critical floor
        if trust_score < self.params.get("suspicious_trust_threshold", 0.22):
            return 1

        # Condition 3: Flagged by ML Trust Classifier or ASD AI-spam detector
        if self.ml_classifier is not None and text.strip():
            ml_res = self.ml_classifier.predict_review(review)
            if ml_res.get("is_spam") or ml_res.get("is_ai_heuristic_flag"):
                return 1
        elif text.strip():
            asd_res = self.asd_calculator.calculate_asd(text)
            if asd_res.get("is_likely_ai_spam"):
                return 1

        return 0

    def compute_hotel_trust_metrics(self, reviews: List[Dict[str, Any]], hotel_rating_baseline: float = 8.0) -> Dict[str, Any]:
        """
        Aggregates review-level scores into Hotel-level Trust Metrics adhering to contracts/c1_aspect_scores.json:
        - overall_trust_score: 0.0 to 10.0
        - time_decay_rating: 1.0 to 5.0
        """
        if not reviews:
            return {
                "overall_trust_score": 5.0,
                "time_decay_rating": 3.0,
                "total_reviews_analyzed": 0,
                "suspicious_reviews_count": 0,
                "verified_stay_ratio": 0.0
            }

        total_trust_weighted = 0.0
        total_time_weighted_score = 0.0
        sum_weights = 0.0
        suspicious_count = 0
        verified_count = 0

        for r in reviews:
            t_score = self.calculate_review_trust(r, hotel_baseline_rating=hotel_rating_baseline)
            is_susp = self.detect_suspicious_review(r, t_score)
            if is_susp:
                suspicious_count += 1

            if r.get("is_verified_stay"):
                verified_count += 1

            # Time decay weight
            w_time = self.calculate_time_decay_weight(str(r.get("review_date", "")))

            # Normalized rating on 1.0 to 5.0 scale
            try:
                raw_score = float(r.get("rating_score", 8.0) or 8.0)
            except (ValueError, TypeError):
                raw_score = 8.0
            score_5 = max(1.0, min(5.0, (raw_score / 2.0) if raw_score > 5.0 else raw_score))

            # Effective weight combines trust and recency
            eff_weight = t_score * w_time
            total_time_weighted_score += eff_weight * score_5
            sum_weights += eff_weight
            total_trust_weighted += t_score

        avg_trust = total_trust_weighted / len(reviews)
        # Confidence penalty for very low review counts (< 5 reviews)
        sample_penalty = min(1.0, len(reviews) / 5.0)
        overall_trust_score = round(avg_trust * 10.0 * sample_penalty, 2)
        overall_trust_score = max(0.0, min(10.0, overall_trust_score))

        time_decay_rating = round(total_time_weighted_score / sum_weights, 2) if sum_weights > 0 else 3.5
        time_decay_rating = max(1.0, min(5.0, time_decay_rating))

        return {
            "overall_trust_score": overall_trust_score,
            "time_decay_rating": time_decay_rating,
            "total_reviews_analyzed": len(reviews),
            "suspicious_reviews_count": suspicious_count,
            "verified_stay_ratio": round(verified_count / len(reviews), 2)
        }
