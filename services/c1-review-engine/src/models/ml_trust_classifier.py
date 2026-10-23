"""
ML Trust & Deceptive Spam Classifier Inference Wrapper.
Research Component 1: Trust-Aware Review Analysis Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Provides unified runtime inference for trained LightGBM and XGBoost models
using Aspect Specificity Density (ASD) and tabular review signals.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ENGINE_DIR
from src.models.asd_calculator import AspectSpecificityDensityCalculator

MODELS_DIR = ENGINE_DIR / "models"
LGBM_MODEL_FILE = MODELS_DIR / "lightgbm_trust_classifier.joblib"


class MLTrustClassifier:
    """Inference wrapper for trained tabular gradient boosting trust models."""

    def __init__(self, model_path: Path = LGBM_MODEL_FILE):
        self.model_path = model_path
        self.asd_calculator = AspectSpecificityDensityCalculator()
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads trained LightGBM model weights from disk."""
        if joblib is not None and self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"[!] Warning: Failed to load ML Trust model ({e}). Fallback to heuristic.")
                self.model = None
        else:
            self.model = None


    def extract_features(self, review: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extracts the 8 tabular features from a review dictionary."""
        text = str(review.get("review_text", "") or "")
        asd_res = self.asd_calculator.calculate_asd(text)

        word_count = asd_res["word_count"]
        asd = asd_res["asd_score"]
        entities = asd_res["concrete_entities_count"]
        fluff = asd_res["generic_fluff_count"]
        fluff_ratio = fluff / max(1.0, float(entities))

        # De-leaked realistic stay verification & author history signals
        is_verified_raw = review.get("is_verified_stay")
        platform = str(review.get("source_platform", "Booking.com"))
        if is_verified_raw is not None:
            if isinstance(is_verified_raw, bool):
                is_verified = 1.0 if is_verified_raw else 0.0
            else:
                is_verified = 1.0 if str(is_verified_raw).strip().lower() in ["true", "1", "yes", "t"] else 0.0
        else:
            is_verified = 1.0 if platform == "Booking.com" else 0.0

        # Author review history log
        try:
            total_revs = int(review.get("author_total_reviews", 1) or 1)
            if total_revs > 1:
                author_hist_log = float(np.log10(max(1, total_revs)))
            else:
                author_hist_log = 1.2 if platform == "Booking.com" else (1.0 if platform == "Google Maps" else 0.7)
        except (ValueError, TypeError):
            author_hist_log = 1.0

        # Rating disparity
        try:
            r_score = float(review.get("rating_score", 8.0) or 8.0)
            baseline = float(review.get("platform_overall_rating", 8.0) or 8.0)
            rating_disp = min(1.0, abs(r_score - baseline) / 10.0)
        except (ValueError, TypeError):
            rating_disp = 0.0

        feature_vector = np.array([[
            asd,
            float(word_count),
            float(entities),
            float(fluff),
            fluff_ratio,
            is_verified,
            author_hist_log,
            rating_disp
        ]], dtype=np.float32)

        return feature_vector, asd_res

    def predict_review(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts authenticity and credibility of a review.
        Returns:
            - is_spam (bool): True if classified as Deceptive / AI Spam.
            - spam_probability (float): Probability in [0.0, 1.0] that review is spam.
            - ml_credibility_score (float): Probability in [0.0, 1.0] that review is authentic.
            - asd_details (dict): Aspect specificity breakdown.
        """
        feats, asd_res = self.extract_features(review)

        if self.model is not None:
            try:
                # predict_proba returns [P(authentic), P(spam)]
                probs = self.model.predict_proba(feats)[0]
                p_spam = float(probs[1])
                if asd_res["is_likely_ai_spam"]:
                    p_spam = max(p_spam, 0.85)
                p_authentic = 1.0 - p_spam
            except Exception:
                p_spam = 1.0 if asd_res["is_likely_ai_spam"] else 0.0
                p_authentic = 1.0 - p_spam
        else:
            # Fallback to ASD heuristic if model file not available
            p_spam = 1.0 if asd_res["is_likely_ai_spam"] else 0.0
            p_authentic = 1.0 - p_spam

        is_spam = p_spam >= 0.50

        return {
            "is_spam": is_spam,
            "spam_probability": round(p_spam, 4),
            "ml_credibility_score": round(p_authentic, 4),
            "aspect_specificity_density": asd_res["asd_score"],
            "concrete_entities_count": asd_res["concrete_entities_count"],
            "generic_fluff_count": asd_res["generic_fluff_count"],
            "is_ai_heuristic_flag": asd_res["is_likely_ai_spam"]
        }
