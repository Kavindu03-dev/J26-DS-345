"""
Weak-Aspect Alert & Hotel Owner Operational Monitoring Engine.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine.
Project ID: J26-DS-345
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)
Supervisors: Prof. Samantha Thelijjagoda, Dr. Junius Anjana

Fulfills Requirement FR9 & WBS 6.3:
- Monitors rolling 60-day review streams against all-time aspect baselines.
- Detects operational degradation across 6 hospitality aspects:
    1. Cleanliness (Housekeeping, hygiene, bed bugs, mold, linens)
    2. Food (Breakfast, kitchen hygiene, dining freshness)
    3. Staff (Front-desk courtesy, service delays, responsiveness)
    4. Room (AC/cooling, hot water, bedding, noise insulation)
    5. Location (Access road condition, safety, transport access)
    6. Wi-Fi (Coverage blindspots, bandwidth, disconnects)
- Triggers structured alerts (CRITICAL, WARNING, INFO/COMMENDATION) with
  actionable remediation recommendations and extracted negative guest quotes.
"""

import sys
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ASPECT_CATEGORIES, ASPECT_KEYWORDS
from src.models.trust_engine import TrustEngine
from src.models.absa_engine import ABSAEngine


RECOMMENDATION_MATRIX: Dict[str, Dict[str, str]] = {
    "cleanliness": {
        "critical": "Initiate mandatory housekeeping audit: deep-sanitize all guest bathrooms, inspect mattresses/drapes, and replace worn towels.",
        "warning": "Increase daily room inspection frequency; conduct spot-checks on linen freshness and dust control in public areas.",
        "info": "Cleanliness standards are performing exceptionally well. Acknowledge and reward the housekeeping team."
    },
    "food": {
        "critical": "Immediate kitchen hygiene and supplier audit: inspect buffet holding temperatures, ingredient freshness, and breakfast replenishment speed.",
        "warning": "Review breakfast buffet variety and menu rotation; collect immediate verbal feedback during morning dining service.",
        "info": "Dining and culinary experience has received positive guest commendations. Maintain culinary standards."
    },
    "staff": {
        "critical": "Conduct urgent front-desk and service staff retraining on guest welcoming, de-escalation protocols, and check-in efficiency.",
        "warning": "Ensure adequate staffing during peak check-in/check-out hours (11:00 AM - 2:00 PM) to reduce wait times.",
        "info": "Staff hospitality and friendliness are rated high by verified guests. Excellent team engagement."
    },
    "room": {
        "critical": "Deploy maintenance technician immediately: inspect air conditioner compressors, hot water pressure, and repair faulty fixtures.",
        "warning": "Audit room lighting, mattress firmness, and window soundproofing seals to prevent noise disturbances.",
        "info": "Room comfort, amenities, and views are highly appreciated by recent travelers."
    },
    "location": {
        "critical": "Coordinate direct shuttle/tuk-tuk transport partnerships and provide pre-arrival Google Maps landmark directions to prevent arrival confusion.",
        "warning": "Update arrival instructions on booking portals with clear road condition warnings and prominent physical signage at the turn-off.",
        "info": "Location and scenic surroundings are highlighted as a primary strength by visitors."
    },
    "wifi": {
        "critical": "Upgrade Internet service plan bandwidth and install commercial mesh Wi-Fi access points in upper corridors and room clusters.",
        "warning": "Reset and reposition Wi-Fi repeaters in corridor blind spots; provide guests with dedicated high-speed login instructions.",
        "info": "High-speed Wi-Fi performance is meeting remote worker and traveler expectations."
    }
}


class WeakAspectAlertEngine:
    """
    Analyzes review streams for a hotel to detect acute or persistent weak-aspect failures
    and generates structured operational alerts for SME hotel managers.
    """

    def __init__(self, trust_engine: TrustEngine = None, absa_engine: ABSAEngine = None, rolling_days: int = 60):
        self.trust_engine = trust_engine or TrustEngine()
        self.absa_engine = absa_engine or ABSAEngine()
        self.rolling_days = rolling_days
        self.ref_date = self.trust_engine.ref_date

    def _split_recent_and_historical_reviews(self, reviews: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Separates reviews into rolling window (default 60 days) and historical baseline."""
        if not reviews:
            return [], []

        recent = []
        historical = []

        for r in reviews:
            rev_dt = self.trust_engine.parse_review_date(str(r.get("review_date", "")))
            delta_days = (self.ref_date - rev_dt).total_seconds() / 86400.0
            if 0.0 <= delta_days <= self.rolling_days:
                recent.append(r)
            else:
                historical.append(r)

        # Fallback for hotels with low review velocity: use top 30% most recent reviews if recent count < 3
        if len(recent) < 3 and len(reviews) >= 3:
            # Sort all reviews descending by parsed date
            sorted_all = sorted(
                reviews,
                key=lambda x: self.trust_engine.parse_review_date(str(x.get("review_date", ""))),
                reverse=True
            )
            split_idx = max(3, int(len(reviews) * 0.35))
            recent = sorted_all[:split_idx]
            historical = sorted_all[split_idx:]

        return recent, historical

    def _compute_aspect_scores_for_subset(self, reviews: List[Dict[str, Any]], hotel_baseline_10: float = 8.0) -> Dict[str, Dict[str, Any]]:
        """Calculates aspect scores, mention counts, and negative quotes for a given subset of reviews."""
        baseline_5 = max(2.5, min(4.8, hotel_baseline_10 / 2.0))
        weighted_sums = {cat: 0.0 for cat in ASPECT_CATEGORIES}
        weight_totals = {cat: 0.0 for cat in ASPECT_CATEGORIES}
        mention_counts = {cat: 0 for cat in ASPECT_CATEGORIES}
        negative_quotes = {cat: [] for cat in ASPECT_CATEGORIES}

        for r in reviews:
            t = self.trust_engine.calculate_review_trust(r, hotel_baseline_rating=hotel_baseline_10)
            text = str(r.get("review_text", "") or "")
            absa_res = self.absa_engine.analyze_review(r, baseline_rating_10=hotel_baseline_10)

            for cat in ASPECT_CATEGORIES:
                cat_info = absa_res.get(cat, {})
                s_5 = float(cat_info.get("score_5", baseline_5))
                m_count = int(cat_info.get("mentions_count", 0))

                # Weight proportional to trust and mention presence
                eff_w = t * (1.0 if m_count > 0 else 0.20)
                weighted_sums[cat] += eff_w * s_5
                weight_totals[cat] += eff_w
                mention_counts[cat] += m_count

                # Collect negative quote snippet if aspect was negatively reviewed
                if m_count > 0 and s_5 <= 2.8 and len(negative_quotes[cat]) < 3:
                    # Extract sentence containing aspect keyword
                    sentences = re.split(r'[.!?]+', text)
                    keywords = ASPECT_KEYWORDS.get(cat, [cat])
                    for s in sentences:
                        clean_s = s.strip()
                        if len(clean_s) >= 15 and any(re.search(r'\b' + re.escape(kw) + r'\b', clean_s, re.IGNORECASE) for kw in keywords):
                            if clean_s not in negative_quotes[cat]:
                                negative_quotes[cat].append(clean_s)
                                break

        results = {}
        for cat in ASPECT_CATEGORIES:
            total_w = weight_totals[cat]
            final_s = weighted_sums[cat] / total_w if total_w > 0 else baseline_5
            results[cat] = {
                "score_5": round(max(1.0, min(5.0, final_s)), 2),
                "mentions": mention_counts[cat],
                "negative_quotes": negative_quotes[cat]
            }

        return results

    def evaluate_hotel_alerts(self, hotel_id: str, hotel_name: str, reviews: List[Dict[str, Any]], hotel_baseline_10: float = 8.0) -> List[Dict[str, Any]]:
        """
        Evaluates a hotel's review history and returns a list of active alerts.
        """
        if not reviews:
            return []

        recent_revs, historical_revs = self._split_recent_and_historical_reviews(reviews)

        # Compute aspect scores for recent window vs historical baseline
        recent_aspects = self._compute_aspect_scores_for_subset(recent_revs, hotel_baseline_10)
        historical_aspects = self._compute_aspect_scores_for_subset(historical_revs, hotel_baseline_10) if historical_revs else recent_aspects

        alerts = []
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for aspect in ASPECT_CATEGORIES:
            rec = recent_aspects[aspect]
            hist = historical_aspects[aspect]

            s_recent = rec["score_5"]
            s_hist = hist["score_5"]
            delta = round(s_recent - s_hist, 2)
            mentions = rec["mentions"]
            quotes = rec["negative_quotes"]

            severity: Optional[str] = None
            alert_type: Optional[str] = None

            # Condition 1: CRITICAL ALERT (recent aspect score < 2.5 or acute collapse <= -1.2)
            if s_recent < 2.5 and (mentions > 0 or len(quotes) > 0):
                severity = "CRITICAL"
                alert_type = f"Severe {aspect.capitalize()} Quality Deficit"
            elif delta <= -1.2 and s_recent <= 3.2:
                severity = "CRITICAL"
                alert_type = f"Acute {aspect.capitalize()} Quality Collapse"

            # Condition 2: WARNING ALERT (recent aspect 2.5 - 3.2 or negative drift <= -0.8)
            elif s_recent <= 3.2 and (mentions > 0 or len(quotes) > 0):
                severity = "WARNING"
                alert_type = f"Substandard {aspect.capitalize()} Performance"
            elif delta <= -0.8:
                severity = "WARNING"
                alert_type = f"Negative {aspect.capitalize()} Trend Detected"

            # Condition 3: INFO / COMMENDATION (significant turnaround >= +0.8 or excellent >= 4.7)
            elif delta >= 0.8:
                severity = "INFO"
                alert_type = f"Significant {aspect.capitalize()} Turnaround"
            elif s_recent >= 4.7 and mentions >= 2:
                severity = "INFO"
                alert_type = f"Exemplary {aspect.capitalize()} Rating"

            if severity is not None:
                rec_text = RECOMMENDATION_MATRIX.get(aspect, {}).get(severity.lower(), "Review operational procedures for this aspect.")
                alert_obj = {
                    "alert_id": f"ALT_{hotel_id}_{aspect.upper()}_{severity}",
                    "hotel_id": hotel_id,
                    "hotel_name": hotel_name,
                    "aspect": aspect,
                    "severity": severity,
                    "alert_type": alert_type,
                    "recent_score_60d": s_recent,
                    "historical_baseline_score": s_hist,
                    "score_delta": delta,
                    "recent_mentions_count": mentions,
                    "sample_negative_quotes": quotes if severity != "INFO" else [],
                    "actionable_recommendation": rec_text,
                    "created_at": now_iso
                }
                alerts.append(alert_obj)

        return alerts
