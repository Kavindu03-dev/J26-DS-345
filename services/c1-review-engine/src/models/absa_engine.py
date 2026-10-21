"""
Aspect-Based Sentiment Analysis (ABSA) Engine for Research Component 1.
Extracts sentiment for the 6 core hospitality aspects defined in contracts/c1_aspect_scores.json:
- food
- staff
- room
- cleanliness
- location
- wifi

Uses contextual phrase & sentence parsing, negation windowing, intensifiers,
and multi-platform text segment handling (Booking positive/negative vs Google/Trip full text).
"""

import re
from typing import Dict, List, Tuple, Any

from src.config import ASPECT_CATEGORIES, ASPECT_KEYWORDS

# Sentiment Lexicon
POSITIVE_LEXICON = {
    # Strong positive (weight 2.5 - 3.0)
    "exceptional": 3.0, "outstanding": 3.0, "superb": 3.0, "flawless": 3.0, "impeccable": 3.0,
    "perfect": 2.8, "amazing": 2.8, "incredible": 2.8, "fantastic": 2.8, "wonderful": 2.5,
    "excellent": 2.5, "delicious": 2.5, "spotless": 2.5, "magical": 2.5,
    
    # Moderate positive (weight 1.5 - 2.0)
    "great": 2.0, "lovely": 2.0, "tasty": 2.0, "spacious": 2.0, "comfortable": 2.0,
    "friendly": 2.0, "helpful": 2.0, "clean": 2.0, "attentive": 2.0, "polite": 1.8,
    "welcoming": 1.8, "hospitable": 1.8, "cozy": 1.8, "quiet": 1.8, "peaceful": 1.8,
    "convenient": 1.8, "central": 1.8, "beautiful": 2.0, "pristine": 2.0, "fast": 1.8,
    
    # Mild positive (weight 1.0 - 1.4)
    "good": 1.4, "nice": 1.4, "pleasant": 1.4, "fine": 1.2, "adequate": 1.0,
    "fresh": 1.5, "decent": 1.2, "sufficient": 1.0, "helpful": 1.5, "safe": 1.4,
    "working": 1.2, "functioning": 1.2, "usable": 1.2, "reliable": 1.6, "stable": 1.6
}

NEGATIVE_LEXICON = {
    # Extreme negative (weight -3.0)
    "horrible": -3.0, "terrible": -3.0, "awful": -3.0, "disgusting": -3.0, "atrocious": -3.0,
    "unacceptable": -3.0, "nightmare": -3.0, "worst": -3.0,
    
    # Strong negative (weight -2.0 to -2.5)
    "dirty": -2.5, "filthy": -2.5, "rude": -2.5, "smelly": -2.5, "unfriendly": -2.2,
    "noisy": -2.0, "broken": -2.0, "stained": -2.0, "cockroach": -2.5, "bugs": -2.0,
    "dusty": -2.0, "uncomfortable": -2.0, "poor": -2.0, "bad": -2.0, "overpriced": -2.0,
    
    # Moderate negative (weight -1.2 to -1.8)
    "slow": -1.5, "small": -1.4, "cramped": -1.6, "dark": -1.4, "cold": -1.4,
    "leaking": -1.8, "disappointing": -1.8, "disappointed": -1.8, "bland": -1.5,
    "dodgy": -1.6, "shabby": -1.8, "unhelpful": -1.8, "weak": -1.5, "lacking": -1.4
}

NEGATION_TERMS = {
    "not", "no", "never", "hardly", "scarcely", "barely", "wasn't", "weren't",
    "didn't", "don't", "doesn't", "isn't", "aren't", "couldn't", "wouldn't",
    "cannot", "can't", "without", "lack", "lacked", "lacking"
}

INTENSIFIERS = {
    "very": 1.4, "extremely": 1.6, "super": 1.5, "really": 1.3, "exceptionally": 1.6,
    "highly": 1.5, "incredibly": 1.6, "absolutely": 1.6, "totally": 1.5, "completely": 1.5,
    "so": 1.3, "extra": 1.4, "particularly": 1.4, "especially": 1.4
}

DIMINISHERS = {
    "slightly": 0.6, "somewhat": 0.7, "a bit": 0.65, "little": 0.7, "fairly": 0.8,
    "rather": 0.85, "barely": 0.5
}


class ABSAEngine:
    def __init__(self, categories: List[str] = None, keywords: Dict[str, List[str]] = None, use_transformer: bool = True):
        self.categories = categories or ASPECT_CATEGORIES
        self.keywords = keywords or ASPECT_KEYWORDS
        self.use_transformer = use_transformer
        self.transformer_model = None
        if self.use_transformer:
            try:
                from src.models.transformer_absa_model import ContextualTransformerABSA
                self.transformer_model = ContextualTransformerABSA(architecture="deberta-v3-small")
            except Exception:
                self.transformer_model = None

    def split_into_clauses_and_sentences(self, text: str) -> List[str]:
        """Splits review text into cohesive clauses based on sentence and punctuation boundaries."""
        if not text:
            return []
        # Split on period, exclamation, question mark, newline, or major conjunctions like 'but', 'however', 'although'
        raw_chunks = re.split(r'[\.\!\?\n\;]|\s+(?:but|however|although|though|whereas)\s+', str(text), flags=re.IGNORECASE)
        cleaned = [c.strip() for c in raw_chunks if c and len(c.strip()) > 2]
        return cleaned

    def detect_aspects_in_clause(self, clause: str) -> List[str]:
        """Identifies which aspect categories are referenced in a given clause."""
        matched_aspects = []
        clause_lower = f" {clause.lower()} "

        for aspect, kw_list in self.keywords.items():
            for kw in kw_list:
                # Word boundary search
                pattern = r'(?:^|\W)' + re.escape(kw) + r'(?:$|\W)'
                if re.search(pattern, clause_lower):
                    matched_aspects.append(aspect)
                    break
        return matched_aspects

    def score_clause_sentiment(self, clause: str, prior_bias: float = 0.0) -> Tuple[float, str]:
        """
        Computes sentiment polarity for a clause in [-3.0, 3.0].
        Applies negation windowing and intensifier modifiers.
        """
        words = re.findall(r'\b[a-zA-Z\']+\b', clause.lower())
        if not words and prior_bias == 0.0:
            return 0.0, "Neutral"

        clause_lower = clause.lower()

        # Handle positive idioms
        if "nothing was too much trouble" in clause_lower or "could not fault" in clause_lower or "can't complain" in clause_lower:
            return 2.5, "Positive"
        if "not bad" in clause_lower or "not too bad" in clause_lower:
            return 0.8, "Positive"

        net_score = prior_bias
        i = 0
        while i < len(words):
            word = words[i]
            multiplier = 1.0

            # Check for intensifier in previous 2 tokens
            for back_step in [1, 2]:
                if i - back_step >= 0 and words[i - back_step] in INTENSIFIERS:
                    multiplier *= INTENSIFIERS[words[i - back_step]]
                elif i - back_step >= 0 and words[i - back_step] in DIMINISHERS:
                    multiplier *= DIMINISHERS[words[i - back_step]]

            # Check for negation in previous 3 tokens
            is_negated = False
            for back_step in [1, 2, 3]:
                if i - back_step >= 0 and words[i - back_step] in NEGATION_TERMS:
                    is_negated = True
                    break

            if word in POSITIVE_LEXICON:
                val = POSITIVE_LEXICON[word] * multiplier
                net_score += (-1.2 * val if is_negated else val)
            elif word in NEGATIVE_LEXICON:
                val = NEGATIVE_LEXICON[word] * multiplier
                net_score += (0.8 * abs(val) if is_negated else val)

            i += 1

        label = "Positive" if net_score >= 0.5 else ("Negative" if net_score <= -0.5 else "Neutral")
        return max(-3.0, min(3.0, net_score)), label

    def polarity_to_5_scale(self, net_score: float, baseline_hotel_5: float = 4.0) -> float:
        """
        Converts clause polarity in [-3.0, 3.0] into standardized 0.0 to 5.0 aspect score.
        A score of 0.0 (neutral) aligns with the hotel's baseline performance.
        """
        # Neutral sits at baseline_hotel_5 (e.g. 4.0)
        # Positive pushes toward 5.0; Negative pushes toward 1.0 - 0.5
        if net_score >= 0.0:
            boost = (net_score / 3.0) * (5.0 - baseline_hotel_5)
            score_5 = baseline_hotel_5 + boost
        else:
            drop = (abs(net_score) / 3.0) * (baseline_hotel_5 - 1.0)
            score_5 = baseline_hotel_5 - drop

        return max(0.5, min(5.0, round(score_5, 2)))

    def analyze_review(self, review: Dict[str, Any], baseline_rating_10: float = 8.0) -> Dict[str, Any]:
        """
        Analyzes a single review across all 6 aspects.
        Returns a dictionary of aspect scores, sentiment labels, and extracted snippets.
        """
        baseline_5 = max(2.5, min(4.8, baseline_rating_10 / 2.0))
        aspect_results = {
            cat: {
                "score_5": baseline_5,
                "sentiment": "Neutral",
                "mentions_count": 0,
                "snippets": []
            }
            for cat in self.categories
        }

        # Analyze separate positive and negative fields if available (Booking.com format)
        pos_text = str(review.get("positive_text") or "").strip()
        neg_text = str(review.get("negative_text") or "").strip()
        full_text = str(review.get("review_text") or "").strip()

        chunks_to_analyze = []
        if pos_text:
            for c in self.split_into_clauses_and_sentences(pos_text):
                chunks_to_analyze.append((c, 0.8))  # positive prior
        if neg_text:
            for c in self.split_into_clauses_and_sentences(neg_text):
                chunks_to_analyze.append((c, -0.8)) # negative prior
        if not pos_text and not neg_text and full_text:
            for c in self.split_into_clauses_and_sentences(full_text):
                chunks_to_analyze.append((c, 0.0))

        aspect_clause_scores = {cat: [] for cat in self.categories}

        for clause, prior in chunks_to_analyze:
            matched_aspects = self.detect_aspects_in_clause(clause)
            if not matched_aspects:
                continue

            score, label = self.score_clause_sentiment(clause, prior_bias=prior)
            for asp in matched_aspects:
                # Cascaded Hybrid Escalation (Tier 1 Fast Lexicon -> Tier 2 DeBERTa Contextual Escalation)
                if self.use_transformer and self.transformer_model is not None:
                    signals = self.transformer_model.extract_contextual_pragmatic_signals(clause)
                    has_complex_pragmatics = any(v > 0 for v in signals.values())
                    if has_complex_pragmatics or abs(score) < 0.5:
                        tf_label, tf_conf = self.transformer_model.predict_sentiment(
                            clause, aspect=asp, lexical_fallback_fn=self.score_clause_sentiment
                        )
                        if tf_conf >= 0.70:
                            label = tf_label
                            score = 2.0 if tf_label == "Positive" else (-2.0 if tf_label == "Negative" else 0.0)

                aspect_clause_scores[asp].append(score)
                aspect_results[asp]["snippets"].append(clause)
                aspect_results[asp]["mentions_count"] += 1

        for cat in self.categories:
            scores = aspect_clause_scores[cat]
            if scores:
                avg_pol = sum(scores) / len(scores)
                score_5 = self.polarity_to_5_scale(avg_pol, baseline_hotel_5=baseline_5)
                label = "Positive" if score_5 >= 3.8 else ("Negative" if score_5 <= 2.6 else "Neutral")
                aspect_results[cat]["score_5"] = score_5
                aspect_results[cat]["sentiment"] = label

        return aspect_results
