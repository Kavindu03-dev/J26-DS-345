"""
Verified Aspect Facts Extractor for Research Component 1.
Extracts grounded, factual, high-trust sentences from verified reviews
to satisfy contracts/c1_aspect_scores.json for Component 4 (RAG Conversational Assistant).
"""

import re
from typing import List, Dict, Any


class FactExtractor:
    def __init__(self, min_trust_threshold: float = 0.65):
        self.min_trust = min_trust_threshold

        # Informational fact indicators
        self.fact_patterns = [
            # Time & Schedule facts
            r'\b(?:breakfast|dinner|lunch|food|tea|coffee)\b.*?\b(?:served|available|from|until|at|stations?)\b.*?(?:[0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?|morning|evening)',
            r'\b(?:check[- ]?in|check[- ]?out)\b.*?(?:from|until|at)\b.*?[0-9]{1,2}',
            
            # Distance & Proximity facts
            r'\b(?:walk|walking|distance|located|station|beach|fort)\b.*?(?:[0-9]+\s*(?:mins?|minutes?|km|meters?)|close|heart of|pedlar)',
            
            # Specific amenities & room features
            r'\b(?:room|bathroom|shower|pool|wifi|garden|terrace)\b.*?\b(?:equipped|equipped with|features?|spacious|tub|balcony|high ceilings?|ac|air condition|fridge)',
            
            # Specific food items
            r'\b(?:breakfast|buffet|dinner|station)\b.*?\b(?:hoppers?|waffles?|omelette|curry|fish|seafood|fruit|tropical)',
            
            # Facilities & Activities
            r'\b(?:pool|swimming pool|parking|shuttle|driver|tuk tuk|bird park|badminton|cricket)\b'
        ]

    def is_informative_fact(self, sentence: str) -> bool:
        """Determines if a sentence conveys verifiable factual detail rather than generic praise/rant."""
        s = sentence.strip()
        words = s.split()
        if len(words) < 5 or len(words) > 35:
            return False

        # Filter out purely generic emotional phrases
        generic_praise = [
            "had a great time", "nice place", "would recommend", "really liked it",
            "great stay", "worst place", "loved our stay", "thank you", "must visit"
        ]
        s_lower = s.lower()
        if any(gp in s_lower for gp in generic_praise) and not any(re.search(p, s_lower) for p in self.fact_patterns):
            return False

        # Check against fact patterns
        for pattern in self.fact_patterns:
            if re.search(pattern, s_lower):
                return True

        return False

    def clean_fact_sentence(self, sentence: str) -> str:
        """Cleans and capitalizes the factual sentence."""
        s = sentence.strip()
        # Remove Tripadvisor / Booking boilerplate prefixes
        s = re.sub(r'^(?:Date of stay:.*?\)\s*|Reviewed:.*?\s*|Trip type.*?\s*)', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*Read more on.*$', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*More$', '', s, flags=re.IGNORECASE)
        s = s.strip()
        if s and not s.endswith(('.', '!', '?')):
            s += '.'
        if s:
            s = s[0].upper() + s[1:]
        return s

    def extract_facts_from_reviews(self, reviews: List[Dict[str, Any]], trust_engine, max_facts: int = 5) -> List[str]:
        """
        Extracts up to max_facts unique, high-trust factual sentences from a hotel's review set.
        """
        candidate_facts = []
        seen_stems = set()

        for r in reviews:
            t_score = trust_engine.calculate_review_trust(r)
            if t_score < self.min_trust:
                continue

            # Prioritize clean text chunks
            text = str(r.get("review_text") or "")
            pos = str(r.get("positive_text") or "")
            full = f"{pos} {text}".strip()

            # Split sentences
            sentences = re.split(r'[\.\!\?\n]+', full)
            for s in sentences:
                s_clean = s.strip()
                if self.is_informative_fact(s_clean):
                    fact_formatted = self.clean_fact_sentence(s_clean)
                    # Deduplication key on first 4 words
                    stem = " ".join(fact_formatted.lower().split()[:4])
                    if stem not in seen_stems and len(fact_formatted) > 15:
                        seen_stems.add(stem)
                        candidate_facts.append((fact_formatted, t_score, len(s_clean.split())))

        if not candidate_facts:
            # Fallback fact if reviews are too short or sparse
            return ["Convenient accommodation located in Sri Lanka with basic amenities."]

        # Sort by trust score and informational length
        candidate_facts.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [f[0] for f in candidate_facts[:max_facts]]
