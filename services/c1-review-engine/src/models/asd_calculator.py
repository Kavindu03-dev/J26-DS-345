"""
Aspect Specificity Density (ASD) Engine for Research Component 1.
Project ID: J26-DS-345
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Formulates the Aspect Specificity Density metric (SO2 / FR4) to identify
LLM (ChatGPT)-generated hotel reviews and generic astroturfing spam.

Core Research Intuition:
- Genuine human reviews mention specific, verifiable domain entities:
  dishes (hoppers, pol sambol), appliances (split AC, moka pot), landmarks (Galle Fort),
  and concrete measurements (10 mins walk, 200m, room 204).
- AI bots and promotional spam use high frequencies of generic, flowery superlatives
  (breathtaking, magnificent, extraordinary, immaculate, oasis, paradise) with near-zero entity density.
"""

import re
import math
from typing import Dict, List, Tuple, Any, Set


# Concrete Hospitality Entities in Sri Lankan Tourism
CONCRETE_FOOD_ENTITIES: Set[str] = {
    "hopper", "hoppers", "egg hopper", "egg hoppers", "string hopper", "string hoppers",
    "pol sambol", "sambol", "kottu", "curry", "dhal", "roti", "paratha",

    "rice and curry", "seafood", "prawns", "crab", "tuna", "calamari",
    "papaya", "mango", "banana", "pineapple", "passion fruit", "watermelon",
    "omelette", "toast", "pancake", "pancakes", "jam", "butter",
    "ceylon tea", "tea", "coffee", "moka pot", "espresso", "latte",
    "coconut water", "king coconut", "fresh juice", "smoothie", "buffet"
}

CONCRETE_FACILITY_ENTITIES: Set[str] = {
    "infinity pool", "plunge pool", "swimming pool", "pool", "rooftop pool",
    "balcony", "terrace", "veranda", "patio", "garden", "lawn",
    "split ac", "air conditioning", "ac unit", "ceiling fan", "fan",
    "hot water", "water heater", "geyser", "shower", "rain shower", "bathtub",
    "hair dryer", "mini fridge", "refrigerator", "kettle", "electric kettle",
    "mosquito net", "wardrobe", "closet", "hangers", "safe", "locker",
    "king bed", "queen bed", "mattress", "linen", "towels", "pillows",
    "generator", "power backup", "solar", "parking", "garage", "tuk tuk"
}

CONCRETE_LANDMARK_ENTITIES: Set[str] = {
    "galle fort", "lighthouse", "beach", "beachfront", "coral reef", "harbor",
    "railway station", "train station", "bus stand", "bus stop", "main road",
    "sigiriya", "pidurangala", "lion rock", "nine arches", "nine arch bridge",
    "little adam's peak", "adams peak", "ella rock", "ravana falls",
    "temple of the tooth", "kandy lake", "botanical garden", "peradeniya",
    "horton plains", "worlds end", "gregory lake", "tea factory", "plantation",
    "yala", "udawalawe", "wilpattu", "minneriya", "safari park",
    "mirissa", "unawatuna", "weligama", "bentota", "hikkaduwa", "arugam bay",
    "trincomalee", "nilaveli", "jaffna fort", "nallur", "knuckles"
}

# Generic Hyperbolic Superlatives & Fluff (frequent in zero-shot AI reviews)
GENERIC_FLUFF_ADJECTIVES: Set[str] = {
    "amazing", "wonderful", "fantastic", "breathtaking", "unmatched",
    "extraordinary", "unbelievable", "magnificent", "splendid", "superb",
    "flawless", "impeccable", "delightful", "incredible", "charming",
    "mesmerizing", "heavenly", "paradise", "magical", "top-notch",
    "world-class", "oasis", "gem", "haven", "perfection", "spectacular",
    "fabulous", "gorgeous", "phenomenal", "sublime", "stellar", "peerless",
    "marvelous", "sensational", "impressive", "outstanding", "exceptional"
}

GENERIC_FLUFF_ADVERBS: Set[str] = {
    "absolutely", "truly", "deeply", "utterly", "extremely", "incredibly",
    "thoroughly", "exceptionally", "remarkably", "immensely", "profoundly",
    "unquestionably", "undeniably", "genuinely", "exceedingly"
}


class AspectSpecificityDensityCalculator:
    """
    Computes Aspect Specificity Density (ASD) to catch LLM-generated
    or generic astroturfing reviews.
    """

    def __init__(self,
                 ai_spam_asd_threshold: float = 0.25,
                 min_word_count_for_ai_test: int = 15):
        self.ai_spam_threshold = ai_spam_asd_threshold
        self.min_word_count = min_word_count_for_ai_test

        # Compile regexes for fast entity & measurement matching
        self.measurement_regex = re.compile(
            r'\b(\d+(?:\.\d+)?\s*(?:min|mins|minute|minutes|hour|hours|km|m|meters|meter|miles|ft|yards|nights?|days?|am|pm|lkr|usd|\$|room|floor))\b',
            re.IGNORECASE
        )
        self.room_number_regex = re.compile(r'\b(?:room|suite|villa|cottage)\s*(?:no\.?|#)?\s*(\d+[a-z]?)\b', re.IGNORECASE)

    def extract_concrete_entities(self, text_lower: str) -> Tuple[List[str], int]:
        """Identifies tangible food, facility, and landmark entities in text."""
        detected = []
        all_concrete = sorted(
            list(CONCRETE_FOOD_ENTITIES | CONCRETE_FACILITY_ENTITIES | CONCRETE_LANDMARK_ENTITIES),
            key=len,
            reverse=True
        )

        for entity in all_concrete:
            pattern = r'(?:^|\W)' + re.escape(entity) + r'(?:$|\W)'
            if re.search(pattern, text_lower):
                detected.append(entity)

        return detected, len(detected)


    def extract_measurements(self, text: str) -> Tuple[List[str], int]:
        """Detects physical numbers, distances, walk times, and room codes."""
        measurements = []
        for m in self.measurement_regex.findall(text):
            measurements.append(m.strip())
        for r in self.room_number_regex.findall(text):
            measurements.append(f"room {r}")
        return measurements, len(measurements)

    def extract_generic_fluff(self, text_lower: str) -> Tuple[List[str], int]:
        """Finds hyperbolic adjectives and adverbs common in AI / fake reviews."""
        words = re.findall(r'\b[a-z\-]+\b', text_lower)
        fluff_found = []
        for w in words:
            if w in GENERIC_FLUFF_ADJECTIVES or w in GENERIC_FLUFF_ADVERBS:
                fluff_found.append(w)
        return fluff_found, len(fluff_found)

    def calculate_asd(self, review_text: str) -> Dict[str, Any]:
        """
        Calculates Aspect Specificity Density for a given review.

        Formula:
            Entity_Weight = N_entities + (0.6 * N_measurements)
            Fluff_Weight  = max(1.0, N_fluff)
            Raw_ASD       = (Entity_Weight / Fluff_Weight) * log_scaling(word_count)
            ASD_Score     = clip(Raw_ASD / 2.0, 0.02, 1.0)
        """
        if not review_text or not isinstance(review_text, str):
            return {
                "asd_score": 0.10,
                "concrete_entities_count": 0,
                "measurements_count": 0,
                "generic_fluff_count": 0,
                "detected_entities": [],
                "detected_fluff": [],
                "detected_measurements": [],
                "is_likely_ai_spam": False,
                "word_count": 0
            }

        text_clean = review_text.strip()
        text_lower = text_clean.lower()
        words = text_clean.split()
        word_count = len(words)

        # Extract features
        entities, n_entities = self.extract_concrete_entities(text_lower)
        measurements, n_measurements = self.extract_measurements(text_clean)
        fluff, n_fluff = self.extract_generic_fluff(text_lower)

        # Mathematical Formulation
        entity_signal = float(n_entities) + (0.6 * float(n_measurements))
        fluff_penalty = max(1.0, float(n_fluff))

        # Density ratio
        base_density = entity_signal / fluff_penalty

        # Length normalization factor
        length_factor = math.log(1.0 + min(150, max(5, word_count))) / math.log(1.0 + 150)

        raw_asd = (base_density * 0.70) + (length_factor * 0.30)
        # Bounded between 0.02 and 1.0
        asd_score = max(0.02, min(1.0, round(raw_asd / 2.0, 4)))

        # Classification Heuristic for LLM / AI-generated review:
        is_ai_spam = (
            word_count >= 8
            and n_fluff >= 2
            and n_entities == 0
            and n_measurements == 0
        ) or (
            word_count >= 15
            and asd_score < self.ai_spam_threshold
            and n_fluff > (n_entities + n_measurements) * 2
        )

        return {
            "asd_score": asd_score,
            "concrete_entities_count": n_entities,
            "measurements_count": n_measurements,
            "generic_fluff_count": n_fluff,
            "detected_entities": entities,
            "detected_fluff": fluff,
            "detected_measurements": measurements,
            "is_likely_ai_spam": bool(is_ai_spam),
            "word_count": word_count
        }
