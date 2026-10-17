"""
Language Validation & English Text Filter Module.
Filters out non-English reviews using langdetect with fast heuristic fallbacks.
"""

import re
import sys
from pathlib import Path

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "hotel", "room", "stay", "clean", "staff", "breakfast", "good", "great",
    "place", "view", "location", "nice", "service", "food", "pool", "bed"
}

def is_english_heuristic(text: str) -> bool:
    """Fast lexical heuristic fallback to check if a review is in English."""
    if not text:
        return False
    
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    if not words:
        return False
    
    match_count = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
    ratio = match_count / len(words)
    return ratio >= 0.12 or match_count >= 3

def is_english(text: str) -> bool:
    """
    Validates if the review text is written in English.
    Uses langdetect where available, falling back to lexical heuristics.
    """
    if not text or len(text.strip()) < 5:
        return False
    
    clean = " ".join(text.split())
    
    if LANGDETECT_AVAILABLE:
        try:
            detected_lang = detect(clean)
            if detected_lang == "en":
                return True
        except Exception:
            pass
            
    return is_english_heuristic(clean)
