"""
Utility helpers for scraping, stealth, delay, hash generation, and checkpoint state management.
"""

import os
import sys
import json
import time
import hashlib
import random
import re
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    RAW_DIR, LOG_DIR, CHECKPOINT_FILE,
    PAGE_LOAD_DELAY_MIN, PAGE_LOAD_DELAY_MAX,
    ACTION_DELAY_MIN, ACTION_DELAY_MAX,
    HOTELS_SEED_FILE, REVIEWS_BOOKING_FILE
)

def generate_entity_id(entity_name: str, city: str, prefix: str = "HTL") -> str:
    """Generate a clean deterministic ID for a hotel or destination."""
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', entity_name)[:8].upper()
    clean_city = re.sub(r'[^a-zA-Z0-9]', '', city)[:3].upper()
    hash_suffix = hashlib.md5(f"{entity_name}_{city}".encode("utf-8")).hexdigest()[:4].upper()
    return f"{prefix}_{clean_city}_{clean_name}_{hash_suffix}"

def generate_review_id(hotel_id: str, author: str, review_date: str, review_title: str) -> str:
    """Generate a unique SHA-256 hash for deduplicating individual reviews."""
    raw_str = f"{hotel_id}|{author.strip().lower()}|{review_date.strip()}|{review_title.strip().lower()}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

def anonymize_author(author_name: str, hotel_id: str = "") -> str:
    """
    Hash reviewer name to ensure strict PII compliance (FR3, NFR1).
    Uses a deterministic salted SHA-256 hash so user identity is fully protected
    while preserving cross-review behavioral patterns for the trust engine.
    """
    if not author_name or author_name.strip().lower() in ["guest", "google user", "tripadvisor traveler", "traveler", "unknown", ""]:
        return "USR_ANONYMOUS"
    salt = "SLIIT_J26_DS_345_SALT"
    raw = f"{salt}_{author_name.strip().lower()}"
    return f"USR_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:10].upper()}"

def compute_time_decay_weight(review_date_str: str, half_life_days: float = 365.0) -> float:
    """
    Computes mathematical exponential time-decay weight for a review (SO4).
    W_time = exp(-lambda * delta_days), where lambda = ln(2) / half_life_days.
    """
    import math
    from datetime import datetime
    if not review_date_str:
        return 0.5
    try:
        # Match common date patterns, e.g., 'August 24, 2026' or '2026-08-24' or 'March 2026'
        clean_date = review_date_str.replace("Reviewed:", "").replace("·", "").strip()
        review_dt = None
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%B %Y", "%b %Y"):
            try:
                review_dt = datetime.strptime(clean_date, fmt)
                break
            except ValueError:
                continue
        if not review_dt:
            return 0.5
        current_dt = datetime.now()
        delta_days = max(0, (current_dt - review_dt).days)
        decay_lambda = math.log(2) / half_life_days
        return round(math.exp(-decay_lambda * delta_days), 4)
    except Exception:
        return 0.5


def random_delay(min_sec=ACTION_DELAY_MIN, max_sec=ACTION_DELAY_MAX):
    """Gaussian-like random delay to simulate human pauses."""
    mid = (min_sec + max_sec) / 2.0
    std = (max_sec - min_sec) / 4.0
    delay = max(min_sec, min(max_sec, random.gauss(mid, std)))
    time.sleep(delay)

def long_delay(min_sec=PAGE_LOAD_DELAY_MIN, max_sec=PAGE_LOAD_DELAY_MAX):
    """Longer pause between major navigation actions."""
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)

def clean_text(text: str) -> str:
    """Normalize whitespace, tabs, and newlines in extracted text."""
    if not text:
        return ""
    # Strip HTML tags if any remain
    text = re.sub(r'<[^>]+>', ' ', str(text))
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())

def parse_price(price_str: str):
    """Extract numeric value and currency from price string."""
    if not price_str:
        return 0.0, "USD"
    
    clean = price_str.replace(",", "").strip()
    match = re.search(r'([0-9]+(\.[0-9]+)?)', clean)
    val = float(match.group(1)) if match else 0.0
    
    curr = "LKR" if "LKR" in price_str or "Rs" in price_str else "USD"
    return val, curr

def apply_stealth(page):
    """Apply stealth plugin to Playwright page (compatible with v1 and v2)."""
    try:
        from playwright_stealth import Stealth
        Stealth().apply_stealth_sync(page)
    except (ImportError, AttributeError):
        try:
            from playwright_stealth import stealth_sync
            stealth_sync(page)
        except ImportError:
            pass

class CheckpointManager:
    """Manages scraper checkpointing for zero-loss pause and resume."""
    def __init__(self, checkpoint_path=CHECKPOINT_FILE):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self):
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "completed_districts": [],
            "completed_hotels": [],
            "completed_destinations": [],
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def save(self):
        self.state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def is_hotel_completed(self, hotel_url: str) -> bool:
        return hotel_url in self.state.get("completed_hotels", [])

    def mark_hotel_completed(self, hotel_url: str):
        if "completed_hotels" not in self.state:
            self.state["completed_hotels"] = []
        if hotel_url not in self.state["completed_hotels"]:
            self.state["completed_hotels"].append(hotel_url)
            self.save()

    def is_district_completed(self, district: str) -> bool:
        return district in self.state.get("completed_districts", [])

    def mark_district_completed(self, district: str):
        if "completed_districts" not in self.state:
            self.state["completed_districts"] = []
        if district not in self.state["completed_districts"]:
            self.state["completed_districts"].append(district)
            self.save()
