"""
Global configuration file for the Research-Grade Tourism Review & Destination Scraper.
Research Component 1: Trust-Aware Review Analysis Engine (Project ID: J26-DS-345)
"""

import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
LOG_DIR = DATA_DIR / "logs"


# Ensure all directories exist
for d in [RAW_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
HOTELS_SEED_FILE = RAW_DIR / "hotels_seed.csv"
DESTINATIONS_SEED_FILE = RAW_DIR / "destinations_seed.csv"
REVIEWS_BOOKING_FILE = RAW_DIR / "reviews_raw_booking.csv"
REVIEWS_TRIPADVISOR_FILE = RAW_DIR / "reviews_raw_tripadvisor.csv"
REVIEWS_GOOGLEMAPS_FILE = RAW_DIR / "reviews_raw_googlemaps.csv"
MASTER_DATASET_FILE = PROCESSED_DIR / "research_dataset_unified.csv"
DATA_REPORT_FILE = PROCESSED_DIR / "data_summary_report.json"
CHECKPOINT_FILE = CHECKPOINTS_DIR / "scraper_state.json"

# Sampling & Extraction Defaults (Target: 900-1000 hotels, 35k-50k reviews)
DEFAULT_HOTELS_PER_DISTRICT = 42
DEFAULT_REVIEWS_PER_ENTITY = 60
DEFAULT_MAX_PAGES_PER_ENTITY = 6
MAX_DISCOVERY_PAGES = 4

# Browser Automation & Anti-Bot Settings
DEFAULT_HEADLESS = True
BROWSER_TIMEOUT_MS = 60000
PAGE_LOAD_DELAY_MIN = 2.0
PAGE_LOAD_DELAY_MAX = 5.0
ACTION_DELAY_MIN = 1.0
ACTION_DELAY_MAX = 2.5
LONG_PAUSE_MIN = 15.0
LONG_PAUSE_MAX = 30.0

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# 1. Hotel Entity Metadata Fieldnames (Scraped Once per Property)
HOTEL_ENTITY_FIELDS = [
    "hotel_id",
    "hotel_name",
    "hotel_url",
    "latitude",
    "longitude",
    "address_street",
    "city",
    "district",
    "province",
    "property_type",
    "star_rating",
    "price_tier",
    "avg_price_per_night",
    "amenities",
    "platform_overall_rating",
    "platform_total_reviews_count"
]

# Destination Entity Metadata Fieldnames
DESTINATION_ENTITY_FIELDS = [
    "destination_id",
    "destination_name",
    "destination_url",
    "category",
    "latitude",
    "longitude",
    "city",
    "district",
    "province",
    "description",
    "platform_overall_rating",
    "platform_total_reviews_count"
]

# 2. Complete 35-Field Unified Research Dataset Schema
# (Sections 1-4 Scraped/Extracted; Section 5 Initialized for Model Pipeline)
RESEARCH_DATASET_FIELDS = [
    # 1. Hotel Entity Metadata (16 fields)
    "hotel_id",
    "hotel_name",
    "hotel_url",
    "latitude",
    "longitude",
    "address_street",
    "city",
    "district",
    "province",
    "property_type",
    "star_rating",
    "price_tier",
    "avg_price_per_night",
    "amenities",
    "platform_overall_rating",
    "platform_total_reviews_count",
    
    # 2. Core Review Fields (9 fields)
    "review_id",
    "source_platform",
    "review_date",
    "rating_score",
    "review_title",
    "positive_text",
    "negative_text",
    "review_text",
    
    # 3. Reviewer & Behavioral Features for Trust Scoring (6 fields)
    "author_id",
    "author_country",
    "author_total_reviews",
    "author_account_age",
    "helpful_votes",
    "is_verified_stay",
    
    # 4. Stay & Contextual Metadata for Recommender (4 fields)
    "stay_date",
    "nights_stayed",
    "traveler_type",
    "room_type",
    
    # 5. Derived & Output Fields (11 fields for downstream models)
    "text_length_words",
    "sentiment_rating_disparity",
    "trust_score",
    "is_suspicious",
    "aspect_food_sentiment",
    "aspect_staff_sentiment",
    "aspect_cleanliness_sentiment",
    "aspect_room_sentiment",
    "aspect_location_sentiment",
    "aspect_wifi_sentiment",
    "trust_weighted_aspect_score"
]
