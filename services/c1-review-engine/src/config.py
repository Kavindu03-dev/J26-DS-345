"""
Configuration for Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine.
Maintains data paths, aspect definitions, and hyper-parameters for trust & time-decay scoring.
"""

from pathlib import Path

# Data Directories
ENGINE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ENGINE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
MODELS_DIR = ENGINE_DIR / "models"

# Primary Raw Data Files
HOTELS_SEED_FILE = (RAW_DATA_DIR / "hotels_seed.csv") if (RAW_DATA_DIR / "hotels_seed.csv").exists() else (DATA_DIR / "hotels_seed.csv")
DESTINATIONS_SEED_FILE = (RAW_DATA_DIR / "destinations_seed.csv") if (RAW_DATA_DIR / "destinations_seed.csv").exists() else (DATA_DIR / "destinations_seed.csv")
REVIEWS_BOOKING_FILE = (RAW_DATA_DIR / "reviews_raw_booking.csv") if (RAW_DATA_DIR / "reviews_raw_booking.csv").exists() else (DATA_DIR / "reviews_raw_booking.csv")
REVIEWS_GOOGLEMAPS_FILE = (RAW_DATA_DIR / "reviews_raw_googlemaps.csv") if (RAW_DATA_DIR / "reviews_raw_googlemaps.csv").exists() else (DATA_DIR / "reviews_raw_googlemaps.csv")
REVIEWS_TRIPADVISOR_FILE = (RAW_DATA_DIR / "reviews_raw_tripadvisor.csv") if (RAW_DATA_DIR / "reviews_raw_tripadvisor.csv").exists() else (DATA_DIR / "reviews_raw_tripadvisor.csv")

# Processed & Unified Datasets
MASTER_DATASET_FILE = (PROCESSED_DATA_DIR / "research_dataset_unified.csv") if (PROCESSED_DATA_DIR / "research_dataset_unified.csv").exists() else (DATA_DIR / "research_dataset_unified.csv")
REVIEWS_BY_HOTEL_JSON = (PROCESSED_DATA_DIR / "reviews_by_hotel.json") if (PROCESSED_DATA_DIR / "reviews_by_hotel.json").exists() else (DATA_DIR / "reviews_by_hotel.json")
DATA_SUMMARY_REPORT_FILE = (PROCESSED_DATA_DIR / "data_summary_report.json") if (PROCESSED_DATA_DIR / "data_summary_report.json").exists() else (DATA_DIR / "data_summary_report.json")
ENRICHED_REVIEWS_FILE = PROCESSED_DATA_DIR / "reviews_enriched.csv"


# Pre-computed Profiles & Benchmark Reports
HOTEL_PROFILES_JSON = DATA_DIR / "hotel_aspect_profiles.json"
HOTEL_PROFILES_CSV = DATA_DIR / "hotel_aspect_profiles.csv"
EVALUATION_METRICS_REPORT_FILE = DATA_DIR / "evaluation_metrics_report.json"


# Hotel Profile Schema (for Component 2 Recommender)
HOTEL_PROFILE_FIELDS = [
    "hotel_id",
    "hotel_name",
    "city",
    "district",
    "province",
    "latitude",
    "longitude",
    "property_type",
    "star_rating",
    "price_tier",
    "avg_price_per_night",
    "platform_overall_rating",
    "overall_trust_score",
    "time_decay_rating",
    "aspect_food",
    "aspect_staff",
    "aspect_room",
    "aspect_cleanliness",
    "aspect_location",
    "aspect_wifi",
    "total_reviews",
    "suspicious_reviews_count",
    "verified_stay_ratio",
    "top_verified_facts",
    "last_updated"
]

# 35-Field Unified Research Review Schema
RESEARCH_DATASET_FIELDS = [
    "hotel_id", "hotel_name", "hotel_url", "latitude", "longitude",
    "address_street", "city", "district", "province", "property_type",
    "star_rating", "price_tier", "avg_price_per_night", "amenities",
    "platform_overall_rating", "platform_total_reviews_count",
    "review_id", "source_platform", "review_date", "rating_score",
    "review_title", "positive_text", "negative_text", "review_text",
    "author_id", "author_country", "author_total_reviews", "author_account_age",
    "helpful_votes", "is_verified_stay", "stay_date", "nights_stayed",
    "traveler_type", "room_type", "text_length_words",
    "sentiment_rating_disparity", "trust_score", "is_suspicious",
    "aspect_food_sentiment", "aspect_staff_sentiment", "aspect_cleanliness_sentiment",
    "aspect_room_sentiment", "aspect_location_sentiment", "aspect_wifi_sentiment",
    "trust_weighted_aspect_score"
]

# Aspects defined in contracts/c1_aspect_scores.json
ASPECT_CATEGORIES = [
    "food",
    "staff",
    "room",
    "cleanliness",
    "location",
    "wifi"
]

# Aspect keyword seeds for hospitality ABSA
ASPECT_KEYWORDS = {
    "food": [
        "food", "breakfast", "dinner", "lunch", "meal", "buffet", "restaurant",
        "delicious", "tasty", "coffee", "tea", "omelette", "curry", "hopper",
        "chef", "cuisine", "dishes", "menu", "seafood", "eating"
    ],
    "staff": [
        "staff", "service", "friendly", "helpful", "host", "manager", "reception",
        "welcome", "hospitality", "attentive", "polite", "courteous", "waiter",
        "rude", "unfriendly", "prompt", "assistance", "care"
    ],
    "room": [
        "room", "bed", "bathroom", "shower", "balcony", "view", "ac", "air condition",
        "spacious", "comfortable", "furniture", "pillow", "mattress", "ceiling fan",
        "hot water", "toilet", "amenities", "tub", "suite"
    ],
    "cleanliness": [
        "clean", "cleanliness", "spotless", "dirty", "dusty", "stain", "smell",
        "hygiene", "tidy", "neat", "maintained", "cockroach", "bugs", "linen",
        "towels", "fresh"
    ],
    "location": [
        "location", "beach", "fort", "center", "centre", "walking distance", "central",
        "close", "accessible", "train station", "bus stand", "surroundings",
        "scenery", "quiet", "peaceful", "noisy", "traffic", "convenient"
    ],
    "wifi": [
        "wifi", "wi-fi", "internet", "connection", "signal", "speed", "fast wifi",
        "poor wifi", "slow internet", "network"
    ]
}

# Trust Engine Hyper-parameters
TRUST_PARAMS = {
    "half_life_days": 365.0,           # Half-life lambda parameter for time decay
    "verified_stay_weight": 1.0,       # Multiplier for verified platform reviews
    "unverified_stay_weight": 0.80,    # Multiplier for open platform reviews
    "min_trust_score": 0.05,
    "max_trust_score": 1.0,
    "suspicious_trust_threshold": 0.22, # Flagged as suspicious if trust < threshold
    "suspicious_short_word_len": 6,    # Short unverified extreme reviews flagged
}
