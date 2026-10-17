"""
Dataset Normalization, Quality Audit & Master Export Engine.
Unifies multi-source reviews, validates English text, removes duplicates,
and produces the master research dataset and statistical summary report.
"""

import csv
import sys
import json
import re
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    HOTELS_SEED_FILE,
    REVIEWS_BOOKING_FILE, REVIEWS_TRIPADVISOR_FILE, REVIEWS_GOOGLEMAPS_FILE,
    MASTER_DATASET_FILE, DATA_REPORT_FILE,
    RESEARCH_DATASET_FIELDS
)
from src.processing.language_filter import is_english
from src.discovery.geo_seeds import get_fallback_coordinates
from src.utils import generate_entity_id, generate_review_id, clean_text, anonymize_author

def load_hotel_entities_map() -> dict:
    """Build a lookup map of hotel_url/hotel_name -> hotel metadata dictionary."""
    entities_map = {}
    
    if HOTELS_SEED_FILE.exists():
        with open(HOTELS_SEED_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("hotel_url"):
                    entities_map[row["hotel_url"]] = row
                if row.get("hotel_name"):
                    entities_map[row["hotel_name"].lower()] = row
                    
    return entities_map

def normalize_and_export_dataset():
    """
    Consolidates all raw review files, enforces schema, applies deduplication,
    filters for English text, and exports the unified research dataset.
    """
    print("=================================================================")
    print("      Starting Dataset Normalization & Export Engine             ")
    print("=================================================================")

    entities_map = load_hotel_entities_map()
    print(f"Loaded {len(entities_map)} hotel entities in metadata lookup index.")

    raw_files = [
        ("Booking.com", REVIEWS_BOOKING_FILE),
        ("TripAdvisor", REVIEWS_TRIPADVISOR_FILE),
        ("Google Maps", REVIEWS_GOOGLEMAPS_FILE)
    ]

    seen_review_ids = set()
    master_records = []
    skipped_non_english = 0
    skipped_duplicates = 0

    for source_name, f_path in raw_files:
        if not f_path.exists():
            continue
            
        print(f"Reading {source_name} data from: {f_path.name}")
        with open(f_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hotel_name = clean_text(row.get("hotel_name", "Unknown Hotel"))
                hotel_url = row.get("hotel_url", "")
                
                # Fetch entity metadata
                entity_meta = entities_map.get(hotel_url) or entities_map.get(hotel_name.lower()) or {}
                city = entity_meta.get("city") or row.get("city") or "Sri Lanka"
                hotel_id = entity_meta.get("hotel_id") or generate_entity_id(hotel_name, city)
                
                # Full review text
                pos_text = clean_text(row.get("positive_text", ""))
                neg_text = clean_text(row.get("negative_text", ""))
                title = clean_text(row.get("review_title", ""))
                raw_text = clean_text(row.get("review_text", ""))

                if not raw_text:
                    parts = []
                    if title: parts.append(title)
                    if pos_text: parts.append(pos_text)
                    if neg_text: parts.append(neg_text)
                    full_text = " ".join(parts)
                else:
                    full_text = raw_text

                if not full_text:
                    continue

                # English language validation
                if not is_english(full_text):
                    skipped_non_english += 1
                    continue

                author_name = row.get("author_id") or row.get("author", "Guest")
                rev_date = row.get("review_date", "2026")
                
                # Deduplication
                rev_id = row.get("review_id") or generate_review_id(hotel_id, author_name, rev_date, title or full_text[:20])
                if rev_id in seen_review_ids:
                    skipped_duplicates += 1
                    continue
                seen_review_ids.add(rev_id)

                # Parse rating score
                score_str = str(row.get("rating_score", "8.0"))
                match = re.search(r'([0-9]+(\.[0-9]+)?)', score_str)
                score_val = float(match.group(1)) if match else 8.0

                # Author review count
                auth_rev_str = str(row.get("author_total_reviews") or row.get("author_reviews_count", "1"))
                auth_match = re.search(r'([0-9]+)', auth_rev_str)
                author_reviews_int = int(auth_match.group(1)) if auth_match else 1

                # Coordinates validation (never 0.0)
                lat = float(entity_meta.get("latitude", 0.0) or row.get("latitude", 0.0) or 0.0)
                lng = float(entity_meta.get("longitude", 0.0) or row.get("longitude", 0.0) or 0.0)
                if lat == 0.0 or lng == 0.0:
                    fb_lat, fb_lng = get_fallback_coordinates(city, entity_meta.get("district", city))
                    lat = fb_lat
                    lng = fb_lng

                price_val = float(entity_meta.get("avg_price_per_night", 0.0) or row.get("avg_price_per_night", 0.0) or 0.0)
                if price_val <= 0.0:
                    tier = entity_meta.get("price_tier", "Mid-Range")
                    price_val = 28.0 if tier == "Budget" else (145.0 if tier == "Luxury" else 55.0)

                # Construct unified record
                unified_row = {
                    "hotel_id": hotel_id,
                    "hotel_name": hotel_name,
                    "hotel_url": hotel_url or entity_meta.get("hotel_url", ""),
                    "latitude": lat,
                    "longitude": lng,
                    "address_street": entity_meta.get("address_street", city),
                    "city": city,
                    "district": entity_meta.get("district", city),
                    "province": entity_meta.get("province", "Sri Lanka"),
                    "property_type": entity_meta.get("property_type", "Hotel"),
                    "star_rating": float(entity_meta.get("star_rating", 3.0) or 3.0),
                    "price_tier": entity_meta.get("price_tier", "Mid-Range"),
                    "avg_price_per_night": price_val,
                    "amenities": entity_meta.get("amenities", "Free Wi-Fi, Air conditioning, Private Bathroom"),
                    "platform_overall_rating": float(entity_meta.get("platform_overall_rating", 8.0) or 8.0),
                    "platform_total_reviews_count": int(entity_meta.get("platform_total_reviews_count", 100) or 100),
                    "review_id": rev_id,
                    "source_platform": row.get("source_platform", source_name.split()[0]),
                    "review_date": rev_date,
                    "rating_score": score_val,
                    "review_title": title,
                    "positive_text": pos_text,
                    "negative_text": neg_text,
                    "review_text": full_text,
                    "author_id": anonymize_author(author_name),
                    "author_country": row.get("author_country", "Unknown"),
                    "author_total_reviews": author_reviews_int,
                    "author_account_age": row.get("author_account_age", "Verified Guest"),
                    "helpful_votes": int(row.get("helpful_votes", 0) or 0),
                    "is_verified_stay": bool(row.get("is_verified_stay", True)),
                    "stay_date": row.get("stay_date", ""),
                    "nights_stayed": row.get("nights_stayed", ""),
                    "traveler_type": row.get("traveler_type", "Solo traveller"),
                    "room_type": row.get("room_type", ""),
                    "text_length_words": len(full_text.split()),
                    "sentiment_rating_disparity": round(abs(score_val - float(entity_meta.get("platform_overall_rating", 8.0))), 2),
                    "trust_score": float(row.get("trust_score", 0.0) or 0.0),
                    "is_suspicious": int(row.get("is_suspicious", 0) or 0),
                    "aspect_food_sentiment": row.get("aspect_food_sentiment", "Neutral"),
                    "aspect_staff_sentiment": row.get("aspect_staff_sentiment", "Neutral"),
                    "aspect_cleanliness_sentiment": row.get("aspect_cleanliness_sentiment", "Neutral"),
                    "aspect_room_sentiment": row.get("aspect_room_sentiment", "Neutral"),
                    "aspect_location_sentiment": row.get("aspect_location_sentiment", "Neutral"),
                    "aspect_wifi_sentiment": row.get("aspect_wifi_sentiment", "Neutral"),
                    "trust_weighted_aspect_score": float(row.get("trust_weighted_aspect_score", 0.0) or 0.0)
                }
                master_records.append(unified_row)

    # Sort records so all reviews for the same hotel are grouped together contiguously
    master_records.sort(key=lambda r: (r.get("hotel_name", "").lower(), r.get("source_platform", ""), r.get("review_date", "")))

    # Write unified dataset to CSV
    MASTER_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_DATASET_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESEARCH_DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(master_records)

    # Also build and export hotel-grouped review dictionary (reviews_by_hotel.json)
    grouped_by_hotel = {}
    for r in master_records:
        h_id = r["hotel_id"]
        if h_id not in grouped_by_hotel:
            grouped_by_hotel[h_id] = {
                "hotel_id": h_id,
                "hotel_name": r["hotel_name"],
                "city": r["city"],
                "district": r["district"],
                "star_rating": r["star_rating"],
                "price_tier": r["price_tier"],
                "total_reviews": 0,
                "platform_breakdown": Counter(),
                "reviews": []
            }
        grouped_by_hotel[h_id]["total_reviews"] += 1
        grouped_by_hotel[h_id]["platform_breakdown"][r["source_platform"]] += 1
        grouped_by_hotel[h_id]["reviews"].append(r)

    # Convert Counter to dict for JSON serialization
    for h_id in grouped_by_hotel:
        grouped_by_hotel[h_id]["platform_breakdown"] = dict(grouped_by_hotel[h_id]["platform_breakdown"])

    grouped_json_file = MASTER_DATASET_FILE.parent / "reviews_by_hotel.json"
    with open(grouped_json_file, "w", encoding="utf-8") as f:
        json.dump(grouped_by_hotel, f, indent=2, ensure_ascii=False)

    # Compute Statistical Quality Audit
    districts_counter = Counter(r["district"] for r in master_records)
    price_tier_counter = Counter(r["price_tier"] for r in master_records)
    platform_counter = Counter(r["source_platform"] for r in master_records)
    word_counts = [r["text_length_words"] for r in master_records] if master_records else [0]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

    audit_report = {
        "dataset_name": "SLIIT Tourism Review Research Master Dataset",
        "component": "Component 1: Trust-Aware Review Analysis Engine",
        "total_unique_reviews": len(master_records),
        "total_unique_hotels": len(set(r["hotel_id"] for r in master_records)),
        "skipped_non_english_reviews": skipped_non_english,
        "skipped_duplicate_reviews": skipped_duplicates,
        "average_review_word_count": round(avg_words, 1),
        "platform_distribution": dict(platform_counter),
        "price_tier_distribution": dict(price_tier_counter),
        "district_distribution": dict(districts_counter),
        "export_path": str(MASTER_DATASET_FILE),
        "grouped_by_hotel_path": str(grouped_json_file)
    }

    with open(DATA_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\n[SUCCESS] Master Research Dataset Generated: {len(master_records)} clean reviews.")
    print(f"-> Saved to: {MASTER_DATASET_FILE}")
    print(f"-> Grouped by Hotel: {grouped_json_file}")
    print(f"-> Quality Audit Report: {DATA_REPORT_FILE}")
    return audit_report

if __name__ == "__main__":
    normalize_and_export_dataset()
