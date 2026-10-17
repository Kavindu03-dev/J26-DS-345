"""
Batch Dataset Enrichment & Profile Pre-Computation Pipeline for Component 1.
Processes all multi-source raw reviews, executes ML Trust Scoring & ABSA,
and pre-computes hotel aspect profiles for Component 2 (Recommender) and Component 4 (Chatbot).
"""

import csv
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    HOTELS_SEED_FILE,
    REVIEWS_BOOKING_FILE,
    REVIEWS_GOOGLEMAPS_FILE,
    REVIEWS_TRIPADVISOR_FILE,
    ENRICHED_REVIEWS_FILE,
    HOTEL_PROFILES_JSON,
    HOTEL_PROFILES_CSV,
    RESEARCH_DATASET_FIELDS,
    HOTEL_PROFILE_FIELDS,
    ASPECT_CATEGORIES
)
from src.models.trust_engine import TrustEngine
from src.models.absa_engine import ABSAEngine
from src.models.fact_extractor import FactExtractor
from src.models.aggregator import AspectScoreAggregator


class BatchEnricher:
    def __init__(self):
        self.trust_engine = TrustEngine()
        self.absa_engine = ABSAEngine()
        self.fact_extractor = FactExtractor()
        self.aggregator = AspectScoreAggregator(
            trust_engine=self.trust_engine,
            absa_engine=self.absa_engine,
            fact_extractor=self.fact_extractor
        )

    def load_hotel_entities(self) -> Dict[str, Dict[str, Any]]:
        """Loads all hotel entities from seed registry into lookup dictionaries."""
        hotels_by_id = {}
        if not HOTELS_SEED_FILE.exists():
            print(f"[!] Warning: Hotels seed file not found: {HOTELS_SEED_FILE}")
            return {}

        with open(HOTELS_SEED_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h_id = row.get("hotel_id")
                if h_id:
                    hotels_by_id[h_id] = row

        print(f"[+] Loaded {len(hotels_by_id)} hotels from seed file.")
        return hotels_by_id

    def load_raw_reviews(self, hotels_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidates reviews from Booking, Google Maps, and TripAdvisor."""
        raw_sources = [
            ("Booking.com", REVIEWS_BOOKING_FILE),
            ("Google Maps", REVIEWS_GOOGLEMAPS_FILE),
            ("TripAdvisor", REVIEWS_TRIPADVISOR_FILE)
        ]

        all_reviews = []
        seen_ids = set()

        for platform_name, file_path in raw_sources:
            if not file_path.exists():
                continue

            print(f"[+] Reading {platform_name} from: {file_path.name}...")
            count = 0
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r_id = row.get("review_id", "")
                    # Generate deterministic key if missing
                    if not r_id:
                        r_id = f"{row.get('hotel_id')}_{row.get('author_id')}_{row.get('review_date')}"

                    if r_id in seen_ids:
                        continue
                    seen_ids.add(r_id)

                    row["source_platform"] = row.get("source_platform") or platform_name
                    all_reviews.append(row)
                    count += 1
            print(f"    Loaded {count} unique reviews from {platform_name}.")

        print(f"[+] Total unique raw reviews gathered: {len(all_reviews)}")
        return all_reviews

    def enrich_reviews(self, reviews: List[Dict[str, Any]], hotels_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes Trust Scoring and ABSA on all reviews to populate the 35-field schema."""
        enriched = []
        total = len(reviews)
        print(f"\n>>> Starting Review-Level ML Enrichment on {total} reviews...")

        start_time = time.time()
        suspicious_count = 0

        for idx, r in enumerate(reviews, 1):
            h_id = r.get("hotel_id", "")
            hotel_meta = hotels_by_id.get(h_id, {})
            baseline_10 = float(hotel_meta.get("platform_overall_rating", 8.0) or 8.0)

            # 1. Trust Scoring & Suspicion Flag
            t_score = self.trust_engine.calculate_review_trust(r, hotel_baseline_rating=baseline_10)
            is_susp = self.trust_engine.detect_suspicious_review(r, t_score)
            if is_susp:
                suspicious_count += 1

            # 2. Rating Disparity
            try:
                r_score = float(r.get("rating_score", 8.0) or 8.0)
            except (ValueError, TypeError):
                r_score = 8.0
            _, p_disp = self.trust_engine.calculate_rating_disparity(r_score, baseline_10)

            # 3. ABSA 6-Aspect Sentiment Analysis
            absa_res = self.absa_engine.analyze_review(r, baseline_rating_10=baseline_10)

            # Aspect average for review-level trust-weighted aspect score
            aspect_ratings = [absa_res[cat]["score_5"] for cat in ASPECT_CATEGORIES if absa_res[cat]["mentions_count"] > 0]
            if aspect_ratings:
                avg_aspect = sum(aspect_ratings) / len(aspect_ratings)
            else:
                avg_aspect = max(2.5, min(4.8, baseline_10 / 2.0))
            trust_weighted_aspect = round(avg_aspect * t_score, 2)

            # Assemble full 35-field record
            text = str(r.get("review_text") or "")
            w_count = len(text.split())

            enriched_row = {
                # 1. Hotel Entity Metadata
                "hotel_id": h_id,
                "hotel_name": hotel_meta.get("hotel_name") or r.get("hotel_name", "Unknown Hotel"),
                "hotel_url": hotel_meta.get("hotel_url") or r.get("hotel_url", ""),
                "latitude": float(hotel_meta.get("latitude", 0.0) or r.get("latitude", 0.0) or 0.0),
                "longitude": float(hotel_meta.get("longitude", 0.0) or r.get("longitude", 0.0) or 0.0),
                "address_street": hotel_meta.get("address_street") or r.get("address_street", ""),
                "city": hotel_meta.get("city") or r.get("city", "Sri Lanka"),
                "district": hotel_meta.get("district") or r.get("district", "Sri Lanka"),
                "province": hotel_meta.get("province") or r.get("province", "Sri Lanka"),
                "property_type": hotel_meta.get("property_type") or r.get("property_type", "Hotel"),
                "star_rating": float(hotel_meta.get("star_rating", 3.0) or r.get("star_rating", 3.0) or 3.0),
                "price_tier": hotel_meta.get("price_tier") or r.get("price_tier", "Mid-Range"),
                "avg_price_per_night": float(hotel_meta.get("avg_price_per_night", 55.0) or r.get("avg_price_per_night", 55.0) or 55.0),
                "amenities": hotel_meta.get("amenities") or r.get("amenities", "Free Wi-Fi, Air conditioning"),
                "platform_overall_rating": baseline_10,
                "platform_total_reviews_count": int(hotel_meta.get("platform_total_reviews_count", 50) or 50),

                # 2. Core Review Fields
                "review_id": r.get("review_id", ""),
                "source_platform": r.get("source_platform", "Booking.com"),
                "review_date": r.get("review_date", ""),
                "rating_score": r_score,
                "review_title": r.get("review_title", ""),
                "positive_text": r.get("positive_text", ""),
                "negative_text": r.get("negative_text", ""),
                "review_text": text,

                # 3. Reviewer & Behavioral Features
                "author_id": r.get("author_id", "USR_ANONYMOUS"),
                "author_country": r.get("author_country", "Unknown"),
                "author_total_reviews": int(r.get("author_total_reviews", 1) or 1),
                "author_account_age": r.get("author_account_age", "Guest"),
                "helpful_votes": int(r.get("helpful_votes", 0) or 0),
                "is_verified_stay": bool(r.get("is_verified_stay", True)),

                # 4. Stay Metadata
                "stay_date": r.get("stay_date", ""),
                "nights_stayed": r.get("nights_stayed", ""),
                "traveler_type": r.get("traveler_type", "Solo traveller"),
                "room_type": r.get("room_type", ""),

                # 5. Derived & Output ML Fields
                "text_length_words": w_count,
                "sentiment_rating_disparity": round(p_disp * 10.0, 2),
                "trust_score": t_score,
                "is_suspicious": is_susp,
                "aspect_food_sentiment": absa_res["food"]["sentiment"],
                "aspect_staff_sentiment": absa_res["staff"]["sentiment"],
                "aspect_cleanliness_sentiment": absa_res["cleanliness"]["sentiment"],
                "aspect_room_sentiment": absa_res["room"]["sentiment"],
                "aspect_location_sentiment": absa_res["location"]["sentiment"],
                "aspect_wifi_sentiment": absa_res["wifi"]["sentiment"],
                "trust_weighted_aspect_score": trust_weighted_aspect
            }
            enriched.append(enriched_row)

            if idx % 3000 == 0 or idx == total:
                elapsed = time.time() - start_time
                print(f"    Progress: {idx}/{total} reviews ({int(idx/total*100)}%) - {elapsed:.1f}s")

        print(f"[+] Enrichment complete: {total} reviews processed. Flagged {suspicious_count} suspicious reviews.")
        return enriched

    def precompute_hotel_profiles(
        self,
        hotels_by_id: Dict[str, Dict[str, Any]],
        enriched_reviews: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Pre-computes aspect ratings, trust scores, and verified facts for all 1,090 hotels.
        Returns:
          1. Dict keyed by hotel_id matching contracts/c1_aspect_scores.json
          2. Tabular records list for Component 2 Recommender CSV
        """
        print(f"\n>>> Pre-computing Profiles for all {len(hotels_by_id)} Hotels...")
        reviews_by_hotel = {h_id: [] for h_id in hotels_by_id.keys()}

        for r in enriched_reviews:
            h_id = r.get("hotel_id")
            if h_id in reviews_by_hotel:
                reviews_by_hotel[h_id].append(r)

        profiles_json = {}
        profiles_tabular = []

        for h_id, hotel in hotels_by_id.items():
            h_reviews = reviews_by_hotel.get(h_id, [])
            contract_dict = self.aggregator.aggregate_hotel_aspect_scores(hotel, h_reviews)
            profiles_json[h_id] = contract_dict

            # Construct tabular record for Recommender
            aspects = contract_dict["aspect_scores"]
            verified_count = sum(1 for r in h_reviews if r.get("is_verified_stay"))
            suspicious_count = sum(1 for r in h_reviews if r.get("is_suspicious") == 1)
            verified_ratio = round(verified_count / len(h_reviews), 2) if h_reviews else 1.0

            facts_text = " | ".join(contract_dict.get("verified_aspect_facts", [])[:3])

            tabular_row = {
                "hotel_id": h_id,
                "hotel_name": hotel.get("hotel_name", "Unknown Hotel"),
                "city": hotel.get("city", ""),
                "district": hotel.get("district", ""),
                "province": hotel.get("province", ""),
                "latitude": float(hotel.get("latitude", 0.0) or 0.0),
                "longitude": float(hotel.get("longitude", 0.0) or 0.0),
                "property_type": hotel.get("property_type", "Hotel"),
                "star_rating": float(hotel.get("star_rating", 3.0) or 3.0),
                "price_tier": hotel.get("price_tier", "Mid-Range"),
                "avg_price_per_night": float(hotel.get("avg_price_per_night", 55.0) or 55.0),
                "platform_overall_rating": float(hotel.get("platform_overall_rating", 8.0) or 8.0),
                "overall_trust_score": contract_dict["overall_trust_score"],
                "time_decay_rating": contract_dict["time_decay_rating"],
                "aspect_food": aspects.get("food", 4.0),
                "aspect_staff": aspects.get("staff", 4.0),
                "aspect_room": aspects.get("room", 4.0),
                "aspect_cleanliness": aspects.get("cleanliness", 4.0),
                "aspect_location": aspects.get("location", 4.0),
                "aspect_wifi": aspects.get("wifi", 4.0),
                "total_reviews": len(h_reviews),
                "suspicious_reviews_count": suspicious_count,
                "verified_stay_ratio": verified_ratio,
                "top_verified_facts": facts_text,
                "last_updated": contract_dict["last_updated"]
            }
            profiles_tabular.append(tabular_row)

        print(f"[+] Pre-computed profiles for {len(profiles_tabular)} properties.")
        return profiles_json, profiles_tabular

    def export_all(
        self,
        enriched_reviews: List[Dict[str, Any]],
        profiles_json: Dict[str, Any],
        profiles_tabular: List[Dict[str, Any]]
    ):
        """Exports all three enriched artifact files."""
        # 1. Export Enriched Reviews CSV
        print(f"\n[+] Saving Enriched Reviews ({len(enriched_reviews)} rows) -> {ENRICHED_REVIEWS_FILE.name}")
        with open(ENRICHED_REVIEWS_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=RESEARCH_DATASET_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(enriched_reviews)

        # 2. Export Hotel Profiles JSON (for C1 FastAPI & C4 Chatbot)
        print(f"[+] Saving Hotel Contract Profiles JSON ({len(profiles_json)} hotels) -> {HOTEL_PROFILES_JSON.name}")
        with open(HOTEL_PROFILES_JSON, "w", encoding="utf-8") as f:
            json.dump(profiles_json, f, indent=2)

        # 3. Export Hotel Profiles CSV (for C2 Recommender)
        print(f"[+] Saving Hotel Aspect Profiles CSV ({len(profiles_tabular)} hotels) -> {HOTEL_PROFILES_CSV.name}")
        with open(HOTEL_PROFILES_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=HOTEL_PROFILE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(profiles_tabular)

        print("\n" + "=" * 75)
        print("[OK] BATCH ENRICHMENT PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"   1. Enriched Reviews CSV   : {ENRICHED_REVIEWS_FILE}")
        print(f"   2. Hotel Profiles JSON    : {HOTEL_PROFILES_JSON}")
        print(f"   3. Hotel Profiles CSV     : {HOTEL_PROFILES_CSV}")
        print("=" * 75)

    def run(self):
        """Main execution flow."""
        hotels = self.load_hotel_entities()
        if not hotels:
            return

        reviews = self.load_raw_reviews(hotels)
        if not reviews:
            print("[!] No reviews found to enrich.")
            return

        enriched = self.enrich_reviews(reviews, hotels)
        p_json, p_tabular = self.precompute_hotel_profiles(hotels, enriched)
        self.export_all(enriched, p_json, p_tabular)


if __name__ == "__main__":
    enricher = BatchEnricher()
    enricher.run()
