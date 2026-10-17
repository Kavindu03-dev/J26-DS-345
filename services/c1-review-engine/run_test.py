"""
Interactive Test & Evaluation CLI for C1 Review Engine.
Allows you to test individual reviews or entire hotels directly from the command line.

Usage examples:
  python run_test.py                        # Runs analysis on a default hotel
  python run_test.py --sample 5             # Runs analysis on 5 different hotels
  python run_test.py --hotel "Ivy Lane"     # Analyzes a specific hotel by name
  python run_test.py --unit-tests           # Runs the automated test suite
"""

import csv
import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import HOTELS_SEED_FILE, REVIEWS_BOOKING_FILE, REVIEWS_GOOGLEMAPS_FILE
from src.models.trust_engine import TrustEngine
from src.models.aggregator import AspectScoreAggregator


def load_reviews_for_hotel(hotel_id_or_name: str):
    """Finds all reviews matching a hotel id or name from both Booking and Google Maps."""
    matched_hotel = None
    all_hotels = []

    # 1. Search hotel in seed file
    if HOTELS_SEED_FILE.exists():
        with open(HOTELS_SEED_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                all_hotels.append(row)
                h_id = row.get("hotel_id", "")
                h_name = row.get("hotel_name", "")
                if hotel_id_or_name.lower() in h_name.lower() or hotel_id_or_name.lower() == h_id.lower():
                    matched_hotel = row
                    break

    if not matched_hotel and all_hotels:
        matched_hotel = all_hotels[0]

    if not matched_hotel:
        print("⚠️ No hotels found in data/hotels_seed.csv")
        return None, []

    target_id = matched_hotel.get("hotel_id")
    target_name = matched_hotel.get("hotel_name")

    reviews = []
    # 2. Gather from reviews_raw_booking.csv
    if REVIEWS_BOOKING_FILE.exists():
        with open(REVIEWS_BOOKING_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("hotel_id") == target_id or target_name.lower() in row.get("hotel_name", "").lower():
                    row["source_platform"] = "Booking.com"
                    reviews.append(row)

    # 3. Gather from reviews_raw_googlemaps.csv
    if REVIEWS_GOOGLEMAPS_FILE.exists():
        with open(REVIEWS_GOOGLEMAPS_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("hotel_id") == target_id or target_name.lower() in row.get("hotel_name", "").lower():
                    row["source_platform"] = "Google Maps"
                    reviews.append(row)

    return matched_hotel, reviews


def run_hotel_analysis(hotel_dict, reviews, aggregator):
    """Computes and displays detailed trust metrics and aspect scores for a given hotel."""
    name = hotel_dict.get("hotel_name", "Unknown Hotel")
    city = hotel_dict.get("city", "")
    h_id = hotel_dict.get("hotel_id", "")
    baseline_rating = float(hotel_dict.get("platform_overall_rating", 8.0) or 8.0)

    print("\n" + "=" * 75)
    print(f"[HOTEL] {name} ({city})")
    print(f"   ID: {h_id} | Baseline Rating: {baseline_rating}/10.0")
    print(f"   Total Reviews Found: {len(reviews)}")
    print("=" * 75)

    if not reviews:
        print("   [!] No reviews found for this hotel yet.")
        return

    # Compute full contract response via aggregator
    contract_res = aggregator.aggregate_hotel_aspect_scores(hotel_dict, reviews)
    aspects = contract_res["aspect_scores"]

    print("\n>>> 1. CORE TRUST & RECENCY METRICS (contracts/c1_aspect_scores.json):")
    print(f"   * Overall Trust Score  : {contract_res['overall_trust_score']} / 10.0   (Target range: 0.0 - 10.0)")
    print(f"   * Time-Decay Rating    : {contract_res['time_decay_rating']} / 5.0     (Target range: 1.0 - 5.0)")

    print("\n>>> 2. TRUST-WEIGHTED ASPECT RATINGS (0.0 to 5.0 scale):")
    print(f"   * [Food & Dining]    : {aspects.get('food', 4.0)} / 5.0")
    print(f"   * [Staff & Service]  : {aspects.get('staff', 4.0)} / 5.0")
    print(f"   * [Room & Comfort]   : {aspects.get('room', 4.0)} / 5.0")
    print(f"   * [Cleanliness]      : {aspects.get('cleanliness', 4.0)} / 5.0")
    print(f"   * [Location]         : {aspects.get('location', 4.0)} / 5.0")
    print(f"   * [Wi-Fi & Internet] : {aspects.get('wifi', 4.0)} / 5.0")

    print("\n>>> 3. VERIFIED ASPECT FACTS (Grounding for Component 4 RAG Chatbot):")
    for f_idx, fact in enumerate(contract_res.get("verified_aspect_facts", []), 1):
        print(f"   [{f_idx}] \"{fact}\"")

    # Sample reviews breakdown
    print("\n>>> 4. SAMPLE REVIEW DETAILS (First 2 reviews):")
    for idx, r in enumerate(reviews[:2], 1):
        platform = r.get("source_platform", "Booking.com")
        t_score = aggregator.trust_engine.calculate_review_trust(r, hotel_baseline_rating=baseline_rating)
        w_time = aggregator.trust_engine.calculate_time_decay_weight(str(r.get("review_date", "")))
        author = r.get("author_id", "Guest")
        badge = r.get("author_account_age", "")
        rating = r.get("rating_score", "8.0")
        text = (r.get("review_text") or "")[:85].replace("\n", " ")

        print(f"   ({idx}) [{platform}] Rating: {rating} | Date: {r.get('review_date')}")
        print(f"       Author: {author} ({badge})")
        print(f"       Trust Score T(r): {t_score:.2f} | Time Decay Weight: {w_time:.2f}")
        print(f"       Snippet: \"{text}...\"")


def main():
    parser = argparse.ArgumentParser(description="Test C1 Trust Engine on Local Data")
    parser.add_argument("--hotel", type=str, default="Ivy Lane", help="Search and evaluate a specific hotel by name")
    parser.add_argument("--sample", type=int, default=1, help="Evaluate N distinct hotels")
    parser.add_argument("--unit-tests", action="store_true", help="Run automated test suite")

    args = parser.parse_args()
    aggregator = AspectScoreAggregator()

    if args.unit_tests:
        print("\n>>> Running Automated Unit Tests...")
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT_DIR / "tests"))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
        return

    if args.sample > 1:
        # Load N hotels
        hotels = []
        if HOTELS_SEED_FILE.exists():
            with open(HOTELS_SEED_FILE, "r", encoding="utf-8-sig") as f:
                hotels = list(csv.DictReader(f))[:args.sample]

        for h in hotels:
            _, revs = load_reviews_for_hotel(h.get("hotel_id"))
            run_hotel_analysis(h, revs, aggregator)
    else:
        hotel_dict, revs = load_reviews_for_hotel(args.hotel)
        if hotel_dict:
            run_hotel_analysis(hotel_dict, revs, aggregator)
        else:
            print(f"Could not find hotel matching '{args.hotel}'")


if __name__ == "__main__":
    main()
