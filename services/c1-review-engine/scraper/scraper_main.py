"""
Master CLI & Pipeline Orchestrator for Research Review Scraper.
Research Component 1: Trust-Aware Review Analysis Engine (Project ID: J26-DS-345)
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DEFAULT_HOTELS_PER_DISTRICT, DEFAULT_REVIEWS_PER_ENTITY,
    DEFAULT_HEADLESS
)
from src.discovery.hotel_discovery import discover_hotels
from src.discovery.destination_discovery import initialize_destination_seeds
from src.extractors.booking_extractor import extract_reviews as extract_booking_reviews
from src.extractors.google_maps_extractor import extract_google_maps_reviews
from src.extractors.tripadvisor_extractor import extract_tripadvisor_reviews
from src.processing.normalizer import normalize_and_export_dataset
from src.utils import long_delay

def run_extraction_platforms(platform="all", max_reviews=DEFAULT_REVIEWS_PER_ENTITY, headless=DEFAULT_HEADLESS, limit_hotels=None, reset_ta_checkpoint=False, reset_gmap_checkpoint=False):
    """Execute review extraction across specified platforms for the same hotels."""
    print("=================================================================")
    print(f"      PHASE 3: MULTI-SOURCE REVIEW & TRUST EXTRACTION ({platform.upper()}) ")
    print("=================================================================")
    
    if platform in ["all", "booking"]:
        print("\n>>> [1/3] Extracting Verified Reviews from Booking.com...")
        booking_cap = max(10, int(max_reviews * 0.50)) if platform == "all" else max_reviews
        extract_booking_reviews(max_reviews_per_hotel=booking_cap, headless=headless, limit_hotels=limit_hotels)
        long_delay(3.0, 6.0)

    if platform in ["all", "google"]:
        print("\n>>> [2/3] Extracting Open Reviews & Local Guide Badges from Google Maps...")
        extract_google_maps_reviews(max_reviews_per_hotel=max_reviews, headless=headless, limit_hotels=limit_hotels, reset_checkpoint=reset_gmap_checkpoint)
        long_delay(3.0, 6.0)

    if platform in ["all", "tripadvisor"]:
        print("\n>>> [3/3] Extracting In-Depth Reviews from TripAdvisor...")
        ta_cap = max(5, int(max_reviews * 0.20)) if platform == "all" else max_reviews
        ta_headless = False if platform == "tripadvisor" else headless
        extract_tripadvisor_reviews(max_reviews_per_hotel=ta_cap, headless=ta_headless, limit_hotels=limit_hotels, reset_checkpoint=reset_ta_checkpoint)

def run_full_pipeline(
    hotels_per_district=DEFAULT_HOTELS_PER_DISTRICT,
    max_reviews_per_hotel=DEFAULT_REVIEWS_PER_ENTITY,
    platform="all",
    headless=DEFAULT_HEADLESS
):
    print("=================================================================")
    print("      RESEARCH COMPONENT 1: DATA ACQUISITION & AUDIT PIPELINE   ")
    print("=================================================================")
    
    # Phase 1: Stratified Hotel Discovery (25 Administrative Districts)
    print("\n>>> [PHASE 1] Stratified Hotel & SME Discovery across Sri Lanka...")
    discover_hotels(hotels_per_district=hotels_per_district, headless=headless)
    
    # Phase 2: Top 50 Sri Lankan Destination Landmark Seeding
    print("\n>>> [PHASE 2] Initializing Top 50 Destination Landmark Seeds...")
    initialize_destination_seeds()
    
    print("\nPausing briefly between phases...")
    long_delay(5.0, 10.0)
    
    # Phase 3: Multi-Platform Review & Trust Signal Extraction
    run_extraction_platforms(platform=platform, max_reviews=max_reviews_per_hotel, headless=headless)
    
    # Phase 4: Data Normalization, Deduplication, Language Filtering & Export
    print("\n>>> [PHASE 4] Normalizing, Deduplicating & Exporting Master Research Dataset...")
    normalize_and_export_dataset()
    
    print("\n=================================================================")
    print("               PIPELINE RUN SUCCESSFULLY COMPLETED               ")
    print("=================================================================")

def main():
    parser = argparse.ArgumentParser(description="Trust-Aware Review Scraper & Dataset Pipeline")
    parser.add_argument("--phase", choices=["all", "discovery", "destinations", "extraction", "export"], default="all", help="Pipeline phase to execute")
    parser.add_argument("--platform", choices=["all", "booking", "google", "tripadvisor"], default="all", help="Target review platform for extraction (default: all)")
    parser.add_argument("--hotels-per-district", type=int, default=DEFAULT_HOTELS_PER_DISTRICT, help="Number of properties to discover per district")
    parser.add_argument("--max-reviews", type=int, default=DEFAULT_REVIEWS_PER_ENTITY, help="Max reviews to scrape per property")
    parser.add_argument("--limit-hotels", type=int, default=None, help="Optional limit on number of hotels to scrape (useful for multi-platform batches)")
    parser.add_argument("--reset-ta-checkpoint", action="store_true", help="Reset previous TripAdvisor checkpoint flags (useful after DataDome unblock)")
    parser.add_argument("--reset-gmap-checkpoint", action="store_true", help="Reset previous Google Maps checkpoint flags (to extract 20 reviews for previously visited properties)")
    parser.add_argument("--headful", action="store_true", help="Run browser in visible mode (default is headless)")

    args = parser.parse_args()
    headless = not args.headful

    if args.phase == "all":
        run_full_pipeline(
            hotels_per_district=args.hotels_per_district,
            max_reviews_per_hotel=args.max_reviews,
            platform=args.platform,
            headless=headless
        )
    elif args.phase == "discovery":
        discover_hotels(hotels_per_district=args.hotels_per_district, headless=headless)
    elif args.phase == "destinations":
        initialize_destination_seeds()
    elif args.phase == "extraction":
        run_extraction_platforms(
            platform=args.platform,
            max_reviews=args.max_reviews,
            headless=headless,
            limit_hotels=args.limit_hotels,
            reset_ta_checkpoint=args.reset_ta_checkpoint,
            reset_gmap_checkpoint=args.reset_gmap_checkpoint
        )
    elif args.phase == "export":
        normalize_and_export_dataset()

if __name__ == "__main__":
    main()
