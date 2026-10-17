"""
Enhanced Booking.com Multi-Page Review Extractor.
Extracts complete Trust Signals, Behavioral Attributes, Aspect Feedback, and Geolocation metadata.
"""

import csv
import sys
import time
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    HOTELS_SEED_FILE,
    REVIEWS_BOOKING_FILE,
    RESEARCH_DATASET_FIELDS, DEFAULT_REVIEWS_PER_ENTITY,
    DEFAULT_HEADLESS, DEFAULT_USER_AGENT
)
from src.extractors.base_extractor import BaseExtractor
from src.utils import (
    random_delay, long_delay, apply_stealth,
    clean_text, generate_review_id, generate_entity_id,
    anonymize_author
)

class BookingReviewExtractor(BaseExtractor):
    def __init__(self, output_file=REVIEWS_BOOKING_FILE):
        super().__init__(output_file=output_file, fieldnames=RESEARCH_DATASET_FIELDS)

    def extract_all(self, max_reviews_per_hotel=DEFAULT_REVIEWS_PER_ENTITY, headless=DEFAULT_HEADLESS, limit_hotels=None):
        print("==========================================================")
        print("   Starting Phase 3: Booking.com Trust-Aware Extraction   ")
        print(f" Target: Up to {max_reviews_per_hotel} reviews per property")
        print("==========================================================")

        # 1. Load hotel entity seeds
        hotels = []
        if HOTELS_SEED_FILE.exists():
            with open(HOTELS_SEED_FILE, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("hotel_url"):
                        hotels.append(row)

        if not hotels:
            print("⚠️ No hotels found in seed files. Run discovery first.")
            return 0

        if limit_hotels and limit_hotels > 0:
            hotels = hotels[:limit_hotels]

        print(f"Loaded {len(hotels)} hotels from seed registry.")
        max_pages_per_hotel = max(1, (max_reviews_per_hotel + 9) // 10)
        total_extracted_reviews = 0

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                user_agent=DEFAULT_USER_AGENT,
                viewport={"width": 1920, "height": 1080},
                locale="en-US"
            )
            page = context.new_page()
            apply_stealth(page)

            for h_idx, hotel in enumerate(hotels, 1):
                hotel_name = hotel.get("hotel_name", "Unknown")
                hotel_url = hotel.get("hotel_url", "")
                hotel_id = hotel.get("hotel_id", generate_entity_id(hotel_name, hotel.get("city", "")))
                
                print(f"\n[{h_idx}/{len(hotels)}] Processing: {hotel_name}")
                
                # Check if already processed
                if self.checkpoint.is_hotel_completed(hotel_url):
                    print(f"  ⏩ Hotel already marked completed in checkpoint. Skipping.")
                    continue

                try:
                    # Append English query params to force English locale
                    target_url = hotel_url
                    if "?" in target_url:
                        target_url += "&lang=en-us&selected_currency=USD"
                    else:
                        target_url += "?lang=en-us&selected_currency=USD"

                    page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                    random_delay(2.0, 4.0)

                    # Bot challenge check
                    if "challenge" in page.url.lower() or "verify you are human" in page.content().lower():
                        print("⚠️ Blocked by CAPTCHA. Pausing extraction.")
                        break

                    # Handle cookie banner
                    try:
                        cookie_btn = page.locator("button:has-text('Accept'), button:has-text('Accept all')").first
                        if cookie_btn.is_visible(timeout=1500):
                            cookie_btn.click()
                            time.sleep(1)
                    except Exception:
                        pass

                    # Extract coordinates from page if not already populated
                    if float(hotel.get("latitude", 0.0)) == 0.0:
                        try:
                            coords = page.evaluate(r"""() => {
                                const mapEl = document.querySelector('[data-atlas-latlng], [data-lat]');
                                if (mapEl) {
                                    const lat = mapEl.getAttribute('data-atlas-latlng') || mapEl.getAttribute('data-lat');
                                    return lat;
                                }
                                return '';
                            }""")
                            if coords and "," in coords:
                                parts = coords.split(",")
                                hotel["latitude"] = float(parts[0].strip())
                                hotel["longitude"] = float(parts[1].strip())
                        except Exception:
                            pass

                    # Trigger Guest Reviews modal / sidebar
                    opened_sidebar = False
                    trigger_selectors = [
                        "a[data-decider-header='reviews']",
                        "a:has-text('Guest reviews')",
                        "button:has-text('Guest reviews')",
                        "button:has-text('Read all reviews')",
                        "[data-testid='review-score-component']",
                        "[data-testid='review-score-right-component']",
                        "a.js-hotel-review-score"
                    ]

                    for sel in trigger_selectors:
                        try:
                            btn = page.locator(sel).first
                            if btn.count() > 0 and btn.is_visible():
                                btn.scroll_into_view_if_needed()
                                time.sleep(0.5)
                                btn.click()
                                time.sleep(2.5)
                                if page.locator("[data-testid='review-card']").count() > 0:
                                    opened_sidebar = True
                                    break
                        except Exception:
                            continue

                    if not opened_sidebar:
                        if page.locator("[data-testid='review-card']").count() > 0:
                            opened_sidebar = True
                        else:
                            print(f"  ℹ️ No reviews sidebar found or property has 0 reviews.")
                            self.checkpoint.mark_hotel_completed(hotel_url)
                            continue

                    hotel_reviews_count = 0

                    # Pagination loop
                    for page_num in range(1, max_pages_per_hotel + 1):
                        raw_cards = page.evaluate(r"""() => {
                            const cards = Array.from(document.querySelectorAll('[data-testid="review-card"]'));
                            return cards.map(c => {
                                const getTxt = (sel) => {
                                    const el = c.querySelector(sel);
                                    return el ? el.innerText.trim() : '';
                                };
                                
                                const avatarArea = c.querySelector('[data-testid="review-avatar"]');
                                let author = 'Guest';
                                let authorReviews = '1';
                                let authorCountry = 'Unknown';
                                if (avatarArea) {
                                    const lines = avatarArea.innerText.trim().split('\\n').map(l => l.trim()).filter(l => l.length > 0);
                                    if (lines.length >= 1) author = lines[0];
                                    if (lines.length >= 2) authorReviews = lines[1];
                                    if (lines.length >= 3) authorCountry = lines[2];
                                }
                                
                                let rawScore = getTxt('[data-testid="review-score"]');
                                let score = 'N/A';
                                if (rawScore) {
                                    const match = rawScore.match(/\\b(10|[0-9](\\.[0-9])?)\\b/);
                                    score = match ? match[0] : rawScore;
                                }
                                
                                let rawDate = getTxt('[data-testid="review-date"]');
                                let reviewDate = rawDate.replace(/^Reviewed:\\s*/i, '').trim();
                                
                                const title = getTxt('[data-testid="review-title"]');
                                const positive = getTxt('[data-testid="review-positive-text"]');
                                const negative = getTxt('[data-testid="review-negative-text"]');
                                const room = getTxt('[data-testid="review-room-name"]');
                                const nights = getTxt('[data-testid="review-num-nights"]');
                                const travelerType = getTxt('[data-testid="review-traveler-type"]');
                                const stayDate = getTxt('[data-testid="review-stay-date"]');
                                const helpfulCount = getTxt('[data-testid="review-helpful-count"]');
                                
                                return {
                                    author,
                                    authorReviews,
                                    authorCountry,
                                    score,
                                    reviewDate,
                                    title,
                                    positive,
                                    negative,
                                    room,
                                    nights,
                                    travelerType,
                                    stayDate,
                                    helpfulCount
                                };
                            });
                        }""")

                        batch_records = []
                        for card in raw_cards:
                            if hotel_reviews_count >= max_reviews_per_hotel:
                                break

                            author_name = card.get("author", "Guest")
                            rev_date = card.get("reviewDate", "")
                            title = clean_text(card.get("title", ""))
                            pos_text = clean_text(card.get("positive", ""))
                            neg_text = clean_text(card.get("negative", ""))

                            # Build unified text
                            full_parts = []
                            if title: full_parts.append(title)
                            if pos_text: full_parts.append(pos_text)
                            if neg_text: full_parts.append(neg_text)
                            full_review_text = " ".join(full_parts)

                            if not full_review_text:
                                continue

                            # Deduplication key check
                            rev_id = generate_review_id(hotel_id, author_name, rev_date, title)
                            compound_key = (hotel_id, author_name, rev_date, title)
                            if compound_key in self.existing_keys:
                                continue

                            # Parse author review count
                            rev_num_match = re.search(r'([0-9]+)', card.get("authorReviews", "1"))
                            author_reviews_int = int(rev_num_match.group(1)) if rev_num_match else 1

                            # Parse score
                            score_val = 0.0
                            try:
                                score_val = float(card.get("score", 0.0))
                            except Exception:
                                score_val = 8.0 # Fallback default

                            # Parse helpful votes
                            helpful_match = re.search(r'([0-9]+)', card.get("helpfulCount", "0"))
                            helpful_int = int(helpful_match.group(1)) if helpful_match else 0

                            # Construct complete 35-field row
                            row_record = {
                                # 1. Hotel Entity Metadata (16 fields)
                                "hotel_id": hotel_id,
                                "hotel_name": hotel_name,
                                "hotel_url": hotel_url,
                                "latitude": hotel.get("latitude", 0.0),
                                "longitude": hotel.get("longitude", 0.0),
                                "address_street": hotel.get("address_street", ""),
                                "city": hotel.get("city", ""),
                                "district": hotel.get("district", ""),
                                "province": hotel.get("province", ""),
                                "property_type": hotel.get("property_type", "Hotel"),
                                "star_rating": hotel.get("star_rating", 0.0),
                                "price_tier": hotel.get("price_tier", "Mid-Range"),
                                "avg_price_per_night": hotel.get("avg_price_per_night", 0.0),
                                "amenities": hotel.get("amenities", "Free Wi-Fi, AC"),
                                "platform_overall_rating": hotel.get("platform_overall_rating", 0.0),
                                "platform_total_reviews_count": hotel.get("platform_total_reviews_count", 0),
                                
                                # 2. Core Review Fields (9 fields)
                                "review_id": rev_id,
                                "source_platform": "Booking.com",
                                "review_date": rev_date,
                                "rating_score": score_val,
                                "review_title": title,
                                "positive_text": pos_text,
                                "negative_text": neg_text,
                                "review_text": full_review_text,
                                
                                # 3. Reviewer & Behavioral Features for Trust Scoring (6 fields)
                                "author_id": anonymize_author(author_name),
                                "author_country": card.get("authorCountry", "Unknown"),
                                "author_total_reviews": author_reviews_int,
                                "author_account_age": "Verified Guest",
                                "helpful_votes": helpful_int,
                                "is_verified_stay": True,
                                
                                # 4. Stay & Contextual Metadata for Recommender (4 fields)
                                "stay_date": card.get("stayDate", ""),
                                "nights_stayed": clean_text(card.get("nights", "")),
                                "traveler_type": clean_text(card.get("travelerType", "Solo traveller")),
                                "room_type": clean_text(card.get("room", "")),
                                
                                # 5. Derived & Output Fields (Placeholders for NLP/Model pipeline)
                                "text_length_words": len(full_review_text.split()),
                                "sentiment_rating_disparity": round(abs(score_val - float(hotel.get("platform_overall_rating", 8.0))), 2),
                                "trust_score": 0.0,
                                "is_suspicious": 0,
                                "aspect_food_sentiment": "Neutral",
                                "aspect_staff_sentiment": "Neutral",
                                "aspect_cleanliness_sentiment": "Neutral",
                                "aspect_room_sentiment": "Neutral",
                                "aspect_location_sentiment": "Neutral",
                                "aspect_wifi_sentiment": "Neutral",
                                "trust_weighted_aspect_score": 0.0
                            }

                            batch_records.append(row_record)
                            self.existing_keys.add(compound_key)
                            hotel_reviews_count += 1

                        # Save batch atomically
                        if batch_records:
                            self.append_batch(batch_records)
                            total_extracted_reviews += len(batch_records)

                        print(f"  -> Page {page_num}: Extracted {len(batch_records)} new reviews (Hotel Total: {hotel_reviews_count})")

                        if hotel_reviews_count >= max_reviews_per_hotel:
                            break

                        # Click Next Page
                        next_btn = page.locator("button[aria-label='Next page'], button:has-text('Next page')").first
                        if next_btn.count() > 0 and next_btn.is_visible() and not next_btn.is_disabled():
                            next_btn.click()
                            random_delay(1.5, 3.0)
                        else:
                            break

                    self.checkpoint.mark_hotel_completed(hotel_url)
                    print(f"  [OK] Completed {hotel_name}: {hotel_reviews_count} reviews saved.")

                except Exception as e:
                    print(f"  [!] Error extracting reviews for {hotel_name}: {e}")

                random_delay(3.0, 6.0)

            browser.close()

        print("\n==========================================================")
        print(f" Extraction Phase Finished: {total_extracted_reviews} reviews collected!")
        print(f" Output Location: {self.output_file}")
        print("==========================================================")
        return total_extracted_reviews

def extract_reviews(max_reviews_per_hotel=DEFAULT_REVIEWS_PER_ENTITY, headless=DEFAULT_HEADLESS, limit_hotels=None):
    extractor = BookingReviewExtractor()
    return extractor.extract_all(max_reviews_per_hotel=max_reviews_per_hotel, headless=headless, limit_hotels=limit_hotels)

if __name__ == "__main__":
    extract_reviews(max_reviews_per_hotel=30)
