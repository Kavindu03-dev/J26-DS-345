"""
Google Maps Places & Reviews Extractor Module.
Extracts Local Guide badges, review likes, photos count, and multi-source reviews
for the same physical hotels discovered in hotels_seed.csv.
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
    HOTELS_SEED_FILE, REVIEWS_GOOGLEMAPS_FILE, RESEARCH_DATASET_FIELDS,
    DEFAULT_HEADLESS, DEFAULT_USER_AGENT
)
from src.extractors.base_extractor import BaseExtractor
from src.utils import (
    random_delay, apply_stealth, clean_text,
    generate_review_id, generate_entity_id, anonymize_author
)

class GoogleMapsReviewExtractor(BaseExtractor):
    def __init__(self, output_file=REVIEWS_GOOGLEMAPS_FILE):
        super().__init__(output_file=output_file, fieldnames=RESEARCH_DATASET_FIELDS)

    def extract_reviews_for_hotel_record(self, page, hotel: dict, max_reviews: int = 20):
        """Scrape Google Maps reviews for a specific hotel from hotels_seed.csv."""
        hotel_id = hotel.get("hotel_id", "")
        hotel_name = hotel.get("hotel_name", "")
        city = hotel.get("city", "")
        district = hotel.get("district", city)

        search_query = f"{hotel_name} {city} Sri Lanka"
        gmaps_url = f"https://www.google.com/maps/search/{search_query.replace(' ', '+')}"
        collected_records = []

        try:
            page.goto(gmaps_url, wait_until="commit", timeout=30000)
            random_delay(3.0, 4.5)

            # Check if search gave a list of results instead of direct place card
            first_result = page.locator("div[role='feed'] a[href*='/maps/place/'], a.hfpxzc").first
            if first_result.count() > 0:
                try:
                    first_result.click(force=True, timeout=4000)
                    random_delay(2.5, 3.5)
                except Exception:
                    pass

            def dismiss_popups():
                try:
                    dismiss_btn = page.locator("button:has-text('Dismiss'), button[aria-label='Dismiss']").first
                    if dismiss_btn.count() > 0 and dismiss_btn.is_visible():
                        dismiss_btn.click(timeout=1500)
                except Exception:
                    pass

            # Multi-trigger strategy to open full Reviews pane (works for hotels and general places)
            opened_reviews = False
            
            # Trigger 1: Click 'Reviews' tab (standard hotel and place layout)
            try:
                rev_tab = page.locator("button[role='tab'][aria-label*='Reviews'], button[role='tab']:has-text('Reviews')").first
                if rev_tab.count() > 0:
                    rev_tab.click(force=True, timeout=4000)
                    opened_reviews = True
                    random_delay(2.0, 3.0)
            except Exception:
                pass

            # Trigger 2: Click the reviews counter button (e.g. "460 reviews")
            if not opened_reviews:
                try:
                    rev_btn = page.locator("button[jsaction*='moreReviews']").first
                    if rev_btn.count() > 0 and rev_btn.is_visible():
                        rev_btn.click()
                        opened_reviews = True
                        random_delay(2.0, 3.0)
                except Exception:
                    pass

            # Trigger 3: Regex click on any button with 'X reviews' or 'More reviews'
            if not opened_reviews:
                try:
                    opened_reviews = page.evaluate(r"""() => {
                        const btns = Array.from(document.querySelectorAll('button, div[role="button"]'));
                        for (const b of btns) {
                            const txt = (b.innerText || '').trim();
                            if (/^[0-9,]+\s+(reviews|ratings)/i.test(txt) || /^more\s+reviews/i.test(txt)) {
                                b.click();
                                return true;
                            }
                        }
                        return false;
                    }""")
                    if opened_reviews:
                        random_delay(2.0, 3.0)
                except Exception:
                    pass

            dismiss_popups()

            def harvest_cards():
                dismiss_popups()
                # Expand 'More' on truncated reviews
                try:
                    page.evaluate(r"""() => {
                        Array.from(document.querySelectorAll('button')).filter(b => 
                            (b.innerText || '').trim() === 'More' || (b.innerText || '').trim() === 'See more'
                        ).forEach(b => b.click());
                    }""")
                except Exception:
                    pass

                raw_cards = page.evaluate(r"""() => {
                    const cards = Array.from(document.querySelectorAll('div.jftiEf, div[data-review-id]'));
                    return cards.map(c => {
                        const getTxt = (sel) => {
                            const el = c.querySelector(sel);
                            return el ? el.innerText.trim() : '';
                        };
                        const author = getTxt('.d4r55, .WNx5Ec');
                        const authorStats = getTxt('.RfnDt');
                        const text = getTxt('.wiI7pd, .MyEned');
                        const date = getTxt('.rsqaWe');
                        const likes = getTxt('span[aria-label*="Like"], .GBkF3d');
                        let score = 5.0;
                        const starEl = c.querySelector('span.kvMYJc, span[aria-label*="star"]');
                        if (starEl) {
                            const aria = starEl.getAttribute('aria-label') || '';
                            const match = aria.match(/([0-9]+(\.[0-9]+)?)/);
                            if (match) score = parseFloat(match[1]);
                        }
                        return {
                            author: author || 'Google User',
                            authorStats,
                            text,
                            date,
                            likes,
                            score
                        };
                    });
                }""")

                newly_added = 0
                for card in raw_cards:
                    if len(collected_records) >= max_reviews:
                        break

                    text = clean_text(card.get("text", ""))
                    if not text:
                        continue

                    author_raw = card.get("author", "Google User")
                    author_hashed = anonymize_author(author_raw)
                    rev_date = card.get("date", "Recent")
                    rev_id = generate_review_id(hotel_id, author_raw, rev_date, text[:25])
                    compound_key = (hotel_id, author_hashed, rev_date, "")

                    if compound_key in self.existing_keys:
                        continue

                    stats_str = card.get("authorStats", "")
                    rev_count_match = re.search(r'([0-9]+)\s+reviews', stats_str)
                    author_total_reviews = int(rev_count_match.group(1)) if rev_count_match else (10 if "Local Guide" in stats_str else 2)

                    record = {
                        "hotel_id": hotel_id,
                        "hotel_name": hotel_name,
                        "hotel_url": hotel.get("hotel_url", gmaps_url),
                        "latitude": float(hotel.get("latitude", 0.0) or 0.0),
                        "longitude": float(hotel.get("longitude", 0.0) or 0.0),
                        "address_street": hotel.get("address_street", f"{city}, {district}"),
                        "city": city,
                        "district": district,
                        "province": hotel.get("province", "Sri Lanka"),
                        "property_type": hotel.get("property_type", "Guest House / Accommodation"),
                        "star_rating": float(hotel.get("star_rating", 3.0) or 3.0),
                        "price_tier": hotel.get("price_tier", "Mid-Range"),
                        "avg_price_per_night": float(hotel.get("avg_price_per_night", 55.0) or 55.0),
                        "amenities": hotel.get("amenities", "Free Wi-Fi, Air conditioning, Private Bathroom"),
                        "platform_overall_rating": float(hotel.get("platform_overall_rating", 8.0) or 8.0),
                        "platform_total_reviews_count": int(hotel.get("platform_total_reviews_count", 50) or 50),
                        "review_id": rev_id,
                        "source_platform": "Google Maps",
                        "review_date": rev_date,
                        "rating_score": float(card.get("score", 5.0)),
                        "review_title": "",
                        "positive_text": text,
                        "negative_text": "",
                        "review_text": text,
                        "author_id": author_hashed,
                        "author_country": "International / Domestic",
                        "author_total_reviews": author_total_reviews,
                        "author_account_age": stats_str or "Google User",
                        "helpful_votes": 1 if card.get("likes") else 0,
                        "is_verified_stay": False,
                        "stay_date": rev_date,
                        "nights_stayed": "1 night",
                        "traveler_type": "Solo traveller",
                        "room_type": "Standard",
                        "text_length_words": len(text.split()),
                        "sentiment_rating_disparity": round(abs(float(card.get("score", 5.0)) - (float(hotel.get("platform_overall_rating", 8.0)) / 2.0)), 2),
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

                    collected_records.append(record)
                    self.existing_keys.add(compound_key)
                    newly_added += 1

                return newly_added

            # Initial harvest (Most relevant default)
            harvest_cards()

            # Multi-Sort Cycling (1=Newest, 2=Highest rating, 3=Lowest rating)
            if len(collected_records) < max_reviews:
                for sort_idx in [1, 2, 3]:
                    if len(collected_records) >= max_reviews:
                        break
                    try:
                        dismiss_popups()
                        sort_btn = page.locator("button.FNe70d:not(:has-text('All reviews'))").first
                        if sort_btn.count() > 0:
                            sort_btn.click(force=True, timeout=3000)
                            try:
                                page.wait_for_selector(f"div[data-index='{sort_idx}']", timeout=3000)
                                page.click(f"div[data-index='{sort_idx}']", force=True, timeout=3000)
                                time.sleep(2.0)
                                harvest_cards()
                            except Exception:
                                pass
                    except Exception:
                        pass

            # Platform Filter Cycling if still < max_reviews (TripAdvisor syndicated, Trip.com, Google Only)
            if len(collected_records) < max_reviews:
                for plat_idx in [2, 3, 1]:
                    if len(collected_records) >= max_reviews:
                        break
                    try:
                        dismiss_popups()
                        plat_btn = page.locator("button.FNe70d:has-text('All reviews'), button.FNe70d:has-text('Google'), button.FNe70d:has-text('Tripadvisor')").first
                        if plat_btn.count() > 0:
                            plat_btn.click(force=True, timeout=3000)
                            try:
                                page.wait_for_selector(f"div[data-index='{plat_idx}']", timeout=3000)
                                page.click(f"div[data-index='{plat_idx}']", force=True, timeout=3000)
                                time.sleep(2.0)
                                harvest_cards()
                            except Exception:
                                pass
                    except Exception:
                        pass

            if collected_records:
                self.append_batch(collected_records)

        except Exception as e:
            print(f"    [!] Error extracting Google Maps for {hotel_name}: {e}")

        return len(collected_records)

    def extract_all(self, max_reviews_per_hotel: int = 20, headless: bool = DEFAULT_HEADLESS, limit_hotels: int = None, reset_checkpoint: bool = False):
        """Batch extraction of Google Maps reviews for the same hotels in hotels_seed.csv."""
        print("==========================================================")
        print("   Starting Phase 3: Google Maps Multi-Source Extraction  ")
        print(f" Target: Up to {max_reviews_per_hotel} reviews per property")
        print("==========================================================")

        if reset_checkpoint:
            gmap_keys = [k for k in list(self.checkpoint.state.keys()) if k.startswith("gmap_")]
            for k in gmap_keys:
                self.checkpoint.state.pop(k, None)
            self.checkpoint.save()
            print(f"[+] Cleared {len(gmap_keys)} old Google Maps checkpoint flags.")

        hotels = []
        if HOTELS_SEED_FILE.exists():
            with open(HOTELS_SEED_FILE, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                hotels = [r for r in reader if r.get("hotel_name")]

        if not hotels:
            print("⚠️ No hotels found in seed files. Run discovery first.")
            return 0

        if limit_hotels:
            hotels = hotels[:limit_hotels]

        print(f"Loaded {len(hotels)} hotels from seed registry for Google Maps matching.")
        total_extracted = 0

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(viewport={"width": 1400, "height": 950}, user_agent=DEFAULT_USER_AGENT, locale="en-US")
            page = context.new_page()
            apply_stealth(page)

            for h_idx, hotel in enumerate(hotels, 1):
                hotel_id = hotel.get("hotel_id")
                hotel_name = hotel.get("hotel_name", "Unknown")
                city = hotel.get("city", "")

                gmap_ckpt_key = f"gmap_{hotel_id}"
                if self.checkpoint.state.get(gmap_ckpt_key):
                    continue

                print(f"[{h_idx}/{len(hotels)}] Google Maps: {hotel_name} ({city})...", end="", flush=True)
                count = self.extract_reviews_for_hotel_record(page, hotel, max_reviews=max_reviews_per_hotel)
                total_extracted += count
                print(f" -> {count} reviews saved.")

                self.checkpoint.state[gmap_ckpt_key] = True
                self.checkpoint.save()
                random_delay(2.0, 4.0)

            browser.close()

        print(f"\n[OK] Google Maps Extraction Finished: {total_extracted} reviews collected!")
        return total_extracted

def extract_google_maps_reviews(max_reviews_per_hotel=20, headless=DEFAULT_HEADLESS, limit_hotels=None, reset_checkpoint=False):
    extractor = GoogleMapsReviewExtractor()
    return extractor.extract_all(max_reviews_per_hotel=max_reviews_per_hotel, headless=headless, limit_hotels=limit_hotels, reset_checkpoint=reset_checkpoint)

if __name__ == "__main__":
    extract_google_maps_reviews(max_reviews_per_hotel=10, limit_hotels=5)
