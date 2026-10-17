# Master Review Scraping & Extraction Architecture Plan

**Research Component 1: Trust-Aware Review Analysis Engine (Project ID: J26-DS-345)**  
**Author:** Research Scraper Engineering Team  
**Date:** 15 October 2026  
**Status:** Operational Production System  
**Reference Document:** Retaining original preliminary concept in [`review_scraping_plan.md`](file:///d:/RESEARCH%20PROJECT/review_scraper/review_scraping_plan.md)

---

## 1. Executive Summary & Evaluation of the Preliminary Plan

The initial document [`review_scraping_plan.md`](file:///d:/RESEARCH%20PROJECT/review_scraper/review_scraping_plan.md) was an early, preliminary conceptual outline written before live scraper implementation. While its broad goal (acquiring reviews for fake review detection and sentiment analysis) was correct, **it is no longer up to date with the reality of the production system**.

### Critical Differences: Preliminary Plan vs. Actual Production System

| Aspect | Preliminary Plan (`review_scraping_plan.md`) | Actual Operational System (`MASTER_REVIEW_SCRAPING_PLAN.md`) |
| :--- | :--- | :--- |
| **TripAdvisor Strategy** | Assumed simple `BeautifulSoup` / `requests` parsing. | Direct scraping blocked by **DataDome 403** (TLS JA3/JA4 fingerprinting & slider puzzles). Solved via **Google Maps Third-Party Review Syndication** and persistent Edge profiling. |
| **Google Maps Strategy** | Assumed basic Playwright scrolling. | Google Maps unauthenticated layout caps reviews to 5 in standard DOM; solved via **Multi-Sort Cycling** (4 sort modes) yielding 20+ reviews per property. |
| **Data Schema** | Only 9 basic fields (`review_id`, `rating`, `review_text`, etc.). | **45 standardized research fields** covering hotel geo-coordinates, verified guest badges, author history, room types, aspect sentiments, and trust scores. |
| **Hotel Coverage** | Generic plan for 100 hotels. | **Comprehensive national census**: 1,090 hotels discovered and verified across all **25 administrative districts** of Sri Lanka. |
| **Dataset Scale** | Not quantified. | **39,399+ unified reviews** (52,427 raw Booking.com, 7,037 raw Google Maps, 425 TripAdvisor). |
| **Data Pipeline** | Proposed hypothetical database (MongoDB / PostgreSQL). | **Reproducible, version-controlled research pipeline** using atomic CSVs, state checkpoints (`scraper_state.json`), and unified exports for ML downstream tasks. |

---

## 2. System Architecture & Pipeline Workflow

The complete review acquisition and analysis pipeline operates in four coordinated phases:

```
[ PHASE 1: GEOGRAPHIC & HOTEL DISCOVERY ]
  │  Input: 25 Sri Lankan administrative districts (geo_seeds.py)
  │  Engine: src/discovery/hotel_discovery.py (Playwright + Booking.com catalog)
  ▼
  Output: data/raw/hotels_seed.csv (1,090 verified hotel entities with coordinates & stars)
  
[ PHASE 2: MULTI-PLATFORM REVIEW EXTRACTION ]
  │  Orchestrator: scraper_main.py --phase extraction
  ├────────────────────────────────┬───────────────────────────────┐
  ▼                                ▼                               ▼
Booking.com Extractor          Google Maps Extractor           TripAdvisor Extractor
(src/extractors/booking_       (src/extractors/google_maps_    (Google Maps Syndication /
 extractor.py)                  extractor.py)                   DataDome Edge fallback)
  │                              │                               │
  ▼                              ▼                               ▼
reviews_raw_booking.csv        reviews_raw_googlemaps.csv      reviews_raw_tripadvisor.csv
(52,427+ raw records)          (7,037+ raw records)            (425 syndicated records)

[ PHASE 3: NORMALIZATION & DEDUPLICATION ]
  │  Engine: src/processing/normalizer.py (scraper_main.py --phase export)
  │  - Enforces 45-field unified research schema
  │  - Strips non-English reviews (src/processing/language_filter.py)
  │  - Deterministic deduplication via compound keys (hotel_id, author_id, review_date)
  ▼
  Output: data/processed/research_dataset_unified.csv (39,399+ pristine research rows)
  Output: data/processed/reviews_by_hotel.json

[ PHASE 4: TRUST & ASPECT-BASED SENTIMENT (ABSA) ENRICHMENT ]
  │  Engine: c1_review_engine/src/batch_enricher.py
  │  - Trust Model: Multi-signal heuristic (verified stays, text length, rating disparity)
  │  - Aspect ABSA: 6 dimensions (Food, Staff, Cleanliness, Room, Location, WiFi)
  ▼
  Output: c1_review_engine/data/reviews_enriched.csv & hotel_aspect_profiles.json
```

---

## 3. Platform Extraction Strategies & Anti-Bot Defense Solutions

### A. Booking.com Extraction Engine
- **Implementation:** [`src/extractors/booking_extractor.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/src/extractors/booking_extractor.py)
- **Target:** Verified guest reviews with structured positive/negative text splits, room types, stay durations, and reviewer origin countries.
- **Defenses Bypassed:**
  - Automated dismissal of Genius sign-in overlays and GDPR cookie banners.
  - Review modal targeting: Triggers `button[data-testid='read-all-reviews-button']`.
  - Pagination automation: Iterates through review modal pagination buttons up to configured limit.
- **Yield:** 52,427+ raw reviews across Sri Lankan properties.

### B. Google Maps Extraction Engine
- **Implementation:** [`src/extractors/google_maps_extractor.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/src/extractors/google_maps_extractor.py)
- **Target:** Broad public sentiment, local guide reviews, and star ratings.
- **Defenses Bypassed:**
  - Fast page navigation: Uses `wait_until="commit"` to bypass perpetual asset streaming.
  - Desktop Viewport Setting (`1400x950`): Ensures the reviews sidebar and dropdowns remain expanded.
  - **The 5-Review Virtual DOM Ceiling**: Unauthenticated Google Maps renders only 5 reviews per view and locks infinite scroll. Solved via **Multi-Sort Cycling**:
    1. Sort 0: *Most relevant* (extracts 5)
    2. Sort 1: *Newest* (extracts 5)
    3. Sort 2: *Highest rating* (extracts 5)
    4. Sort 3: *Lowest rating* (extracts 5)
  - Result: Extracts 15–25 unique reviews per hotel instead of stopping at 5.

### C. TripAdvisor Extraction Engine
- **Implementation:** [`src/extractors/tripadvisor_extractor.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/src/extractors/tripadvisor_extractor.py) & Syndicated Feed
- **Target:** Detailed long-form descriptive aspect reviews.
- **Defenses Bypassed:**
  - TripAdvisor deploys **DataDome** (JA3/JA4 TLS signature enforcement, device sensor tracking, slider puzzles).
  - Primary Solution: Extracted authentic TripAdvisor reviews syndicated through Google Maps partner feeds (`button.FNe70d:has-text('All reviews')` -> TripAdvisor).
  - Secondary Solution: Persistent Microsoft Edge browser context with interactive puzzle detection for direct crawls.
- **Yield:** 425 high-quality authentic reviews across 334 matched hotels.

---

## 4. The Complete 45-Field Unified Research Dataset Schema

The schema defined in [`src/config.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/src/config.py#L90-L145) covers every signal necessary for **Component 1 (Trust-Aware Review Analysis Engine)**:

### Section 1: Hotel Entity Metadata (16 Fields)
| Column Name | Type | Description | Research Relevance |
| :--- | :--- | :--- | :--- |
| `hotel_id` | String | Unique hash format: `HTL_DIS_NAME_XXXX` | Entity linking and cross-platform alignment |
| `hotel_name` | String | Full hotel business name | Entity recognition |
| `hotel_url` | String | Canonical URL of hotel listing | Provenance and auditing |
| `latitude` | Float | Geographic latitude coordinates | Spatial clustering and map visualization |
| `longitude` | Float | Geographic longitude coordinates | Spatial clustering and map visualization |
| `address_street` | String | Street or postal address | Geographic precision |
| `city` | String | City / Town (e.g., Galle Fort, Colombo) | Urban vs. rural stratification |
| `district` | String | Administrative district (25 in Sri Lanka) | Balanced national sample validation |
| `province` | String | Administrative province (9 in Sri Lanka) | Macro-regional comparisons |
| `property_type` | String | Hotel, Villa, Guest House, Resort | Property segment control |
| `star_rating` | Float | Official or estimated star rating (1.0–5.0) | Quality baseline comparison |
| `price_tier` | String | Budget, Mid-Range, Luxury | Price-performance normalization |
| `avg_price_per_night` | Float | Price in USD | Economic correlation analysis |
| `amenities` | String | Comma-separated list (WiFi, Pool, etc.) | Feature expectation baseline |
| `platform_overall_rating` | Float | Platform aggregate score (normalized to 10) | Rating inflation detection |
| `platform_total_reviews_count`| Integer | Total lifetime review count | Hotel popularity weight |

### Section 2: Core Review Fields (9 Fields)
| Column Name | Type | Description | Research Relevance |
| :--- | :--- | :--- | :--- |
| `review_id` | String | SHA-256 unique review identifier | Deduplication and traceability |
| `source_platform` | String | `Booking.com`, `Google Maps`, or `TripAdvisor` | Cross-platform bias evaluation |
| `review_date` | String | Date review was published | Temporal decay and trend modeling |
| `rating_score` | Float | Normalized review score (1.0–10.0 or 1.0–5.0) | Ground truth rating |
| `review_title` | String | Headline summary of the review | High-weight sentiment feature |
| `positive_text` | String | Explicit positive comments | Structured polarity training |
| `negative_text` | String | Explicit negative comments | Pain point and defect detection |
| `review_text` | String | Full composite text | Primary input for NLP / Transformers |
| `text_length_words` | Integer | Total word count of the review text | Verbosity and effort indicator |

### Section 3: Reviewer & Trust Features (6 Fields)
| Column Name | Type | Description | Research Relevance |
| :--- | :--- | :--- | :--- |
| `author_id` | String | Anonymized hash `USR_XXXXXXXXXX` | User behavior tracking (privacy-safe) |
| `author_country` | String | Reviewer origin country | Domestic vs. International tourist analysis |
| `author_total_reviews` | Integer | Total reviews authored by this user | Account credibility signal |
| `author_account_age` | String | Badge / Level / Age indicator | New-account sockpuppet detection |
| `helpful_votes` | Integer | Number of community upvotes | Peer validation metric |
| `is_verified_stay` | Boolean | True for verified bookings | Critical fake review discriminator |

### Section 4: Stay & Contextual Metadata (4 Fields)
| Column Name | Type | Description | Research Relevance |
| :--- | :--- | :--- | :--- |
| `stay_date` | String | Month/Year stay occurred | Seasonality and lag time analysis |
| `nights_stayed` | String | Number of nights stayed | Trip depth indicator |
| `traveler_type` | String | Solo, Couple, Family, Group, Business | Persona-based recommendation filter |
| `room_type` | String | Room booked (Deluxe, Standard, Suite) | Quality expectation baseline |

### Section 5: Trust Engine & ABSA Output Fields (10 Fields)
| Column Name | Type | Description | Research Relevance |
| :--- | :--- | :--- | :--- |
| `sentiment_rating_disparity` | Float | Gap between text sentiment and star rating | Suspicious review heuristic |
| `trust_score` | Float | Trust confidence score (0.00 – 1.00) | Core Component 1 metric |
| `is_suspicious` | Integer | Flag: 1 if suspicious, 0 if authentic | Fake review classification target |
| `aspect_food_sentiment` | String | Positive / Neutral / Negative | Food & Dining sentiment |
| `aspect_staff_sentiment` | String | Positive / Neutral / Negative | Hospitality & Service sentiment |
| `aspect_cleanliness_sentiment` | String | Positive / Neutral / Negative | Hygiene & Cleanliness sentiment |
| `aspect_room_sentiment` | String | Positive / Neutral / Negative | Comfort & Room Quality sentiment |
| `aspect_location_sentiment` | String | Positive / Neutral / Negative | Convenience & Location sentiment |
| `aspect_wifi_sentiment` | String | Positive / Neutral / Negative | Connectivity sentiment |
| `trust_weighted_aspect_score` | Float | Final aspect score discounted by trust | Component 1 Deliverable output |

---

## 5. Execution Procedures & CLI Reference

All execution is managed through [`scraper_main.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/scraper_main.py).

### 1. Hotel Discovery (Populates `hotels_seed.csv`)
Discovers hotels across Sri Lankan districts using Booking.com directory structures:
```powershell
python scraper_main.py --phase discovery --hotels-per-district 42
```

### 2. Review Extraction (Individual Platforms)
- **Extract Google Maps reviews (up to 20 reviews per hotel):**
  ```powershell
  python scraper_main.py --phase extraction --platform google --max-reviews 20
  ```
- **Reset Google Maps checkpoint to re-extract with latest multi-sort improvements:**
  ```powershell
  python scraper_main.py --phase extraction --platform google --max-reviews 20 --reset-gmap-checkpoint
  ```
- **Extract Booking.com reviews:**
  ```powershell
  python scraper_main.py --phase extraction --platform booking --max-reviews 30
  ```
- **Run in visible browser window (headed mode for debugging):**
  ```powershell
  python scraper_main.py --phase extraction --platform google --no-headless --limit-hotels 5
  ```

### 3. Master Dataset Export
Merges all platform CSVs, applies English filtering, removes duplicates, and generates the master unified research dataset:
```powershell
python scraper_main.py --phase export
```

### 4. Downstream Model Enrichment (Component 1 Engine)
Computes trust scores, sentiment-rating disparities, and 6-dimension aspect sentiments:
```powershell
python c1_review_engine/src/batch_enricher.py
```

---

## 6. Current Dataset Metrics & Verification

As of October 2026, the scraper system has captured:

- **Total Unified Reviews:** **39,399 rows** in [`data/processed/research_dataset_unified.csv`](file:///d:/RESEARCH%20PROJECT/review_scraper/data/processed/research_dataset_unified.csv)
  - **Booking.com:** 32,382 verified reviews
  - **Google Maps:** 6,592 public reviews (expanding to 15,000+ with 20-review multi-sort run)
  - **TripAdvisor:** 425 syndicated in-depth reviews
- **Hotels Covered:** **1,090 verified properties** across all 25 districts of Sri Lanka in [`data/raw/hotels_seed.csv`](file:///d:/RESEARCH%20PROJECT/review_scraper/data/raw/hotels_seed.csv)
- **Enriched Model Outputs:** Fully generated in `c1_review_engine/data/hotel_aspect_profiles.json` for frontend and research evaluation.
