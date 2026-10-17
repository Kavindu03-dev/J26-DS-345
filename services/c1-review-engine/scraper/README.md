# Component 1 Review Scraper & Data Ingestion Pipeline

This subsystem handles the stratified discovery and multi-source review extraction for **Component 1 (Trust-Aware Review Analysis Engine)**.

## Supported Platforms
- **Booking.com**: Verified reservation reviews, stay metadata (room type, nights, guest type), positive/negative splits.
- **Google Maps**: Unauthenticated public reviews using Multi-Sort Cycling (extracts 15-25 reviews/hotel).
- **TripAdvisor**: Syndicated reviews via Google Maps partner feeds + persistent Edge profiling.

## Execution Commands

Run from `services/c1-review-engine/scraper/`:

```powershell
# 1. Hotel Discovery (discovers hotels across all 25 Sri Lankan districts)
python scraper_main.py --phase discovery --hotels-per-district 42

# 2. Review Extraction across platforms
python scraper_main.py --phase extraction --platform booking --max-reviews 30
python scraper_main.py --phase extraction --platform google --max-reviews 20

# 3. Normalization & Unified Dataset Export
python scraper_main.py --phase export
```
