# Permanent Conversation History, System Memory & Decision Ledger
**Project ID:** J26-DS-345  
**Component 1:** Trust-Aware Review Analysis and Aspect Sentiment Engine  
**Candidate:** DISSANAYAKE L.G.D.P.M. (IT23252554)  
**Supervisors:** Prof. Samantha Thelijjagoda, Dr. Junius Anjana  
**Degree:** B.Sc. (Hons) in Information Technology Specialized in Data Science, SLIIT  
**File Created:** October 18, 2026 (Live Project Ledger)

---

## 1. Why This File Exists
This document serves as the **unbreakable, version-controlled memory log** for Component 1. It bridges all Antigravity / IDE conversation sessions so that **context, architectural breakthroughs, scraping milestones, and next-step roadmaps are never lost**, even if chat UI sessions are refreshed or reset.

### Active & Past Session Directory Map
All raw agent transcripts and thinking trajectories are permanently archived by the IDE under:
- Master Directory: `C:\Users\pasin\.gemini\antigravity-ide\brain\`
- Current Active Session ID: `b3d1de04-6698-4f36-a6e3-970f7b85a26f` (October 18, 2026)
- Immediate Predecessor Sessions:
  - `4a1c65eb-e5df-4b52-8b52-126cf3f9ee89` (Oct 18, 2026 12:21 AM)
  - `102eb2b2-2e93-4605-be6f-92812b8c648c` (Oct 17, 2026 11:05 AM)
  - `0c72ccc5-b4bc-44c2-9df0-420f2fe8752b` (Oct 16, 2026 07:42 AM)
  - `dfaccae8-09a6-47f1-b143-3fdf11bdd26e` (Oct 15, 2026 07:45 AM)
  - `40cf848b-841d-40f8-827e-578548441054` (Oct 14, 2026 11:18 AM)
  - `e1da2793-3a21-4317-a3d3-0ecf21f7edb0` (Oct 12, 2026 10:12 PM)
  - `cce6ec49-a13f-433b-9043-37e2f858c6a6` (Sept 16, 2026 10:13 AM)
  - `958bde84-6b4b-4013-ac5f-2aa5daf856e7` (Aug 23, 2026 10:05 PM)
  - `7696c7cd-9d00-4d00-aaa7-5b432b476597` (Aug 16, 2026 07:48 PM)

---

## 2. Key Historical Milestones & Decisions Log

### Milestone 1: Genesis & Schema Definition (August 2026)
- **Problem:** Initial script only scraped basic hotel listings (1 description per hotel).
- **Decision:** Pivot to deep multi-page review crawling. Designed the **45-Field Universal Schema** spanning Hotel Metadata, Review Content, Reviewer Credibility, Stay Context, and Trust/ABSA outputs.

### Milestone 2: Anti-Bot & Scraper Engineering Breakthroughs (October 2026)
1. **Google Maps 5-Review DOM Ceiling:** Unauthenticated Google Maps only displayed 5 reviews in the virtual DOM. We engineered **Multi-Sort Cycling** (`Most relevant`, `Newest`, `Highest rating`, `Lowest rating`), force-hydrating 4 distinct subsets to harvest 15–25 reviews per property.
2. **Booking.com Dynamic Modals:** Automated popup dismissal for "Genius" overlays and GDPR banners, paginating directly through review parameters.
3. **TripAdvisor DataDome Evasion:** Extracted syndicated TripAdvisor reviews via Google Maps integration selectors and Edge browser profile persistence.

### Milestone 3: Dataset Completion & Audit (October 15–17, 2026)
- **Target in Proposal (SO1):** 35,000–50,000 reviews across 750–1,000 hotels in all 25 districts.
- **Final Achievement:**
  - **39,399 unique verified English reviews** (from 59,889 raw reviews; 13,298 duplicates and 7,192 non-English purged).
  - **1,025 hotels** across **all 25 administrative districts** of Sri Lanka.
  - Platforms: Booking.com (32,382), Google Maps (6,592), TripAdvisor (425).
  - **Status: 100% COMPLETE & VERIFIED.** No further scraping is required.

### Milestone 4: Baseline Engine & Contract Compliance (October 16–17, 2026)
- Implemented `trust_engine.py` (author credibility, stay verification, text richness, rating disparity, exponential time decay with 365d half-life).
- Implemented `absa_engine.py` (6 aspects: Food, Staff, Room, Cleanliness, Location, Wi-Fi; clause splitting, negation windowing).
- Implemented `fact_extractor.py` (factual grounding sentences for Component 4 RAG).
- Implemented `aggregator.py` and FastAPI service matching `contracts/c1_aspect_scores.json`.
- Unit tests: 10/10 passing in 0.365s.

---

## 3. Current System State & Status
- **Scraper Status:** Finished and sealed.
- **Dataset Status:** Ingested, cleaned, deduplicated, and normalized at `D:\RESEARCH PROJECT\review_scraper\data\processed\research_dataset_unified.csv`.
- **Seed Profiles:** Synchronized to `d:\RESEARCH PROJECT\J26-DS-345\data\` (`hotel_aspect_profiles.csv`, `hotel_aspect_profiles.json`, `hotels_seed.csv`, `destinations_seed.csv`, `data_summary_report.json`).
- **Baseline Metrics:** Aspect detection accuracy 93.33%, polarity accuracy 80.00%, macro-F1 73.64%.
- **Storage Cleanup:** Removed redundant `reviews_enriched.csv` files (44 MB freed), duplicate markdown docs from `scraper/docs/`, and obsolete `sync_to_main.py`.
- **Path Architecture:** `config.py` refactored to direct workspace relative paths (`DATA_DIR / "raw"`, `DATA_DIR / "processed"`). 10/10 automated tests pass.

---

## 4. Phase-by-Phase Roadmap to Dissertation Defense

```
+-------------------------------------------------------------------------------------------------+
|                                 MASTER EXECUTION ROADMAP                                        |
+-------------------------------------------------------------------------------------------------+
| Step 1 | Ground-Truth Annotation Pipeline (300-500 sentences for 6 aspects + spam)              |
| Step 2 | Aspect Specificity Density (ASD) Engine Formulation (SO2 / FR4)                        |
| Step 3 | Tabular Machine Learning Trust Classifier: LightGBM / XGBoost (SO2 / FR5)              |
| Step 4 | Transformer ABSA Fine-Tuning: DeBERTa-v3 / RoBERTa (SO3 / FR6)                         |
| Step 5 | Formal Empirical Ablation Study: Ranking Stability Analysis (SO5)                      |
| Step 6 | Microservice Integration & Hotel Owner Weak-Aspect Alerts (FR9, FR10)                  |
| Step 7 | Dissertation Chapters (Ch 3, 4, 5, 6) & IEEE/Springer Conference Paper (WBS 6.4)       |
+-------------------------------------------------------------------------------------------------+
```
