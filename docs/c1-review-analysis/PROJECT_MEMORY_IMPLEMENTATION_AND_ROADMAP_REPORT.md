# Comprehensive Project Memory, Implementation Audit, and Future Roadmap Report

**Project Title:** Tourism Review Analysis and Recommendation System for Sri Lankan Hotels  
**Component 1:** Trust-Aware Review Analysis and Aspect Sentiment Engine  
**Project ID:** J26-DS-345  
**Candidate:** DISSANAYAKE L.G.D.P.M. – IT23252554  
**Degree:** B.Sc. (Hons) in Information Technology Specialized in Data Science  
**Faculty / University:** Department of Computer Science, Faculty of Computing, Sri Lanka Institute of Information Technology (SLIIT)  
**Supervisors:** Prof. Samantha Thelijjagoda (Supervisor), Dr. Junius Anjana (Co-Supervisor)  
**Report Date:** October 17, 2026  
**Document Version:** 1.0 (Master Historical & Technical Review)

---

## 1. Executive Summary & Academic Project Context

### 1.1 Project Overview & Purpose
The Sri Lankan tourism sector is experiencing rapid digital transformation; however, both international and domestic travelers face acute **information overload** from unorganized, unstructured online reviews. Furthermore, small and medium enterprises (SMEs) across Sri Lanka are frequently penalized by **outdated negative reviews** or corrupted by **deceptive, AI-generated spam** that distorts traditional all-time average ratings.

Within the four-member research group project **"Tourism Review Analysis and Recommendation System for Sri Lankan Hotels" (J26-DS-345)**, **Component 1 (This Research)** serves as the **foundational data and intelligence layer**. Its primary mandate is to collect, clean, filter, and continuously transform raw, multi-source online customer reviews into **trust-weighted, time-decayed aspect scores and verified facts** for downstream consumption.

```
========================================================================================
                       OVERALL SYSTEM ARCHITECTURE & INTEGRATION
========================================================================================

   [ Multi-Platform Review Sources ]
   (Booking.com, Google Maps, TripAdvisor)
                  │
                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │         COMPONENT 1 (This Research - IT23252554)            │
   │  Trust-Aware Review Analysis & Aspect Sentiment Engine       │
   │  - PII Scrubbing & Multi-Source Review Normalization        │
   │  - Multi-Factor Trust Scoring & AI/Spam Anomaly Filtering   │
   │  - 6-Aspect ABSA (Food, Staff, Room, Cleanliness, Loc, WiFi)│
   │  - Exponential Time-Decay Mathematical Aggregation           │
   │  - Grounded Fact Extraction (Hallucination Prevention)      │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │ (aspect_scores,        │ (verified_hotels,      │ (verified_aspect_facts)
         │  time_decay_rating,    │  geo_coordinates,      │
         │  overall_trust_score)  │  location_ratings)     │
         ▼                        ▼                        ▼
  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
  │ COMPONENT 2  │         │ COMPONENT 3  │         │ COMPONENT 4  │
  │ Context-     │         │ Optimized    │         │ RAG-Based    │
  │ Aware Recom- │         │ Itinerary &  │         │ Conversational│
  │ mendations   │         │ Food Planner │         │ Assistant    │
  │ (Hansada)    │         │ (Bandara)    │         │ (Gajanayake) │
  └──────────────┘         └──────────────┘         └──────────────┘
```

### 1.2 Individual Research Scope Demarcation (Table 5 Boundary)
As formalized in Table 5 of the approved Proposal Report:
- **My Strict Responsibilities:**
  1. Multi-source scraping & review ingestion pipeline (Booking.com, Google Maps, TripAdvisor).
  2. Text normalization, non-English filtering, and PII anonymization.
  3. Deceptive review detection & aspect-specificity density trust scoring.
  4. Aspect-Based Sentiment Analysis (ABSA) for the 6 core hospitality aspects.
  5. Mathematical exponential time-decay modeling and aspect score aggregation.
  6. Pre-computed data hand-off and REST microservice compliance with `contracts/c1_aspect_scores.json`.
- **Downstream Boundaries (Handled by Team Members):**
  - **Component 2 (Hansada S.A.B.):** Context-aware recommendation engine, tourist personalization, price-performance matching.
  - **Component 3 (Bandara H.A.N.D.):** Itinerary route planning, traveling salesperson optimization, food preference matching.
  - **Component 4 (Gajanayake K.R.K.V.):** RAG chatbot interface, vector embedding retrieval, dynamic conversational itinerary updates.

---

## 2. Chronological Memory Review: The Conversations That Built the Project

Across multiple in-depth collaborative sessions from August 2026 to October 2026, the project evolved through distinct architectural transformations:

```
+--------------------------------------------------------------------------------------------------+
|                              EVOLUTIONARY TIMELINE OF CONVERSATIONS                              |
+--------------------------------------------------------------------------------------------------+
|  Mid-Aug 2026   | Genesis & Scraper Concept Evaluation (Friend's 1-review script rejected)      |
|  Late-Aug 2026  | Discovery Engine & The 45-Field Universal Schema Architecture                  |
|  Mid-Sept 2026  | Academic Proposal Alignment & CRISP-DM Formulation (Supervisor guidance)       |
|  Early-Oct 2026 | Overcoming Anti-Bot Barriers: Booking Modals, DataDome, & G-Maps Multi-Sort   |
|  Mid-Oct 2026   | Dataset Unification (39,399 reviews across 1,025 hotels in all 25 districts)   |
|  Oct 14-15 2026 | Engine Development (TrustEngine, ABSAEngine, Aggregator, FactExtractor)        |
|  Oct 16-17 2026 | Microservice Packaging, J26-DS-345 Sync, Unit Testing, & Metric Benchmarks     |
+--------------------------------------------------------------------------------------------------+
```

### Conversation Milestone 1: Genesis & Conception (Aug 16, 2026 — Session `7696c7cd`)
- **User Prompt & Challenge:** The user presented the first draft of the research component and shared a simple Python script provided by a colleague to scrape basic hotel info.
- **Key Decision:** The colleague's script only scraped basic listing descriptions without user reviews. We realized that an AI data science project on deceptive spam and aspect sentiment requires *thousands of rich, multi-sentence consumer reviews*. We formulated the initial plan documented in [`review_scraping_plan.md`](file:///d:/RESEARCH%20PROJECT/review_scraper/review_scraping_plan.md).

### Conversation Milestone 2: Pagination Breakdown & Multi-Page Discovery (Aug 23–25, 2026 — Sessions `958bde84`, `55999ddc`)
- **User Prompt & Challenge:** After the first test run, the user noted: *"Why you got only 1 review per hotel? How can I possibly get an idea with just 1 review? We need more than 1 review per hotel."*
- **Key Decision & Implementation:**
  - Re-engineered the scraper from a single-page scraper into a **multi-page, deep-paginating crawler**.
  - Engineered the **45-Field Unified Research Dataset Schema** covering 5 logical tiers: Hotel Metadata (16 fields), Core Review Fields (9 fields), Reviewer & Trust Features (6 fields), Stay Context (4 fields), and Trust Engine/ABSA Outputs (10 fields).
  - Implemented fault-tolerant checkpointing via [`scraper_state.json`](file:///d:/RESEARCH%20PROJECT/review_scraper/data/checkpoints/scraper_state.json) so multi-hour scraping runs could recover seamlessly from network drops.

### Conversation Milestone 3: Academic Proposal Integration & Supervisor Alignment (Sept 2026 — Sessions `cce6ec49`, `93a665cb`)
- **Key Discussions:** Alignment with research supervisors Prof. Samantha Thelijjagoda and Dr. Junius Anjana.
- **Decisions Made (Recorded in Appendix 2 Decision Log of Proposal):**
  1. *Time-Decay Rationale:* Instead of static all-time averages that permanently penalize recovering Sri Lankan SMEs, introduce a mathematical exponential time-decay function $w(t) = \exp(-\lambda \Delta t)$ with a 365-day half-life.
  2. *AI Spam Defense:* Recognize that modern LLMs (ChatGPT) generate grammatically flawless but generic reviews. Therefore, aspect specificity density (measuring the density of concrete entities vs. generic fluff) must be included to identify AI spam.
  3. *Downstream Contract:* Standardize output JSON schema (`c1_aspect_scores.json`) to decouple development between Component 1 and the other 3 teammates.

### Conversation Milestone 4: Anti-Bot Defense Engineering & Scraping Breakthroughs (Oct 13–15, 2026 — Sessions `e1da2793`, `40cf848b`, `dfaccae8`)
This period witnessed our most intense technical debugging across 56 conversational turns:
1. **Google Maps 5-Review Virtual DOM Ceiling:**
   - *Problem:* In unauthenticated sessions, Google Maps only rendered 5 reviews into the virtual DOM and refused to expand via standard scroll events.
   - *Breakthrough:* We engineered **Multi-Sort Cycling**. By programmatically clicking and cycling through 4 sort buttons (`Most relevant`, `Newest`, `Highest rating`, `Lowest rating`), the extractor force-hydrated 4 different DOM subsets, successfully yielding **15–25 unique reviews per hotel**!
2. **Booking.com Modal & Overlay Handling:**
   - *Problem:* Dynamic "Sign in to Genius" overlays and GDPR consent prompts blocked review tab clicks.
   - *Breakthrough:* Engineered automated popup dismissal and targeted the underlying `read-all-reviews-button`, paginating via URL parameters.
3. **TripAdvisor DataDome Mitigation:**
   - *Problem:* Direct requests were intercepted by DataDome TLS JA3/JA4 fingerprinting and cryptographic slider challenges.
   - *Breakthrough:* Leveraged Google Maps third-party syndicated review feeds (`button.FNe70d:has-text('All reviews')`) to extract authentic TripAdvisor reviews directly without hitting the DataDome wall, complemented by persistent Microsoft Edge profiling.

### Conversation Milestone 5: Component 1 Engine Implementation & Group Sync (Oct 14–16, 2026 — Sessions `dfaccae8`, `0c72ccc5`)
- Built the standalone analytics package in [`c1_review_engine/`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/):
  - [`trust_engine.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/src/models/trust_engine.py): Multi-factor author credibility, stay verification, text richness, rating disparity, helpful votes boost, and exponential time-decay.
  - [`absa_engine.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/src/models/absa_engine.py): 6-aspect sentiment analyzer using clause splitting, negation windowing (3-token lookback), and intensifiers/diminishers.
  - [`fact_extractor.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/src/models/fact_extractor.py): Grounded factual sentence extractor for Component 4 RAG.
  - [`aggregator.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/src/models/aggregator.py): Aggregates all signals into exact compliance with the shared team contract.
  - Synchronized microservice code and pre-computed datasets into the group repository `D:\RESEARCH PROJECT\J26-DS-345` via [`sync_to_main.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/sync_to_main.py).

---

## 3. Detailed Audit: What Has Been Implemented

### 3.1 Empirical Dataset Collection & Geography
Our actual data collection directly matches and exceeds the targets laid out in the Proposal Report:

| Proposal Specification (SO1 / Section 3.2) | Target in Proposal | Implemented Production System | Status |
| :--- | :--- | :--- | :--- |
| **Total Review Volume** | 35,000 – 50,000 reviews | **39,399 unique verified reviews** (from 59,889 raw) | **PASSED** (Optimal band) |
| **Hotel Property Coverage** | 750 – 1,000 hotels | **1,025 unique hotels** (1,090 discovered entities) | **PASSED** (Exceeded target) |
| **District Distribution** | All 25 districts | **All 25 administrative districts** represented | **PASSED** (100% island-wide) |
| **Multi-Source Ingestion** | Booking.com, Google Maps, TripAdvisor | Booking.com (32,382), Google (6,592), TripAdvisor (425) | **PASSED** (Tri-source dataset) |
| **Language Filtering** | English only | Removed **7,192 non-English reviews** | **PASSED** (Strictly filtered) |
| **Deduplication** | Unique review records | Removed **13,298 duplicate reviews** | **PASSED** (Deduplicated) |
| **PII Anonymization** | Ethical privacy compliance | Author IDs hashed as `USR_XXXXXXXXXX` | **PASSED** (Redacted) |

#### Geographic Stratification Across Sri Lanka's 25 Districts:
- **Southern Province:** Galle (2,632), Matara (2,359), Hambantota (2,224)
- **Central Province:** Kandy (1,535), Matale (2,162), Nuwara Eliya (2,141)
- **Western Province:** Colombo (2,171), Gampaha (1,517), Kalutara (2,224)
- **Uva Province:** Badulla (2,245), Monaragala (1,283)
- **Eastern Province:** Trincomalee (2,352), Batticaloa (1,906), Ampara (1,178)
- **North Central Province:** Anuradhapura (1,879), Polonnaruwa (1,650)
- **North Western Province:** Puttalam (1,546), Kurunegala (705)
- **Sabaragamuwa Province:** Ratnapura (1,783), Kegalle (1,529)
- **Northern Province:** Jaffna (1,250), Mannar (505), Kilinochchi (319), Vavuniya (250), Mullaitivu (54)

---

### 3.2 Mathematical Formulations Implemented in Code

#### A. Review-Level Trust Scoring $T(r)$
Implemented in [`trust_engine.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/src/models/trust_engine.py#L128-L165):
$$T(r) = \operatorname{clip}\left( \left(0.35 \cdot C_{\text{author}} + 0.65 \cdot I_{\text{text}}\right) \cdot V_{\text{stay}} \cdot (1.0 - P_{\text{disp}}^{\text{adj}}) + H_{\text{boost}}, \, 0.05, \, 1.0 \right)$$
- **Author Credibility ($C_{\text{author}}$):** $0.45 + 0.12 \log_{10}(\text{reviews}) + \text{badges}$ (Local Guide $+0.15$, Verified Guest $+0.10$).
- **Informational Richness ($I_{\text{text}}$):** Logarithmic saturation at 120 words: $0.30 + 0.70 \cdot \frac{\ln(1 + \min(120, W))}{\ln(1 + 120)}$.
- **Verification Multiplier ($V_{\text{stay}}$):** $1.0$ for verified stay bookings, $0.80$ for unverified open platforms.
- **Rating Disparity Penalty ($P_{\text{disp}}^{\text{adj}}$):** Penalizes extreme ratings that contradict property baselines, scaled down if the review text is rich and specific.

#### B. Mathematical Exponential Time-Decay Weighting $w(t)$
Implemented in [`trust_engine.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/src/models/trust_engine.py#L67-L76):
$$w(t) = \exp(-\lambda \cdot \Delta t), \quad \text{where } \lambda = \frac{\ln(2)}{t_{\text{half-life}}}$$
- Half-life $t_{\text{half-life}} = 365$ days.
- Reviews submitted today have $w(t) \to 1.0$; reviews from 1 year ago have $w(t) \approx 0.50$; reviews older than 3 years are floored at $0.15$.
- **Academic Impact:** Prevents historic management failures from permanently penalizing newly renovated or improving SME hotels.

#### C. Aspect-Based Sentiment Analysis (ABSA) Engine
Implemented in [`absa_engine.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/src/models/absa_engine.py):
- **Core Aspects:** `food`, `staff`, `room`, `cleanliness`, `location`, `wifi`.
- **Clause Splitting:** Splits sentences across punctuation and contrastive conjunctions (*"but", "however", "although"*).
- **Negation Windowing:** 3-token lookback window (*"not clean"*, *"never friendly"*).
- **Intensifier & Diminisher Modifiers:** Scales lexical sentiment from $[-3.0, +3.0]$ and maps to the standardized $0.0 \text{ to } 5.0$ rating scale.

#### D. Verified Aspect Fact Extraction
Implemented in [`fact_extractor.py`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/src/models/fact_extractor.py):
- Regex pattern filters for grounded facts (e.g., breakfast timings, distances to attractions, pool availability, room amenities).
- Filters out generic emotional fluff (*"had a great time"*, *"worst place"*) and delivers high-trust factual sentences to **Component 4 (RAG Assistant)** to prevent LLM hallucinations.

---

### 3.3 System Verification & Test Results
- **Automated Unit Tests:** 10/10 test cases passing in 0.365s ([`run_test.py --unit-tests`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/run_test.py)).
- **ABSA Benchmark Metrics** ([`evaluation_metrics_report.json`](file:///d:/RESEARCH%20PROJECT/review_scraper/c1_review_engine/data/evaluation_metrics_report.json)):
  - Aspect Detection Accuracy: **93.33%**
  - Sentiment Polarity Accuracy: **80.00%**
  - Macro F1-Score: **73.64%** (Negative F1: 0.909, Positive F1: 0.800, Neutral F1: 0.500)
- **Empirical Trust Distribution:**
  - Mean Overall Trust Score: **0.729 / 1.00**
  - Booking.com (Verified): Mean 0.712
  - Google Maps (Open): Mean 0.812

---

## 4. Proposal Compliance Audit: What Was Promised vs. What Exists

| Proposal Code | Requirement Description | Target Specification | Current Implementation Status | Compliance Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **FR1** | Multi-source scraping & ingestion | Booking, Google Maps, TripAdvisor | 39,399 reviews across 3 platforms | **100% COMPLETE** |
| **FR2** | Text cleaning & normalization | Duplicates, misspellings, noise | Normalizer engine executed; 13k duplicates purged | **100% COMPLETE** |
| **FR3** | PII data anonymization | Redact personal names/identifiers | User hashes `USR_XXXXXXXXXX` used everywhere | **100% COMPLETE** |
| **FR4** | Aspect specificity density metric | Identify LLM / AI-generated spam | Rule-based heuristic implemented; needs ML model | **PARTIAL (Heuristic operational, ML pending)** |
| **FR5** | Tabular ML trust score | LightGBM / XGBoost classifier | Convex mathematical formula; ML tree model pending | **PARTIAL (Formula operational, ML pending)** |
| **FR6** | 6-Aspect ABSA extraction | Food, Staff, Room, Cleanliness, Loc, WiFi | Rule-based lexical ABSA operational; DeBERTa pending | **PARTIAL (Lexical operational, Transformer pending)** |
| **FR7** | Exponential time-decay | Weight recent reviews over past | Implemented via $\exp(-\lambda \Delta t)$ with 365d half-life | **100% COMPLETE** |
| **FR8** | Normalized aspect ratings (0.0–5.0) | Clamped 0.0 to 5.0 scale | Full contract compatibility in Aggregator | **100% COMPLETE** |
| **FR9** | Weak-aspect alerts for SME dashboard | Cleanliness $< 2.5$ alert triggers | Schema field defined in Table 4; alert logic ready | **READY FOR DASHBOARD HOOKUP** |
| **FR10** | Continuous JSON forwarding | Contract schema delivery to shared DB | Synchronized to `J26-DS-345/data` and fast REST API | **100% COMPLETE** |
| **NFR1** | Security & Privacy (PII) | Strict PII redaction | Fully enforced in processing pipeline | **100% COMPLETE** |
| **NFR2** | Pipeline Performance | Process up to 50,000 reviews | Successfully parsed 59,889 raw reviews | **100% COMPLETE** |
| **NFR3** | Fake Review Classifier ROC-AUC | High discriminative accuracy | Tested on heuristic; formal ROC-AUC benchmark pending | **NEEDS FORMAL BENCHMARK** |
| **NFR4** | ABSA Micro/Macro F1-Score | Higher F1 than VADER baseline | Current Macro-F1: 73.64%; fine-tuned DeBERTa target $>85\%$ | **NEEDS DEBERTA FINE-TUNING** |
| **NFR5** | Reliability & Schema Consistency | Prevent downstream pipeline breaks | Validated against `contracts/c1_aspect_scores.json` | **100% COMPLETE** |
| **NFR6** | Audit Explainability | Traceable review counts & trust weights | Review count, trust stats & fact snippets preserved | **100% COMPLETE** |
| **NFR7** | Schema Extensibility | Add new aspects (e.g., "Pool") | Config-driven `ASPECT_CATEGORIES` allows simple addition | **100% COMPLETE** |
| **NFR8** | Database Query Latency | Sub-second retrieval (<1.0s) | Pre-computed JSON cache responds in $< 15$ ms | **100% COMPLETE** |

---

## 5. What We Need to Do: The Research & Dissertation Roadmap

While the operational data scraping, preprocessing, formulaic trust engine, and microservice integration are complete, **the academic dissertation and proposal timeline (M5–M12)** require transitioning from our current heuristic baseline to **full machine learning and deep learning models**, accompanied by rigorous empirical evaluations.

```
========================================================================================
                          NEXT PHASE EXECUTION ROADMAP
========================================================================================

   [ PHASE 1: EXPAND GROUND TRUTH ANNOTATION ] (WBS 2.3)
   - Annotate 300–500 Sri Lankan hotel review sentences across 6 aspects
   - Create balanced positive, negative, and neutral ground-truth sets
   
   [ PHASE 2: TRAIN & FINE-TUNE NLP & ML MODELS ] (SO2, SO3 / WBS 3.2, 4.1)
   - Train LightGBM / XGBoost on aspect specificity density + reviewer features
   - Fine-tune DeBERTa-v3 / RoBERTa on annotated hospitality reviews
   - Benchmark against baseline (VADER / TextBlob)
   
   [ PHASE 3: CONDUCT SCIENTIFIC ABLATION & EVALUATION ] (SO5 / WBS 5.2, 5.3)
   - K-Fold Cross-Validation for Fake Review Classifier (ROC-AUC, Precision, Recall)
   - Macro/Micro F1-score evaluation on ABSA aspects
   - Ablation Study: Quantify rank shifts before vs. after trust & time-decay
   
   [ PHASE 4: FULL REPO INTEGRATION & OWNER DASHBOARD ] (WBS 6.1, 6.2, 6.3)
   - Connect live FastAPI/MongoDB endpoints to Components 2, 3, and 4
   - Operationalize weak-aspect alert webhooks (e.g., Cleanliness < 2.5)
   
   [ PHASE 5: THESIS DISSERTATION & RESEARCH PAPER ] (WBS 6.4)
   - Write Chapters 4 (Design), 5 (Implementation), 6 (Evaluation)
   - Draft and submit research paper to IEEE/Springer conference
```

### 5.1 Work Package 1: Ground Truth Dataset Expansion (WBS 2.3 / Milestone M3–M4)
- **Current State:** 30 gold-standard benchmark sentences in `evaluate_models.py`.
- **What Needs to Be Done:**
  1. Sample 300 to 500 review sentences from our 39,399-review dataset.
  2. Manually annotate each sentence with:
     - Target aspect(s): `food`, `staff`, `room`, `cleanliness`, `location`, `wifi`.
     - Sentiment label: `Positive`, `Negative`, `Neutral`.
     - Deceptive label: `Authentic` vs. `Suspicious/AI-generated`.
  3. Save as `data/processed/ground_truth_annotated_hospitality.csv` to serve as the gold-standard test set for your dissertation defense.

### 5.2 Work Package 2: Train Tabular Trust Classifier (LightGBM / XGBoost) (SO2 / WBS 3.2 / Milestone M5–M6)
- **Current State:** Convex formula combines author history, stay verification, and text richness.
- **What Needs to Be Done:**
  1. **Compute Aspect Specificity Density:** For each review, calculate:
     $$\text{Aspect Specificity Density} = \frac{\text{Count of specific named entities and aspect nouns}}{\text{Total generic adjectives and adverbs}}$$
     *(Research insight: AI bots use generic adjectives like "wonderful", "great", "flawless", whereas humans mention specific items like "live egg hopper station", "noisy AC compressor", "200m from Fort gate").*
  2. **Feature Engineering:** Build feature matrix containing:
     - `aspect_specificity_density`
     - `text_length_words`
     - `author_total_reviews`
     - `is_verified_stay`
     - `sentiment_rating_disparity`
     - `helpful_votes`
     - `review_recency_delta`
  3. **Train LightGBM & XGBoost:** Fit models on labeled benchmark data, perform hyperparameter tuning, and export model weights to `c1_review_engine/models/`.

### 5.3 Work Package 3: Fine-Tune Transformer Model (DeBERTa-v3 / RoBERTa) (SO3 / WBS 4.1 / Milestone M5–M7)
- **Current State:** Rule-based lexicons with negation windowing achieving 93.3% aspect detection and 73.6% macro F1.
- **What Needs to Be Done:**
  1. Use Hugging Face Transformers (`microsoft/deberta-v3-small` or `roberta-base`).
  2. Set up fine-tuning script in Google Colab Pro (as budgeted: LKR 14,000 in Table 11) using PyTorch.
  3. Train for aspect extraction and 3-class sentiment polarity classification.
  4. Compare performance: Lexical Baseline vs. VADER vs. DeBERTa-v3.

### 5.4 Work Package 4: Formal Empirical Evaluation & Ablation Study (SO5 / WBS 5.2, 5.3 / Milestone M8–M10)
This is the most critical scientific contribution of the dissertation:
1. **K-Fold Cross-Validation:** Run 5-fold cross-validation on the fake review detection model and generate:
   - Confusion Matrix
   - ROC-AUC Curve
   - Precision-Recall Curve
2. **Ablation Study (Ranking Stability Analysis):**
   - Rank all 1,025 hotels under 3 experimental conditions:
     - *Condition A (Raw Baseline):* Simple unweighted arithmetic mean of all-time star ratings (current Google/Booking method).
     - *Condition B (Trust-Only):* Aspect ratings filtered by trust score without time-decay.
     - *Condition C (Full Proposed System):* Trust-weighted + exponential time-decay aggregation.
   - **Measure Metric:** Spearman's Rank Correlation Coefficient ($\rho$) and shift in ranking for recovering SME hotels. Show that hotels with recent verified improvements jump up the ranks, while spam-boosted hotels drop.

### 5.5 Work Package 5: Downstream Microservice Integration & Dashboard Alerts (WBS 6.1–6.3 / Milestone M8–M11)
1. **Shared Database Connection:** Deploy PostgreSQL / MongoDB instance for the group project and push pre-computed JSON aspect records.
2. **REST Endpoints:** Ensure FastAPI endpoints in `c1_review_engine/src/main.py` respond in $< 1.0$ second (NFR8).
   - `GET /api/v1/aspects/{hotel_id}`
   - `GET /api/v1/hotels/verified-summary`
3. **Weak-Aspect Alert Generation (FR9):** Implement an automated alert trigger:
   - If an aspect rating (e.g., `cleanliness` or `wifi`) drops below $2.5 / 5.0$ in the last 60 days, trigger a structured alert for the Hotel Owner Dashboard.

### 5.6 Work Package 6: Dissertation Chapters & Research Paper Writing (WBS 6.4 / Milestone M10–M12)
1. **Thesis Chapters:**
   - *Chapter 3: Methodology* (Dataset sampling, feature engineering, mathematical formulation of $T(r)$ and $w(t)$).
   - *Chapter 4: Implementation* (Multi-source Playwright scraping, Multi-Sort Cycling breakthrough, DeBERTa fine-tuning).
   - *Chapter 5: Results & Empirical Discussion* (ROC-AUC curves, F1 comparisons, ablation study tables).
2. **Research Publication:** Prepare manuscript for an IEEE or Springer conference (budget allocated: LKR 15,000 for publication fee).

---

## 6. Actionable Next-Step Task Checklist

| Priority | Task Description | Target Milestone | Tool / Framework |
| :---: | :--- | :---: | :--- |
| **P1** | Create 300-sentence annotated Sri Lankan ground truth dataset | M3–M4 | CSV / Excel annotation |
| **P2** | Implement aspect specificity density calculation module | M4–M5 | Python / SpaCy / NLTK |
| **P3** | Train LightGBM & XGBoost classifiers for fake review detection | M5–M6 | Scikit-learn / LightGBM |
| **P4** | Fine-tune DeBERTa-v3 on hospitality ABSA dataset | M5–M7 | PyTorch / Hugging Face (Colab) |
| **P5** | Run formal Ablation Study comparing Condition A, B, and C rankings | M8–M9 | Scipy / Pandas (Spearman $\rho$) |
| **P6** | Operationalize weak-aspect alerts (`Cleanliness < 2.5`) | M10 | FastAPI / JSON Webhooks |
| **P7** | Draft Dissertation Chapter 4 (Design) & Chapter 5 (Evaluation) | M10–M12 | LaTeX / Word SLIIT Template |

---
*Report compiled from complete project repository memory, code implementations, test suites, and academic proposal guidelines.*
