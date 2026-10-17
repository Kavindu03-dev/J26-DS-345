# Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine

**Project ID:** J26-DS-345  
**Owner:** DISSANAYAKE L.G.D.P.M. – IT23252554  
**Supervisors:** Prof. Samantha Thelijjagoda, Dr. Junius Anjana  
**Port:** `8001`

---

## 1. Overview
Component 1 is the foundational data and intelligence engine for the Sri Lankan Hotel Recommendation System. It continuously transforms multi-source customer reviews into trust-weighted, time-decayed aspect ratings (Food, Staff, Room, Cleanliness, Location, WiFi) and grounded factual snippets for downstream RAG assistance.

## 2. Directory Layout
```
services/c1-review-engine/
├── src/                    # FastAPI microservice & inference engines
│   ├── models/             # TrustEngine, ABSAEngine, FactExtractor, Aggregator
│   ├── main.py             # FastAPI entrypoint (Port 8001)
│   ├── config.py           # Configuration & filepaths
│   ├── batch_enricher.py   # Bulk enrichment pipeline
│   └── evaluate_models.py  # Model evaluation benchmark runner
├── scraper/                # Multi-source Playwright review scraping pipeline
├── tests/                  # Automated pytest / unittest test suite
├── data/                   # Pre-computed aspect profiles & seeds
├── notebooks/              # Research notebooks & experiments
├── run_test.py             # Interactive CLI evaluation tool
├── Dockerfile              # Docker container definition
└── requirements.txt        # Python dependencies
```

## 3. Running the Service Locally

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Run Automated Unit Tests
```powershell
python run_test.py --unit-tests
# or
pytest
```

### Start the Microservice API
```powershell
uvicorn src.main:app --reload --port 8001
```
- Health Check: `http://localhost:8001/health`
- Aspect Scores API: `http://localhost:8001/hotels/{hotel_id}/aspect-scores`

## 4. Contract Conformance
Conforms 100% to `contracts/c1_aspect_scores.json`:
- `hotel_id`: String
- `aspect_scores`: Normalized 0.0 to 5.0 for 6 core aspects
- `overall_trust_score`: 0.0 to 10.0
- `time_decay_rating`: 1.0 to 5.0
- `verified_aspect_facts`: List of factual sentences for Component 4
- `last_updated`: ISO Timestamp
