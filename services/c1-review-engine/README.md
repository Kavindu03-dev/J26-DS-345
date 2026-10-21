# Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine

**Project ID:** J26-DS-345  
**Owner:** DISSANAYAKE L.G.D.P.M. – IT23252554  
**Supervisors:** Prof. Samantha Thelijjagoda, Dr. Junius Anjana  
**Port:** `8001`

---

## 1. Overview
Component 1 is the foundational data and intelligence engine for the Sri Lankan Hotel Recommendation System. It continuously transforms multi-source customer reviews into trust-weighted, time-decayed aspect ratings (Food, Staff, Room, Cleanliness, Location, WiFi) and grounded factual snippets for downstream RAG assistance.

> 🎓 **Complete Academic Defense Guide (A to Z):** For an all-in-one plain English explanation of Component 1 prepared for your supervisor and examination panel, see **[COMPONENT_1_COMPLETE_EXPLANATION_A_TO_Z.md](file:///d:/RESEARCH%20PROJECT/J26-DS-345/services/c1-review-engine/COMPONENT_1_COMPLETE_EXPLANATION_A_TO_Z.md)**.  
> 📖 **Operational & Terminal Manual:** For a file-by-file terminal command reference and troubleshooting, see [COMPONENT_1_OPERATIONAL_GUIDE.md](file:///d:/RESEARCH%20PROJECT/J26-DS-345/services/c1-review-engine/COMPONENT_1_OPERATIONAL_GUIDE.md).

## 2. Directory Layout
```
services/c1-review-engine/
├── src/
│   ├── 01_data_preparation/      # Ground truth, 50 hard challenges & Kappa study
│   ├── 02_trust_classification/  # ASD metrics, LightGBM/XGBoost & feature ablation
│   ├── 03_aspect_sentiment/      # DeBERTa-v3 ABSA & 4-way benchmark comparison
│   ├── 04_aggregation_alerts/    # 39k bulk enrichment & weak aspect alert engine
│   ├── 05_scientific_evaluation/ # Half-life sensitivity, backtesting & ranking ablation
│   ├── models/                   # Runtime engines (ASD, Trust, ABSA, Alert, Aggregator)
│   ├── __init__.py               # Central dynamic loader & module exports
│   ├── config.py                 # File paths, thresholds & hyperparameters
│   └── main.py                   # FastAPI microservice endpoint (Port 8001)
├── tests/                        # 36 Automated pytest / contract tests (100% passing)
├── data/                         # Processed research datasets & precomputed profiles
├── notebooks/                    # Research notebooks & Colab GPU fine-tuning
├── Dockerfile                    # Container configuration
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
