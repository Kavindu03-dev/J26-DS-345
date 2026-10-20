# Component 1: Operational Guide & Terminal Execution Handbook

**Project Title:** Tourism Review Analysis and Recommendation System for Sri Lankan Hotels  
**Project ID:** J26-DS-345  
**Component 1:** Trust-Aware Review Analysis & Aspect Sentiment Engine  
**Candidate:** DISSANAYAKE L.G.D.P.M. (IT23252554)  
**Supervisors:** Prof. Samantha Thelijjagoda, Dr. Junius Anjana  
**Degree:** B.Sc. (Hons) in Information Technology Specializing in Data Science, SLIIT  

---

## 1. Executive Architecture & Data Flow

Component 1 transforms raw, messy, multi-platform hotel reviews into **trust-weighted, time-decayed aspect profiles** and **operational weakness alerts**.

```
[Raw Scraped Reviews] (Booking.com, Google Maps, TripAdvisor: 59.8k raw -> 39.4k clean)
         │
         ▼
[Preprocessing & Normalization] (research_dataset_unified.csv: 1,025 hotels, 25 districts)
         │
         ├───► [ASD Calculator & Feature Engineering] ──► [ML Trust Classifier (LightGBM/XGBoost)]
         │                                                            │
         ├───► [6-Aspect ABSA Engine] (Lexical / DeBERTa) ◄───────────┘ (Trust-Weighting)
         │           │
         │           ▼
         ├───► [Exponential Time-Decay w(t)] (365-day half-life recency weighting)
         │           │
         │           ▼
         ├───► [AspectScoreAggregator] ──► [contracts/c1_aspect_scores.json]
         │           │                           │
         │           ▼                           ├──► Component 2 (Collaborative Recommender)
         │     [FastAPI Server] (Port 8001)       └──► Component 4 (RAG Chatbot Verified Facts)
         │           │
         │           ▼
         ├───► [WeakAspectAlertEngine] ──► Hotel Owner Dashboard (Cleanliness, WiFi < 2.5)
         │
         └───► [RankingAblationStudy] ──► Scientific Validation (Spearman rho, MARS for Chapter 5)
```

---

## 2. Environment Setup & Prerequisites

### Working Directory Rule
> [!IMPORTANT]
> Always open your terminal and ensure your current working directory is `services/c1-review-engine`:
> ```powershell
> cd "d:\RESEARCH PROJECT\J26-DS-345\services\c1-review-engine"
> ```
> All relative paths (e.g., `data/processed/...`, `models/...`) expect this directory as the root.

### Install All Required Dependencies
Run the following in PowerShell:
```powershell
pip install -r requirements.txt
```

Verify your installation:
```powershell
python -c "import fastapi, uvicorn, lightgbm, xgboost, sklearn, scipy, vaderSentiment; print('[OK] All dependencies installed successfully!')"
```

---

## 3. Complete File-by-File Inventory

Every single file in this component has a dedicated purpose. Here is the complete breakdown:

### A. Root Directory Files
| File Name | Purpose | What It Does / What to Look For |
| :--- | :--- | :--- |
| `requirements.txt` | Dependency Manifest | Contains all Python libraries needed (`fastapi`, `uvicorn`, `scikit-learn`, `lightgbm`, `xgboost`, `scipy`, `vaderSentiment`, `pytest`, `joblib`). |
| `pytest.ini` | Test Configuration | Tells `pytest` where to find tests (`testpaths = tests`). |
| `run_test.py` | Standalone CLI Tester | Interactive command-line script to test single reviews or run the test suite. |
| `Dockerfile` | Container Specification | Docker configuration to package Component 1 as an independent container. |
| `README.md` | Quickstart Overview | High-level summary of the service and endpoints. |

### B. Core Inference Engines (`src/models/`)
| File Name | Purpose | Inputs | Logic & Outputs |
| :--- | :--- | :--- | :--- |
| `asd_calculator.py` | Aspect Specificity Density | Raw review text | Scans for Sri Lankan hospitality entities (hoppers, AC, bathroom, Sigiriya) vs. generic promotional fluff. Outputs `asd_score` $\in [0, 1]$ and flags AI-generated spam. |
| `trust_engine.py` | Heuristic Trust & Decay | Review dictionary | Computes Author Credibility $C_{\text{author}}$, Stay Verification $V_{\text{stay}}$, Text Richness $I_{\text{text}}$, Disparity Penalty $P_{\text{disp}}$, and Exponential Decay $w(t) = \exp(-\lambda \Delta t)$. Outputs $T(r) \in [0.05, 1.0]$. |
| `ml_trust_classifier.py` | ML Trust Inference Wrapper | Review dictionary | Extracts 8 tabular features, loads `lightgbm_trust_classifier.joblib`, and predicts `spam_probability` and `ml_credibility_score`. Fallback to ASD heuristic if model missing. |
| `absa_engine.py` | Aspect-Based Sentiment | Review text & rating | Splits review into clauses, detects target aspects (*food, staff, room, cleanliness, location, wifi*), applies negation windows, and scores sentiment 0.0 to 5.0. |
| `fact_extractor.py` | Grounded Fact Extraction | Hotel reviews list | Filters for high-trust reviews ($T(r) \ge 0.65$), extracts concrete factual sentences (e.g. *"Pool is open until 10 PM"*), and formats them for Component 4 RAG. |
| `aggregator.py` | Unified Contract Aggregator | Hotel metadata + reviews | Merges Trust, ABSA, Time-Decay, and Facts into the exact JSON contract schema required by `contracts/c1_aspect_scores.json`. |
| `alert_engine.py` | Weak-Aspect Alert Engine | Hotel review streams | Compares rolling 60-day aspect scores against historical baselines. Triggers `CRITICAL`, `WARNING`, or `INFO` alerts, extracts negative guest quotes, and prescribes remediation actions (FR9). |

### C. The 5 Numbered Pipeline Stages (`src/`)

To match the modular step-by-step engineering architecture of Component 3, Component 1 is organized into **5 chronological, numbered pipeline stages**:

```text
src/
├── 01_data_preparation/          <── Data & Benchmark Pipeline (Numbered)
│   ├── 01_build_ground_truth.py
│   ├── 02_generate_hard_challenges.py
│   └── 03_audit_annotation_agreement.py
│
├── 02_trust_classification/      <── Trust & Anti-Spam Pipeline (Numbered)
│   ├── 01_calculate_asd_metrics.py
│   ├── 02_train_trust_classifiers.py
│   ├── 03_audit_feature_ablation.py
│   └── 04_evaluate_trust_engine.py
│
├── 03_aspect_sentiment/          <── ABSA NLP Pipeline (Numbered)
│   ├── 01_lexical_clause_engine.py
│   ├── 02_train_deberta_transformer.py
│   └── 03_benchmark_absa_comparison.py
│
├── 04_aggregation_alerts/        <── Aggregation & Alerts Pipeline (Numbered)
│   ├── 01_extract_verified_facts.py
│   ├── 02_aggregate_contract_scores.py
│   ├── 03_detect_weak_aspect_alerts.py
│   └── 04_batch_precompute_profiles.py
│
├── 05_scientific_evaluation/     <── Academic Ablation & Backtesting (Numbered)
│   ├── 01_evaluate_halflife_sensitivity.py
│   ├── 02_evaluate_temporal_backtest.py
│   └── 03_run_master_ablation_study.py
│
├── models/                       <── Reusable OOP Engines (FastAPI & Test Backend)
├── config.py                     <── Central Paths & Hyperparameters
└── main.py                       <── FastAPI Live Microservice (Port 8001)
```

| Stage | Script Name | What Command Runs It | Artifacts Generated |
| :--- | :--- | :--- | :--- |
| **01** | `01_build_ground_truth.py` | `python src/01_data_preparation/01_build_ground_truth.py` | Stratified 400-sentence benchmark (`data/processed/ground_truth_annotated_hospitality.csv`). |
| **01** | `02_generate_hard_challenges.py` | `python src/01_data_preparation/02_generate_hard_challenges.py` | 50 adversarial challenge sentences (`data/processed/hard_linguistic_challenges.csv`). |
| **01** | `03_audit_annotation_agreement.py`| `python src/01_data_preparation/03_audit_annotation_agreement.py` | Cohen's Kappa $\kappa$ metrics (`annotation_agreement_report.json` & `.csv`). |
| **02** | `01_calculate_asd_metrics.py` | `python src/02_trust_classification/01_calculate_asd_metrics.py` | Demonstrates ASD entity extraction vs. AI promotional fluff detection. |
| **02** | `02_train_trust_classifiers.py` | `python src/02_trust_classification/02_train_trust_classifiers.py` | Trains LightGBM & XGBoost with 5-fold CV (`models/lightgbm_trust_classifier.joblib`). |
| **02** | `03_audit_feature_ablation.py` | `python src/02_trust_classification/03_audit_feature_ablation.py` | Evaluates 4 feature configs, proving +48.04% Macro-F1 lift (`trust_classifier_ablation_report.csv`). |
| **02** | `04_evaluate_trust_engine.py` | `python src/02_trust_classification/04_evaluate_trust_engine.py` | Demonstrates review-level trust scoring $T(r) \in [0.05, 1.0]$ and anomaly detection. |
| **03** | `01_lexical_clause_engine.py` | `python src/03_aspect_sentiment/01_lexical_clause_engine.py` | Demonstrates clause splitting, negation windowing, and 0.0 to 5.0 aspect scoring. |
| **03** | `02_train_deberta_transformer.py` | `python src/03_aspect_sentiment/02_train_deberta_transformer.py` | Fine-tunes DeBERTa-v3 on 450 sentences (`models/deberta_absa_config.json`). |
| **03** | `03_benchmark_absa_comparison.py`| `python src/03_aspect_sentiment/03_benchmark_absa_comparison.py` | Compares VADER vs Lexical vs RoBERTa vs DeBERTa across splits (`absa_models_comparison_report.csv`). |
| **04** | `01_extract_verified_facts.py` | `python src/04_aggregation_alerts/01_extract_verified_facts.py` | Extracts high-trust factual sentences for Component 4 RAG chatbot grounding. |
| **04** | `02_aggregate_contract_scores.py`| `python src/04_aggregation_alerts/02_aggregate_contract_scores.py` | Aggregates reviews into `contracts/c1_aspect_scores.json` canonical format. |
| **04** | `03_detect_weak_aspect_alerts.py`| `python src/04_aggregation_alerts/03_detect_weak_aspect_alerts.py` | Evaluates rolling 60-day degradation alerts (`CRITICAL`, `WARNING`, `INFO`) for hotel managers. |
| **04** | `04_batch_precompute_profiles.py`| `python src/04_aggregation_alerts/04_batch_precompute_profiles.py` | Ingests 39,399 reviews, precomputes `data/hotel_aspect_profiles.json` and `.csv`. |
| **05** | `01_evaluate_halflife_sensitivity.py`| `python src/05_scientific_evaluation/01_evaluate_halflife_sensitivity.py` | Sweeps $\tau \in \{90, 180, 365, 730\}$ days on 1,025 hotels (`halflife_sensitivity_report.csv`). |
| **05** | `02_evaluate_temporal_backtest.py`| `python src/05_scientific_evaluation/02_evaluate_temporal_backtest.py` | Backtests on 724 hotels $<2026$ vs $\ge 2026$, proving lower error (`temporal_backtest_report.csv`). |
| **05** | `03_run_master_ablation_study.py`| `python src/05_scientific_evaluation/03_run_master_ablation_study.py` | Ranks 1,025 hotels under A, B, and C, computing Spearman $\rho$, $\tau$, and MARS. |

> 📊 **Scientific Audit & Backtesting Documentation:**
> - [PHASE_1_SCIENTIFIC_AUDIT_AND_EMPIRICAL_RESULTS.md](file:///d:/RESEARCH%20PROJECT/J26-DS-345/docs/c1-review-analysis/PHASE_1_SCIENTIFIC_AUDIT_AND_EMPIRICAL_RESULTS.md): 100% ROC-AUC investigation, feature gain audit, sensitivity curves, and temporal backtest.
> - [PHASE_2_NLP_RIGOR_AND_ANNOTATION_AGREEMENT.md](file:///d:/RESEARCH%20PROJECT/J26-DS-345/docs/c1-review-analysis/PHASE_2_NLP_RIGOR_AND_ANNOTATION_AGREEMENT.md): 50 Hard Linguistic Challenges, Cohen's Kappa, and multi-model benchmark across splits.
> - [PHASE_3_TRANSFORMER_TRAINING_AND_SYSTEM_HARMONIZATION.md](file:///d:/RESEARCH%20PROJECT/J26-DS-345/docs/c1-review-analysis/PHASE_3_TRANSFORMER_TRAINING_AND_SYSTEM_HARMONIZATION.md): DeBERTa-v3 training pipeline, Colab GPU notebook, rating scale harmonization, and literature audit.

### D. Automated Test Suite (`tests/`)
| Test File Name | What It Validates |
| :--- | :--- |
| `test_health.py` | Verifies microservice `/health` endpoint responds HTTP 200 with indexed statistics. |
| `test_api_contract.py` | Verifies `/hotels/{hotel_id}/aspect-scores` adheres 100% to `contracts/c1_aspect_scores.json`, plus tests alert endpoints. |
| `test_trust_engine.py` | Validates trust bounds $[0.05, 1.0]$, time decay recency ordering, and suspicious review flagging. |
| `test_asd_calculator.py` | Validates entity recognition, promotional fluff detection, and ASD scoring. |
| `test_ml_trust_classifier.py` | Validates LightGBM model inference on authentic vs synthetic spam text. |
| `test_absa_engine.py` | Validates aspect keyword detection, clause negation (*"not clean"*), and contract score clipping $[0.0, 5.0]$. |
| `test_alert_engine.py` | Validates rolling 60-day degradation alerts, severity assignment, negative quote extraction, and recommendations. |
| `test_ablation_study.py` | Validates ranking mathematics, Spearman $\rho$, Kendall $\tau$, and Mean Absolute Rank Shift (MARS). |
| `test_nlp_rigor_phase2.py` | Validates 50 hard challenges schema, Cohen's Kappa bounds, and the empirical Lexicon collapse. |
| `test_transformer_phase3.py` | Validates 450-sample unified training pool, DeBERTa model manifest, and 0.0-5.0 score scale bounds. |
| `test_numbered_pipeline.py` | Validates that all 5 numbered pipeline stages exist, are importable, and expose all 17 scripts. |

### E. Notebooks (`notebooks/`)
| File Name | Purpose | How to Use |
| :--- | :--- | :--- |
| `03_transformer_absa_finetuning.ipynb` | Interactive Colab/VS Code Notebook | Open in VS Code or upload to Google Colab Pro. Runs on 450 unified sentences with GPU for interactive training, confusion matrix plots, and ONNX export. |
| `03_transformer_absa_finetuning.py` | Headless Python Training Script | CLI alternative: `python notebooks/03_transformer_absa_finetuning.py --model_name microsoft/deberta-v3-small` |

---

## 4. Terminal Command Playbook (How to Run Everything Independently)

Below are the exact commands you can copy-paste into your PowerShell terminal to operate and inspect Component 1.

---

### Step 1: Run the Full Test Suite (Health Check)
Verify that all 36 unit and contract tests are passing.

```powershell
pytest -v
```

**Expected Terminal Output:**
```text
tests/test_ablation_study.py::TestRankingAblationStudy::test_statistical_metrics_calculation PASSED
tests/test_absa_engine.py::TestABSAEngine::test_aspect_detection_and_negation PASSED
...
tests/test_trust_engine.py::TestTrustEngine::test_trust_score_range PASSED
======================= 23 passed, 1 warning in ~3s =======================
```
- **If it passes:** All engines, models, and contract rules are operational.
- **If a test fails:** PyTest will display the exact file and line number. Check section 5 below for troubleshooting.

---

### Step 2: Run the Tabular ML Trust Classifier Training
Trains the LightGBM and XGBoost models on the 8-feature ground truth matrix using 5-fold Stratified Cross-Validation.

```powershell
python src/train_trust_classifier.py
```

**What Happens:**
1. Loads `data/processed/ground_truth_annotated_hospitality.csv`.
2. Computes the 8 tabular features (ASD, fluff ratio, verification status, author history log, disparity).
3. Trains LightGBM and XGBoost across 5 folds.
4. Saves model weights to:
   - `models/lightgbm_trust_classifier.joblib`
   - `models/xgboost_trust_classifier.json`
5. Saves evaluation metrics to:
   - `data/processed/trust_classifier_evaluation.json`

**How to Inspect the Results:**
```powershell
python -c "import json; d=json.load(open('data/processed/trust_classifier_evaluation.json')); print('LightGBM ROC-AUC:', d['models']['LightGBM']['mean_roc_auc']); print('XGBoost ROC-AUC:', d['models']['XGBoost']['mean_roc_auc'])"
```

---

### Step 3: Run the Multi-Model ABSA Benchmark
Compares VADER Baseline vs. Rule-Based Lexicon vs. Pre-trained RoBERTa vs. DeBERTa-v3 on the 400-sentence hospitality benchmark.

```powershell
python src/benchmark_absa_comparison.py
```

**What Happens:**
1. Evaluates all 4 models on identical sentences.
2. Computes Accuracy, Macro-Precision, Macro-Recall, and Macro-F1.
3. Outputs comparison table directly in the console.
4. Saves results to:
   - `data/processed/absa_models_comparison_report.csv`
   - `data/processed/absa_models_comparison_report.json`

**How to Inspect the CSV Table:**
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/absa_models_comparison_report.csv'); print(df[['model_name', 'accuracy_percentage', 'macro_f1_percentage']])"
```

---

### Step 4: Run the Scientific Ranking Ablation Study
Evaluates the 39,399 reviews across all 1,025 hotels in Sri Lanka to demonstrate the empirical impact of Trust Scoring and Exponential Time-Decay (SO5 / WBS 5.2).

```powershell
python src/ablation_study.py
```

**What Happens:**
1. Computes scores for all 1,025 hotels under:
   - **Condition A:** Naive all-time average (Status Quo).
   - **Condition B:** Trust-weighted average (Deceptive review filtered).
   - **Condition C:** Full proposed system (Trust-weighted + Exponential time-decay).
2. Computes Spearman's $\rho$, Kendall's $\tau$, and Mean Absolute Rank Shift (MARS).
3. Identifies Top Recovering SME Hotels (rank gainers) and Top Spam-Penalized Hotels (rank droppers).
4. Saves full rankings to:
   - `data/processed/ablation_ranking_results.csv`
   - `data/processed/ablation_metrics_summary.json`

**How to Inspect Top Recovering SME Hotels:**
```powershell
python -c "import json; d=json.load(open('data/processed/ablation_metrics_summary.json')); print(json.dumps(d['case_study_recovering_sme_hotels'][:3], indent=2))"
```

---

### Step 5: Start the Live FastAPI Microservice Server
Launches the Component 1 REST API server on port `8001`.

```powershell
uvicorn src.main:app --reload --port 8001
```

**What Happens:**
1. Loads and indexes hotels and reviews into high-speed memory.
2. Serves endpoints at `http://127.0.0.1:8001`.
3. Auto-reloads whenever code files are modified.

> [!TIP]
> Keep this terminal open while testing API endpoints in a second terminal or in your web browser.

---

### Step 6: Query and Test the REST API Endpoints
Open a **new** PowerShell terminal (or browser) and test the live endpoints:

#### A. Health Check
```powershell
curl http://localhost:8001/health
```
*Expected Response:*
```json
{"service":"c1-review-engine","status":"ok","indexed_hotels":1025,"total_reviews":39399}
```

#### B. Hotel Aspect Scores (Contract Verification)
Test with a real hotel (*Ivy Lane Galle Fort*):
```powershell
curl http://localhost:8001/hotels/HTL_GAL_IVYLANEG_EE74/aspect-scores
```
*Expected Response:*
```json
{
  "hotel_id": "HTL_GAL_IVYLANEG_EE74",
  "aspect_scores": {
    "food": 4.2,
    "staff": 4.6,
    "room": 4.1,
    "cleanliness": 4.5,
    "location": 4.8,
    "wifi": 3.9
  },
  "overall_trust_score": 8.74,
  "time_decay_rating": 4.41,
  "verified_aspect_facts": [
    "Location is inside the historic Dutch Fort within walking distance to lighthouse.",
    "Staff helped carry heavy luggage up the stairs."
  ],
  "last_updated": "2026-10-19T03:45:00Z"
}
```

#### C. Hotel Weak-Aspect Alerts (Requirement FR9)
```powershell
curl http://localhost:8001/hotels/HTL_GAL_IVYLANEG_EE74/alerts
```
*Expected Response:*
```json
{
  "hotel_id": "HTL_GAL_IVYLANEG_EE74",
  "hotel_name": "Ivy Lane Galle Fort",
  "active_alerts_count": 0,
  "alerts": []
}
```

#### D. Island-Wide Critical Alerts Scan
Scan for hotels in Sri Lanka that have active `CRITICAL` operational alerts:
```powershell
curl http://localhost:8001/api/v1/alerts/critical?limit=5
```
Or filter by district:
```powershell
curl "http://localhost:8001/api/v1/alerts/critical?district=Kandy&limit=3"
```

---

### Step 7: Run the Transformer Fine-Tuning Notebook

You have two ways to run the DeBERTa / RoBERTa fine-tuning:

#### Option A: In VS Code (Local)
1. Open [`notebooks/03_transformer_absa_finetuning.ipynb`](file:///d:/RESEARCH%20PROJECT/J26-DS-345/services/c1-review-engine/notebooks/03_transformer_absa_finetuning.ipynb).
2. Select your Python kernel in the top-right corner.
3. Click **"Run All"** to execute cell-by-cell and generate the Confusion Matrix plot inline.

#### Option B: In Google Colab Pro (GPU Cloud)
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `notebooks/03_transformer_absa_finetuning.ipynb`.
3. In Colab, upload `data/processed/ground_truth_annotated_hospitality.csv` to the session storage.
4. Set **Runtime $\to$ Change runtime type $\to$ T4 GPU** (or A100).
5. Click **Runtime $\to$ Run all**.

---

## 5. Error Diagnosis & Troubleshooting Guide

| Issue / Error Message | Root Cause | Exact Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | Terminal was executed from the wrong folder. | Make sure your working directory is `services/c1-review-engine`. Run: `cd "d:\RESEARCH PROJECT\J26-DS-345\services\c1-review-engine"`. |
| `ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8001)` | Another process is already using port `8001`. | Terminate the old process: Run `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess -Force` in PowerShell, then re-run uvicorn. |
| `FileNotFoundError: ... research_dataset_unified.csv` | File path was moved or looking in root. | Ensure `data/processed/research_dataset_unified.csv` exists. In `config.py`, paths resolve relative to `services/c1-review-engine/data/`. |
| `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated` | Harmless deprecation warning from Starlette/FastAPI internal library. | Can be safely ignored. Does not affect test correctness or runtime execution. |
| `ImportError: cannot import name 'joblib'` | Optional ML dependency missing in active venv. | Run: `pip install joblib scikit-learn lightgbm xgboost`. |

---

## 6. How to Inspect Output Data with 1-Line Commands

You do not need to open large 50 MB CSV files in Excel (which can freeze or alter date formats). Use these quick terminal one-liners to inspect the data yourself:

#### Inspect Master Dataset Summary
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/research_dataset_unified.csv', nrows=10); print('Total Columns:', len(df.columns)); print('Sample Columns:', df.columns.tolist()[:10])"
```

#### Check Review Count per Platform
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/research_dataset_unified.csv', usecols=['source_platform']); print(df['source_platform'].value_counts())"
```

#### Check Reviews Count per District
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/research_dataset_unified.csv', usecols=['district']); print(df['district'].value_counts().head(10))"
```

#### View Sample Generated Alerts
```powershell
python -c "from src.models.alert_engine import WeakAspectAlertEngine; engine = WeakAspectAlertEngine(); print('[OK] WeakAspectAlertEngine initialized and ready for streaming!')"
```

---

## 7. Downstream Contract Integration Map

Component 1 delivers data to three other components in the group research system:

1. **Component 2 (Context-Aware Recommendation Engine - Team Member 2):**
   - Consumes: [`data/hotel_aspect_profiles.json`](file:///d:/RESEARCH%20PROJECT/J26-DS-345/services/c1-review-engine/data/hotel_aspect_profiles.json) and `GET /hotels/{hotel_id}/aspect-scores`.
   - Purpose: Uses trust-weighted aspect ratings (Food, Cleanliness, Staff, WiFi) to match user travel personas and filter out deceptive high-rating properties.

2. **Component 4 (RAG Conversational Chatbot - Team Member 4):**
   - Consumes: The `verified_aspect_facts` array from `GET /hotels/{hotel_id}/aspect-scores`.
   - Purpose: Grounds chatbot responses with verified factual evidence extracted from reviews (e.g., *"Breakfast includes vegan options"*), preventing hallucination.

3. **Hotel Owner Dashboard UI:**
   - Consumes: `GET /hotels/{hotel_id}/alerts` and `GET /api/v1/alerts/critical`.
   - Purpose: Displays real-time operational alerts when Cleanliness, Food, or WiFi quality drops below $2.5 / 5.0$, with direct quotes and suggested remediation actions.

---

*This guide is saved at [`services/c1-review-engine/COMPONENT_1_OPERATIONAL_GUIDE.md`](file:///d:/RESEARCH%20PROJECT/J26-DS-345/services/c1-review-engine/COMPONENT_1_OPERATIONAL_GUIDE.md) as the operational manual for Component 1.*
