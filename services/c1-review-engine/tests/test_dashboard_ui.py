"""
Unit tests for Component 1 Interactive Dashboard and Helper API Endpoints.
"""

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_api_root_served():
    """Verifies that the root API endpoint returns service metadata."""
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    data = resp_root.json()
    assert data["service"] == "C1 Trust-Aware Review & Aspect Sentiment Engine"
    assert data["status"] == "healthy"
    assert data["port"] == 8001


def test_api_pipeline_overview():
    """Verifies that Step 1 pipeline funnel and cleaning examples endpoint returns exact dataset metrics."""
    resp = client.get("/api/v1/pipeline-overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "funnel" in data
    funnel = data["funnel"]
    assert funnel["raw_scraped_reviews"] == 59889
    assert funnel["duplicates_removed"] == 13298
    assert funnel["non_english_removed"] == 7192
    assert funnel["final_verified_reviews"] == 39399
    assert funnel["final_verified_hotels"] == 1025
    assert "cleaning_examples" in data
    assert len(data["cleaning_examples"]) >= 4


def test_api_rankings_national_and_district():
    """Verifies that Step 4 rankings endpoint returns all 1,025 hotels and supports district filtering."""
    resp_all = client.get("/api/v1/rankings")
    assert resp_all.status_code == 200
    data_all = resp_all.json()
    assert data_all["total_national_hotels"] == 1025
    assert len(data_all["hotels"]) == 1025
    first_hotel = data_all["hotels"][0]
    assert "rank_before" in first_hotel
    assert "rank_after" in first_hotel
    assert "rank_shift" in first_hotel
    assert "score_before" in first_hotel
    assert "score_after" in first_hotel

    # Filter by district (e.g., Galle)
    resp_galle = client.get("/api/v1/rankings?district=Galle&sort_by=biggest_gain")
    assert resp_galle.status_code == 200
    data_galle = resp_galle.json()
    assert data_galle["selected_district"] == "Galle"
    assert data_galle["total_matching"] > 0
    for h in data_galle["hotels"]:
        assert h["district"].lower() == "galle"
        assert "district_rank_before" in h
        assert "district_rank_after" in h


def test_api_list_hotels():
    """Verifies that the hotel selector endpoint returns a valid hotel list."""
    resp = client.get("/api/v1/hotels?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "hotel_id" in first
    assert "hotel_name" in first
    assert "district" in first
    assert "rating" in first


def test_api_analyze_review_sandbox():
    """Verifies that the live sandbox endpoint computes ASD and ABSA metrics."""
    # Test authentic review
    sample_text = "The egg hoppers and pol sambol at breakfast were great. Hot water shower in room 204."
    resp = client.post("/api/v1/analyze-review", json={"review_text": sample_text, "hotel_baseline_rating": 8.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["word_count"] > 5
    assert data["concrete_entities_count"] >= 1
    assert "detected_entities" in data
    assert "trust_score" in data
    assert data["trust_score"] >= 5.0
    assert data["is_suspected_ai_spam"] is False

    # Test suspected spam review
    spam_text = "Breathtaking paradise! Heavenly, immaculate, magnificent and extraordinary! Best stay ever!"
    resp_spam = client.post("/api/v1/analyze-review", json={"review_text": spam_text, "hotel_baseline_rating": 8.0})
    assert resp_spam.status_code == 200
    data_spam = resp_spam.json()
    assert data_spam["generic_fluff_count"] >= 3
    assert data_spam["fluff_to_entity_ratio"] >= 3.0


def test_api_benchmarks_summary():
    """Verifies that the benchmarks endpoint returns verified research metrics."""
    resp = client.get("/api/v1/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert "kappa_agreement" in data
    assert data["kappa_agreement"]["aspect_kappa"] == 0.9159
    assert "absa_comparison" in data
    assert "ranking_ablation" in data
    assert data["ranking_ablation"]["spearman_rho"] == 0.9449
