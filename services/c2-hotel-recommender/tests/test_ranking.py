"""The join: must-haves never broken, review checks, prices, where to book, output for Component 3."""
import json

from src.c2.c1_client import DummyC1, FileC1
from src.c2.output import to_c3
from src.c2.pricing import room_plan
from src.c2.ranker import recommend
from tests.helpers import make_request


def ranked(result, i=0):
    return result["results"][i]["ranked_hotels"]


def test_budget_must_never_broken(data):
    result = recommend(make_request(bmax=8000), top_k=10, data=data)
    hotels = ranked(result)
    assert hotels, "some hotels fit Rs 8,000"
    for h in hotels:
        assert h["price"]["price_per_night_lkr"] <= 8000
    assert "k-lux" not in [h["hotel_id"] for h in hotels]
    assert result["results"][0]["removed"]["must-have 'budget' not met"] >= 1


def test_feature_and_hospital_musts(data):
    result = recommend(make_request(must=["pool", "near_hospital"], health=["heart"]), top_k=10, data=data)
    for h in ranked(result):
        assert h["must_check"]["pool"] == "yes"
        assert h["must_check"]["near_hospital"] == "yes"
        assert h["nearest_hospital"]["km"] <= 5
    ids = [h["hotel_id"] for h in ranked(result)]
    assert "k-far" not in ids            # 14 km from the hospital
    assert "k-cheap" not in ids          # no pool


def test_reviews_can_remove_a_listed_feature(data, tmp_path):
    """Gap 2: the listing says pool, but trusted reviews say it is closed -> not 'yes' -> removed for a pool must."""
    dummy = DummyC1(data.hotels)
    items = [dummy.get(h) for h in data.hotels.index]
    for it in items:
        it["verified_aspect_facts"] = []
        if it["hotel_id"] == "k-closed":
            it["verified_aspect_facts"] = ["Reviews from September 2026 say the pool was closed for repairs."]
    path = tmp_path / "c1.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    c1 = FileC1(str(path))
    result = recommend(make_request(must=["pool"]), top_k=10, data=data, c1=c1)
    ids = [h["hotel_id"] for h in ranked(result)]
    assert "k-closed" not in ids and "k-pool" in ids


def test_not_sure_musts_come_last_and_are_marked(data):
    # low_steps: lift listed for k-pool, k-closed, k-lux, g-beach; nothing listed for the others -> not sure
    result = recommend(make_request(must=["low_steps"]), top_k=10, data=data)
    hotels = ranked(result)
    flags = [h["musts_confirmed"] for h in hotels]
    assert flags == sorted(flags, reverse=True)          # confirmed first
    for h in hotels:
        if not h["musts_confirmed"]:
            assert "NOT CONFIRMED" in h["explanation"]


def test_overpriced_and_where_to_book(data):
    result = recommend(make_request(), top_k=10, data=data)
    by_id = {h["hotel_id"]: h for h in ranked(result)}
    assert by_id["k-lux"]["price"]["price_status"] == "Potentially Overpriced"
    assert by_id["k-cheap"]["price"]["price_status"] == "Within Expected Price Range"
    assert by_id["k-pool"]["price"]["best_platform"] == "Trip.com"
    assert "trip.com" in by_id["k-pool"]["booking_url"]
    assert "checkin=2027-01-20" in by_id["k-cheap"]["booking_url"]


def test_groups_become_rooms():
    assert room_plan(4, 0)[0] == [(2, 0), (2, 0)]
    assert room_plan(3, 0)[0] == [(2, 0), (1, 0)]
    assert room_plan(2, 2)[0] == [(2, 2)]
    assert room_plan(None, None)[0] == [(2, 0)]


def test_town_outside_our_data(data):
    result = recommend(make_request(town="Peradeniya"), top_k=3, data=data)
    r = result["results"][0]
    assert r["ranked_hotels"], "Kandy hotels are within 10 km of Peradeniya"
    assert any("not one of our 15 data towns" in n for n in r["notes"])


def test_output_for_component_3(data):
    req = make_request(bmax=10000, health=["heart"], must=["near_hospital"], total=50000)
    result = recommend(req, top_k=3, data=data)
    no_consent = to_c3(req, result, health_consent=False)
    assert no_consent["health"] == {"consent_given": False}             # health needs not sent
    yes = to_c3(req, result, health_consent=True)
    assert yes["health"]["needs"] == ["heart"]
    h = yes["recommended_hotels"][0]
    for key in ("latitude", "longitude", "nights", "total_hotel_cost_lkr", "budget_left_lkr", "booking_url",
                "nearest_hospital_km", "price_label", "book_on"):
        assert key in h
    assert h["budget_left_lkr"] == 50000 - h["total_hotel_cost_lkr"]


def test_dummy_c1_is_marked(data):
    result = recommend(make_request(), top_k=2, data=data)
    assert "DUMMY" in result["c1_source"]
    assert "DUMMY" in ranked(result)[0]["explanation"]
