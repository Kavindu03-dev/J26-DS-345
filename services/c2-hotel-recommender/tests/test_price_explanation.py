"""Where the fair price goes (what-if breakdown) and why it is this number (SHAP), with a tiny trained model."""
from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.c2.data import HotelData, set_data
import pandas as pd

from src.c2.explain import (GROUP_NAMES, _class_words, exact_shap, explain_price, few_reviews_caution, group_of,
                            nice_date, review_count)
from src.c2.ranker import recommend
from src.c2.unseen_models import _make_model
from tests.helpers import make_request

CAT = ["destination_clean"]
NUM = ["num_adults", "num_children", "has_pool", "n_pools", "has_bar", "is_beachfront", "breakfast_included",
       "meal_level", "stars", "free_cancellation", "lead_time_days", "checkin_month", "checkin_weekday",
       "stay_has_weekend_night", "is_event_date"]


@pytest.fixture(scope="module")
def priced(fixture_dirs):
    """The made-up data with a tiny gradient boosting price model (same pipeline as the real one)."""
    d = HotelData(*fixture_dirs, load_model=False)
    rows = d.model_rows
    model = _make_model(CAT, NUM, {"learning_rate": 0.1, "max_iter": 60, "max_leaf_nodes": 7, "min_samples_leaf": 3})
    model.fit(rows[CAT + NUM], np.log(rows.price_incl_tax_lkr_night.values))
    d.model, d.features = model, CAT + NUM
    return d


def test_every_column_has_one_group():
    for c in ["destination_clean", "has_pool", "fac_pool_bar", "fac_pool_table", "breakfast_included", "stars",
              "name_resort", "lead_time_days", "num_adults", "deal_shown", "room_sea_view", "has_spa"]:
        assert group_of(c) in GROUP_NAMES
    assert group_of("fac_pool_table") == "Other facilities"            # a billiard table, not a pool


def test_labels_and_few_reviews_caution():
    row = pd.Series({"stars": np.nan, "review_score": 10.0, "review_count": 1.0, "name_bungalow": 1, "name_hotel": 0})
    assert _class_words(row) == "no star rating, 10/10 from 1 review, 'bungalow' in the name"
    assert few_reviews_caution(review_count(row)).startswith("Only 1 review on Booking.com")
    assert few_reviews_caution(review_count(pd.Series({"review_score": 8.5, "review_count": 250.0}))) is None
    assert few_reviews_caution(review_count(pd.Series({"page_rating": np.nan}))).startswith("No reviews")
    assert few_reviews_caution(None) is None                                  # not known: no warning
    assert nice_date("2026-12-20") == "20 Dec 2026"


def test_shap_parts_add_up_to_the_fair_price(priced):
    rid = int(priced.model_rows[priced.model_rows.booking_hotel_id == "k-pool"].row_id.iloc[0])
    e = explain_price(priced, rid)
    assert e["available"] and e["checks"]["parts_add_up_to_the_fair_price"]
    assert e["checks"]["same_as_textbook_shapley_formula"] in (True, None)     # None: shap not installed
    row = priced.model_rows.loc[[rid], priced.features]
    assert e["fair_price_lkr"] == pytest.approx(float(np.exp(priced.model.predict(row))[0]), rel=1e-6)
    start = np.exp(priced.model.predict(priced.typical_rows[priced.features])).mean()
    assert e["start_lkr"] == pytest.approx(start, rel=1e-6)
    assert e["start_lkr"] + sum(p["lkr"] for p in e["parts"]) == pytest.approx(e["fair_price_lkr"], rel=1e-6)
    assert {p["group"] for p in e["parts"]} <= set(GROUP_NAMES)
    assert e["sentences"] and explain_price(priced, rid)["from_cache"]


def test_textbook_formula_gives_the_same_parts():
    # a made-up price with 3 groups: 100 + 50 if A, + 20 if B, + 30 more if A and B together
    price = lambda known: np.array([100 + 50 * k[0] + 20 * k[1] + 30 * (k[0] and k[1]) for k in np.asarray(known, bool)])
    parts, start, _, _ = exact_shap(price, 3)
    assert start == pytest.approx(100)
    assert parts == pytest.approx([65, 35, 0])                         # the 30 is shared fairly between A and B


def test_training_hotel_needs_the_model_that_never_saw_it(priced):
    rid = int(priced.model_rows[priced.model_rows.booking_hotel_id == "k-far"].row_id.iloc[0])
    e = explain_price(priced, rid)
    assert not e["available"] and "unseen_models" in e["reason"]


def test_breakdown_adds_up(priced):
    req = make_request("Kandy", ci="2026-12-09", nights=2, adults=2, not_needed=["pool"])
    res = recommend(req, top_k=8, today=date(2026, 10, 20), data=priced)
    hotels = res["results"][0]["ranked_hotels"]
    assert hotels
    for h in hotels:
        b = h["price"]["breakdown"]
        assert b is not None
        shown = [p for p in b["parts"] if p.get("lkr") is not None]
        assert b["everything_else_lkr"] + sum(p["lkr"] for p in shown) == b["fair_price_lkr"]
        taken = [p["lkr"] for p in shown if p["taken_off"]]
        assert abs(b["your_fair_price_lkr"] - (b["fair_price_lkr"] - sum(taken))) <= 60 * (len(taken) + 1)   # rounding
        assert b["your_fair_price_lkr"] == h["price"]["your_expected_price_lkr"]
        assert b["fair_price_lkr"] == h["price"]["expected_price_lkr"]
        assert {p["status"] for p in b["parts"]} <= {"absent", "too_small", "not_needed", "not_there", "kept"}
        for p in b["parts"]:
            assert p["has_it"] or p["status"] == "absent"
        e = h["price"]["explanation"]
        assert e["available"] == (h["hotel_id"] != "k-far") and e["url"].startswith(f"/hotels/{h['hotel_id']}/")


def test_price_explanation_endpoint(priced, data):
    old = data                                  # the session data (conftest)
    set_data(priced)
    try:
        c = TestClient(__import__("src.main", fromlist=["app"]).app)
        rid = int(priced.model_rows[priced.model_rows.booking_hotel_id == "g-beach"].row_id.iloc[0])
        r = c.get(f"/hotels/g-beach/price-explanation?row_id={rid}&check_in=2027-01-23&nights=2&today=2026-10-20")
        assert r.status_code == 200 and r.json()["available"]
        assert "check-in 23 Jan 2027" in r.json()["explains"]
        r = c.get("/hotels/g-beach/price-explanation?check_in=2027-01-23&today=2026-10-20")   # no row_id: picks one
        assert r.status_code == 200 and r.json()["available"]
        assert c.get(f"/hotels/k-pool/price-explanation?row_id={rid}").status_code == 404     # not this hotel's price
        assert c.get("/hotels/nope/price-explanation").status_code == 404
    finally:
        set_data(old)
