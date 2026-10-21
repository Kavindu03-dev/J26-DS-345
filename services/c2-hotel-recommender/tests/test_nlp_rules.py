"""The rule check after the NLP model (no Ollama needed)."""
from src.nlp.run_stage1 import RATES_FIXED, apply_rules
from src.nlp.understand import find_not_needed


def model_output(**trip):
    base = {"trip": {"stay_in": [], "want_to_visit": [], "check_in": None, "check_out": None, "nights": None,
                     "dates_fixed": None, "adults": None, "children": None, "travel_style": None,
                     "budget_total_lkr": None},
            "hotel": {"budget_max_lkr": None, "must": [], "wishes": {}}, "health": [], "interests": []}
    base["trip"].update(trip)
    return base


def test_not_needed():
    assert find_not_needed("Group of 6 friends, we don't need breakfast but AC is a must") == ["breakfast"]
    assert set(find_not_needed("We do not need a pool or a bar")) == {"pool", "bar"}
    assert set(find_not_needed("No need for a pool. We don't want a bar either.")) == {"pool", "bar"}
    assert find_not_needed("Room only please") == ["breakfast"]
    assert find_not_needed("A pool would be nice") == []
    assert find_not_needed("We don't need much, just a pool") == []
    assert find_not_needed("We don't need a pool, a bar or breakfast.") == ["pool", "bar", "breakfast"]
    assert find_not_needed("We don't need breakfast, AC is a must") == ["breakfast"]


def test_dates_and_money_rules():
    pred = model_output(stay_in=["kandyy"], check_in="2026-05-10", nights=3)
    pred["hotel"]["budget_max_lkr"] = 60
    final, notes = apply_rules(pred, "hotl in kandyy 3 nights from 10 may under $60 a night", "2026-10-19",
                               RATES_FIXED)
    assert final["trip"]["stay_in"] == ["Kandy"]
    assert final["trip"]["check_in"] == "2027-05-10"           # past date -> next year
    assert final["trip"]["check_out"] == "2027-05-13"          # code counts the nights
    assert final["hotel"]["budget_max_lkr"] == round(60 * RATES_FIXED["USD"])
    assert "budget" in final["hotel"]["must"]


def test_health_becomes_must():
    pred = model_output(stay_in=["Galle"])
    pred["health"] = ["heart"]
    final, _ = apply_rules(pred, "My father has heart problems, hotel in Galle", "2026-10-19", RATES_FIXED)
    assert "near_hospital" in final["hotel"]["must"]
