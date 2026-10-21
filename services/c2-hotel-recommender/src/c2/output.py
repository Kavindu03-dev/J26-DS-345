"""
Turn the ranker's result into the two outputs:
  - for Component 4: contracts/c2_hotel_ranking.json (ranked_hotels with price, explanation) + extra fields
  - for Component 3: Output 2 in output-format.md (location, nights, cost, budget left, hospital, health with consent)
"""
import copy
import math

import numpy as np

CONTRACT_PLATFORMS = ["Booking.com", "Agoda", "Trip.com"]


def plain(obj):
    """numpy numbers -> Python numbers, NaN -> None (so the answer is valid JSON)."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is not None and not isinstance(obj, (str, int, bool)) and str(obj) in ("nan", "NaT", "<NA>"):
        return None
    return obj


def contract_hotel(h):
    """One hotel in the team contract format. Contract fields first; everything else is extra (allowed)."""
    p = h["price"]
    return {
        "hotel_id": h["hotel_id"],
        "name": h["name"],
        "rank": h["rank"],
        "score": h["score"],
        "price": {
            "expected_price_lkr": p["expected_price_lkr"] if p["expected_price_lkr"] is not None
            else p["price_per_night_lkr"],
            "price_status": p["price_status"],
            "best_platform": p["best_platform"],
            "offers": [{"platform": s, "price_lkr": p["offers_seen"][s]} for s in CONTRACT_PLATFORMS
                       if s in p["offers_seen"]],
            # extra
            "seen_checkin": p["seen_checkin"],
            "price_basis": "LKR per night for all rooms, taxes included; offers are the prices collected for "
                           f"check-in {p['seen_checkin']}",
            "price_per_night_lkr": p["price_per_night_lkr"],
            "your_expected_price_lkr": p["your_expected_price_lkr"],
            "price_vs_fair_pct": p["price_vs_fair_pct"],
            "price_vs_your_fair_pct": p["price_vs_your_fair_pct"],
            "what_if": p["what_if"],
            "breakdown": p.get("breakdown"),
            "caution": p.get("caution"),              # very few reviews: the fair price is less certain          # where the fair price goes: pool, bar, breakfast, beach, rest
            "explanation": p.get("explanation"),      # SHAP: GET this url for why the fair price is this number
            "where_to_book": {"advice": p["where_to_book_advice"], "notes": p["where_to_book_notes"]},
            "rooms": p["rooms"],
        },
        "explanation": h["explanation"],
        # extra fields
        "town": h["town"], "latitude": h["latitude"], "longitude": h["longitude"], "km_from_town": h["km_from_town"],
        "nights": h["nights"], "total_hotel_cost_lkr": h["total_hotel_cost_lkr"],
        "budget_left_lkr": h["budget_left_lkr"], "booking_url": h["booking_url"], "links": h["links"],
        "nearest_hospital": h["nearest_hospital"], "musts_confirmed": h["musts_confirmed"],
        "must_check": h["must_check"], "feature_checks": h["feature_checks"],
        "free_cancellation": h["free_cancellation"], "score_parts": h["score_parts"],
        "found_live": h.get("found_live"), "photos": h.get("photos", []),
        "review_scores_c1": h["c1"],
    }


def to_contract(result, town_index=0):
    """Response for POST /recommendations (one town)."""
    if not result["results"]:
        return {"ranked_hotels": [], "notes": ["No town given."]}
    r = result["results"][town_index]
    return {
        "ranked_hotels": [contract_hotel(h) for h in r["ranked_hotels"]],
        "town": r["town"], "hotels_considered": r["hotels_considered"], "removed": r["removed"],
        "with_price": r["with_price"], "meet_all_musts": r["meet_all_musts"], "musts_not_sure": r["musts_not_sure"],
        "live_search": r.get("live_search", {"needed": False, "reason": ""}),
        "notes": r["notes"] + result["data_notes"],
        "c1_source": result["c1_source"], "price_model_used": result["price_model_used"],
        "prices_collected": result["prices_collected"],
    }


def to_c3(request, result, health_consent=False):
    """Output 2: what Component 3 needs to plan the days (output-format.md)."""
    trip = copy.deepcopy(request.get("trip", {}))
    hotels = []
    for r in result["results"]:
        for h in r["ranked_hotels"]:
            hotels.append({
                "rank": h["rank"], "hotel_id": h["hotel_id"], "name": h["name"], "town": h["town"],
                "stay_in": r["town"], "latitude": h["latitude"], "longitude": h["longitude"],
                "price_per_night_lkr": h["price"]["price_per_night_lkr"], "nights": h["nights"],
                "total_hotel_cost_lkr": h["total_hotel_cost_lkr"], "budget_left_lkr": h["budget_left_lkr"],
                "price_label": "overpriced" if h["price"]["price_status"] == "Potentially Overpriced"
                else "suitable",
                "book_on": h["price"]["best_platform"], "booking_url": h["booking_url"],
                "nearest_hospital_km": h["nearest_hospital"]["km"],
                "photo_url": (h.get("photos") or [{}])[0].get("photo_url"),
            })
    health = {"consent_given": bool(health_consent)}
    if health_consent:
        health["needs"] = list(request.get("health", []))     # categories only, never the sentence
    return {"trip": trip, "interests": request.get("interests", []), "health": health,
            "recommended_hotels": hotels}


def contract_request_to_internal(body):
    """POST /recommendations body (team contract) -> the request format the ranker uses (Output 1)."""
    from datetime import date

    from .schemas import ACCESS_WORDS, HEALTH_TO_MUST, INTEREST_WORDS_MAP, STYLE_WORDS

    ci, co = body.get("check_in"), body.get("check_out")
    nights = body.get("nights")
    if ci and co and not nights:
        nights = (date.fromisoformat(co) - date.fromisoformat(ci)).days
    budget = body.get("budget_lkr") or {}
    must = list(body.get("must") or [])
    if budget.get("max") is not None and "budget" not in must:
        must.insert(0, "budget")
    for need in body.get("accessibility_needs") or []:
        word = ACCESS_WORDS.get(str(need).strip().lower().replace(" ", "_"))
        if word and word not in must:
            must.append(word)
    for h in body.get("health") or []:                      # health needs always become must-haves
        word = HEALTH_TO_MUST.get(h)
        if word and word not in must:
            must.append(word)
    interests = []
    for i in body.get("interests") or []:
        key = str(i).strip().lower()
        word = INTEREST_WORDS_MAP.get(key) or next((v for k, v in INTEREST_WORDS_MAP.items() if k in key), None)
        if word and word not in interests:
            interests.append(word)
    style = str(body.get("travel_purpose") or "").strip().lower()
    return {
        "trip": {"stay_in": [body["destination"]], "want_to_visit": body.get("want_to_visit") or [],
                 "check_in": ci, "check_out": co, "nights": nights, "dates_fixed": body.get("dates_fixed"),
                 "adults": body.get("adults"), "children": body.get("children"),
                 "travel_style": STYLE_WORDS.get(style), "budget_total_lkr": body.get("budget_total_lkr")},
        "hotel": {"budget_max_lkr": budget.get("max"), "must": must, "wishes": body.get("wishes") or {}},
        "health": body.get("health") or [],
        "interests": interests,
        "not_needed": body.get("not_needed") or [],
    }
