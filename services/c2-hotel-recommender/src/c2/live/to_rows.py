"""
One live Booking.com hotel (search card + hotel page) -> rows with the SAME columns as the research CSVs:
    hotel_row()   -> hotels_booking.csv          (facts, rating, facilities, location)
    words_row()   -> hotel_words_facilities.csv  (name words and the full facility list, fac_* columns)
    price_row()   -> model_table.csv             (one price: hotel + dates + guests, with every model column)
All the rules come from preprocessing.ipynb (prep_rules.py), applied to one hotel instead of the whole file.
"""
import math
import re
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .. import settings
from . import prep_rules as P

LIVE_ROW_ID_START = 10_000_000          # live prices get row ids far above the research rows


def _num(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def room_facts_raw(page_info, rates):
    """The scraper's combined-file columns room_types_raw / meal_plans_raw / hotel_chain_raw
    (same format as common/rooms.py hotel_room_facts)."""
    rooms, meals = {}, []
    for r in rates:
        name = r.get("room_name_raw")
        if name:
            key = r.get("room_id_raw") or name
            if key not in rooms:
                parts = [name, r.get("room_size_raw"), r.get("room_beds_raw"),
                         f"sleeps {r['room_sleeps_raw']}" if r.get("room_sleeps_raw") else None]
                rooms[key] = " | ".join(p for p in parts if p)
        meal = r.get("meal_plan_raw")
        if meal and meal not in meals:
            meals.append(meal)
    return {"hotel_chain_raw": (page_info or {}).get("hotel_chain_raw") or "",
            "room_types_raw": " || ".join(rooms.values()), "meal_plans_raw": "; ".join(meals)}


def hotel_row(hotel_id, name, cluster, details, facts, centre, known_chains):
    """build_hotel_table() for one hotel. centre = (lat, lon) of the town (median of its hotels or the town point)."""
    g = lambda k: details.get(k) or ""
    lat, lon = _num(details.get("latitude_raw")), _num(details.get("longitude_raw"))
    h = {"booking_hotel_id": hotel_id, "hotel_name": name, "destination_cluster": cluster,
         "property_type": details.get("property_type_raw") or "Hotel", "latitude": lat, "longitude": lon,
         "address_region": details.get("address_region_raw"), "postal_code": details.get("postal_code_raw"),
         "page_rating": _num(details.get("overall_rating_raw")),
         "page_review_count": _num(details.get("overall_review_count_raw"))}
    for s in P.SUBSCORES:
        h[f"subscore_{s}"] = _num(details.get(f"subscore_{s}_raw"))
    h["validation_value_for_money"] = _num(details.get("subscore_value_for_money_raw"))
    h.update(P.amenity_features(g("popular_facilities_raw"), g("room_amenities_raw"), g("listed_facilities_raw"),
                                g("property_highlights_raw"), g("facilities_raw")))
    h["km_from_town_centre"] = (round(float(P.haversine_km(lat, lon, centre[0], centre[1])), 2)
                                if centre and not (np.isnan(lat) or np.isnan(lon)) else np.nan)
    h["details_complete"] = int(not details.get("details_warnings"))
    rf = P.room_facts(facts["room_types_raw"], facts["meal_plans_raw"], facts["hotel_chain_raw"])
    h.update(rf)
    chain = (rf.get("chain_name") or "").strip()
    h["chain_group"] = ("unknown" if not rf["has_room_facts"] else "independent" if not chain else
                        chain if chain in known_chains else "other chain")
    return h


def words_row(hotel_id, name, details, columns):
    """words_and_facilities() for one hotel, using the research file's own column list."""
    out = {"booking_hotel_id": hotel_id}
    low = (name or "").lower()
    facilities = details.get("facilities_raw") or ""
    items = P._facility_items(facilities, details.get("room_amenities_raw") or "")
    slugs = {"fac_" + P._slug(t) for t in items}
    for col in columns:
        if col.startswith("name_"):
            out[col] = int(bool(re.search(rf"\b{re.escape(col[5:])}\b", low)))
        elif col.startswith("fac_"):
            out[col] = int(col.rstrip("_") in slugs) if facilities else np.nan
    return out


def _event(d):
    return any(d.month == m and lo <= d.day <= hi for m, lo, hi in settings.EVENT_WINDOWS)


def price_row(card, *, hotel, words, checkin, checkout, adults, children, scraped_at, destination_clean,
              room_list_raw, row_id):
    """One priced stay with every model column (cell 12, 15, 18 and 32 of preprocessing.ipynb, for one card)."""
    nights = (checkout - checkin).days
    r = {"row_id": row_id, "source_ota": "booking", "hotel_id_on_source": hotel["booking_hotel_id"],
         "booking_hotel_id": hotel["booking_hotel_id"], "hotel_name": hotel["hotel_name"],
         "destination_cluster": hotel["destination_cluster"], "destination_clean": destination_clean,
         "checkin_date": checkin.isoformat(), "checkout_date": checkout.isoformat(), "nights": nights,
         "num_adults": adults, "num_children": children, "date_purpose": "live", "is_extra_date": 0,
         "room_type_name_raw": card.get("room_type_name_raw"), "scraped_at_utc": scraped_at}
    stars = P.score(card.get("star_rating_raw"))
    r["stars"] = stars
    r["stars_is_estimate"] = int(card.get("star_rating_kind_raw") in ("squares", "diamonds"))
    r["star_band"] = P.star_band(pd.Series([stars], dtype=float)).iloc[0]
    r["review_score"] = P.score(card.get("review_score_raw"))
    r["review_count"] = P.count(card.get("review_count_raw"))
    r["log_review_count"] = float(np.log1p(r["review_count"])) if r["review_count"] == r["review_count"] else np.nan
    r["location_score"] = P.location_score(card.get("location_score_raw"))
    r["distance_km"] = P.distance_km(card.get("distance_text_raw"), None)
    r["units_left"] = P.units_left(card.get("units_left_raw"))
    r["cancellation_type"] = P.cancellation_type(card.get("cancellation_policy_raw"))
    r["free_cancellation"] = int(r["cancellation_type"] in ("free", "extended_free"))
    r["pay_at_property"] = P.pay_at_property(card.get("prepayment_policy_raw"))
    r["meal_plan"] = P.meal_plan((card.get("room_config_raw") or "") + " | " + (card.get("badges_raw") or ""))
    r["meal_level"] = P.MEAL_LEVELS[r["meal_plan"]]
    r["breakfast_included"] = float(r["meal_level"] >= 1)
    r["dinner_included"] = float(r["meal_level"] >= 2)
    r["preferred_partner"] = int("preferred" in (card.get("badges_raw") or "").lower())
    r["deal_shown"] = int(bool(card.get("deal_badge_raw")))
    r["new_listing"] = int(bool(re.search(r"Opened in 20", card.get("badges_raw") or "", re.I)))
    r.update(P.price_columns({"source_ota": "booking", "nights": str(nights),
                              "displayed_price_raw": card.get("displayed_price_raw"),
                              "original_price_raw": card.get("original_price_raw"),
                              "taxes_and_fees_raw": card.get("taxes_and_fees_raw"),
                              "deal_badge_raw": card.get("deal_badge_raw")}))
    r.update(P.room_features(card.get("room_type_name_raw"), card.get("room_config_raw")))
    rooms = P.clean_room_list(P.room_list(room_list_raw))
    m2 = P.card_room_m2(card.get("room_type_name_raw"), rooms)
    r["card_room_m2"] = m2 if m2 == m2 else P.size_m2(card.get("room_config_raw"))
    whole = P.whole_place_or_dorm(card.get("room_config_raw"), card.get("room_type_name_raw"))
    r["whole_place_or_dorm_reason"], r["is_whole_place_or_dorm"] = whole, int(bool(whole))
    scraped_day = datetime.fromisoformat(scraped_at).date()
    stay = [checkin + timedelta(days=i) for i in range(nights)]
    r["lead_time_days"] = (checkin - scraped_day).days
    r["checkin_month"], r["checkin_weekday"] = checkin.month, checkin.weekday()
    r["stay_has_weekend_night"] = int(any(d.weekday() in (4, 5) for d in stay))
    r["guests_total"], r["has_children"] = adults + children, int(children > 0)
    r["is_event_date"] = int(any(_event(d) for d in stay))
    price = r["price_incl_tax_lkr_night"]
    r["log_price_incl_tax"] = float(np.log(price)) if price == price and price > 0 else np.nan
    r["is_sold_out"] = int(not (price == price and price > 0))
    # hotel facts and words, like the merge in cell 32
    for k, v in hotel.items():
        if k not in ("hotel_name", "destination_cluster"):
            r.setdefault(k, v)
    for k, v in words.items():
        r.setdefault(k, v)
    r["has_hotel_details"] = int(hotel.get("latitude") == hotel.get("latitude"))
    r["card_room_m2_or_hotel_median"] = r["card_room_m2"] if r["card_room_m2"] == r["card_room_m2"] \
        else hotel.get("room_m2_median", np.nan)
    smallest = hotel.get("room_m2_min", np.nan)
    r["card_room_vs_smallest"] = r["card_room_m2"] / smallest if (smallest == smallest and smallest) else np.nan
    r["hotel_group_id"] = hotel["booking_hotel_id"]
    return r


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
