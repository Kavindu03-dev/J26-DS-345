"""
The join: understood request -> hotels -> checks -> prices -> ranked list.

Input: the NLP output (Output 1 format), plus two optional keys:
    "not_needed": ["pool", "bar", "breakfast", "beachfront"]   things the tourist said they don't need
    "health_consent": true/false                               may health needs go to Component 3?

Order of the steps (research-gaps.md):
    Gap 1 must-haves -> hotels in the town -> Gap 2 feature check (listing + Component 1 reviews)
    -> Gap 5 hospital distance and access -> Gap 3 fair price for this tourist -> score -> Gap 4 where to book

Must-haves are never broken: a hotel that fails one is removed. A hotel where a must-have is only
"not sure" (no information) is shown only after all confirmed hotels, and marked.
"""
from datetime import date
from urllib.parse import urlencode

import numpy as np

from . import settings
from .c1_client import make_c1
from .data import get_data
from .features import NICE_NAME, check_features
from .geo import km
from .explain import few_reviews_caution, nice_date, review_count
from .pricing import (FEATURE_TO_FACILITY, NOT_THERE_MESSAGES, WHAT_IF_MESSAGES, PriceModel, booking_links,
                      cancellation_advice, date_features, has_facility, pick_stay, room_plan)

SITE_COLS = {"Booking.com": "price_incl_tax_lkr_night", "Trip.com": "wtb_tripcom", "Agoda": "wtb_agoda"}


def _num(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _rs(x):
    return f"Rs {int(round(x, -2)):,}"


# ---------------- step 1: hotels in or near the town ----------------
def find_hotels(data, town, top_k):
    """Hotels for this town and a note. Our 15 data towns use their own hotels; other towns use a radius."""
    hotels = data.hotels
    point = data.towns.point(town)
    in_data = data.towns.data_town(town)
    if in_data:
        cand = hotels[hotels.destination_cluster == in_data].copy()
        note = None
    elif point:
        d = km(point[0], point[1], hotels.latitude.to_numpy(float), hotels.longitude.to_numpy(float))
        cand, note = hotels.iloc[0:0].copy(), None
        for radius in settings.SEARCH_RADIUS_KM:
            mask = d <= radius
            if mask.sum() >= top_k or radius == settings.SEARCH_RADIUS_KM[-1]:
                cand = hotels[mask].copy()
                note = (f"{town} is not one of our 15 data towns. Showing hotels within {radius} km of {town}."
                        if mask.sum() else
                        f"{town} is not one of our 15 data towns and we have no hotels within {radius} km of it.")
                if not mask.sum():
                    centres = hotels.groupby("destination_cluster")[["latitude", "longitude"]].median()
                    dist = km(point[0], point[1], centres.latitude.to_numpy(float), centres.longitude.to_numpy(float))
                    i = int(np.argmin(dist))
                    note += f" The nearest town in our data is {centres.index[i]} ({dist[i]:.0f} km)."
                break
    else:
        return hotels.iloc[0:0].copy(), f"Town '{town}' not found."
    if point is not None and len(cand):
        cand["km_from_town"] = np.round(km(point[0], point[1], cand.latitude.to_numpy(float),
                                           cand.longitude.to_numpy(float)), 1)
    else:
        cand["km_from_town"] = cand.get("km_from_town_centre")
    return cand, note


# ---------------- step 2: prices for each hotel ----------------
def price_hotels(data, cand, rooms, check_in, nights, today, not_needed, checks):
    """For each hotel: the collected price, fair price, your-dates estimate and personal fair price."""
    model = PriceModel(data)
    plans = {}
    for hid in cand.index:
        when, rows = pick_stay(data.stay_index.get(hid), rooms, check_in, approx_from_main=data.is_live(hid))
        if not rows:
            continue
        main = rows[rooms[0]]
        rid = main["row_id"]
        your_dates = date_features(check_in, nights, today) if check_in else {}
        # the fair price at your dates: asked to the model that never saw this hotel (it made its fair price)
        model.ask((hid, "seen"), rid, unseen=True)
        model.ask((hid, "yours"), rid, your_dates, unseen=True)
        # what-if (same as the notebooks: the final model): each priced facility the hotel has, switched off
        model.ask((hid, "base"), rid, your_dates)
        mrow = data.model_rows.loc[rid] if model.ready else main
        has = {f: has_facility(mrow, f) for f in settings.PRICED_FACILITIES}
        for facility in settings.PRICED_FACILITIES:
            if has[facility]:
                model.ask((hid, "off", facility), rid, {**your_dates, **model.switches.get(facility, {})})
        # things the tourist doesn't need, and things reviews say are not there now
        wanted_off = {f: "not_needed" for f in not_needed}
        for feature, facility in FEATURE_TO_FACILITY.items():
            if checks[hid][feature]["status"] == "no" and checks[hid][feature]["listing"] == "yes":
                wanted_off.setdefault(facility, "not_there")
        plans[hid] = (when, rows, wanted_off, has)
    answers = model.answer() if model.ready else {}

    out = {}
    for hid, (when, rows, wanted_off, has) in plans.items():
        seen_m, yours_m = answers.get((hid, "seen")), answers.get((hid, "yours"))
        factor = (yours_m / seen_m) if (seen_m and yours_m and check_in) else 1.0
        fair = [_num(r.get("fair_price_lkr")) for r in rows.values()]
        fair_seen = sum(fair) if all(f is not None for f in fair) else None
        expected = fair_seen * factor if fair_seen else None              # for everyone, all rooms, your dates
        # where the money goes: each facility's share of the fair price (what-if)
        parts = facility_parts(answers, hid, has, wanted_off, expected, len(rows))
        messages, shares = [], []
        for p in parts:
            if p.get("taken_off"):
                template = (WHAT_IF_MESSAGES if p["status"] == "not_needed" else NOT_THERE_MESSAGES)[p["facility"]]
                messages.append(template.format(rs=p["lkr"], pct=p["pct"]))
                shares.append(p["pct"] / 100)
        cut = 1 - min(sum(shares), 0.9)

        # prices per site: the sum over all rooms, only if the site has every room
        offers = {}
        for site, col in SITE_COLS.items():
            vals = [_num(r.get(col)) for r in rows.values()]
            if all(v is not None for v in vals):
                offers[site] = round(sum(vals))
        main = rows[rooms[0]]
        best_site = main.get("wtb_site") if isinstance(main.get("wtb_site"), str) else "Booking.com"
        if best_site not in offers:
            best_site = "Booking.com"
        booking_seen = offers.get("Booking.com")
        vs_fair = 100 * (booking_seen / fair_seen - 1) if (fair_seen and booking_seen) else None
        your_fair_seen = fair_seen * cut if fair_seen else None
        vs_your_fair = 100 * (booking_seen / your_fair_seen - 1) if (your_fair_seen and booking_seen) else None
        overpriced = vs_your_fair is not None and vs_your_fair > settings.OVERPRICED_PCT
        price_yours = offers[best_site] * factor
        out[hid] = {
            "seen_checkin": when,
            "rooms": [{"adults": a, "children": c} for a, c in rows],
            "offers_seen": offers,                                     # per night, all rooms, as collected
            "best_platform": best_site,
            "price_per_night_lkr": int(round(price_yours, -1)),        # best site, moved to your dates
            "date_factor": round(factor, 3),
            "expected_price_lkr": int(round(fair_seen * factor, -1)) if fair_seen else None,
            "your_expected_price_lkr": int(round(fair_seen * cut * factor, -1)) if fair_seen else None,
            "price_vs_fair_pct": round(vs_fair, 1) if vs_fair is not None else None,
            "price_vs_your_fair_pct": round(vs_your_fair, 1) if vs_your_fair is not None else None,
            "price_status": "Potentially Overpriced" if overpriced else "Within Expected Price Range",
            "what_if": messages,
            "breakdown": breakdown(parts, expected, cut, len(rows)) if (model.ready and expected) else None,
            "caution": few_reviews_caution(review_count(data.hotels.loc[hid])) if (expected and hid in data.hotels.index)
            else None,
            "explanation": explanation_ref(data, hid, main["row_id"], check_in, nights, rooms),
            "where_to_book_advice": main.get("wtb_advice") if isinstance(main.get("wtb_advice"), str) else
            "Book on Booking.com.",
            "where_to_book_notes": main.get("wtb_notes") if isinstance(main.get("wtb_notes"), str) else "",
            "meal_plan": main.get("meal_plan") if isinstance(main.get("meal_plan"), str) else None,
            "free_cancellation": _num(main.get("free_cancellation")),
        }
    return out


# ---------------- step 3: score ----------------
def facility_parts(answers, hid, has, wanted_off, expected, n_rooms):
    """Pool, bar, breakfast, beachfront: Rs and % of the fair price, and what we do with each (same rules as
    personal_fair_price.ipynb). Amounts are shares of the fair price, so the parts add up to it."""
    base = answers.get((hid, "base"))
    parts = []
    for f in settings.PRICED_FACILITIES:
        part = {"facility": f, "has_it": bool(has.get(f)), "status": "absent", "taken_off": False}
        off = answers.get((hid, "off", f))
        if has.get(f) and off is not None and base:
            rs, pct = base - off, 100 * (base - off) / base               # one room, model price
            part["pct"] = round(pct, 1)
            if rs >= settings.WHAT_IF_MIN_RS and pct >= settings.WHAT_IF_MIN_PCT:
                part["lkr"] = int(round((expected if expected else rs * n_rooms) * pct / 100, -2))
                part["status"] = wanted_off.get(f, "kept")
                part["taken_off"] = f in wanted_off
            else:
                part["status"] = "too_small"
        elif has.get(f):
            part["status"] = "unknown"
        parts.append(part)
    return parts


STATUS_TEXT = {"absent": "the hotel doesn't have it",
               "too_small": "too small to be sure (under Rs 1,000 or 5%): not shown, it stays in everything else",
               "not_needed": "you don't need it: taken off your fair price",
               "not_there": "reviews say it is not there now: taken off your fair price",
               "kept": "you may use it: it stays in your fair price",
               "unknown": "the price model is not loaded"}


def breakdown(parts, expected, cut, n_rooms):
    """The fair price (all rooms, a night, your dates) split into the facilities and everything else."""
    fair = int(round(expected, -1))                          # same rounding as expected_price_lkr
    shown = [p for p in parts if "lkr" in p]
    rest = fair - sum(p["lkr"] for p in shown)
    return {
        "fair_price_lkr": fair,
        "your_fair_price_lkr": int(round(expected * cut, -1)),  # same as your_expected_price_lkr
        "parts": [dict(p, text=STATUS_TEXT[p["status"]],
                       pct_of_fair=round(100 * p["lkr"] / fair, 1) if "lkr" in p and fair else None) for p in parts],
        "everything_else_lkr": rest, "everything_else_pct": round(100 * rest / fair, 1) if fair else None,
        "per": "a night, " + ("all rooms, " if n_rooms > 1 else "") + "taxes included",
        "note": "About how much of the fair price is for each facility (the price model's what-if, rounded to Rs 100). "
                "Only these 4 facilities can be priced: for the other 10 we tested (spa, gym, air conditioning, "
                "airport shuttle, ...) the model can't tell their cost apart from the type of hotel.",
    }


def explanation_ref(data, hid, row_id, check_in, nights, rooms):
    """Where to get the SHAP explanation of this fair price (it takes 5-20 s, so it is made only on request)."""
    model, name, label = data.unseen_model(row_id) if data.model is not None else (None, None, "no price model")
    q = {"row_id": int(row_id)}
    if check_in:
        q.update({"check_in": check_in.isoformat(), "nights": nights or 2})
    return {"available": model is not None, "model": label if model is not None else None,
            "reason": None if model is not None else label,
            "for": "one room" + (f" (your stay is priced as {len(rooms)} rooms)" if len(rooms) > 1 else ""),
            "url": f"/hotels/{hid}/price-explanation?{urlencode(q)}"}


def interest_value(interest, h, check, c1):
    if interest in ("beach", "surfing"):
        return settings.MATCH_VALUE[check["beach"]["status"]]
    if interest == "nightlife":
        near = (_num(h.get("km_from_town_centre")) or 99) <= 1.5
        return 1.0 if (near or (_num(h.get("has_bar")) or 0) == 1) else 0.0
    if interest == "food":
        if c1 and (c1.get("aspect_scores") or {}).get("food") is not None:
            return c1["aspect_scores"]["food"] / 5
        return 1.0 if (_num(h.get("has_restaurant")) or 0) == 1 else 0.5
    if interest in ("nature", "hiking", "tea_country"):
        return settings.MATCH_VALUE[check["nature"]["status"]]
    return None            # wildlife, culture, religious: places to visit (Component 3 plans those)


def score_hotel(h, check, price, c1, wishes, interests, far_town=False):
    parts = {}
    if wishes:
        w = {k: settings.IMPORTANCE_WEIGHT.get(v, 2) for k, v in wishes.items()}
        got = {}
        for k in w:
            status = check[k]["status"]
            if k == "breakfast" and price.get("meal_plan") == "breakfast":
                status = "yes"
            got[k] = settings.MATCH_VALUE[status]
        parts["wishes"] = sum(w[k] * got[k] for k in w) / sum(w.values())
    trust = _num((c1 or {}).get("overall_trust_score"))
    rating = _num(h.get("page_rating"))
    parts["quality"] = (trust / 10) if trust is not None else (rating / 10 if rating is not None else 0.6)
    pct = price.get("price_vs_your_fair_pct")
    # 0.5 = fair price. The price model's typical error is about 26%, and a very big "saving" is more often a
    # smaller room or a data error, so the bonus stops at 30% below the fair price (0.75); 60% above -> 0
    parts["value"] = float(0.5 - np.clip(pct, -30, 60) / 120) if pct is not None else 0.5
    vals = [v for v in (interest_value(i, h, check, c1) for i in interests) if v is not None]
    if vals:
        parts["interests"] = sum(vals) / len(vals)
    if far_town:        # town not in our data: hotels closer to the town the tourist asked for are better
        parts["distance"] = float(np.clip(1 - (_num(h.get("km_from_town")) or 0) / settings.SEARCH_RADIUS_KM[-1],
                                          0, 1))
    weights = {k: settings.SCORE_WEIGHTS[k] for k in parts}
    total = sum(weights[k] * parts[k] for k in parts) / sum(weights.values())
    return round(total, 3), {k: round(v, 3) for k, v in parts.items()}


# ---------------- step 4: explanation ----------------
def explain(h, check, price, must, must_status, wishes, c1, c1_source, budget_max, check_in):
    s = []
    if must:
        ok = []
        for m in must:
            if m == "budget":
                ok.append(f"under your {_rs(budget_max)} a night (about Rs {price['price_per_night_lkr']:,})")
            elif must_status[m] == "yes":
                ok.append(NICE_NAME[m] + (f" ({check[m]['why']})" if m == "near_hospital" else ""))
        unsure = [NICE_NAME[m] for m in must if must_status.get(m) == "not sure"]
        if ok:
            s.append("Has your must-haves: " + "; ".join(ok) + ".")
        if unsure:
            s.append("NOT CONFIRMED: " + ", ".join(unsure) + " (no information) - ask the hotel before booking.")
    good = [NICE_NAME[k] + (" (reviews agree)" if check[k]["reviews"] == "confirmed" else "")
            for k in wishes if check[k]["status"] == "yes"]
    bad = [f"{NICE_NAME[k]}: {check[k]['why']}" for k in wishes if check[k]["status"] == "no"]
    unsure = [NICE_NAME[k] for k in wishes if check[k]["status"] == "not sure"]
    if good:
        s.append("Has " + ", ".join(good) + ".")
    if bad:
        s.append("Missing " + "; ".join(bad) + ".")
    if unsure:
        s.append("Not sure about " + ", ".join(unsure) + ".")
    trust = _num((c1 or {}).get("overall_trust_score"))
    dummy = " [DUMMY review score]" if c1 and "DUMMY" in str(c1_source) else ""
    if trust is not None:
        s.append(f"Trusted review score {trust:.1f}/10{dummy}.")
    elif _num(h.get("page_rating")) is not None:
        s.append(f"Booking.com rating {h['page_rating']:.1f}/10.")
    if price.get("expected_price_lkr"):
        pct = price["price_vs_your_fair_pct"]
        for_you = " for you" if price["what_if"] else ""
        general = price["price_vs_fair_pct"]
        if price["price_status"] == "Potentially Overpriced" and general is not None and \
                general <= settings.OVERPRICED_PCT:
            s.append(f"Overpriced for you: the price is fair for most guests ({general:+.0f}%), but about "
                     f"{pct:.0f}% above the fair price for you once we remove what you don't need "
                     f"(about {_rs(price['your_expected_price_lkr'])} a night).")
        elif price["price_status"] == "Potentially Overpriced":
            s.append(f"Potentially overpriced: about {pct:.0f}% above the fair price{for_you} "
                     f"(about {_rs(price['your_expected_price_lkr'])} a night).")
        elif pct is not None and pct <= -30:
            s.append(f"Good value: about {-pct:.0f}% below the fair price{for_you} "
                     f"(about {_rs(price['your_expected_price_lkr'])} a night)" +
                     (" - unusually low, check the room type before booking." if pct <= -50 else "."))
        else:
            s.append(f"Price is within the expected range (fair price about "
                     f"{_rs(price['your_expected_price_lkr'] or price['expected_price_lkr'])} a night).")
        if price.get("caution"):
            s.append(price["caution"])
    s.extend(price["what_if"])
    if check_in and price["seen_checkin"]:
        seen = date.fromisoformat(price["seen_checkin"])
        change = 100 * (price["date_factor"] - 1)
        if seen != check_in and abs(change) >= 5:
            s.append(f"Prices were collected for check-in {nice_date(seen)}; for your dates the price "
                     f"model expects about {abs(change):.0f}% {'more' if change > 0 else 'less'}.")
    s.append(price["where_to_book_advice"])
    return " ".join(s)


# ---------------- live Booking.com search: needed? ----------------
def live_search_needed(data, town, cand, confirmed, top_k):
    """True when our CSV data cannot give top_k good hotels for this town (see live/search.py)."""
    if not settings.live_enabled():
        return False, ""
    if not len(cand):
        return True, f"We have no hotels near {town} in our data."
    if data.towns.data_town(town) is None:
        near = [c for c in confirmed if (_num(c["h"].get("km_from_town")) or 0) <= settings.LIVE_NEAR_KM]
        if len(near) < top_k:
            return True, (f"{town} is not one of our data towns and only {len(near)} hotel"
                          f"{'' if len(near) == 1 else 's'} in our data within {settings.LIVE_NEAR_KM} km meet "
                          "everything you asked for.")
    if len(confirmed) < top_k:
        return True, (f"Only {len(confirmed)} hotel{'' if len(confirmed) == 1 else 's'} in our data meet"
                      f"{'s' if len(confirmed) == 1 else ''} every must-have.")
    return False, ""


def found_live(data, hotel_id):
    when = data.live_found.get(hotel_id)
    return {"on": "Booking.com", "date": str(when)[:10]} if when else None


def live_sentence(data, hotel_id, price, check_in):
    when = data.live_found.get(hotel_id)
    if not when:
        return ""
    exact = check_in and price.get("seen_checkin") == check_in.isoformat()
    return (f" Found live on Booking.com on {nice_date(str(when)[:10])} and saved to our data" +
            ("; the price is for your exact dates." if exact else "."))


# ---------------- the whole pipeline ----------------
def recommend(request, top_k=5, today=None, data=None, c1=None):
    """request = the NLP output (Output 1 format). Returns the ranked hotels for every stay_in town."""
    data = data or get_data()
    c1 = c1 or make_c1(data.hotels)
    today = date.fromisoformat(today) if isinstance(today, str) else (today or date.today())
    trip, hotel = request.get("trip", {}), request.get("hotel", {})
    check_in = date.fromisoformat(trip["check_in"]) if trip.get("check_in") else None
    nights = trip.get("nights")
    must = [m for m in hotel.get("must", []) if m == "budget" or m in NICE_NAME]
    wishes = {k: v for k, v in (hotel.get("wishes") or {}).items() if k in NICE_NAME and k not in must}
    not_needed = [f for f in request.get("not_needed", []) if f in WHAT_IF_MESSAGES]
    for f in not_needed:                                    # "don't need breakfast" wins over a breakfast wish
        wishes.pop({"beachfront": "beach"}.get(f, f), None)
    interests = request.get("interests", [])
    budget_max = hotel.get("budget_max_lkr")
    rooms, room_notes = room_plan(trip.get("adults"), trip.get("children"))
    adults = trip.get("adults") or 2
    children = trip.get("children") or 0

    towns = trip.get("stay_in") or []
    results = []
    for town in towns:
        notes = list(room_notes)
        cand, note = find_hotels(data, town, top_k)
        if note:
            notes.append(note)
        if not check_in:
            notes.append("No dates given: typical prices are shown.")
        c1_cache = {hid: c1.get(hid) for hid in cand.index}
        checks = {hid: check_features(cand.loc[hid], c1_cache[hid]) for hid in cand.index}
        prices = price_hotels(data, cand, rooms, check_in, nights, today, not_needed, checks)

        removed = {"no reliable price for your group": int(len(cand) - len(prices))}
        confirmed, unsure = [], []
        for hid, price in prices.items():
            h, check = cand.loc[hid], checks[hid]
            must_status = {}
            for m in must:
                if m == "budget":
                    must_status[m] = "yes" if (budget_max and price["price_per_night_lkr"] <= budget_max) else "no"
                else:
                    must_status[m] = check[m]["status"]
            failed = [m for m in must if must_status[m] == "no"]
            if failed:
                for m in failed:
                    removed[f"must-have '{m}' not met"] = removed.get(f"must-have '{m}' not met", 0) + 1
                continue
            score, parts = score_hotel(h, check, price, c1_cache[hid], wishes, interests,
                                       far_town=data.towns.data_town(town) is None)
            item = {"hotel_id": hid, "h": h, "check": check, "price": price, "score": score, "parts": parts,
                    "must_status": must_status, "c1": c1_cache[hid]}
            (unsure if any(v == "not sure" for v in must_status.values()) else confirmed).append(item)
        confirmed.sort(key=lambda x: -x["score"])
        unsure.sort(key=lambda x: -x["score"])
        live_needed, live_why = live_search_needed(data, town, cand, confirmed, top_k)
        chosen = confirmed[:top_k]
        if len(chosen) < top_k and unsure:
            chosen += unsure[:top_k - len(chosen)]
            notes.append("Some must-haves could not be confirmed for the last hotels in the list (marked "
                         "NOT CONFIRMED).")

        ranked = []
        for rank, it in enumerate(chosen, 1):
            h, price, check = it["h"], it["price"], it["check"]
            total = price["price_per_night_lkr"] * nights if nights else None
            budget_total = trip.get("budget_total_lkr")
            links = booking_links(data, it["hotel_id"], check_in, nights, adults, children, len(rooms))
            ranked.append({
                "hotel_id": it["hotel_id"], "name": h.get("hotel_name"), "rank": rank, "score": it["score"],
                "score_parts": it["parts"], "town": h.get("destination_cluster"),
                "latitude": _num(h.get("latitude")), "longitude": _num(h.get("longitude")),
                "km_from_town": _num(h.get("km_from_town")),
                "musts_confirmed": all(v == "yes" for v in it["must_status"].values()),
                "must_check": it["must_status"],
                "feature_checks": {k: v for k, v in check.items() if k in must or k in wishes},
                "nearest_hospital": {"name": h.get("nearest_hospital"), "km": _num(h.get("nearest_hospital_km"))},
                "price": price, "nights": nights, "total_hotel_cost_lkr": total,
                "budget_left_lkr": (budget_total - total) if (budget_total and total) else None,
                "booking_url": links.get(price["best_platform"]) or next(iter(links.values()), None),
                "links": links,
                "free_cancellation": cancellation_advice(check_in, today, trip.get("dates_fixed"), town,
                                                         price["free_cancellation"]),
                "c1": it["c1"],
                "found_live": found_live(data, it["hotel_id"]),
                "photos": data.photos.get(it["hotel_id"], [])[:settings.PHOTOS_PER_HOTEL],
                "explanation": explain(h, check, price, must, it["must_status"], wishes, it["c1"], c1.source,
                                       budget_max, check_in) + live_sentence(data, it["hotel_id"], price, check_in),
            })
        results.append({"town": town, "hotels_considered": int(len(cand)), "removed": removed,
                        "with_price": len(prices), "meet_all_musts": len(confirmed),
                        "musts_not_sure": len(unsure),
                        "live_search": {"needed": live_needed, "reason": live_why},
                        "ranked_hotels": ranked, "notes": notes})
    return {"results": results, "c1_source": c1.source, "data_notes": data.notes,
            "price_model_used": data.model is not None,
            "prices_collected": settings.PRICES_COLLECTED}
