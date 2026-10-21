"""
Why is the fair price this number? A SHAP explanation in rupees (same method as personal_fair_price.ipynb, section 5).

    start (average of 20 typical prices)  + town + hotel class + room + pool + ... + your dates  =  the fair price

- 11 groups a tourist understands, not 518 columns (the pool alone is in 15 columns).
- EXACT: with 11 groups there are only 2^11 = 2,048 combinations, so shap.explainers.Exact computes SHAP exactly.
  (The quick shap.TreeExplainer gives wrong numbers for this model: its parts don't add up, because it can't
  handle the category columns town, star band and chain.)
- In rupees: we explain the price itself, so the parts add up exactly to the fair price.
- With the model that NEVER saw the hotel - the same model that made its fair price (data.unseen_model):
  the final model for test hotels and live hotels, one of the 3 cross-validation models for training hotels.
- "The model doesn't know a group" = that group's values come from 20 typical prices (one from the middle of
  each 5% slice of all prices) and the model's 20 answers are averaged.

One explanation takes about 5-20 seconds (2,048 x 20 = 40,960 model answers), so it is made only when the
tourist asks ("Why this fair price?") and then kept in memory.
"""
import math
import threading
import time
from collections import OrderedDict
from datetime import date

import numpy as np
import pandas as pd

from . import settings

GROUP_NAMES = ["Town and location", "Hotel class and reviews", "Room type and size", "Pool", "Bar", "Beach",
               "Breakfast and meals", "Other facilities", "Your dates", "Your group", "Booking terms and deals"]


def group_of(c):
    """Which of the 11 groups a model column belongs to (same as the notebook)."""
    if c in ("destination_clean", "latitude", "longitude", "km_from_town_centre", "distance_km", "location_score",
             "subscore_location"):
        return "Town and location"
    if ("pool" in c and c != "fac_pool_table") or c == "n_pools":
        return "Pool"
    if c == "has_bar" or (c.startswith("fac_") and "_bar" in c):
        return "Bar"
    if "beach" in c and not c.startswith("name_"):
        return "Beach"
    if "breakfast" in c or "meal" in c or c == "dinner_included" or c.startswith("offers_"):
        return "Breakfast and meals"
    if c in ("lead_time_days", "checkin_month", "checkin_weekday", "stay_has_weekend_night", "is_event_date"):
        return "Your dates"
    if c in ("num_adults", "num_children", "guests_total", "has_children"):
        return "Your group"
    if c in ("free_cancellation", "pay_at_property", "preferred_partner", "deal_shown", "discount_pct", "new_listing",
             "units_left"):
        return "Booking terms and deals"
    if c.startswith(("room", "card_room")) or c == "n_room_types":
        return "Room type and size"
    if c.startswith(("name_", "subscore_")) or c in ("stars", "stars_is_estimate", "star_band", "review_score",
                                                     "review_count", "log_review_count", "page_rating",
                                                     "page_review_count", "chain_group", "is_chain"):
        return "Hotel class and reviews"
    return "Other facilities"


def typical_price_rows(priced, n):
    """n typical prices: from each 1/n slice of the fair prices (cheapest to dearest), the price in the middle."""
    if priced is None or len(priced) < n:
        return None
    priced = priced.reset_index(drop=True)
    rank = priced.fair_price_lkr.rank(pct=True)
    pick = []
    for k in range(n):
        in_slice = priced.index[(rank > k / n) & (rank <= (k + 1) / n)]
        if len(in_slice):
            pick.append((rank[in_slice] - (k + 0.5) / n).abs().idxmin())
    return priced.loc[pick].reset_index(drop=True)


# ---------------- the SHAP sums ----------------
def shapley_by_hand(answers, n):
    """The textbook Shapley formula on the same 2^n answers: an independent check of the shap library."""
    parts = np.zeros(n)
    for known, p in answers.items():
        k = sum(known)
        for g in range(n):
            if known[g]:
                without = known[:g] + (False,) + known[g + 1:]
                parts[g] += math.factorial(k - 1) * math.factorial(n - k) / math.factorial(n) * (p - answers[without])
    return parts


def exact_shap(price, n):
    """SHAP part of each of the n groups. price(known) -> model price when only the `known` groups are this hotel's.
    Uses shap.explainers.Exact when the shap package is installed, else the textbook formula (same numbers)."""
    answers = {}

    def ask(known):
        p = price(known)
        answers.update(zip([tuple(k) for k in np.asarray(known, dtype=bool).tolist()], p))
        return p

    try:
        import shap
    except ImportError:
        all_known = np.array([[(i >> g) & 1 for g in range(n)] for i in range(2 ** n)], dtype=bool)
        ask(all_known)
        parts = shapley_by_hand(answers, n)
        return parts, answers[(False,) * n], "the exact Shapley formula (the shap package is not installed)", None
    explainer = shap.explainers.Exact(ask, shap.maskers.Independent(np.zeros((1, n)), max_samples=1))
    sv = explainer(np.ones((1, n)), silent=True)
    parts = np.asarray(sv.values[0], dtype=float)
    same = bool(np.allclose(shapley_by_hand(answers, n), parts, rtol=1e-6, atol=1e-3))
    return parts, float(sv.base_values[0]), f"shap.explainers.Exact (shap {shap.__version__})", same


# ---------------- one model, ready to explain ----------------
class _Ready:
    """The model split into its two steps, its columns put into groups, and the 20 typical prices prepared."""

    def __init__(self, model, features, typical_rows):
        steps = getattr(model, "named_steps", {})
        if "prepare" not in steps or "model" not in steps:
            raise ValueError("SHAP needs the gradient boosting pipeline (steps 'prepare' and 'model').")
        self.prep, self.trees, self.features = steps["prepare"], steps["model"], features
        cols = [c.split("__", 1)[1] if "__" in c else c for c in self.prep.get_feature_names_out()]
        groups = {g: np.array([i for i, c in enumerate(cols) if group_of(c) == g]) for g in GROUP_NAMES}
        self.groups = {g: v for g, v in groups.items() if len(v)}          # every column is in one group
        self.names = list(self.groups)
        self.background = self.prep.transform(typical_rows[features])

    def price(self, x, known):
        """Average model price (Rs) when only the `known` groups come from this hotel, the rest from the 20 typical
        prices."""
        known = np.asarray(known, dtype=bool)
        bg, out = self.background, np.empty(len(known))
        for s in range(0, len(known), 64):
            k = known[s:s + 64]
            X = np.repeat(bg[None], len(k), axis=0)
            for g, name in enumerate(self.names):
                on = np.flatnonzero(k[:, g])
                if len(on):
                    cols = self.groups[name]
                    X[np.ix_(on, np.arange(len(bg)), cols)] = x[cols]
            out[s:s + 64] = np.exp(self.trees.predict(X.reshape(-1, X.shape[2]))).reshape(len(k), -1).mean(axis=1)
        return out


_lock = threading.Lock()
_ready = {}                         # id(model) -> _Ready
_cache = OrderedDict()              # (model name, row_id, changes) -> explanation


def _ready_for(model, data):
    with _lock:
        r = _ready.get(id(model))
        if r is None:
            r = _ready[id(model)] = _Ready(model, data.features, data.typical_rows)
        return r


# ---------------- what the hotel actually has (shown next to each part) ----------------
OTHER_FACILITIES = {"has_spa": "spa", "has_fitness_center": "gym", "has_restaurant": "restaurant",
                    "has_room_service": "room service", "has_free_wifi": "free wifi", "has_free_parking": "free parking",
                    "has_airport_shuttle": "airport shuttle", "has_garden": "garden", "has_family_rooms": "family rooms"}


def _num(v):
    try:
        v = float(v)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def nice_date(d):
    """'2026-12-20' -> '20 Dec 2026' (written by hand: %-d does not work on Windows)."""
    try:
        d = date.fromisoformat(str(d)[:10])
        return f"{d.day} {d.strftime('%b %Y')}"
    except ValueError:
        return str(d)


def review_count(row):
    """Number of Booking.com reviews (0 when the hotel has no rating at all), or None if not known."""
    n = _num(row.get("review_count"))
    if n is None:
        n = _num(row.get("page_review_count"))
    if n is None and _num(row.get("review_score")) is None and _num(row.get("page_rating")) is None:
        return 0
    return n


def few_reviews_caution(n):
    """A short warning when the hotel has very few reviews (the model knows little about its quality)."""
    if n is None or n >= settings.FEW_REVIEWS:
        return None
    if n == 0:
        return "No reviews on Booking.com yet, so the model knows little about this hotel's quality: this fair " \
               "price is less certain."
    return (f"Only {int(n)} review{'s' if n != 1 else ''} on Booking.com, so the model knows little about this "
            "hotel's quality: this fair price is less certain.")


def _class_words(row):
    """Stars (or none), the review score and how many reviews, and words in the name the model uses."""
    stars, score, n = _num(row.get("stars")), _num(row.get("review_score")), review_count(row)
    if score is None:
        score = _num(row.get("page_rating"))
    out = [f"{stars:.0f} stars" + (" (estimated)" if _num(row.get("stars_is_estimate")) == 1 else "") if stars
           else "no star rating"]
    if score is not None:
        out.append(f"{score:g}/10" + (f" from {int(n):,} review{'s' if n != 1 else ''}" if n else ""))
    elif n == 0:
        out.append("no reviews yet")
    words = [c[5:] for c in row.index if c.startswith("name_") and _num(row.get(c)) == 1][:2]
    if words:
        out.append(" and ".join(f"'{w}'" for w in words) + " in the name")
    return ", ".join(out)


def what_it_is(row, check_in=None):
    yes_no = lambda c: {1.0: "yes", 0.0: "no"}.get(_num(row.get(c)), "not known")
    km_c = _num(row.get("km_from_town_centre"))
    others = [name for c, name in OTHER_FACILITIES.items() if _num(row.get(c)) == 1]
    meal = str(row.get("meal_plan") or "")
    adults, children = int(_num(row.get("num_adults")) or 0), int(_num(row.get("num_children")) or 0)
    cancel = _num(row.get("free_cancellation"))
    return {
        "Town and location": f"{row.get('destination_clean')}" + (f", {km_c:.1f} km from the centre the price model uses" if km_c is not None else ""),
        "Hotel class and reviews": _class_words(row),
        "Room type and size": str(row.get("room_type_name_raw") or row.get("room_tier_name") or "not known")[:40],
        "Pool": yes_no("has_pool"), "Bar": yes_no("has_bar"),
        "Beach": "on the beach" if _num(row.get("is_beachfront")) == 1 else "not on the beach",
        "Breakfast and meals": {"none": "room only", "breakfast": "breakfast included"}.get(meal, meal.replace("_", " ")
                                                                                               or "not known"),
        "Other facilities": ", ".join(others[:4]) + (", ..." if len(others) > 4 else "") if others else "few listed",
        "Your dates": f"check-in {nice_date(check_in or row.get('checkin_date'))}",
        "Your group": f"{adults} adult{'s' if adults != 1 else ''}" + (f" + {children} child{'ren' if children > 1 else ''}"
                                                                      if children else ""),
        "Booking terms and deals": ("free cancellation" if cancel == 1 else "no free cancellation")
                                   + (", deal shown" if _num(row.get("deal_shown")) == 1 else ""),
    }


def sentences(parts, start, end, facts, caution=None, top=3, min_rs=1000):
    up = [p for p in parts if p["lkr"] >= min_rs][:top]
    down = [p for p in parts if p["lkr"] <= -min_rs][:top]
    out = [f"An average hotel price is about Rs {int(round(start, -2)):,}. This fair price is about "
           f"Rs {int(round(end, -1)):,}."]
    if up:
        out.append("Higher mainly because of: " + "; ".join(
            f"{p['group'].lower()} ({facts[p['group']]}): about Rs {int(round(p['lkr'], -2)):,} more" for p in up) + ".")
    if down:
        out.append("Lower because of: " + "; ".join(
            f"{p['group'].lower()} ({facts[p['group']]}): about Rs {int(round(-p['lkr'], -2)):,} less" for p in down) + ".")
    if caution:
        out.append(caution)
    return out


# ---------------- the explanation ----------------
def explain_price(data, row_id, changes=None, check_in=None):
    """SHAP explanation of the fair price of one price row (one room), optionally moved to the tourist's dates."""
    model, name, label = data.unseen_model(row_id)
    if model is None:
        return {"available": False, "reason": label}
    changes = {c: v for c, v in (changes or {}).items() if c in data.features}
    key = (name, int(row_id), tuple(sorted(changes.items())))
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return dict(_cache[key], from_cache=True)
    if data.typical_rows is None:
        return {"available": False, "reason": "booking_predictions.csv not found, so there are no typical prices."}

    t0 = time.time()
    ready = _ready_for(model, data)
    row = data.model_rows.loc[[row_id]].iloc[[0]].copy()
    for c, v in changes.items():
        row[c] = v
    x = ready.prep.transform(row[data.features])[0]
    raw, start, method, same_as_formula = exact_shap(lambda known: ready.price(x, known), len(ready.names))
    model_price = float(np.exp(model.predict(row[data.features]))[0])
    end = start + float(raw.sum())
    r = row.iloc[0]
    facts = what_it_is(r, check_in)
    parts = sorted(({"group": g, "what_it_is": facts.get(g, ""), "lkr": float(v), "pct_of_fair": 100 * float(v) / end}
                    for g, v in zip(ready.names, raw)), key=lambda p: -abs(p["lkr"]))
    adults, children = int(_num(r.get("num_adults")) or 0), int(_num(r.get("num_children")) or 0)
    out = {
        "available": True, "row_id": int(row_id), "hotel_id": r.get("booking_hotel_id"), "name": r.get("hotel_name"),
        "explains": f"for one room ({adults} adult{'s' if adults != 1 else ''}"
                    + (f" + {children} child{'ren' if children > 1 else ''}" if children else "") + ") a night"
                    + f", check-in {nice_date(check_in or r.get('checkin_date'))}",
        "model": label,
        "method": f"exact SHAP over {len(ready.names)} groups ({2 ** len(ready.names):,} combinations), {method}",
        "start_lkr": start, "start_is": f"the average of {len(ready.background)} typical prices "
                                        "(one from the middle of each 5% slice, cheapest to dearest)",
        "parts": parts, "fair_price_lkr": end,
        "checks": {"parts_add_up_to_the_fair_price": bool(np.isclose(end, model_price, rtol=1e-6)),
                   "same_as_textbook_shapley_formula": same_as_formula},
        "sentences": sentences(parts, start, end, facts, few_reviews_caution(review_count(r))),
        "seconds": round(time.time() - t0, 1), "from_cache": False,
    }
    with _lock:
        _cache[key] = out
        while len(_cache) > settings.SHAP_CACHE_SIZE:
            _cache.popitem(last=False)
    return out
