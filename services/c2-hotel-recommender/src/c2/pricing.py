"""
Prices: which collected price to use, the fair price, the personal fair price (Gap 3), where to book (Gap 4).

The prices were collected for 20 check-in dates (Oct 2026 - Sep 2027), 2-night stays, and 4 guest groups
(1 adult, 2 adults, 2 adults + 1 child, 2 adults + 2 children). For a tourist's request:
  1. Rooms: the group is split into rooms we have prices for (4 friends -> 2 rooms for 2 adults).
  2. Date: the collected check-in date closest to the tourist's date in the year (season), for every room.
  3. Fair price: from booking_predictions.csv (the model's price for that hotel, made while the model had NOT seen
     the hotel). "Potentially Overpriced" = Booking.com price more than 30% above it.
  4. Your dates: the model that never saw the hotel (the one that made its fair price) is asked again with the
     tourist's own dates (month, weekday, weekend, festival, days before the trip). The ratio moves the collected
     prices to the tourist's dates.
  5. Where the money goes: the what-if (switch a facility off, predict again) gives the share of the fair price
     for the pool, bar, breakfast and beachfront (the only 4 that can be priced, see personal_fair_price.ipynb).
     Personal fair price: the shares of what the tourist said they don't need, or reviews say is not there, come off.
  6. Where to book: Booking.com vs Trip.com vs Agoda (same-deal rules from where_to_book.ipynb).
"""
import math
from datetime import date, timedelta
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from . import settings

WHAT_IF_MESSAGES = {   # tourist does not need it
    "pool": "About Rs {rs:,} ({pct:.0f}%) of this price is for the pool, which you said you don't need.",
    "bar": "About Rs {rs:,} ({pct:.0f}%) of this price is for the bar, which you said you don't need.",
    "breakfast": "About Rs {rs:,} ({pct:.0f}%) of this price is for breakfast, which you said you don't need. "
                 "A room-only rate may be cheaper.",
    "beachfront": "About Rs {rs:,} ({pct:.0f}%) of this price is for being on the beach, which you said you "
                  "don't need.",
}
NOT_THERE_MESSAGES = {  # reviews say it is not there now
    "pool": "About Rs {rs:,} ({pct:.0f}%) of this price is for a pool that reviews say is not usable now.",
    "beachfront": "About Rs {rs:,} ({pct:.0f}%) of this price is for a beach location that reviews question.",
}
FEATURE_TO_FACILITY = {"pool": "pool", "beach": "beachfront"}


# ---------------- rooms and dates ----------------
def room_plan(adults, children):
    """Split the group into rooms we have prices for. Returns ([(adults, children), ...], notes)."""
    notes = []
    if adults is None:
        adults = 2
        notes.append("Number of adults not given: priced for 2 adults.")
    adults, children = max(int(adults), 1), max(int(children or 0), 0)
    if adults <= 2 and children <= 2:
        if children:
            if adults == 1:
                notes.append("Priced as 2 adults + children (the closest group we have prices for).")
            return [(2, children)], notes
        return [(adults, 0)], notes
    rooms, a, c = [], adults, children
    if c:
        rooms.append((2, min(c, 2)))
        a, c = a - 2, c - min(c, 2)
        if c:
            rooms.append((2, 0))
            notes.append("More than 2 children: one extra room for 2 is priced for them.")
    while a >= 2:
        rooms.append((2, 0))
        a -= 2
    if a == 1:
        rooms.append((1, 0))
    if len(rooms) > 1:
        notes.append(f"{adults} adults" + (f" + {children} children" if children else "") +
                     f": priced as {len(rooms)} rooms.")
    return rooms, notes


def _season_gap(d1, d2):
    """Days between two dates, ignoring the year (20 Dec and 24 Dec -> 4, 28 Dec and 3 Jan -> 6)."""
    a, b = d1.timetuple().tm_yday, d2.timetuple().tm_yday
    g = abs(a - b)
    return min(g, 365 - g)


def pick_stay(hotel_stays, rooms, check_in, approx_from_main=False):
    """The collected date to use for this hotel: every room must have a price that day.
    approx_from_main (live hotels, priced for one room only): other rooms get the first room's price.
    Returns (date text, {room: stay row}) or (None, {})."""
    if hotel_stays is None or not len(hotel_stays):
        return None, {}
    need = set(rooms)
    by_date = {}
    for d, g in hotel_stays.groupby("checkin_date"):
        have = {(int(a), int(c)): row for a, c, (_, row) in zip(g.num_adults, g.num_children, g.iterrows())}
        if approx_from_main and rooms[0] in have:
            have = {r: have.get(r, have[rooms[0]]) for r in need}
        if need <= set(have):
            by_date[d] = have
    if not by_date:
        return None, {}
    main = rooms[0]
    # skip a price far from this hotel's own usual price (a data error or a different deal,
    # e.g. Rs 8,700 for a hotel that is usually Rs 33,000 - 40,000)
    usual = float(hotel_stays["price_incl_tax_lkr_night"].median())
    by_date = {d: rows for d, rows in by_date.items()
               if 0.4 <= rows[main]["price_incl_tax_lkr_night"] / usual <= 2.5}
    if not by_date:
        return None, {}
    if check_in is not None:
        best = min(by_date, key=lambda d: (_season_gap(date.fromisoformat(d), check_in), d))
    else:   # no dates: the date with the middle price (a typical price)
        ordered = sorted(by_date, key=lambda d: by_date[d][main]["price_incl_tax_lkr_night"])
        best = ordered[len(ordered) // 2]
    return best, {r: by_date[best][r] for r in need}


def date_features(check_in, nights, today):
    """The date columns of the price model for the tourist's own dates."""
    nights = nights or 2
    stay_nights = [check_in + timedelta(days=i) for i in range(nights)]

    def in_event(d):
        return any(d.month == m and lo <= d.day <= hi for m, lo, hi in settings.EVENT_WINDOWS)

    return {"lead_time_days": max((check_in - today).days, 0), "checkin_month": check_in.month,
            "checkin_weekday": check_in.weekday(),
            "stay_has_weekend_night": int(any(d.weekday() in (4, 5) for d in stay_nights)),
            "is_event_date": int(any(in_event(d) for d in stay_nights))}


# ---------------- the price model (what-if) ----------------
def _fac_cols(features, *words):
    return [c for c in features if c.startswith("fac_") and any(w in c for w in words)]


def facility_switches(features):
    """Columns to switch off for each facility (same as what_if_explanations.ipynb)."""
    return {
        "pool": {**{c: 0 for c in _fac_cols(features, "pool")}, "has_pool": 0, "n_pools": 0,
                 "rooms_private_pool": 0, "room_private_pool": 0, "room_pool_view": 0},
        "bar": {**{c: 0 for c in _fac_cols(features, "_bar")}, "has_bar": 0},
        "breakfast": {"breakfast_included": 0, "meal_level": 0, "dinner_included": 0},
        "beachfront": {**{c: 0 for c in _fac_cols(features, "beach")}, "is_beachfront": 0, "has_private_beach": 0},
    }


def has_facility(row, facility):
    if facility == "breakfast":
        return row.get("meal_plan") == "breakfast"        # only plain breakfast rates (as in the notebook)
    col = {"pool": "has_pool", "bar": "has_bar", "beachfront": "is_beachfront"}[facility]
    try:
        return float(row.get(col)) == 1
    except (TypeError, ValueError):
        return False


class PriceModel:
    """Batch questions to the saved price model: as collected, at the tourist's dates, and with facilities off.
    unseen=True asks the model that never saw the hotel (the one that made its fair price), if we have it."""

    def __init__(self, data):
        self.data = data
        self.model, self.features = data.model, data.features
        self.rows = data.model_rows
        self.switches = facility_switches(self.features) if self.features else {}
        self.questions = []        # (key, row_id, changes, model)

    @property
    def ready(self):
        return self.model is not None

    def ask(self, key, row_id, changes=None, unseen=False):
        if self.ready and row_id in self.rows.index:
            model = self.model
            if unseen and hasattr(self.data, "unseen_model"):
                model = self.data.unseen_model(row_id)[0] or self.model
            self.questions.append((key, row_id, {c: v for c, v in (changes or {}).items() if c in self.features},
                                   model))

    def answer(self):
        """All questions in one go (one batch per model, fast). Returns {key: price per night in LKR}."""
        out = {}
        for model in {id(q[3]): q[3] for q in self.questions}.values():
            qs = [q for q in self.questions if q[3] is model]
            X = self.rows.loc[[rid for _, rid, _, _ in qs], self.features].copy().reset_index(drop=True)
            by_col = {}                                   # column -> (rows, new values): one write per column
            for i, (_, _, changes, _) in enumerate(qs):
                for c, v in changes.items():
                    by_col.setdefault(c, ([], []))
                    by_col[c][0].append(i)
                    by_col[c][1].append(v)
            for c, (idx, vals) in by_col.items():
                if X[c].dtype.kind in "iub":
                    X[c] = X[c].astype(float)
                X.loc[idx, c] = vals
            prices = np.exp(model.predict(X))
            out.update({k: float(p) for (k, _, _, _), p in zip(qs, prices)})
        self.questions = []
        return out


# ---------------- links with the tourist's dates ----------------
def booking_links(data, hotel_id, check_in, nights, adults, children, n_rooms):
    """Links to the hotel on each site, filled in with the dates and guests where possible."""
    links = {}
    co = (check_in + timedelta(days=nights)) if (check_in and nights) else None
    base = data.urls.get(("booking", hotel_id))
    if base:
        q = {"group_adults": adults, "group_children": children, "no_rooms": n_rooms, "selected_currency": "LKR"}
        if check_in and co:
            q = {"checkin": check_in.isoformat(), "checkout": co.isoformat(), **q}
        links["Booking.com"] = base.split("?")[0] + "?" + urlencode(q)
    same = data.same_hotel.get(hotel_id, {})
    if "tripcom" in same:
        q = {"hotelId": same["tripcom"], "adult": adults, "children": children, "crn": n_rooms, "curr": "LKR"}
        if check_in and co:
            q.update({"checkIn": check_in.isoformat(), "checkOut": co.isoformat()})
        links["Trip.com"] = "https://www.trip.com/hotels/detail/?" + urlencode(q)
    agoda = data.urls.get(("agoda", same.get("agoda"))) if "agoda" in same else None
    if agoda:
        q = {"adults": adults, "children": children, "rooms": n_rooms, "currencyCode": "LKR"}
        if check_in and nights:
            q = {"checkIn": check_in.isoformat(), "los": nights, **q}
        links["Agoda"] = agoda.split("?")[0] + "?" + urlencode(q)
    return links


# ---------------- free cancellation (Gap 6, first version) ----------------
def cancellation_advice(check_in, today, dates_fixed, town, offer_free_cancellation):
    points, reasons = 0, []
    if check_in and (check_in - today).days > 30:
        points += 1
        reasons.append("the trip is more than a month away")
    area = settings.TOWN_AREA.get(town)
    if check_in and area and check_in.month in settings.RAINY_MONTHS[area]:
        points += 1
        reasons.append(f"it can be rainy in {town} in {check_in.strftime('%B')}")
    if dates_fixed is False:
        points += 1
        reasons.append("your dates may change")
    pay_more = points >= 2
    advice = ("Pay a little more for free cancellation" if pay_more else
              "The cheaper price without free cancellation is fine")
    text = advice + (" (" + ", ".join(reasons) + ")." if reasons else ".")
    if offer_free_cancellation is not None and not (isinstance(offer_free_cancellation, float)
                                                    and math.isnan(offer_free_cancellation)):
        text += (" The price shown has free cancellation." if int(offer_free_cancellation) == 1
                 else " The price shown has no free cancellation.")
    return {"recommend_free_cancellation": pay_more, "points": points, "advice": text}
