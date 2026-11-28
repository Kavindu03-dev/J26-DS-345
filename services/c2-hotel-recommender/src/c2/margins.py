"""
"Good deal" and "Looks high" limits for every town (instead of one fixed ±30% for the whole country).

The file is price_margins.csv, made every Sunday by the weekly cloud pipeline (hotel-price-weekly repository:
models/current/price_margins.csv, next to the model it belongs to) or once by notebooks/price_margins_v5.ipynb.
Columns used: town, province, good_deal_ratio, looks_high_ratio (and built_on). Two kinds of special rows:
"(province) <name>" and "(whole country)".

How the limits were made (pipeline/price_margins.py): for every price, ratio = price / fair price from a model
that never saw the hotel. Good deal = the cheapest 10% of a town's ratios, Looks high = the dearest 10%. Small
towns borrow from their province: limit = (n x town + 60 x province) / (n + 60), n = the town's hotels.

A hotel's limits: its town -> its province -> the whole country -> (no file at all) the fixed ±30% rule.
"""
from pathlib import Path

import pandas as pd

from . import settings

ALIASES = {"pasikudah": "pasikuda"}         # Booking.com spells it Pasikuda


def _key(name):
    k = str(name or "").strip().lower()
    return ALIASES.get(k, k)


class PriceLimits:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.towns, self.provinces, self.country, self.built_on, self.note = {}, {}, None, None, None
        if self.path is None or not self.path.exists():
            self.note = (f"No price_margins.csv found{f' ({self.path})' if self.path else ''}: Good deal / Looks high "
                         f"use the fixed rule ({settings.OVERPRICED_PCT}% below / above the fair price).")
            return
        try:
            m = pd.read_csv(self.path)
            for r in m.itertuples(index=False):
                pair = (float(r.good_deal_ratio), float(r.looks_high_ratio))
                name = str(r.town)
                if name == "(whole country)":
                    self.country = pair
                elif name.startswith("(province) "):
                    self.provinces[_key(name[len("(province) "):])] = (name[len("(province) "):], pair)
                else:
                    self.towns[_key(name)] = (name, str(r.province), pair)
            if "built_on" in m and len(m):
                self.built_on = str(m.built_on.iloc[0])
            if self.country is None:
                raise ValueError("no '(whole country)' row")
            self.note = (f"Good deal / Looks high limits for {len(self.towns)} towns from {self.path.name}"
                         + (f" (built {self.built_on})." if self.built_on else "."))
        except Exception as e:                               # a broken file must not stop the service
            self.towns, self.provinces, self.country = {}, {}, None
            self.note = (f"{self.path} could not be read ({type(e).__name__}: {str(e)[:100]}): the fixed rule "
                         f"({settings.OVERPRICED_PCT}% below / above) is used.")

    @property
    def loaded(self):
        return self.country is not None

    def for_town(self, town, province=None):
        """{'good_deal_below_pct', 'looks_high_above_pct', 'place', 'from'} for a hotel in `town`."""
        if not self.loaded:
            lo = hi = settings.OVERPRICED_PCT
            return {"good_deal_below_pct": float(lo), "looks_high_above_pct": float(hi), "place": None,
                    "from": f"fixed rule (±{settings.OVERPRICED_PCT}%)"}
        if _key(town) in self.towns:
            name, prov, (lo, hi) = self.towns[_key(town)]
            place, source = name, f"town ({name}, {prov} province)"
        elif _key(province) in self.provinces:
            name, (lo, hi) = self.provinces[_key(province)]
            place, source = f"{name} province", f"province ({name}): {town or 'this town'} is not in the file"
        else:
            (lo, hi), place = self.country, "Sri Lanka"
            source = f"whole country: {town or 'this town'} is not in the file"
        return {"good_deal_below_pct": round(100 * (1 - lo), 1), "looks_high_above_pct": round(100 * (hi - 1), 1),
                "place": place, "from": source + (f", built {self.built_on}" if self.built_on else "")}