"""
The preprocessing rules - COPIED from preprocessing.ipynb (only the rule definitions: settings, functions and
constants; nothing is run at import). The live search uses them to turn a freshly read Booking.com card and hotel
page into rows with the same columns as the research CSVs, so the price model and the ranking treat a live hotel
exactly like one from September's data.

Copied automatically from the notebook cells listed below (python ast: function definitions and UPPER_CASE settings).
"""
import collections
import difflib
import hashlib
import math
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

# ---- preprocessing.ipynb, cell 3 ----
USD_TO_LKR = 330.0

NON_HOTEL_NAME_WORDS = ["villa", "villas", "guest house", "guesthouse", "home stay", "homestay", "hostel", "apartment",
                        "apartments", "bungalow", "cottage", "cottages", "bnb", "b&b", "bed and breakfast", "cabana",
                        "cabanas", "camping", "campsite", "tent", "dorm", "holiday home", "house"]

MATCH_MIN_SCORE_SAME_TOWN = 0.90

MATCH_MIN_SCORE_NEIGHBOUR = 0.97

MATCH_MIN_MARGIN = 0.05

NEIGHBOUR_TOWNS = [("Galle", "Unawatuna"), ("Unawatuna", "Mirissa"), ("Sigiriya", "Dambulla"), ("Colombo", "Negombo")]

DESCRIPTIVE_NAME_WORDS = {"villa", "villas", "guest", "house", "guesthouse", "home", "homestay", "homes", "stay", "stays",
                          "restaurant", "cottage", "cottages", "bungalow", "bungalows", "cabana", "cabanas", "beach", "lodge",
                          "rest", "holiday", "apartment", "apartments", "hostel", "bnb", "motel", "motels", "chalet", "chalets",
                          "garden", "gardens"}

GENERIC_NAME_WORDS = {"hotel", "hotels", "resort", "resorts", "and", "the", "by", "spa", "&", "a", "at", "of", "sri", "lanka",
                      "boutique", "suites", "suite", "inn", "residence", "residencies", "retreat"}

OUTLIER_ROBUST_Z = 4.0

TEST_SHARE = 0.20

SPLIT_SALT = "IT23295674"

WHOLE_PLACE_WORDS = ["villa", "apartment", "home", "vacation home", "holiday home", "chalet", "studio", "bungalow",
                     "house", "cottage", "place", "cabin", "guest house"]

CHAIN_MIN_HOTELS = 3

NAME_WORD_MIN_HOTELS = 12

NAME_WORD_SKIP = {"the", "by", "and", "s", "a", "of", "at", "de", "la", "on", "in", "with", "hotel", "hotels", "sri",
                  "lanka", "arugambay"}

FACILITY_MIN_HOTELS = 50

ROOM_M2_MAX = {"budget": 100, "standard": 100, "superior": 100, "deluxe": 100, "premium": 150,
               "suite": 250, "villa_chalet": 500, "other": 200}

ROOM_STEP_COLUMNS = ["hotel_chain_raw", "room_types_raw", "meal_plans_raw"]


# ---- preprocessing.ipynb, cell 10 ----
NaN = float("nan")

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

def _to_float(text):
    m = _NUMBER.search(text or "")
    return float(m.group().replace(",", "")) if m else NaN

def money(text):
    """"LKR 20,063" -> 20063.0 ; "රු 11,551" -> 11551.0 ; "US$139" -> 139.0 ; "" -> NaN"""
    return _to_float(text)

def currency_of(text):
    t = (text or "").strip()
    if t.startswith("LKR") or t.startswith("රු") or t.startswith("Rs"):
        return "LKR"
    if t.startswith("US$") or t.startswith("$"):
        return "USD"
    return None

def booking_tax(text):
    """Booking.com taxes line -> (tax for the stay, price already includes taxes?)"""
    t = (text or "").lower()
    if "includes taxes" in t:
        return 0.0, True
    if "taxes" in t:
        return _to_float(text), False
    return NaN, False

_TRIP_TOTAL = re.compile(r"Total price:\s*US\$([\d,.]+)\s+(\d+)\s+(\w+)\s*[×x]\s*(\d+)\s*nights?", re.I)

def tripcom_total(text):
    """"Total price: US$360 1 room × 2 nights incl. taxes & fees" -> total 360, 1 unit, "room", 2 nights.
    The unit word matters: "villa" / "apartment" / "bed" mean it is not a hotel room."""
    m = _TRIP_TOTAL.search(text or "")
    if not m:
        return {"total_usd": NaN, "units": NaN, "unit_word": None, "nights": NaN}
    return {"total_usd": float(m.group(1).replace(",", "")), "units": int(m.group(2)),
            "unit_word": m.group(3).lower(), "nights": int(m.group(4))}

def count(text):
    return _to_float(text)

def score(text):
    v = _to_float(text)                           # all three sites score out of 10
    return v if (math.isnan(v) or 0 <= v <= 10) else NaN

location_score = score

_DIST = re.compile(r"(\d+(?:\.\d+)?)\s*(km|m)\b", re.I)

def distance_km(distance_text, address_text=None):
    """Booking.com "600 m from downtown" -> 0.6 ; Agoda address "... - 2.3 km to center" -> 2.3 ;
    Agoda "... - City center" -> 0 ; Trip.com shows no distance -> NaN"""
    for text in (distance_text, address_text):
        t = text or ""
        m = _DIST.search(t)
        if m and ("from" in t.lower() or "to center" in t.lower() or "to centre" in t.lower()):
            v = float(m.group(1))
            return v / 1000 if m.group(2).lower() == "m" else v
        if re.search(r"-\s*city cent(er|re)\b", t, re.I):
            return 0.0
    return NaN

def units_left(text):
    return _to_float(text)

def cancellation_type(text):
    t = (text or "").lower()
    if "extended free" in t:
        return "extended_free"
    if "free cancellation" in t:
        return "free"
    if "cancellation window" in t:
        return "window"
    return "none_shown"

def pay_at_property(text):
    t = (text or "").lower()
    return int("no prepayment" in t or "pay at the property" in t or "no payment today" in t)

def discount_pct(displayed, original):
    """crossed-out price vs shown price -> % off (NaN when there is none or it isn't higher)"""
    if math.isnan(displayed) or math.isnan(original) or original <= displayed or original <= 0:
        return NaN
    return round((1 - displayed / original) * 100, 1)

def deal_percent(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*off", text or "", re.I)   # Trip.com "Member deal; 18% off" -> 18
    return float(m.group(1)) if m else NaN

def split_list(text, sep=";"):
    return [p.strip() for p in (text or "").split(sep) if p.strip()]

MEAL_LEVELS = {"none": 0, "breakfast": 1, "half_board": 2, "full_board": 3, "all_inclusive": 4}

def meal_plan(text):
    t = (text or "").lower()
    if re.search(r"all[- ]inclusive", t):
        return "all_inclusive"
    if re.search(r"full board|all meals included|breakfast,? lunch (?:&|and) dinner included", t):
        return "full_board"
    if re.search(r"half board|breakfast (?:&|and) dinner included", t):
        return "half_board"
    if "breakfast included" in t:
        return "breakfast"
    return "none"

_BREAKFAST_PRICE = re.compile(r"breakfast\s+LKR\s*([\d,]+)", re.I)

def breakfast_extra_lkr(text):
    m = _BREAKFAST_PRICE.search(text or "")       # "Breakfast LKR 3,302 (optional)" -> 3302
    return float(m.group(1).replace(",", "")) if m else NaN

_M2 = re.compile(r"(\d+(?:\.\d+)?)\s*m²")

def size_m2(text):
    m = _M2.search(text or "")                    # "23 m²" -> 23.0
    return float(m.group(1)) if m else NaN

def room_list(text):
    """"Deluxe Room | 23 m² | 1 king bed | sleeps 2 adults || Family Room | 26 m² || ..." -> list of rooms.
    Each part after the name is recognised by what it says, because pages leave some parts out."""
    rooms = []
    for part in (text or "").split(" || "):
        fields = [f.strip() for f in part.split(" | ")]
        if not fields or not fields[0]:
            continue
        room = {"name": fields[0], "m2": NaN, "beds": None, "sleeps": NaN}
        for f in fields[1:]:
            if "m²" in f:
                room["m2"] = size_m2(f)
            elif f.lower().startswith("sleeps"):
                room["sleeps"] = _to_float(f)
            elif f:
                room["beds"] = f
        rooms.append(room)
    return rooms

_WHOLE_PLACE = re.compile(r"\bentire\s+(" + "|".join(re.escape(w) for w in sorted(
    WHOLE_PLACE_WORDS, key=len, reverse=True)) + r")", re.I)

def whole_place_or_dorm(config_text, room_name):
    """"Entire villa • 4 bedrooms ..." -> "whole villa" ; "Bed in dorm" / "... Dormitory Room" -> "dorm bed" ; else "" """
    c, nm = config_text or "", room_name or ""
    m = _WHOLE_PLACE.search(c)
    if m:
        return "whole " + m.group(1).lower()
    if re.search(r"\bbed in dorm", c, re.I) or re.search(r"\bdorm(itory)?\b", nm, re.I):
        return "dorm bed"
    return ""


# ---- preprocessing.ipynb, cell 12 ----
def star_band(stars):
    band = pd.cut(stars, [0, 2.5, 3.5, 4.5, 5.0], labels=["1-2", "3", "4", "5"])
    return band.astype(object).where(band.notna(), "unrated")


# ---- preprocessing.ipynb, cell 15 ----
def price_columns(row):
    """One raw listing -> its normalised price columns."""
    site = row["source_ota"]
    nights = money(row.get("nights")) or NaN
    shown = money(row.get("displayed_price_raw"))
    original = money(row.get("original_price_raw"))
    out = {"price_shown": shown, "price_shown_currency": currency_of(row.get("displayed_price_raw")),
           "price_pre_tax_lkr_night": NaN, "price_incl_tax_lkr_night": NaN, "tax_ratio": NaN,
           "price_incl_tax_is_estimate": 0, "discount_pct": NaN, "price_basis_note": ""}
    if math.isnan(shown) or shown <= 0:
        out["price_basis_note"] = "no price shown (sold out for this date/guests)"
        return out
    if site == "booking":
        tax, includes = booking_tax(row.get("taxes_and_fees_raw"))
        stay_nights = nights if nights and not math.isnan(nights) else 2
        if includes:
            out["price_incl_tax_lkr_night"] = shown / stay_nights
            out["price_basis_note"] = "Booking.com: stay total incl. taxes / nights"
        elif not math.isnan(tax):
            out["price_pre_tax_lkr_night"] = shown / stay_nights
            out["price_incl_tax_lkr_night"] = (shown + tax) / stay_nights
            out["tax_ratio"] = (shown + tax) / shown
            out["price_basis_note"] = "Booking.com: (stay total + taxes line) / nights"
        else:
            out["price_pre_tax_lkr_night"] = shown / stay_nights
            out["price_basis_note"] = "Booking.com: stay total / nights (no taxes line)"
        out["discount_pct"] = discount_pct(shown, original)
    elif site == "agoda":
        out["price_pre_tax_lkr_night"] = shown
        out["price_basis_note"] = "Agoda: one night before taxes (incl. taxes estimated later)"
        out["discount_pct"] = discount_pct(shown, original)
    elif site == "tripcom":
        total = tripcom_total(row.get("taxes_and_fees_raw"))
        out["price_pre_tax_lkr_night"] = shown * USD_TO_LKR
        if not math.isnan(total["total_usd"]) and total["nights"] and total["units"]:
            incl = total["total_usd"] / (total["nights"] * total["units"])
            out["price_incl_tax_lkr_night"] = incl * USD_TO_LKR
            out["tax_ratio"] = incl / shown
            out["price_basis_note"] = "Trip.com: total incl. taxes / (nights x units) x USD rate"
        else:
            out["price_basis_note"] = "Trip.com: one night before taxes x USD rate (no total line)"
        out["discount_pct"] = discount_pct(shown, original)
        if math.isnan(out["discount_pct"]):
            out["discount_pct"] = deal_percent(row.get("deal_badge_raw"))
    return out


# ---- preprocessing.ipynb, cell 18 ----
TIERS = [   # (tier name, level, words) - checked from the top, first match wins
    ("suite", 5, r"\bsuite\b"),
    ("villa_chalet", 5, r"\b(villa|chalet|bungalow|cabana|cottage)\b"),
    ("premium", 4, r"\b(premium|premier|luxury|executive|club|signature|grand|royal|presidential)\b"),
    ("deluxe", 3, r"\b(deluxe|delux)\b"),
    ("superior", 2, r"\b(superior|comfort)\b"),
    ("budget", 0, r"\b(budget|economy|basic|dormitory|dorm|bed in)\b"),
    ("standard", 1, r"\b(standard|classic|double|twin|triple|quadruple|single|family|king|queen|studio)\b"),
]

_NEAR_VIEW = r"(?:\s+\w+){0,2}\s+views?\b"

VIEWS = {
    "sea_view": r"\b(sea|ocean|beach)\b" + _NEAR_VIEW + r"|\b(sea|ocean)\s*(front|facing)\b|\bbeach\s*front\b",
    "lake_river_view": r"\b(lake|river|lagoon)\b" + _NEAR_VIEW,
    "mountain_view": r"\b(mountain|hill|valley|tea estate|rock)\b" + _NEAR_VIEW,
    "pool_view": r"\bpool\b" + _NEAR_VIEW,
    "garden_view": r"\bgarden\b" + _NEAR_VIEW,
    "city_view": r"\bcity\b" + _NEAR_VIEW,
}

CAPACITY = [("quadruple", 4), ("triple", 3), ("family", 4), ("double", 2), ("twin", 2), ("king", 2), ("queen", 2), ("single", 1)]

def room_features(name, config=None):
    nm = (name or "").lower()
    cfg = (config or "").lower()
    out = {"room_info_available": int(bool(nm))}
    tier_name, tier = (None, NaN)
    if nm:
        tier_name, tier = "other", 1
        for label, level, pattern in TIERS:
            if re.search(pattern, nm):
                tier_name, tier = label, level
                break
    out["room_tier_name"], out["room_tier"] = tier_name, tier
    any_view = 0
    for col, pattern in VIEWS.items():
        v = int(bool(nm) and bool(re.search(pattern, nm)))
        out["room_" + col] = v
        any_view |= v
    out["room_any_view"] = int(any_view or bool(re.search(r"\bview\b", nm)))
    out["room_balcony_terrace"] = int(bool(re.search(r"\b(balcony|terrace|patio|veranda|verandah)\b", nm)))
    out["room_private_pool"] = int(bool(re.search(r"\b(private|plunge)\s+pool\b", nm)))
    out["room_spa_bath"] = int(bool(re.search(r"\b(spa bath|jacuzzi|hot tub|whirlpool)\b", nm)))
    out["room_is_shared_or_dorm"] = int(bool(re.search(r"\b(dormitory|dorm|shared bathroom|bed in)\b", nm)))
    cap = NaN
    for word, c in CAPACITY:
        if re.search(rf"\b{word}\b", nm):
            cap = c
            break
    out["room_capacity_hint"] = cap
    beds = re.search(r"(\d+)\s+(?:\w+\s+)?beds?\b", cfg)          # Booking.com bed line: "1 king bed", "2 beds (...)"
    out["room_beds"] = float(beds.group(1)) if beds else NaN
    out["room_large_bed"] = int(bool(re.search(r"\b(king|queen)\b", cfg + " " + nm)))
    return out

def clean_room_list(rooms):
    """Sizes above ROOM_M2_MAX for the room's tier -> missing (and counted)."""
    out = []
    for r in rooms:
        r = dict(r)
        tier = room_features(r["name"])["room_tier_name"] or "other"
        r["m2_set_aside"] = int(r["m2"] == r["m2"] and r["m2"] > ROOM_M2_MAX.get(tier, ROOM_M2_MAX["other"]))
        if r["m2_set_aside"]:
            r["m2"] = NaN
        out.append(r)
    return out

def card_room_m2(card_room_name, hotel_rooms):
    """Size of the card's room, found by its exact name in the hotel's room list (two rooms with the same name -> median)."""
    want = (card_room_name or "").strip().lower()
    if not want:
        return NaN
    sizes = [r["m2"] for r in hotel_rooms if r["name"].strip().lower() == want and r["m2"] == r["m2"]]
    return float(np.median(sizes)) if sizes else NaN


# ---- preprocessing.ipynb, cell 22 ----
SUBSCORES = ["staff", "facilities", "cleanliness", "comfort", "location", "free_wifi"]

AMENITY_RULES = {   # flag -> test on the hotel's amenity items (exact item names as Booking.com writes them, lower case)
    "has_pool": lambda s: any("swimming pool" in i or i.startswith("pool") for i in s["popular"] + s["highlights"]),
    "has_spa": lambda s: any("spa and wellness" in i for i in s["popular"] + s["highlights"]),
    "has_fitness_center": lambda s: any("fitness" in i for i in s["popular"] + s["highlights"]),
    "has_airport_shuttle": lambda s: any("airport shuttle" in i or "shuttle service" in i for i in s["popular"] + s["highlights"]),
    "has_restaurant": lambda s: any("restaurant" in i for i in s["popular"] + s["highlights"] + s["listed"]),
    "has_bar": lambda s: "bar" in s["popular"] + s["listed"],
    "is_beachfront": lambda s: any(i in ("beachfront", "private beach area") for i in s["popular"] + s["highlights"]),
    "has_private_beach": lambda s: "private beach area" in s["popular"] + s["highlights"],
    "has_free_parking": lambda s: any(i.startswith("free parking") or i.startswith("free on-site parking")
                                      or i.startswith("free private parking") for i in s["popular"] + s["highlights"]),
    "has_free_wifi": lambda s: "free wifi" in s["popular"] + s["highlights"],
    "has_family_rooms": lambda s: "family rooms" in s["popular"] + s["highlights"],
    "has_room_service": lambda s: "room service" in s["popular"] + s["highlights"],
    "has_24h_front_desk": lambda s: "24-hour front desk" in s["popular"],
    "has_disabled_facilities": lambda s: "facilities for disabled guests" in s["popular"],
    "has_garden": lambda s: "garden" in s["popular"] + s["highlights"] + s["listed"],
    "rooms_sea_view": lambda s: any(i in ("sea view", "ocean view", "beachfront") for i in s["rooms"]),
    "rooms_private_pool": lambda s: "private pool" in s["rooms"],
    "rooms_hot_tub": lambda s: any(i in ("hot tub", "spa tub", "private hot tub") for i in s["rooms"]),
    "rooms_balcony": lambda s: any(i in ("balcony", "terrace", "patio") for i in s["rooms"]),
    "rooms_bathtub": lambda s: any(i in ("bath", "bathtub", "bath or shower") for i in s["rooms"]),
    "rooms_minibar": lambda s: "minibar" in s["rooms"] + s["listed"],
    "rooms_air_conditioning": lambda s: "air conditioning" in s["rooms"] + s["popular"] + s["highlights"],
    "rooms_kitchen": lambda s: any(i in ("kitchen", "kitchenette") for i in s["rooms"] + s["listed"]),
    "rooms_soundproofing": lambda s: "soundproofing" in s["rooms"],
}

BREAKFAST_LEVELS = {"good breakfast": 1, "very good breakfast": 2, "excellent breakfast": 3,
                    "wonderful breakfast": 4, "exceptional breakfast": 5}

def _items(text):
    return [i.lower() for i in split_list(text)]

def amenity_features(popular, rooms, listed, highlights, facilities):
    s = {"popular": _items(popular), "rooms": _items(rooms), "listed": _items(listed), "highlights": _items(highlights)}
    out = {flag: int(rule(s)) for flag, rule in AMENITY_RULES.items()}
    everything = s["popular"] + s["highlights"]
    n_rest = [int(m.group(1)) for i in everything for m in [re.match(r"(\d+) restaurants", i)] if m]
    out["n_restaurants"] = max(n_rest) if n_rest else int(out["has_restaurant"])
    n_pools = [int(m.group(1)) for i in everything for m in [re.match(r"(\d+) swimming pools", i)] if m]
    out["n_pools"] = max(n_pools) if n_pools else int(out["has_pool"])
    levels = [BREAKFAST_LEVELS[i] for i in everything if i in BREAKFAST_LEVELS]
    out["breakfast_rating_level"] = max(levels) if levels else 0
    out["n_popular_facilities"] = len(s["popular"])
    out["n_room_amenities"] = len(s["rooms"]) if s["rooms"] else np.nan
    out["n_listed_facilities"] = len(s["listed"]) if s["listed"] else np.nan
    out["n_facility_groups"] = (facilities.count("||") + 1) if facilities else np.nan
    return out

def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def room_facts(room_types, meal_plans, chain):
    """One hotel's facts from its page opened with dates. Blank = no free room on any date tried."""
    rooms = clean_room_list(room_list(room_types))
    sizes = [r["m2"] for r in rooms if r["m2"] == r["m2"]]
    plans = split_list(meal_plans)
    levels = [MEAL_LEVELS[meal_plan(p)] for p in plans]
    bf_price = [v for v in map(breakfast_extra_lkr, plans) if v == v]
    seen = bool(rooms)
    na = np.nan
    return {"has_room_facts": int(seen), "room_sizes_set_aside": sum(r["m2_set_aside"] for r in rooms),
            "n_room_types": len(rooms) if seen else na,
            "room_m2_min": min(sizes) if sizes else na, "room_m2_median": float(np.median(sizes)) if sizes else na,
            "room_m2_max": max(sizes) if sizes else na,
            "offers_breakfast_rate": int(1 in levels) if seen else na, "offers_half_board": int(2 in levels) if seen else na,
            "offers_full_board": int(3 in levels) if seen else na, "offers_all_inclusive": int(4 in levels) if seen else na,
            "max_meal_level_offered": max(levels, default=0) if seen else na,
            "breakfast_extra_lkr": float(np.median(bf_price)) if bf_price else na,
            "chain_name": (chain or "").strip() or None, "is_chain": int(bool((chain or "").strip())) if seen else na}

def build_hotel_table(raw):
    b = raw[raw.source_ota == "booking"].drop_duplicates("hotel_id_on_source")
    num = lambda col: pd.to_numeric(b[col].replace("", np.nan), errors="coerce")
    h = pd.DataFrame({"booking_hotel_id": b.hotel_id_on_source.values, "hotel_name": b.hotel_name.values,
                      "destination_cluster": b.destination_cluster.values}, index=b.index)
    h["property_type"] = b.property_type_raw
    h["latitude"], h["longitude"] = num("latitude_raw"), num("longitude_raw")
    h["address_region"] = b.address_region_raw.replace("", np.nan)
    h["postal_code"] = b.postal_code_raw.replace("", np.nan)
    h["page_rating"] = num("overall_rating_raw")
    h["page_review_count"] = num("overall_review_count_raw")
    for s in SUBSCORES:
        h[f"subscore_{s}"] = num(f"subscore_{s}_raw")
    h["validation_value_for_money"] = num("subscore_value_for_money_raw")          # NOT a model input
    am = pd.DataFrame([amenity_features(*vals) for vals in zip(b.popular_facilities_raw, b.room_amenities_raw,
                       b.listed_facilities_raw, b.property_highlights_raw, b.facilities_raw)], index=b.index)
    h = h.join(am)
    centre = h.groupby("destination_cluster")[["latitude", "longitude"]].transform("median")
    h["km_from_town_centre"] = haversine_km(h.latitude, h.longitude, centre.latitude, centre.longitude).round(2)
    h["details_complete"] = (b.details_warnings == "").astype(int)
    h = h.join(pd.DataFrame([room_facts(*vals) for vals in zip(b.room_types_raw, b.meal_plans_raw, b.hotel_chain_raw)],
                            index=b.index))
    chains = h.chain_name.fillna("")
    big = chains.map(chains[chains != ""].value_counts()).fillna(0) >= CHAIN_MIN_HOTELS
    h["chain_group"] = np.where(h.has_room_facts == 0, "unknown",
                                np.where(chains == "", "independent", np.where(big, chains, "other chain")))
    return h.reset_index(drop=True)


# ---- preprocessing.ipynb, cell 23 ----
def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]

def _facility_items(facilities, room_amenities):
    items = set()
    for group in (facilities or "").split("||"):
        items |= {t.strip().lower() for t in group.split(":", 1)[-1].split(";") if t.strip()}
    items |= {"room: " + t.lower() for t in split_list(room_amenities)}
    return items

def words_and_facilities(raw):
    b = raw[raw.source_ota == "booking"].drop_duplicates("hotel_id_on_source")
    names = b.hotel_name.str.lower().fillna("")
    towns = {w for t in raw.destination_cluster.unique() for w in re.findall(r"[a-z]+", t.lower())}
    all_names = set(raw.hotel_name.str.lower().fillna(""))           # distinct hotel names on all three sites
    counts = collections.Counter(w for nm in all_names for w in set(re.findall(r"[a-z]+", nm)))
    words = sorted(w for w, c in counts.items() if c >= NAME_WORD_MIN_HOTELS and w not in towns and w not in NAME_WORD_SKIP)
    out, meaning = {"booking_hotel_id": b.hotel_id_on_source.values}, {}
    for w in words:
        out[f"name_{w}"] = names.str.contains(rf"\b{w}\b", regex=True).astype(int).values
        meaning[f"name_{w}"] = f'hotel name contains "{w}"'
    fac = [_facility_items(f, r) for f, r in zip(b.facilities_raw, b.room_amenities_raw)]
    has_list = (b.facilities_raw != "").values
    fc = collections.Counter(t for s, ok in zip(fac, has_list) if ok for t in s)
    n_list = int(has_list.sum())
    items = sorted(t for t, c in fc.items() if FACILITY_MIN_HOTELS <= c <= n_list - FACILITY_MIN_HOTELS)
    for t in items:
        col = "fac_" + _slug(t)
        while col in out:
            col += "_"
        out[col] = np.where(has_list, [int(t in s) for s in fac], np.nan)
        meaning[col] = f'facility list has "{t}"'
    return pd.DataFrame(out), meaning


# ---- preprocessing.ipynb, cell 27 ----
def normalise_name(name, town=None):
    t = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    drop = set(GENERIC_NAME_WORDS)
    if town:
        drop |= set(normalise_name(town).split())
    return " ".join(w for w in t.split() if w not in drop)

def key_words(normalised):
    return " ".join(w for w in normalised.split() if w not in DESCRIPTIVE_NAME_WORDS)

def similarity(a, b):
    """0..1 on the identifying words; also tried with the words sorted ("Nira Villa" = "Villa Nira")."""
    ka, kb = key_words(a), key_words(b)
    if not ka and not kb:
        return 1.0 if a and a == b else 0.0
    if not ka or not kb:
        return 0.0
    plain = difflib.SequenceMatcher(None, ka, kb).ratio()
    ordered = difflib.SequenceMatcher(None, " ".join(sorted(ka.split())), " ".join(sorted(kb.split()))).ratio()
    return max(plain, ordered)

def match_hotels(listings, booking_hotels):
    nb = defaultdict(set)
    for a, b in NEIGHBOUR_TOWNS:
        nb[a].add(b); nb[b].add(a)
    town = dict(zip(booking_hotels.booking_hotel_id, booking_hotels.destination_cluster))
    bname = dict(zip(booking_hotels.booking_hotel_id, booking_hotels.hotel_name))
    b_same = {h: normalise_name(bname[h], town[h]) for h in bname}
    b_full = {h: normalise_name(bname[h]) for h in bname}
    by_town = defaultdict(list)
    for h, t in town.items():
        by_town[t].append(h)
    others = listings[listings.source_ota != "booking"].drop_duplicates(["source_ota", "hotel_id_on_source"])
    rows = []
    for r in others.itertuples():
        t = r.destination_clean
        n_same, n_full = normalise_name(r.hotel_name, t), normalise_name(r.hotel_name)
        scored = [(similarity(n_same, b_same[c]), c, "same town") for c in by_town.get(t, [])]
        scored += [(similarity(n_full, b_full[c]), c, "neighbouring town") for nt in nb.get(t, ()) for c in by_town.get(nt, [])]
        scored.sort(key=lambda x: -x[0])
        best_score, best, where = scored[0] if scored and scored[0][0] > 0 else (0.0, None, "")
        runner = scored[1][0] if len(scored) > 1 else 0.0
        need = MATCH_MIN_SCORE_SAME_TOWN if where == "same town" else MATCH_MIN_SCORE_NEIGHBOUR
        if best is None:
            status = "no candidate"
        elif best_score < need:
            status = "below threshold"
        elif best_score - runner < MATCH_MIN_MARGIN:
            status = "ambiguous"
        else:
            status = "matched"
        rows.append({"source_ota": r.source_ota, "hotel_id_on_source": r.hotel_id_on_source, "hotel_name": r.hotel_name,
                     "destination_clean": t, "normalised_name": n_same if where != "neighbouring town" else n_full,
                     "best_booking_hotel_id": best, "best_booking_name": bname.get(best), "best_booking_town": town.get(best),
                     "candidate_town": where, "score": round(best_score, 3), "runner_up_score": round(runner, 3),
                     "status": status, "booking_hotel_id": best if status == "matched" else None, "manual_check": ""})
    return pd.DataFrame(rows)
