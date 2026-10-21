"""
Loads the hotel data once and keeps it in memory.

Files (made by the preprocessing, model and where-to-book notebooks):
    <C2_DATA_DIR>/output_clean/hotels_booking.csv          one row per Booking.com hotel (facts, rating, location)
    <C2_DATA_DIR>/output_clean/hotel_words_facilities.csv  the full facility list per hotel (fac_* columns)
    <C2_DATA_DIR>/output_clean/model_table.csv             every collected price (one row per hotel + date + guests + site)
    <C2_DATA_DIR>/output_clean/where_to_book.csv           Booking.com vs Trip.com vs Agoda for each stay
    <C2_DATA_DIR>/output_clean/hotel_matches.csv           the same hotel on Agoda / Trip.com
    <C2_DATA_DIR>/models/booking_predictions.csv           fair price for every Booking.com price (hotel not seen)
    <C2_DATA_DIR>/models/booking_best_model.joblib         the price model (optional: needs scikit-learn 1.5.2)
    <C2_DATA_DIR>/models/booking_cv_models.joblib          the 3 cross-validation models (optional, for SHAP on
                                                           training hotels; made by python -m src.c2.unseen_models)
    <C2_EXTRA_DIR>/sl_hospitals_osm.csv                    hospitals from OpenStreetMap
    <C2_EXTRA_DIR>/sl_places_osm.csv                       towns and villages from OpenStreetMap
    <C2_EXTRA_DIR>/hotel_urls.csv                          the hotel's page on each site
    <C2_DATA_DIR>/output_clean/live/*.csv                  hotels and prices found by the LIVE Booking.com search,
                                                           and every hotel's photos (added over time; see live/search.py)
"""
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings
from .geo import TownFinder, nearest_hospitals

# Hotel columns the code uses. Missing ones are filled, so a smaller test file also works.
HOTEL_NUMBER_COLS = [
    "latitude", "longitude", "page_rating", "page_review_count", "subscore_staff", "subscore_facilities",
    "subscore_cleanliness", "subscore_comfort", "subscore_location", "subscore_free_wifi", "km_from_town_centre",
    "has_pool", "has_spa", "has_bar", "has_restaurant", "is_beachfront", "has_private_beach", "has_free_parking",
    "has_free_wifi", "has_family_rooms", "has_disabled_facilities", "has_garden", "rooms_sea_view",
    "rooms_air_conditioning", "rooms_soundproofing", "offers_breakfast_rate", "breakfast_rating_level",
    "details_complete", "n_pools",
]
FAC_COLS_USED = [   # facility-list columns used to check features (from hotel_words_facilities.csv)
    "fac_beachfront", "fac_private_beach_area", "fac_soundproof_rooms", "fac_soundproofing", "fac_room_soundproofing",
    "fac_free_wifi", "fac_internet_no_internet_access_available", "fac_free_parking", "fac_parking",
    "fac_private_parking", "fac_parking_no_parking_available", "fac_sea_view", "fac_room_sea_view",
    "fac_mountain_view", "fac_room_mountain_view", "fac_room_lake_view", "fac_room_river_view", "fac_landmark_view",
    "fac_room_landmark_view", "fac_view", "fac_room_view", "fac_air_conditioning", "fac_room_air_conditioning",
    "fac_spa_and_wellness_center", "fac_spa_facilities", "fac_garden", "fac_garden_view", "fac_room_garden_view",
    "fac_elevator", "fac_upper_floors_accessible_by_elevator", "fac_room_upper_floors_accessible_by_elevator",
    "fac_entire_unit_located_on_ground_floor", "fac_room_entire_unit_located_on_ground_floor",
    "fac_upper_floors_accessible_by_stairs_only", "fac_room_upper_floors_accessible_by_stairs_only",
    "fac_wheelchair_accessible", "fac_entire_unit_wheelchair_accessible",
    "fac_room_entire_unit_wheelchair_accessible", "fac_facilities_for_disabled_guests", "fac_toilet_with_grab_rails",
]
STAY_COLS = ["row_id", "booking_hotel_id", "hotel_name", "destination_clean", "checkin_date", "num_adults",
             "num_children", "price_incl_tax_lkr_night", "meal_plan", "free_cancellation", "room_tier_name"]


OTHER_COLS = ["source_ota", "hotel_id_on_source", "hotel_name", "destination_clean", "booking_hotel_id", "checkin_date",
              "num_adults", "num_children", "price_incl_tax_lkr_night", "meal_plan", "free_cancellation"]
LIVE_STAY_EXTRA = ["fair_price_lkr", "price_vs_fair_pct", "wtb_tripcom", "wtb_agoda", "wtb_notes", "wtb_advice",
                   "wtb_site", "scraped_at_utc"]


class DataMissing(RuntimeError):
    pass


def guests_text(adults, children):
    """Same wording as where_to_book.csv: '2 adults + 1 child'."""
    a, c = int(adults), int(children)
    text = f"{a} adult{'s' if a > 1 else ''}"
    return text + (f" + {c} child{'ren' if c > 1 else ''}" if c else "")


def _read(path, **kw):
    if not Path(path).exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kw)


class HotelData:
    def __init__(self, data_dir=None, extra_dir=None, load_model=True, model_from=None):
        self.data_dir = Path(data_dir or settings.data_dir())
        self.extra_dir = Path(extra_dir or settings.extra_dir())
        self.notes = []
        clean = self.data_dir / "output_clean"
        models = self.data_dir / "models"
        self.live_dir = clean / "live"
        live = lambda name, **kw: _read(self.live_dir / name, **kw)

        hotels = _read(clean / "hotels_booking.csv")
        stays = _read(clean / "model_table.csv")
        if hotels is None or stays is None:
            raise DataMissing(f"Hotel data not found in {self.data_dir}. Set C2_DATA_DIR to the hotel_preprocessing "
                              "folder (the one with output_clean/ and models/).")
        hotels_live = live("hotels_booking_live.csv")
        self.live_found = {}                     # booking_hotel_id -> when the live search found it
        if hotels_live is not None and len(hotels_live):
            hotels_live = hotels_live[~hotels_live.booking_hotel_id.isin(hotels.booking_hotel_id)]
            self.live_found = dict(zip(hotels_live.booking_hotel_id, hotels_live.get("scraped_at_utc", "")))
            hotels = pd.concat([hotels, hotels_live], ignore_index=True)

        # ---- hotels: facts + facility list + nearest hospital ----
        for c in HOTEL_NUMBER_COLS:
            if c not in hotels:
                hotels[c] = np.nan
        words = _read(clean / "hotel_words_facilities.csv")
        words_live = live("hotel_words_facilities_live.csv")
        if words is not None and words_live is not None and len(words_live):
            words_live = words_live[~words_live.booking_hotel_id.isin(words.booking_hotel_id)]
            words = pd.concat([words, words_live[[c for c in words.columns if c in words_live]]], ignore_index=True)
        self.words = words
        if words is not None:
            keep = ["booking_hotel_id"] + [c for c in FAC_COLS_USED if c in words]
            hotels = hotels.merge(words[keep], on="booking_hotel_id", how="left")
        for c in FAC_COLS_USED:
            if c not in hotels:
                hotels[c] = 0
            hotels[c] = hotels[c].fillna(0)
        hotels = hotels.drop_duplicates("booking_hotel_id").set_index("booking_hotel_id", drop=False)

        self.hospitals = _read(self.extra_dir / "sl_hospitals_osm.csv")
        if self.hospitals is None:
            self.notes.append("sl_hospitals_osm.csv not found: hospital distances are unknown.")
        hotels = hotels.join(nearest_hospitals(hotels, self.hospitals))
        self.hotels = hotels

        places = _read(self.extra_dir / "sl_places_osm.csv")
        if places is None:
            self.notes.append("sl_places_osm.csv not found: only our 15 data towns can be searched.")
        self.towns = TownFinder(places, hotels)

        # ---- stays: one Booking.com price per hotel + date + guests, with its fair price and where to book ----
        b = stays[(stays.source_ota == "booking")]
        if "is_extra_date" in b:
            b = b[b.is_extra_date == 0]
        b = b[b.booking_hotel_id.notna()]
        if "is_sold_out" in b:
            b = b[b.is_sold_out.fillna(0) == 0]
        b = b.sort_values("price_incl_tax_lkr_night").drop_duplicates(
            ["booking_hotel_id", "checkin_date", "num_adults", "num_children"])
        self.model_rows = b.set_index("row_id", drop=False)      # every column: what the price model needs
        self.others = stays[stays.source_ota.isin(["agoda", "tripcom"])][[c for c in OTHER_COLS if c in stays]].copy()
        s = b[[c for c in STAY_COLS if c in b]].copy()
        for c in STAY_COLS:
            if c not in s:
                s[c] = np.nan
        s["guests"] = [guests_text(a, c) for a, c in zip(s.num_adults, s.num_children)]

        pred = _read(models / "booking_predictions.csv")
        self.typical_rows = None                 # the starting point of the SHAP explanation (see explain.py)
        if pred is not None:
            s = s.merge(pred[["row_id", "fair_price_lkr", "price_vs_fair_pct"]], on="row_id", how="left")
            from .explain import typical_price_rows
            priced = stays[stays.source_ota == "booking"]
            if "is_extra_date" in priced:
                priced = priced[priced.is_extra_date == 0]
            self.typical_rows = typical_price_rows(priced.merge(pred[["row_id", "fair_price_lkr"]], on="row_id"),
                                                   settings.SHAP_BACKGROUND)
        else:
            s["fair_price_lkr"] = np.nan
            s["price_vs_fair_pct"] = np.nan
            self.notes.append("booking_predictions.csv not found: no fair prices.")

        wtb = _read(clean / "where_to_book.csv")
        if wtb is not None:
            wtb = wtb.rename(columns={"hotel": "hotel_name", "town": "destination_clean", "check-in": "checkin_date",
                                      "Booking.com": "wtb_booking", "Trip.com": "wtb_tripcom",
                                      "Agoda (from)": "wtb_agoda", "notes": "wtb_notes", "advice": "wtb_advice",
                                      "recommended site": "wtb_site"})
            wtb = wtb.drop_duplicates(["hotel_name", "destination_clean", "checkin_date", "guests"])
            s = s.merge(wtb[["hotel_name", "destination_clean", "checkin_date", "guests", "wtb_tripcom", "wtb_agoda",
                             "wtb_notes", "wtb_advice", "wtb_site"]],
                        on=["hotel_name", "destination_clean", "checkin_date", "guests"], how="left")
        else:
            for c in ["wtb_tripcom", "wtb_agoda", "wtb_notes", "wtb_advice", "wtb_site"]:
                s[c] = np.nan
            self.notes.append("where_to_book.csv not found: only Booking.com prices are shown.")
        prices_live = live("prices_live.csv")
        self.live_rows = None
        if prices_live is not None and len(prices_live):
            prices_live = prices_live[prices_live.is_sold_out.fillna(0) == 0]
            self.live_rows = prices_live
            sl = prices_live[[c for c in STAY_COLS + LIVE_STAY_EXTRA if c in prices_live]].copy()
            sl["guests"] = [guests_text(a, c) for a, c in zip(sl.num_adults, sl.num_children)]
            s = pd.concat([s, sl], ignore_index=True)
        s["checkin"] = pd.to_datetime(s.checkin_date)
        self.stays = s
        self.stay_index = {k: g for k, g in s.groupby("booking_hotel_id")}

        # ---- links to the hotel on each site ----
        self.urls = {}
        urls = _read(self.extra_dir / "hotel_urls.csv", dtype={"hotel_id_on_source": str})
        urls_live = live("hotel_urls_live.csv", dtype={"hotel_id_on_source": str})
        if urls is not None and urls_live is not None:
            urls = pd.concat([urls, urls_live], ignore_index=True)
        if urls is not None:
            self.urls = {(o, i): u for o, i, u in zip(urls.source_ota, urls.hotel_id_on_source, urls.url)}
        else:
            self.notes.append("hotel_urls.csv not found: no booking links.")
        self.same_hotel = {}                    # booking_hotel_id -> {"tripcom": id, "agoda": id}
        matches = _read(clean / "hotel_matches.csv", dtype={"hotel_id_on_source": str})
        matches_live = live("hotel_matches_live.csv", dtype={"hotel_id_on_source": str})
        if matches is not None and matches_live is not None:
            matches = pd.concat([matches, matches_live], ignore_index=True)
        if matches is not None:
            m = matches[matches.status == "matched"].sort_values("score", ascending=False)
            for bid, ota, hid in zip(m.booking_hotel_id, m.source_ota, m.hotel_id_on_source):
                self.same_hotel.setdefault(bid, {}).setdefault(ota, hid)

        # ---- photos (filled by the live search and the photo step) ----
        self.photos = {}
        photos = live("hotel_photos.csv", dtype={"booking_hotel_id": str})
        if photos is not None:
            for hid, g in photos.sort_values(["booking_hotel_id", "position"]).groupby("booking_hotel_id"):
                self.photos[hid] = g[["photo_url", "thumb_url"]].to_dict("records")

        # ---- the price model (optional) ----
        self.model, self.features = None, None
        self.cv, self.cv_note = None, None        # the cross-validation models (optional)
        if model_from is not None and model_from.model is not None:
            self.model, self.features = model_from.model, model_from.features
            self.cv, self.cv_note = model_from.cv, model_from.cv_note
            self._add_word_columns()
        elif load_model:
            self._load_model(models)
            self._load_cv_models(models)
        if self.typical_rows is not None and self.words is not None:
            self.typical_rows = self.typical_rows.merge(self.words, on="booking_hotel_id", how="left",
                                                        suffixes=("", "_w"))
        if self.live_rows is not None:          # live prices already carry every model column
            live_rows = self.live_rows.set_index("row_id", drop=False)
            self.model_rows = pd.concat([self.model_rows, live_rows[[c for c in self.model_rows.columns
                                                                     if c in live_rows]]])

    def _add_word_columns(self):
        if self.words is not None:
            extra = [c for c in self.features if c in self.words.columns and c not in self.model_rows.columns]
            rows = self.model_rows.merge(self.words[["booking_hotel_id"] + extra], on="booking_hotel_id", how="left")
            self.model_rows = rows.set_index("row_id", drop=False)

    # ---- which model never saw this price's hotel (its fair price came from that model) ----
    def unseen_model(self, row_id):
        """(model, short name, label) for the model that made this row's fair price; (None, None, reason) if none."""
        if self.model is None:
            return None, None, "The price model is not loaded."
        if row_id not in self.model_rows.index:
            return None, None, "Unknown price (row_id)."
        r = self.model_rows.loc[row_id]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        if self.is_live(r.get("booking_hotel_id")):
            return self.model, "final", "the price model (this hotel was found live, so the model never saw it)"
        split = r.get("split")
        if split == "test":
            return self.model, "final", "the final price model (a test hotel: the model never saw it)"
        if split == "train":
            if self.cv is None:
                return None, None, ("This hotel was used to train the price model, and the 3 cross-validation models "
                                    "that never saw it are not built yet (run: python -m src.c2.unseen_models).")
            k = self.cv["hotel_fold"].get(r.get("hotel_group_id"))
            if k is not None:
                return (self.cv["models"][k], f"cv{k + 1}",
                        f"cross-validation model {k + 1} of {len(self.cv['models'])} (it never saw this hotel)")
        return None, None, "We don't know which model made this fair price."

    def _load_cv_models(self, models):
        path = models / "booking_cv_models.joblib"
        if self.model is None:
            return
        if not path.exists():
            self.cv_note = "booking_cv_models.joblib not found: SHAP works only for test hotels and live hotels."
            return
        try:
            import joblib
            bundle = joblib.load(path)
            if list(bundle["features"]) != list(self.features):
                raise ValueError("made for a different price model")
            self.cv = bundle
        except Exception as e:
            self.cv_note = f"Cross-validation models could not be loaded ({type(e).__name__}: {str(e)[:120]})."

    # ---- small helpers for the live search ----
    def is_live(self, hotel_id):
        return hotel_id in self.live_found

    def town_centre(self, cluster):
        g = self.hotels[self.hotels.destination_cluster == cluster]
        if len(g):
            return float(g.latitude.median()), float(g.longitude.median())
        p = self.towns.point(cluster)
        return (p[0], p[1]) if p else None

    def known_chains(self):
        if "chain_group" in self.model_rows:
            return set(self.model_rows.chain_group.dropna().unique()) - {"independent", "other chain", "unknown"}
        return set()

    def _load_model(self, models):
        path, info_path = models / "booking_best_model.joblib", models / "booking_best_model_info.json"
        if not path.exists() or not info_path.exists():
            self.notes.append("Price model not found: fair prices for your dates and the what-if amounts are not shown.")
            return
        try:
            import joblib
            info = json.loads(info_path.read_text(encoding="utf-8"))
            self.model = joblib.load(path)
            self.features = info["features"]
            self._add_word_columns()
            missing = [c for c in self.features if c not in self.model_rows.columns]
            if missing:
                raise ValueError(f"{len(missing)} model columns missing, e.g. {missing[:3]}")
        except Exception as e:     # wrong scikit-learn version, missing file ...
            self.model, self.features = None, None
            self.notes.append(f"Price model could not be loaded ({type(e).__name__}: {str(e)[:120]}). "
                              "It needs scikit-learn 1.5.2. Fair prices from booking_predictions.csv are used instead.")


_lock = threading.Lock()
_data = {"obj": None}


def get_data():
    """The data, loaded on first use (about 10 seconds), then kept in memory."""
    with _lock:
        if _data["obj"] is None:
            _data["obj"] = HotelData()
        return _data["obj"]


def reload_data():
    """Read the files again (after the live search added hotels), keeping the loaded price model."""
    with _lock:
        old = _data["obj"]
        _data["obj"] = HotelData(old.data_dir, old.extra_dir, model_from=old) if old else HotelData()
        return _data["obj"]


def set_data(obj):
    """For tests: use a different data object."""
    with _lock:
        _data["obj"] = obj
