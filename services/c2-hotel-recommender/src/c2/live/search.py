"""
LIVE Booking.com search - only when our CSV data is not enough for this tourist.

When does it run? (ranker.py decides and says why)
    - fewer than top_k hotels in our data meet every must-have, or
    - the town is not in our data (or our hotels are more than 10 km away).

What it does (one search page + a few hotel pages, never the whole town):
    1. Opens ONE Booking.com search for the tourist's town, dates and group, with the budget and the must-haves
       as Booking.com's own filters (pool, air conditioning, beachfront ...), hotels only.
    2. Keeps the photo of every hotel on that page (also hotels we already have).
    3. Takes the first few hotels (default 4) that are NOT in our CSV files and opens their hotel pages
       (with the dates) to read the facts, facilities, rooms and photos - the same reading rules as the scraper.
    4. Turns them into rows with the same columns as the research CSVs (preprocessing rules), asks the price model
       for their fair price, and estimates the Agoda / Trip.com price when the same hotel is in our September data.
    5. Saves everything in output_clean/live/*.csv, so next time these hotels come from the CSV, not the web.

Never: answering a "are you a robot?" check (the search just stops and says so), or scraping Agoda / Trip.com.
Needs: pip install playwright beautifulsoup4 lxml   and   python -m playwright install chromium
"""
import random
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .. import settings
from ..data import get_data, reload_data
from ..pricing import _season_gap, date_features, room_plan
from . import booking_pages as bp
from . import prep_rules as P
from .to_rows import LIVE_ROW_ID_START, hotel_row, now_utc, price_row, room_facts_raw, words_row

SCRAPE_LOCK = threading.Lock()          # one browser at a time
HUMAN_CHECK_WORDS = ("captcha", "are you a robot", "verify you are human", "unusual traffic", "press and hold")


class LiveUnavailable(RuntimeError):
    pass


# ---------------- the browser part (replaced by a fake in the tests) ----------------
class PlaywrightPages:
    """Opens Booking.com pages in a real Chromium browser (Playwright), like the scraper does."""

    def __init__(self, page):
        self.page = page

    def _text(self):
        return bp.page_text(self.page).lower()

    def human_check(self):
        return any(w in self._text() for w in HUMAN_CHECK_WORDS)

    def search(self, url):
        """The cards' HTML, [] for no results, None if the page did not load."""
        try:
            ok = bp.polite_goto(self.page, url, wait_selector=bp.CARD_SELECTOR, timeout_ms=settings.LIVE_TIMEOUT_MS)
            if not ok:
                return [] if "0 properties found" in self._text() or "no properties found" in self._text() else None
            return [c.inner_html() for c in self.page.query_selector_all(bp.CARD_SELECTOR)[:25]]
        except Exception as e:
            print(f"[live search] search page failed: {type(e).__name__}: {str(e)[:200]}")
            return None

    def hotel(self, url, facilities=True):
        """The hotel page's HTML, or None if it did not load in time (the hotel is then skipped)."""
        try:
            ok = bp.polite_goto(self.page, url, wait_selector=f"{bp.STRUCTURED_DATA_SELECTOR}, {bp.LATLNG_SELECTOR}",
                                timeout_ms=settings.LIVE_TIMEOUT_MS, wait_state="attached", wait_until="commit",
                                interstitial_retries=1)
            if not ok:
                return None
            if facilities:
                try:        # the facility list needs the page's scripts; give them a little time, never fail on it
                    self.page.wait_for_load_state("domcontentloaded", timeout=15000)
                except Exception:
                    pass
                try:
                    self.page.wait_for_selector(bp.SUBSCORE_SELECTOR, timeout=3000)
                except Exception:
                    pass
                bp.load_facilities(self.page, wait_ms=4000)
            return self.page.content()
        except Exception as e:
            print(f"[live search] hotel page failed: {url[:90]} ({type(e).__name__}: {str(e)[:150]})")
            return None

    def pause(self):
        time.sleep(random.uniform(1.0, 2.5))          # polite gap between pages


@contextmanager
def open_browser():
    if bp.sync_playwright is None:
        raise LiveUnavailable("Live search needs Playwright: pip install playwright beautifulsoup4 lxml, then "
                              "python -m playwright install chromium")
    try:
        with bp.new_browser_context(headless=settings.live_headless()) as page:
            yield PlaywrightPages(page)
    except LiveUnavailable:
        raise
    except Exception as e:
        if "Executable doesn't exist" in str(e):
            raise LiveUnavailable("Chromium for Playwright is not installed: python -m playwright install chromium")
        raise


# ---------------- saving ----------------
def _append(path, rows, key):
    """Add rows to a CSV file, replacing rows with the same key (the newest wins)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={k: str for k in key})
        new = pd.concat([old, new], ignore_index=True)
    for k in key:
        new[k] = new[k].astype(str)
    new = new.drop_duplicates(key, keep="last")
    new.to_csv(path, index=False, encoding="utf-8-sig")


def save_photos(data, hotel_id, photos, source):
    if not photos:
        return
    rows = [{"booking_hotel_id": hotel_id, "position": i, "photo_url": p["photo_url"], "thumb_url": p["thumb_url"],
             "source": source, "scraped_at_utc": now_utc()} for i, p in enumerate(photos)]
    _append(data.live_dir / "hotel_photos.csv", rows, ["booking_hotel_id", "position"])
    data.photos[hotel_id] = [{"photo_url": p["photo_url"], "thumb_url": p["thumb_url"]} for p in photos]


# ---------------- price model and other sites ----------------
def _predict(data, rows):
    if data.model is None or not rows:
        return [np.nan] * len(rows)
    X = pd.DataFrame(rows)
    for c in data.features:
        if c not in X:
            X[c] = np.nan
    X = X[data.features].copy()
    cats = ("destination_clean", "star_band", "chain_group")
    for c in data.features:
        X[c] = X[c].astype(object) if c in cats else pd.to_numeric(X[c], errors="coerce")
    return list(np.exp(data.model.predict(X)))


def _model_town(data, lat, lon, cluster):
    """destination_clean for the price model: the town if it is one of the 15 research towns, else the nearest."""
    research = {"Anuradhapura", "Arugam Bay", "Bentota", "Colombo", "Dambulla", "Ella", "Galle", "Jaffna", "Kandy",
                "Mirissa", "Negombo", "Nuwara Eliya", "Sigiriya", "Trincomalee", "Unawatuna"}
    if cluster in research:
        return cluster
    h = data.hotels[data.hotels.destination_cluster.isin(research)]
    centres = h.groupby("destination_cluster")[["latitude", "longitude"]].median()
    if lat != lat or not len(centres):
        return None
    d = P.haversine_km(lat, lon, centres.latitude.values, centres.longitude.values)
    return centres.index[int(np.argmin(d))]


def other_sites(data, name, model_town, combo, check_in, live_row, today):
    """Agoda / Trip.com: the same hotel in our September data (name match in the same town, the preprocessing
    rule: similarity >= 0.90 and clearly better than the next one), its price at the closest date for the same
    group, moved to the tourist's dates with the price model. Returns {site: {...}}."""
    out = {}
    if model_town is None or not len(data.others):
        return out
    o = data.others[(data.others.destination_clean == model_town) & data.others.booking_hotel_id.isna()]
    target = P.normalise_name(name, model_town)
    for site in ("tripcom", "agoda"):
        cands = o[o.source_ota == site].drop_duplicates("hotel_id_on_source")
        scored = sorted(((P.similarity(target, P.normalise_name(n, model_town)), str(h), n)
                         for h, n in zip(cands.hotel_id_on_source, cands.hotel_name)), reverse=True)
        if not scored or scored[0][0] < P.MATCH_MIN_SCORE_SAME_TOWN or \
                (len(scored) > 1 and scored[0][0] - scored[1][0] < P.MATCH_MIN_MARGIN):
            continue
        score, hid, other_name = scored[0]
        rows = o[(o.source_ota == site) & (o.hotel_id_on_source.astype(str) == hid) &
                 (o.num_adults == combo[0]) & (o.num_children == combo[1]) & o.price_incl_tax_lkr_night.notna()]
        if not len(rows):
            continue
        rows = rows.assign(gap=[_season_gap(date.fromisoformat(d), check_in) for d in rows.checkin_date])
        best = rows.sort_values(["gap", "price_incl_tax_lkr_night"]).iloc[0]
        seen = date.fromisoformat(best.checkin_date)
        at_seen = dict(live_row, **date_features(seen, live_row.get("nights") or 2, today))
        fair_yours, fair_seen = _predict(data, [live_row, at_seen])
        factor = fair_yours / fair_seen if fair_yours == fair_yours and fair_seen else 1.0
        out[site] = {"price": float(best.price_incl_tax_lkr_night) * factor, "hotel_id": hid, "name": other_name,
                     "score": round(score, 3), "seen": best.checkin_date}
    return out


def where_to_book(booking, sites):
    """The where_to_book.ipynb rules, with estimated Agoda / Trip.com prices."""
    out = {"wtb_tripcom": np.nan, "wtb_agoda": np.nan, "wtb_site": "Booking.com", "wtb_advice": "Book on Booking.com.",
           "wtb_notes": ""}
    notes = []
    if "tripcom" in sites:
        t = sites["tripcom"]["price"]
        out["wtb_tripcom"] = round(t)
        saving = 100 * (1 - t / booking)
        if 5 <= saving <= 40:
            out["wtb_site"] = "Trip.com"
            out["wtb_advice"] = (f"Trip.com may be about {saving:.0f}% cheaper (estimated from our September prices) - "
                                 "check it is the same room, breakfast and cancellation before booking.")
        elif saving > 40:
            notes.append(f"Trip.com looks {saving:.0f}% cheaper (estimate) - unusually big, check before booking.")
    if "agoda" in sites:
        a = sites["agoda"]["price"]
        out["wtb_agoda"] = round(a)
        saving = 100 * (1 - a / booking)
        if saving >= 5:
            notes.append(f"Agoda shows from about Rs {round(a, -2):,.0f} (estimate, {saving:.0f}% less) - room type and "
                         "breakfast are not shown, so check before booking.")
    if sites:
        notes.append("Booking.com price is live for your dates; Agoda / Trip.com prices are estimates from our "
                     "September data, moved to your dates with the price model.")
    out["wtb_notes"] = " | ".join(notes)
    return out


# ---------------- the live search ----------------
def _dated_hotel_url(card_url, ci, co, adults, children):
    base = (card_url or "").split("?")[0]
    q = [("checkin", ci.isoformat()), ("checkout", co.isoformat()), ("group_adults", adults),
         ("group_children", children), ("no_rooms", 1), ("selected_currency", "LKR"), ("lang", "en-us")]
    q += [("age", a) for a in bp.child_ages_for(children)]
    return base + "?" + urlencode(q)


def run_live_search(request, town, today=None, pages=None, data=None):
    """Search Booking.com for this town and request, save new hotels to the CSV files. Returns a report."""
    data = data or get_data()
    today = date.fromisoformat(today) if isinstance(today, str) else (today or date.today())
    started = time.time()
    trip, hotel = request.get("trip", {}), request.get("hotel", {})
    example = not trip.get("check_in")
    ci = date.fromisoformat(trip["check_in"]) if trip.get("check_in") else today + timedelta(days=30)
    nights = int(trip.get("nights") or 2)
    co = ci + timedelta(days=nights)
    rooms, _ = room_plan(trip.get("adults"), trip.get("children"))
    combo = rooms[0]
    budget = hotel.get("budget_max_lkr")
    per_room = budget / len(rooms) if budget else None
    must = [m for m in hotel.get("must", []) if m in bp.MUST_FILTERS]
    cluster = data.towns.data_town(town) or town.strip().title()
    url = bp.search_url(f"{town}, Sri Lanka", ci.isoformat(), co.isoformat(), combo[0], combo[1], 1, per_room, must)
    report = {"town": town, "search_url": url, "check_in": ci.isoformat(), "nights": nights,
              "example_dates": example, "group_searched": {"adults": combo[0], "children": combo[1]},
              "filters": ["hotels only"] + (["budget"] if per_room else []) + must,
              "cards_seen": 0, "already_in_our_data": 0, "new_hotels": [], "skipped": [], "status": "ok",
              "message": "", "saved_to": str(data.live_dir)}

    if not SCRAPE_LOCK.acquire(timeout=120):
        report.update(status="busy", message="Another live search is still running. Try again in a minute.")
        return report
    try:
        visits = []
        with (open_browser() if pages is None else _given(pages)) as web:
            cards_html = web.search(url)
            if cards_html is None:
                blocked = web.human_check()
                report.update(status="blocked" if blocked else "failed",
                              message=("Booking.com asked for a human check, so the live search stopped (we never "
                                       "answer these checks). Showing hotels from our data only." if blocked else
                                       "The Booking.com search page did not load. Showing hotels from our data only."))
                return _finish(report, data, started, [])
            cards = [(bp.parse_booking_card(h), bp.photo_urls_from_card(h)) for h in cards_html]
            report["cards_seen"] = len(cards)
            chosen = []
            for card, photo in cards:
                hid = card.get("hotel_id_on_source")
                if not hid:
                    continue
                if photo and hid not in data.photos:
                    save_photos(data, hid, photo, "search card")
                if hid in data.hotels.index:
                    report["already_in_our_data"] += 1
                    continue
                price = P.price_columns({"source_ota": "booking", "nights": str(nights),
                                         "displayed_price_raw": card.get("displayed_price_raw"),
                                         "taxes_and_fees_raw": card.get("taxes_and_fees_raw")})
                night = price["price_incl_tax_lkr_night"]
                if night != night:
                    night = price["price_pre_tax_lkr_night"]
                if night != night:
                    report["skipped"].append(f"{card.get('hotel_name')}: no price (sold out)")
                    continue
                if P.whole_place_or_dorm(card.get("room_config_raw"), card.get("room_type_name_raw")):
                    report["skipped"].append(f"{card.get('hotel_name')}: a whole villa or a dorm bed, not a hotel room")
                    continue
                if per_room and night > per_room:
                    report["skipped"].append(f"{card.get('hotel_name')}: over the budget")
                    continue
                if len(chosen) < settings.LIVE_MAX_NEW_HOTELS:
                    chosen.append(card)
            for card in chosen:
                if time.time() - started > settings.LIVE_MAX_SECONDS:
                    report["skipped"].append(f"{card.get('hotel_name')}: not read (time limit reached)")
                    continue
                page_url = _dated_hotel_url(card.get("hotel_url"), ci, co, combo[0], combo[1])
                try:
                    html = web.hotel(page_url)
                except Exception as e:                      # never let one slow page stop the search
                    html = None
                    print(f"[live search] {card.get('hotel_name')}: {type(e).__name__}: {str(e)[:150]}")
                if html is None:
                    report["skipped"].append(f"{card.get('hotel_name')}: hotel page did not load in time")
                    continue
                visits.append((card, html))
                web.pause()
        return _finish(report, data, started, visits, ci=ci, co=co, combo=combo, cluster=cluster, today=today)
    finally:
        SCRAPE_LOCK.release()


@contextmanager
def _given(pages):
    yield pages


def _finish(report, data, started, visits, ci=None, co=None, combo=None, cluster=None, today=None):
    """Turn the hotel pages into CSV rows, save them, reload the data, log the search."""
    hotels, words, prices, urls, matches = [], [], [], [], []
    known_chains = data.known_chains()
    word_cols = [c for c in (data.words.columns if data.words is not None else []) if c != "booking_hotel_id"]
    next_id = int(max(LIVE_ROW_ID_START, (data.model_rows.row_id.max() + 1) if len(data.model_rows) else 0))
    scraped = now_utc()
    centre = data.town_centre(cluster) if cluster else None
    for card, html in visits:
        hid, name = card["hotel_id_on_source"], card.get("hotel_name") or card["hotel_id_on_source"]
        details = bp.parse_booking_hotel_page(html)
        info, rates = bp.parse_booking_rooms_page(html, (co - ci).days)
        facts = room_facts_raw(info, rates)
        h = hotel_row(hid, name, cluster, details, facts, centre, known_chains)
        h["scraped_at_utc"] = scraped
        w = words_row(hid, name, details, word_cols)
        mtown = _model_town(data, h["latitude"], h["longitude"], cluster)
        r = price_row(card, hotel=h, words=w, checkin=ci, checkout=co, adults=combo[0], children=combo[1],
                      scraped_at=scraped, destination_clean=mtown, room_list_raw=facts["room_types_raw"], row_id=next_id)
        next_id += 1
        fair = _predict(data, [r])[0]
        r["fair_price_lkr"] = fair
        price = r["price_incl_tax_lkr_night"]
        r["price_vs_fair_pct"] = 100 * (price / fair - 1) if (fair == fair and price == price) else np.nan
        sites = other_sites(data, name, mtown, combo, ci, r, today) if price == price else {}
        r.update(where_to_book(price, sites) if price == price else {})
        for site, s in sites.items():
            matches.append({"source_ota": site, "hotel_id_on_source": s["hotel_id"], "hotel_name": s["name"],
                            "destination_clean": mtown, "best_booking_hotel_id": hid, "candidate_town": "same town",
                            "score": s["score"], "status": "matched", "booking_hotel_id": hid,
                            "manual_check": "live search"})
        hotels.append(h)
        words.append(w)
        prices.append(r)
        urls.append({"source_ota": "booking", "hotel_id_on_source": hid, "hotel_name": name,
                     "url": (card.get("hotel_url") or "").split("?")[0]})
        save_photos(data, hid, bp.extract_photos(html, settings.PHOTOS_PER_HOTEL), "hotel page")
        report["new_hotels"].append({"hotel_id": hid, "name": name, "price_per_night_lkr": round(price)
                                     if price == price else None, "photos": len(data.photos.get(hid, [])),
                                     "same_hotel_on": sorted(sites)})
    d = data.live_dir
    _append(d / "hotels_booking_live.csv", hotels, ["booking_hotel_id"])
    _append(d / "hotel_words_facilities_live.csv", words, ["booking_hotel_id"])
    _append(d / "prices_live.csv", prices, ["booking_hotel_id", "checkin_date", "num_adults", "num_children"])
    _append(d / "hotel_urls_live.csv", urls, ["source_ota", "hotel_id_on_source"])
    _append(d / "hotel_matches_live.csv", matches, ["source_ota", "hotel_id_on_source"])
    report["seconds"] = round(time.time() - started, 1)
    if report["status"] == "ok":
        n = len(report["new_hotels"])
        report["message"] = (f"Searched Booking.com: {report['cards_seen']} hotels on the first page, "
                             f"{report['already_in_our_data']} already in our data, {n} new hotel"
                             f"{'' if n == 1 else 's'} read and saved to the CSV files.")
    # what was searched (no names, no health): for the research log
    _append(d / "live_search_log.csv", [{
        "searched_at_utc": now_utc(), "town": report["town"], "check_in": report["check_in"],
        "nights": report["nights"], "group": f"{report['group_searched']['adults']}+"
                                            f"{report['group_searched']['children']}",
        "n_filters": len(report["filters"]), "status": report["status"], "cards_seen": report["cards_seen"],
        "already_in_data": report["already_in_our_data"], "new_hotels": len(report["new_hotels"]),
        "seconds": report["seconds"]}], ["searched_at_utc"])
    if hotels:
        reload_data()
    return report


def fetch_photos(hotel_id, pages=None, data=None):
    """Photos for one hotel: from hotel_photos.csv, or read once from its Booking.com page and saved."""
    data = data or get_data()
    if hotel_id in data.photos and len(data.photos[hotel_id]) > 1:
        return data.photos[hotel_id], "saved"
    url = data.urls.get(("booking", hotel_id)) or f"https://www.booking.com/hotel/lk/{hotel_id}.html"
    if not SCRAPE_LOCK.acquire(timeout=120):
        return data.photos.get(hotel_id, []), "busy"
    try:
        with (open_browser() if pages is None else _given(pages)) as web:
            html = web.hotel(url.split("?")[0] + "?lang=en-us", facilities=False)
        photos = bp.extract_photos(html, settings.PHOTOS_PER_HOTEL) if html else []
        if photos:
            save_photos(data, hotel_id, photos, "hotel page")
            return data.photos[hotel_id], "booking.com"
        return data.photos.get(hotel_id, []), "not found"
    finally:
        SCRAPE_LOCK.release()
