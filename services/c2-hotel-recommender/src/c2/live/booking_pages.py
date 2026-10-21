"""
Reading Booking.com pages - COPIED from the team's scraper (ota_scraper_project v10), so the live data is read
with exactly the same rules as the research data:
    parse_booking_card()        scrapers/booking.py          one search-result card
    parse_booking_hotel_page()  scrapers/booking_details.py  hotel facts: location, sub-scores, facilities
    parse_booking_rooms_page()  scrapers/booking_rooms.py    room table of a hotel page opened with dates
    new_browser_context(), polite_goto()                     common/browser.py
Only new here: extract_photos() and photo_urls_from_card() (the hotel's pictures), load_facilities() (the
fetch() scrolling step as a function) and a few Booking.com search filters.

The page structure was checked live on 2026-10-19 (Tangalle search, Grand Samudra hotel page): property cards,
price filter, facility filters, coordinates, sub-scores, page data store, room table and photos all as expected.
"""
from __future__ import annotations

import json
import random
import re
import time
from contextlib import contextmanager

from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
except ImportError:          # the ranking still works without Playwright; only live search needs it
    sync_playwright = None


def _clean(text):
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()
    return text or None


# ---- common/browser.py ----
USER_AGENT_TEMPLATE = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
)
FALLBACK_CHROME_MAJOR = "140"


def _user_agent_for(browser_version: str) -> str:
    """A normal desktop-Chrome user agent whose version matches the
    Chromium actually running. (An earlier version hardcoded Chrome/124
    while Playwright ships a much newer Chromium - a UA that disagrees with
    the browser's own version hints is an easy "this is a bot" signal.)
    Headless Chromium's default UA says "HeadlessChrome", which is worse
    still, so we always set one explicitly."""
    major = (browser_version or "").split(".")[0]
    return USER_AGENT_TEMPLATE.format(major=major if major.isdigit() else FALLBACK_CHROME_MAJOR)

# Text fragments that show up on the bot-check / loading interstitials
# these three sites use. If we see one of these where we expected real
# search results, we wait and re-check rather than parsing garbage.
INTERSTITIAL_MARKERS = [
    "just a moment",
    "checking your browser",
    "we're finding great stays",
    "verifying you are human",
]


@contextmanager
def new_browser_context(headless: bool = True, locale: str = "en-US", block_media: bool = True):
    """Yields a ready-to-use Playwright `page`. Usage:

        with new_browser_context() as page:
            page.goto(url)
            ...
    """
    with sync_playwright() as p:
        # --disable-blink-features=AutomationControlled stops Chromium from
        # setting navigator.webdriver = true, the most basic "this browser
        # is automated" flag sites check. It changes nothing else about
        # how pages load or render.
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=_user_agent_for(browser.version),
            locale=locale,
            viewport={"width": 1366, "height": 900},
        )
        if block_media:
            context.route("**/*", lambda route: route.abort()
                          if route.request.resource_type in ("image", "media", "font") else route.fallback())
        page = context.new_page()
        try:
            yield page
        finally:
            context.close()
            browser.close()


def page_text(page, timeout_ms: int = 2000) -> str:
    """The page's visible text, or "" - with a time limit (page.evaluate can wait forever on a page that is
    still loading)."""
    try:
        return page.locator("body").inner_text(timeout=timeout_ms) or ""
    except Exception:
        return ""


def looks_like_interstitial(page_text: str) -> bool:
    lowered = (page_text or "").lower()
    return any(marker in lowered for marker in INTERSTITIAL_MARKERS)


def polite_goto(page, url: str, wait_selector: str | None = None,
                 timeout_ms: int = 30000, interstitial_retries: int = 3,
                 wait_state: str = "visible", wait_until: str = "domcontentloaded") -> bool:
    """Navigate to `url`, then wait for `wait_selector` to appear (the
    real content) - if we land on a bot-check interstitial instead, wait
    a couple of seconds and reload, up to `interstitial_retries` times.

    `wait_state`: "visible" (default - a card the user can see) or
    "attached" (merely present in the page, e.g. a <script> data block,
    which is never "visible"; used by the details step).

    Returns True if `wait_selector` was found, False if we never got past
    the interstitial / the selector never appeared (caller should log a
    parse_warning and move on rather than crash the whole run)."""
    for attempt in range(interstitial_retries + 1):
        # CHANGED for the live search (2026-10-19): a slow page used to raise a timeout here and stop everything.
        # Now a slow load is not an error by itself - what counts is whether the content we need appears.
        # wait_until="commit" (used for hotel pages) returns as soon as the page starts arriving.
        selector_ms = timeout_ms
        try:
            page.goto(url, timeout=timeout_ms, wait_until=wait_until)
        except Exception as exc:
            if "timeout" not in f"{type(exc).__name__} {exc}".lower():
                raise
            # timed out: with "commit" the site never answered (nothing to wait for); otherwise the content may
            # already be there, so look for it a little longer
            selector_ms = 1000 if wait_until == "commit" else 5000

        if wait_selector:
            try:
                page.wait_for_selector(wait_selector, timeout=selector_ms, state=wait_state)
                return True
            except Exception:
                body_text = page_text(page)
                if looks_like_interstitial(body_text) and attempt < interstitial_retries:
                    time.sleep(2 + attempt * 2 + random.random())
                    continue
                return False
        return True
    return False



# ---- scrapers/booking.py ----
def parse_booking_card(card_html: str) -> dict:
    """Parse ONE Booking.com property-card's inner HTML into our schema's
    HOTEL/ROOM/PRICE fields. Pure function - see tests/test_parsers.py."""
    soup = BeautifulSoup(card_html, "lxml")
    out: dict = {}
    warnings: list[str] = []

    # ---- identity -----------------------------------------------------
    title_link = soup.select_one('[data-testid="title-link"]')
    title_el = soup.select_one('[data-testid="title"]')
    out["hotel_name"] = _clean(title_el.get_text()) if title_el else None
    href = title_link.get("href") if title_link else None
    out["hotel_url"] = href
    if href:
        m = re.search(r"/hotel/[a-z]{2}/([^./?]+)", href)
        out["hotel_id_on_source"] = m.group(1) if m else None
    else:
        out["hotel_id_on_source"] = None
    if not out["hotel_name"] or not out["hotel_url"]:
        warnings.append("missing hotel name or url")

    # ---- location -------------------------------------------------------
    addr = soup.select_one('[data-testid="address-link"]')
    out["address_text_raw"] = _clean(addr.get_text()) if addr else None
    dist = soup.select_one('[data-testid="distance"]')
    out["distance_text_raw"] = _clean(dist.get_text()) if dist else None

    # ---- star rating (from the accessible aria-label, not the SVG icons)
    # The icon inside the button says WHAT kind of rating it is:
    # data-testid="rating-stars" = official star classification,
    # "rating-squares" = Booking.com's own quality estimate for properties
    # without one (the saved "The Country House Chalets" card is squares).
    # Kept in star_rating_kind_raw so the two aren't silently mixed.
    star_btn = soup.select_one('button[aria-label*="Property rating"]')
    out["star_rating_raw"] = out["star_rating_kind_raw"] = None  # many guesthouses genuinely have none
    if star_btn:
        m = re.search(r"Property rating:\s*(\d+(?:\.\d+)?)\s*out of\s*5", star_btn.get("aria-label", ""))
        out["star_rating_raw"] = m.group(1) if m else None
        icon = star_btn.select_one('[data-testid^="rating-"]')
        if out["star_rating_raw"] and icon:
            out["star_rating_kind_raw"] = icon["data-testid"].replace("rating-", "", 1)

    # ---- location sub-score - confirmed live 2026-09-24 on a Galle search
    # (9 of 15 cards): <a data-testid="secondary-review-score-link"><span>
    # <span>Location 9.8</span></span></a>. Kept as shown. -----------------
    loc = soup.select_one('[data-testid="secondary-review-score-link"]')
    out["location_score_raw"] = _clean(loc.get_text()) if loc else None
    out["review_highlights_raw"] = None  # not shown on Booking.com's search cards

    # ---- review score (from the accessible aria-label, e.g.
    # "Scored 8.2, Very Good, 579 reviews, Opens X information") --------
    review_link = soup.select_one('[data-testid="review-score-link"]')
    if review_link:
        aria = review_link.get("aria-label", "")
        m = re.search(r"Scored\s*([\d.]+),\s*([^,]+),\s*([\d,]+)\s*reviews?", aria)
        if m:
            out["review_score_raw"] = m.group(1)
            out["review_word_raw"] = m.group(2).strip()
            out["review_count_raw"] = f"{m.group(3)} reviews"
        else:
            out["review_score_raw"] = out["review_word_raw"] = out["review_count_raw"] = None
            warnings.append("review-score element present but aria-label format unrecognized")
    else:
        out["review_score_raw"] = out["review_word_raw"] = out["review_count_raw"] = None

    # ---- badges ---------------------------------------------------------
    badges = []
    preferred = soup.select_one('[data-testid="preferred-badge"]')
    if preferred:
        aria = preferred.get("aria-label", "")
        badges.append("Preferred Plus Partner" if "Preferred Plus" in aria else "Preferred Partner")
    deal_el = soup.select_one('[data-testid="property-card-deal"]')
    deal_text = None
    if deal_el:
        inner_span = deal_el.find("span")
        deal_text = _clean(inner_span.get_text()) if inner_span else _clean(deal_el.get_text())
        if deal_text:
            badges.append(deal_text)
    out["deal_badge_raw"] = deal_text
    out["badges_raw"] = "; ".join(badges) if badges else None

    # ---- room-level (the "recommended unit" Booking.com is offering) --
    units = soup.select_one('[data-testid="recommended-units"]')
    room_type, config_parts, cancellation, prepayment, units_left = None, [], None, None, None
    if units:
        h4 = units.find("h4")
        room_type = _clean(h4.get_text()) if h4 else None
        for li in units.select("ul > li"):
            li_text = _clean(li.get_text())
            if not li_text:
                continue
            if li.select_one('[data-testid="cancellation-policy-icon"]'):
                cancellation = li_text
            elif li.select_one('[data-testid="prepayment-policy-icon"]'):
                prepayment = li_text
            elif "left at this price" in li_text.lower():
                units_left = li_text
            else:
                config_parts.append(li_text)
    else:
        warnings.append("no recommended-units block (likely sold out for these dates)")
    out["room_type_name_raw"] = room_type
    out["room_config_raw"] = " | ".join(config_parts) if config_parts else None
    out["cancellation_policy_raw"] = cancellation
    out["prepayment_policy_raw"] = prepayment
    out["units_left_raw"] = units_left

    # ---- price (verbatim strings - no numeric parsing here) ------------
    price_el = soup.select_one('[data-testid="price-and-discounted-price"]')
    out["displayed_price_raw"] = _clean(price_el.get_text()) if price_el else None
    taxes_el = soup.select_one('[data-testid="taxes-and-charges"]')
    out["taxes_and_fees_raw"] = _clean(taxes_el.get_text()) if taxes_el else None

    # The strike-through "was" price has no stable data-testid, but
    # Booking.com always exposes it as hidden, screen-reader text like
    # "Original price LKR 16,550. Current price LKR 13,240." - that
    # text is far more stable than the obfuscated CSS classes nearby.
    original_price = None
    rate_info = soup.select_one('[data-testid="availability-rate-information"]')
    if rate_info:
        for el in rate_info.find_all(True):
            txt = el.get_text()
            if txt and "Original price" in txt and "Current price" in txt:
                m = re.search(r"Original price\s*([^.]+)\.\s*Current price", txt)
                if m:
                    original_price = _clean(m.group(1))
                break
    out["original_price_raw"] = original_price

    if not out["displayed_price_raw"]:
        warnings.append("no price found (sold out for these dates/occupancy)")

    out["parse_warnings"] = "; ".join(warnings)
    return out



# ---- scrapers/booking_details.py ----
SUBSCORE_SELECTOR = '[data-testid="review-subscore"]'
FACILITY_GROUP_SELECTOR = '[data-testid="facility-group-container"]'
FACILITIES_BLOCK_SELECTOR = '[data-testid="property-facilities-block-container"]'
# The whole facilities SECTION. Confirmed live 2026-09-26 on Marino Beach
# Colombo: the block above is only a zero-height placeholder; the groups are
# rendered beside it, inside this section.
FACILITIES_SECTION_SELECTOR = "#hp_facilities_box"
POPULAR_FACILITIES_SELECTOR = '[data-testid="property-most-popular-facilities-wrapper"]'
HIGHLIGHTS_SELECTOR = '[data-testid="property-highlights"]'
LATLNG_SELECTOR = "[data-atlas-latlng]"
STRUCTURED_DATA_SELECTOR = 'script[type="application/ld+json"]'
# The page's own data store (JSON), always sent with the page - no scrolling
# needed. Confirmed live 2026-09-26 (Le Grand Galle, Marino Beach).
PAGE_DATA_SELECTOR = 'script[data-capla-store-data="apollo"]'

# "Value for money, 8.5, Average rating out of 10" -> ("Value for money", "8.5")
_SUBSCORE_PATTERN = re.compile(r"^\s*(.+?),\s*(\d+(?:\.\d+)?)\s*,\s*Average rating", re.IGNORECASE)
KNOWN_SUBSCORES = {
    "staff": "subscore_staff_raw",
    "facilities": "subscore_facilities_raw",
    "cleanliness": "subscore_cleanliness_raw",
    "comfort": "subscore_comfort_raw",
    "value for money": "subscore_value_for_money_raw",
    "location": "subscore_location_raw",
    "free wifi": "subscore_free_wifi_raw",
}
# A warning with this text means the page itself failed - such hotels are
# retried on the next run (see hotel_details.py); anything else is final.
FATAL_WARNING = "page did not load"



def _structured_data(soup) -> dict | None:
    """The page's lodging entry in its structured-data block(s): the one
    that has an address or a rating (other blocks, e.g. breadcrumbs, can
    exist alongside it)."""
    for script in soup.select(STRUCTURED_DATA_SELECTOR):
        try:
            data = json.loads(script.string or script.get_text() or "")
        except (ValueError, TypeError):
            continue
        for obj in data if isinstance(data, list) else [data]:
            if isinstance(obj, dict) and ("address" in obj or "aggregateRating" in obj):
                return obj
    return None


def parse_booking_hotel_page(html: str) -> dict:
    """One Booking.com hotel page -> the parsed DETAILS_FIELDS (minus the
    bookkeeping ones fetch() adds). Pure function - see
    tests/test_booking_details.py."""
    soup = BeautifulSoup(html, "lxml")
    out: dict = {}
    warnings: list[str] = []

    # ---- structured data: type, address, overall rating, description -----
    ld = _structured_data(soup)
    addr = (ld or {}).get("address") or {}
    rating = (ld or {}).get("aggregateRating") or {}
    kind = (ld or {}).get("@type")
    out["property_type_raw"] = ", ".join(kind) if isinstance(kind, list) else _clean(kind)
    out["street_address_raw"] = _clean(addr.get("streetAddress")) if isinstance(addr, dict) else None
    out["address_region_raw"] = _clean(addr.get("addressRegion")) if isinstance(addr, dict) else None
    out["postal_code_raw"] = _clean(addr.get("postalCode")) if isinstance(addr, dict) else None
    out["overall_rating_raw"] = _clean(rating.get("ratingValue")) if isinstance(rating, dict) else None
    out["overall_review_count_raw"] = _clean(rating.get("reviewCount")) if isinstance(rating, dict) else None
    out["description_raw"] = _clean((ld or {}).get("description"))
    if ld is None:
        warnings.append("no structured data block")

    # ---- exact coordinates -----------------------------------------------
    out["latitude_raw"] = out["longitude_raw"] = None
    latlng = soup.select_one(LATLNG_SELECTOR)
    parts = (latlng.get("data-atlas-latlng") or "").split(",") if latlng else []
    if len(parts) >= 2:
        out["latitude_raw"], out["longitude_raw"] = _clean(parts[0]), _clean(parts[1])
    else:
        warnings.append("no coordinates")

    # ---- guest sub-scores (none at all when there are no reviews yet) -----
    for col in KNOWN_SUBSCORES.values():
        out[col] = None
    others = []
    for el in soup.select(SUBSCORE_SELECTOR):
        first = el.find("span")
        m = _SUBSCORE_PATTERN.match(_clean(first.get_text()) if first else "") or \
            _SUBSCORE_PATTERN.match(_clean(el.get_text(" ")) or "")
        if not m:
            continue
        name, score = _clean(m.group(1)), m.group(2)
        col = KNOWN_SUBSCORES.get(name.lower())
        if col:
            out[col] = score
        else:
            others.append(f"{name}: {score}")
    out["subscores_other_raw"] = "; ".join(others) or None

    # ---- top amenities / highlights (the page shows the top list twice) ---
    popular = []
    for li in soup.select(f"{POPULAR_FACILITIES_SELECTOR} li"):
        text = _clean(li.get_text(" "))
        if text and text not in popular:
            popular.append(text)
    out["popular_facilities_raw"] = "; ".join(popular) or None
    highlights = [t for t in (_clean(li.get_text(" ")) for li in soup.select(f"{HIGHLIGHTS_SELECTOR} li")) if t]
    out["property_highlights_raw"] = "; ".join(highlights) or None

    # ---- full amenities list, grouped, with each item's notes -------------
    groups = []
    for group in soup.select(FACILITY_GROUP_SELECTOR):
        heading = group.find("h3")
        head = list(heading.stripped_strings) if heading else []
        if not head:
            continue
        label = head[0] + (f" [{' '.join(head[1:])}]" if len(head) > 1 else "")
        items = []
        for li in group.select("li"):
            bits = list(li.stripped_strings)
            if bits:
                items.append(bits[0] + (f" ({', '.join(bits[1:])})" if len(bits) > 1 else ""))
        groups.append(label + (": " + "; ".join(items) if items else ""))
    out["facilities_raw"] = " || ".join(groups) or None
    if not groups:
        warnings.append("full facilities list not found")

    # ---- the page's own data store: room amenities + listed facilities ---
    store = _page_data_store(soup)
    out["room_amenities_raw"], out["listed_facilities_raw"] = _store_amenities(store)
    if store is None:
        warnings.append("no page data store")

    out["details_warnings"] = "; ".join(warnings)
    return out


def _page_data_store(soup) -> dict | None:
    script = soup.select_one(PAGE_DATA_SELECTOR)
    if script is None:
        return None
    try:
        data = json.loads(script.string or script.get_text() or "")
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _facility_title(store: dict, ref) -> str | None:
    """A facility reference -> its title(s), e.g. "Sea view", or
    "Airport shuttle / Airport pickup / Airport drop-off" for one facility
    listed with several variants (repeats dropped)."""
    fac = store.get(ref.get("__ref")) if isinstance(ref, dict) and "__ref" in ref else ref
    if not isinstance(fac, dict):
        return None
    titles = []
    for inst in fac.get("instances") or []:
        inst = store.get(inst.get("__ref")) if isinstance(inst, dict) and "__ref" in inst else inst
        title = _clean((inst or {}).get("title")) if isinstance(inst, dict) else None
        if title and title not in titles:
            titles.append(title)
    return " / ".join(titles) or _clean(fac.get("title") or fac.get("name"))


def _store_amenities(store: dict | None) -> tuple[str | None, str | None]:
    """(room_amenities_raw, listed_facilities_raw) from the page's data store.
    Rooms: every amenity of every room type, first-seen order, no repeats.
    Listed: the property's facilities(...) and commonAmenities(...) lists
    (not its top-10 "highlights" - that's popular_facilities_raw)."""
    if not store:
        return None, None
    room, listed = [], []
    for key, value in store.items():
        if key.startswith("RoomData:") and isinstance(value, dict):
            for ref in value.get("amenities") or []:
                title = _facility_title(store, ref)
                if title and title not in room:
                    room.append(title)
    for key, value in store.items():
        if key.startswith("Property:") and isinstance(value, dict):
            for field, items in value.items():
                if (field.startswith("facilities(") or field.startswith("commonAmenities(")) and isinstance(items, list):
                    for ref in items:
                        title = _facility_title(store, ref)
                        if title and title not in listed:
                            listed.append(title)
    return "; ".join(room) or None, "; ".join(listed) or None



# ---- scrapers/booking_rooms.py ----
ROOM_TABLE_SELECTOR = "#hprt-table"
DESCRIPTION_SELECTOR = '[data-testid="property-description"]'
CHAIN_LABEL = "Hotel chain/brand"
CHEAPEST_CLASS = "hprt-table-cheapest-block"
HIDDEN_CONDITION_CLASS = "hprt-table-hidden-condition"

FATAL_WARNING = "page did not load"
NO_ROOMS_WARNING = "no rooms free on these dates"

_MEAL = re.compile(r"breakfast|dinner|lunch|all[- ]inclusive|half board|full board|meals", re.IGNORECASE)
_UNITS_LEFT = re.compile(r"(?:We have|Only)\s+\d+\s+(?:rooms?\s+)?left[^.;|]*", re.IGNORECASE)
_ORIGINAL = re.compile(r"Original price\s*(.+?)\s*Current price", re.IGNORECASE)



def _chain(soup) -> str | None:
    """"Jetwing Hotels Limited" from  <div><div>Jetwing Hotels Limited</div>
    <div>Hotel chain/brand</div></div>  (the name is the line before the label)."""
    label = soup.find(string=lambda s: s is not None and s.strip() == CHAIN_LABEL)
    if label is None:
        return None
    el = label.parent
    before = [_clean(sib.get_text(" ")) for sib in reversed(list(el.find_previous_siblings()))]
    name = " ".join(t for t in before if t)
    if not name and el.parent is not None:
        name = (_clean(el.parent.get_text(" ")) or "").replace(CHAIN_LABEL, "")
    return _clean(name)[:120] if _clean(name) else None


def _beds(th) -> str | None:
    """Each <ul class="rt-bed-types"> is one choice; items inside are joined
    with "and": "2 twin beds and 1 queen bed or 4 twin beds"."""
    choices = []
    for ul in th.select("ul.rt-bed-types"):
        items = []
        for li in ul.select("li.rt-bed-type"):
            text = _clean(li.get_text(" "))
            if text:
                items.append(re.sub(r"\s+and$", "", text))
        if items:
            choices.append(" and ".join(items))
    return " or ".join(choices) or None


def _room(th) -> tuple[str | None, dict]:
    link = th.select_one("a.hprt-roomtype-link[data-room-id]")
    room_id = link.get("data-room-id") if link else None
    size_el = th.select_one('[data-name-en="room size"]')
    sleeps_el = th.select_one(".hprt-roomtype-occupancy-text")
    highlights = []
    for f in th.select(".hprt-facilities-facility"):
        if f.get("data-name-en") == "room size":
            continue
        text = _clean(f.get_text(" "))
        if text and text not in highlights:
            highlights.append(text)
    return room_id, {
        "room_id_raw": room_id,
        "room_name_raw": _clean(link.get_text(" ")) if link else None,
        "room_size_raw": _clean(size_el.get_text(" ")) if size_el else None,
        "room_beds_raw": _beds(th),
        "room_sleeps_raw": _clean(sleeps_el.get_text(" ")) if sleeps_el else None,
        "room_highlights_raw": "; ".join(highlights) or None,
    }


def _rate(tr) -> dict:
    price_cell = tr.select_one(".hprt-table-cell-price") or tr
    price_el = price_cell.select_one(".prco-valign-middle-helper")
    taxes_el = price_cell.select_one(".prd-taxes-and-fees-under-price")
    m = None
    for sr in price_cell.select(".bui-u-sr-only"):      # screen-reader lines; one may carry the old price
        m = _ORIGINAL.search(_clean(sr.get_text(" ")) or "")
        if m:
            break
    deals = []
    for box in price_cell.select('[id^="c-deals-container"]'):
        leaves = [d for d in box.find_all("div") if d.find("div") is None and _clean(d.get_text(" "))]
        title = _clean(leaves[0].get_text(" ")) if leaves else None
        if title and title not in deals:
            deals.append(title)

    meal = cancel = prepay = None
    others = []
    for li in tr.select("ul.hprt-conditions-bui > li"):
        if HIDDEN_CONDITION_CLASS in (li.get("class") or []):
            continue
        text = _clean(li.get_text(" "))
        if not text:
            continue
        tid = li.get("data-testid") or ""
        if tid == "cancellation-subtitle":
            cancel = text
        elif tid == "prepayment-subtitle":
            prepay = _clean(text.lstrip("•"))
        elif meal is None and (tid.startswith("rt-rate-") or _MEAL.search(text)):
            meal = text
        else:
            others.append(text)
    units = _UNITS_LEFT.search(_clean(tr.get_text(" ")) or "")
    return {
        "rate_block_id_raw": tr.get("data-block-id"),
        "is_cheapest_rate": "yes" if CHEAPEST_CLASS in (tr.get("class") or []) else "",
        "displayed_price_raw": _clean(price_el.get_text(" ")) if price_el else None,
        "original_price_raw": _clean(m.group(1)) if m else None,
        "taxes_and_fees_raw": _clean(taxes_el.get_text(" ")) if taxes_el else None,
        "deal_badges_raw": "; ".join(deals) or None,
        "meal_plan_raw": meal,
        "cancellation_policy_raw": cancel,
        "prepayment_policy_raw": prepay,
        "other_conditions_raw": "; ".join(others) or None,
        "units_left_raw": _clean(units.group(0)) if units else None,
    }


def parse_booking_rooms_page(html: str, nights: int | None = None) -> tuple[dict, list[dict]]:
    """One dated Booking.com hotel page -> (page facts, one dict per rate).
    Page facts: {"hotel_chain_raw", "rooms_warnings"}. Each rate dict has the
    room and rate columns of ROOM_RATE_FIELDS. Pure function - see
    tests/test_booking_rooms.py."""
    soup = BeautifulSoup(html, "lxml")
    warnings: list[str] = []
    page = {"hotel_chain_raw": _chain(soup)}
    table = soup.select_one(ROOM_TABLE_SELECTOR)
    if table is None:
        warnings.append(NO_ROOMS_WARNING if soup.select_one(DESCRIPTION_SELECTOR) else FATAL_WARNING)
        page["rooms_warnings"] = "; ".join(warnings)
        return page, []

    head = _clean(table.select_one("thead").get_text(" ")) if table.select_one("thead") else ""
    if nights and f"Price for {nights} nights" not in (head or ""):
        warnings.append(f"price column is not 'Price for {nights} nights' (header: {head[:80]})")

    rooms: dict[str, dict] = {}
    rates: list[dict] = []
    for tr in table.select("tbody tr[data-block-id]"):
        th = tr.select_one("th.hprt-table-cell-roomtype")
        if th is not None:
            rid, info = _room(th)
            if rid:
                rooms[rid] = info
        rid = (tr.get("data-block-id") or "").split("_")[0]
        rate = dict(rooms.get(rid) or {"room_id_raw": rid or None})
        rate.update(_rate(tr))
        rates.append(rate)
    if not rates:
        warnings.append(NO_ROOMS_WARNING)
    elif not any(r["is_cheapest_rate"] for r in rates):
        warnings.append("no rate marked as the cheapest")
    page["rooms_warnings"] = "; ".join(warnings)
    return page, rates


def rooms_page_url(hotel_url: str, checkin: str, checkout: str, adults: int = 2, children: int = 0,
                   rooms: int = 1, currency: str = "LKR") -> str:
    """The plain hotel page (https://www.booking.com/hotel/lk/<id>.html) with
    the stay, guests, currency and English set in the address."""
    return (f"{hotel_url}?checkin={checkin}&checkout={checkout}&group_adults={adults}"
            f"&group_children={children}&no_rooms={rooms}&selected_currency={currency}&lang=en-us")


# ---------------------------------------------------------------------------------------------------------
# new for the live search
# ---------------------------------------------------------------------------------------------------------
SEARCH_URL = "https://www.booking.com/searchresults.html"
CARD_SELECTOR = '[data-testid="property-card"]'
HOTELS_ONLY = "ht_id=204"            # Booking.com's "Hotels" property type (same as the research data)
# Booking.com filter codes, read off a live Tangalle search page on 2026-10-19
MUST_FILTERS = {
    "pool": "hotelfacility=433", "parking": "hotelfacility=2", "spa": "hotelfacility=54",
    "wifi": "hotelfacility=107", "ac": "roomfacility=11", "beach": "ht_beach=1", "breakfast": "mealplan=1",
    "wheelchair_access": "accessible_room_facilities=134", "low_steps": "accessible_room_facilities=131",
}
CHILD_AGES = [8, 10]                 # same fixed child ages as the scraper (config/settings.py)


def child_ages_for(children: int) -> list[int]:
    return [CHILD_AGES[i % len(CHILD_AGES)] for i in range(children)]


def search_url(search_text: str, checkin: str, checkout: str, adults: int, children: int, rooms: int = 1,
               max_price_per_night: float | None = None, must: list[str] = ()) -> str:
    """Booking.com search with the tourist's dates, guests, budget and must-haves as filters."""
    from urllib.parse import quote, urlencode
    params = [("ss", search_text), ("checkin", checkin), ("checkout", checkout), ("group_adults", adults),
              ("group_children", children), ("no_rooms", rooms), ("selected_currency", "LKR"), ("lang", "en-us")]
    params += [("age", a) for a in child_ages_for(children)]
    filters = [HOTELS_ONLY]
    if max_price_per_night:
        filters.append(f"price=LKR-min-{int(max_price_per_night)}-1")
    filters += [MUST_FILTERS[m] for m in must if m in MUST_FILTERS]
    return f"{SEARCH_URL}?{urlencode(params)}&nflt={quote(';'.join(filters), safe='')}"


def hotel_id_from_url(url: str | None) -> str | None:
    m = re.search(r"/hotel/[a-z]{2}/([^./?]+)", url or "")
    return m.group(1) if m else None


_PHOTO = re.compile(r"https://cf\.bstatic\.com/xdata/images/hotel/([a-z0-9x]+)/(\d+)\.(jpg|webp)\?k=([0-9a-f]+)(?:&amp;|&)o=")


def _photo(pid: str, ext: str, k: str, size: str) -> str:
    # the size part of the address can be changed (checked live: square240 -> max500 / max1024x768 work)
    return f"https://cf.bstatic.com/xdata/images/hotel/{size}/{pid}.{ext}?k={k}&o="


def extract_photos(html: str, limit: int = 10) -> list[dict]:
    """The hotel's photos from its Booking.com page, in page order (the gallery comes first)."""
    seen, out = set(), []
    for m in _PHOTO.finditer(html or ""):
        pid, ext, k = m.group(2), m.group(3), m.group(4)
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"photo_url": _photo(pid, ext, k, "max1024x768"), "thumb_url": _photo(pid, ext, k, "max500")})
        if len(out) >= limit:
            break
    return out


def photo_urls_from_card(card_html: str) -> list[dict]:
    """The one photo on a search-result card."""
    return extract_photos(card_html, limit=1)


def load_facilities(page, wait_ms: int = 5000) -> bool:
    """The full facility list loads only once it is scrolled into view (same idea as the scraper's
    BookingDetailsScraper._load_facilities, shortened: one scroll to the section, then one walk down the page)."""
    scroll_to = """(sels) => { for (const s of sels) { const el = document.querySelector(s);
                                    if (el) { el.scrollIntoView({block: 'center'}); return s; } } return null; }"""
    targets = [FACILITIES_SECTION_SELECTOR, FACILITIES_BLOCK_SELECTOR]
    for attempt in range(2):
        try:
            if attempt:
                page.evaluate("window.scrollTo(0, 0)")
                for _ in range(25):
                    page.mouse.wheel(0, 900)
                    page.wait_for_timeout(200)
            page.evaluate(scroll_to, targets)
            page.wait_for_selector(FACILITY_GROUP_SELECTOR, state="attached", timeout=wait_ms)
            return True
        except Exception:
            continue
    return False
