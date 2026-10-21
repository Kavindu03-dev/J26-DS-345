"""Live Booking.com search, tested offline: the 'browser' returns real Booking.com pages saved by the scraper
(tests/fixtures, copied from ota_scraper_project/tests/fixtures)."""
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.c2.data import HotelData, get_data, set_data
from src.c2.live import booking_pages as bp
from src.c2.live import search as S
from src.c2.ranker import recommend
from tests.helpers import make_files, make_request

FIX = Path(__file__).with_name("fixtures")
CARDS = [c.decode_contents() for c in BeautifulSoup((FIX / "booking_sample_cards.html").read_text(encoding="utf-8"),
                                                     "lxml").select(bp.CARD_SELECTOR)]
PAGE = ((FIX / "booking_hotel_page.html").read_text(encoding="utf-8") +
        (FIX / "booking_rooms_jetwing_jaffna.html").read_text(encoding="utf-8") +
        "".join(f'<img src="https://cf.bstatic.com/xdata/images/hotel/max500/{100 + i}.jpg?k=ab{i}&amp;o=">'
                for i in range(3)))


class FakePages:
    """Stands in for the Playwright browser."""

    def __init__(self, cards=CARDS, blocked=False):
        self.cards, self.blocked, self.opened = cards, blocked, []

    def search(self, url):
        self.opened.append(url)
        return None if self.blocked else self.cards

    def hotel(self, url, facilities=True):
        self.opened.append(url)
        return PAGE

    def human_check(self):
        return self.blocked

    def pause(self):
        pass


def test_page_readers_and_photos():
    card = bp.parse_booking_card(CARDS[2])
    assert card["hotel_id_on_source"] == "sholay-villa" and card["displayed_price_raw"] == "LKR 13,240"
    photos = bp.extract_photos(PAGE)
    assert len(photos) == 3 and "max1024x768/100.jpg?k=ab0&o=" in photos[0]["photo_url"]
    url = bp.search_url("Ella, Sri Lanka", "2026-12-10", "2026-12-12", 2, 1, 1, 15000, ["pool", "near_hospital"])
    assert "price%3DLKR-min-15000-1" in url and "hotelfacility%3D433" in url and "age=8" in url


def test_live_search_saves_hotels_and_they_come_from_the_csv_next_time(tmp_path):
    old = get_data()
    try:
        data = HotelData(*make_files(tmp_path))
        set_data(data)
        req = make_request(town="Kandy", ci="2027-01-20", nights=2, bmax=20000, must=["pool"])
        pages = FakePages()
        report = S.run_live_search(req, "Kandy", today="2026-10-19", pages=pages, data=data)
        assert report["status"] == "ok"
        assert len(report["new_hotels"]) == 3                       # the whole chalet card is skipped
        assert any("whole" in s for s in report["skipped"])
        assert "hotelfacility%3D433" in pages.opened[0]             # the pool must-have became Booking's filter
        live_dir = data.live_dir
        for f in ("hotels_booking_live.csv", "prices_live.csv", "hotel_words_facilities_live.csv",
                  "hotel_photos.csv", "hotel_urls_live.csv", "live_search_log.csv"):
            assert (live_dir / f).exists(), f
        log = pd.read_csv(live_dir / "live_search_log.csv")
        assert "pool" not in " ".join(map(str, log.iloc[0].values))  # the log keeps no request details
        fresh = get_data()                                          # reloaded with the new hotels
        assert "sholay-villa" in fresh.hotels.index and fresh.is_live("sholay-villa")
        assert len(fresh.photos["sholay-villa"]) == 3
        ranked = recommend(req, top_k=10, today="2026-10-19", data=fresh)["results"][0]["ranked_hotels"]
        live = [h for h in ranked if h["found_live"]]
        assert live and all(h["price"]["seen_checkin"] == "2027-01-20" for h in live)
        assert all(h["price"]["price_per_night_lkr"] <= 20000 for h in ranked)
        again = S.run_live_search(req, "Kandy", today="2026-10-19", pages=FakePages(), data=fresh)
        assert again["new_hotels"] == [] and again["already_in_our_data"] == 3     # not scraped twice
    finally:
        set_data(old)


def test_live_search_stops_at_a_human_check(tmp_path):
    data = HotelData(*make_files(tmp_path))
    report = S.run_live_search(make_request(town="Kandy"), "Kandy", today="2026-10-19",
                               pages=FakePages(blocked=True), data=data)
    assert report["status"] == "blocked" and "never" in report["message"]
    assert not (data.live_dir / "hotels_booking_live.csv").exists()


def test_ranker_says_when_live_search_is_needed(data):
    few = recommend(make_request(must=["pool", "near_hospital"]), top_k=8, data=data)["results"][0]
    assert few["live_search"]["needed"] and "must-have" in few["live_search"]["reason"]
    enough = recommend(make_request(), top_k=2, data=data)["results"][0]
    assert not enough["live_search"]["needed"]


def test_switch_off_live_search(data, monkeypatch):
    monkeypatch.setenv("C2_LIVE", "0")
    r = recommend(make_request(must=["pool", "near_hospital"]), top_k=8, data=data)["results"][0]
    assert not r["live_search"]["needed"]


class SlowPages(FakePages):
    """One hotel page times out (like palm-paradise-cabanas-villas on 2026-10-19)."""

    def hotel(self, url, facilities=True):
        if "sholay" in url:
            raise TimeoutError("Page.goto: Timeout 30000ms exceeded.")
        return super().hotel(url, facilities)


def test_a_slow_hotel_page_is_skipped_not_fatal(tmp_path):
    old = get_data()
    try:
        data = HotelData(*make_files(tmp_path))
        set_data(data)
        report = S.run_live_search(make_request(town="Kandy"), "Kandy", today="2026-10-19", pages=SlowPages(),
                                   data=data)
        assert report["status"] == "ok"
        assert [h["hotel_id"] for h in report["new_hotels"]] == ["smile-beach-hostel-amp-cabanas", "mednest-hostel"]
        assert any("did not load in time" in s for s in report["skipped"])
    finally:
        set_data(old)
