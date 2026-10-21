"""
Component 1 (trusted review scores) for each hotel, in the format of contracts/c1_aspect_scores.json.

Three ways to get them (environment variable C1_MODE):
    dummy  (default)  MADE-UP scores, until Component 1 is ready. Every answer says source = "DUMMY".
    file              Component 1's scores saved in a JSON file (C1_FILE). Easiest first step with your friend:
                      a list of objects in the contract format, or {hotel_id: object}.
    http              ask Component 1's service: GET {C1_URL}/hotels/{hotel_id}/aspect-scores

To switch to the real Component 1 later, nothing else in the code changes:
    $env:C1_MODE="file"; $env:C1_FILE="D:\\Research\\c1_aspect_scores.json"      (PowerShell)
    or  $env:C1_MODE="http"; $env:C1_URL="http://localhost:8001"

Hotel ids: Component 2 uses the Booking.com id (booking_hotel_id, e.g. "cinnamon-red-colombo").
Component 1 must use the same id, or give a mapping.

Export the dummy file (to show your friend the exact format):
    python -m src.c2.c1_client --export dummy_c1_aspect_scores.json
"""
import hashlib
import json
import math
import random
import urllib.error
import urllib.request

from . import settings

DUMMY_LABEL = "DUMMY - made-up values until Component 1 is ready"
ASPECTS = ["food", "staff", "room", "cleanliness", "location", "wifi"]
HILL_TOWNS = {"Ella", "Nuwara Eliya", "Kandy"}


def _num(x, default=None):
    try:
        x = float(x)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default


def _clip(x, lo, hi):
    return round(max(lo, min(hi, x)), 1)


class DummyC1:
    """Made-up but believable scores, the same every time for the same hotel.

    The base comes from the hotel's Booking.com sub-scores (0-10 -> 0-5) with a little noise. A few hotels get a
    'reviews disagree' fact (pool closed, WiFi not working, view blocked, lift broken, noisy) so the truth check
    (Gap 2) can be shown before Component 1 exists. These are NOT real reviews.
    """
    source = DUMMY_LABEL

    def __init__(self, hotels):
        self.hotels = hotels

    def get(self, hotel_id):
        if hotel_id not in self.hotels.index:
            return None
        h = self.hotels.loc[hotel_id]
        rnd = random.Random(int(hashlib.md5(str(hotel_id).encode()).hexdigest()[:8], 16))
        noise = lambda: rnd.gauss(0, 0.25)
        rating = _num(h.get("page_rating"), 7.5)

        def sub(col):
            return _num(h.get(col), rating) / 2

        food_level = _num(h.get("breakfast_rating_level"), 0)
        scores = {
            "food": _clip((food_level if food_level > 0 else rating / 2 - 0.2) + noise(), 0, 5),
            "staff": _clip(sub("subscore_staff") + noise(), 0, 5),
            "room": _clip(sub("subscore_comfort") + noise(), 0, 5),
            "cleanliness": _clip(sub("subscore_cleanliness") + noise(), 0, 5),
            "location": _clip(sub("subscore_location") + noise(), 0, 5),
            "wifi": _clip(sub("subscore_free_wifi") + noise(), 0, 5),
        }
        facts = []
        has = lambda col: _num(h.get(col), 0) == 1
        if has("has_pool"):
            r = rnd.random()
            if r < 0.10:
                facts.append("Reviews from September 2026 say the pool was closed for repairs.")
            elif r < 0.50:
                facts.append("Guests say the pool is clean and well kept.")
        if has("has_free_wifi"):
            r = rnd.random()
            if r < 0.12:
                facts.append("Several guests say the WiFi did not work in the rooms.")
                scores["wifi"] = min(scores["wifi"], 2.0)
            elif r < 0.40:
                facts.append("WiFi was fast in the rooms.")
        if has("rooms_sea_view") and rnd.random() < 0.15:
            facts.append("Some guests say the sea view is blocked by new buildings.")
        town_km = _num(h.get("km_from_town_centre"), 1.0)
        r = rnd.random()
        if town_km < 1.0 and r < 0.20:
            facts.append("Guests mention traffic noise at night.")
        elif town_km >= 2.0 and r < 0.30:
            facts.append("Guests say the hotel is quiet and peaceful.")
        if has("fac_elevator") and rnd.random() < 0.15:
            facts.append("Guests say the lift was not working and they had to use the stairs.")
        elif not has("fac_elevator") and h.get("destination_cluster") in HILL_TOWNS and rnd.random() < 0.25:
            facts.append("Guests mention many steps to reach the rooms.")
        if _num(h.get("offers_breakfast_rate"), 0) == 1 and rnd.random() < 0.35:
            facts.append("Breakfast was fresh and served until 10 am.")
        if has("rooms_air_conditioning") and rnd.random() < 0.08:
            facts.append("A few guests say the air conditioning was weak.")

        trust = _clip(rating - rnd.uniform(0, 1.0), 0, 10)
        return {
            "hotel_id": str(hotel_id),
            "aspect_scores": scores,
            "overall_trust_score": trust,
            "time_decay_rating": _clip(trust / 2 + rnd.gauss(0, 0.2), 1, 5),
            "verified_aspect_facts": facts,
            "last_updated": "2026-10-15T00:00:00Z",
            "source": DUMMY_LABEL,
        }


class FileC1:
    """Component 1's scores from a JSON file (a list in the contract format, or {hotel_id: object})."""

    def __init__(self, path):
        with open(path, encoding="utf-8-sig") as f:
            text = f.read().strip()
        if text.startswith("["):
            items = json.loads(text)
        elif text.startswith("{") and "\n{" not in text:
            obj = json.loads(text)
            items = list(obj.values()) if "hotel_id" not in obj else [obj]
        else:                                   # one JSON object per line
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.by_id = {str(x["hotel_id"]): x for x in items}
        self.source = f"Component 1 file ({path})"

    def get(self, hotel_id):
        return self.by_id.get(str(hotel_id))


class HttpC1:
    """Asks Component 1's service. If it does not answer, the hotel is ranked without C1 (Booking.com rating)."""

    def __init__(self, url, timeout=3):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.source = f"Component 1 service ({self.url})"
        self.cache = {}

    def get(self, hotel_id):
        if hotel_id in self.cache:
            return self.cache[hotel_id]
        try:
            with urllib.request.urlopen(f"{self.url}/hotels/{hotel_id}/aspect-scores", timeout=self.timeout) as r:
                result = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            result = None
        self.cache[hotel_id] = result
        return result


def make_c1(hotels):
    mode = settings.c1_mode()
    if mode == "file" and settings.c1_file():
        return FileC1(settings.c1_file())
    if mode == "http":
        return HttpC1(settings.c1_url())
    return DummyC1(hotels)


if __name__ == "__main__":
    import argparse

    from .data import HotelData

    ap = argparse.ArgumentParser(description="Export the DUMMY Component 1 scores for all hotels.")
    ap.add_argument("--export", default="dummy_c1_aspect_scores.json")
    args = ap.parse_args()
    data = HotelData(load_model=False)
    dummy = DummyC1(data.hotels)
    items = [dummy.get(h) for h in data.hotels.index]
    with open(args.export, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1, ensure_ascii=False)
    print(f"{len(items)} hotels written to {args.export} (DUMMY values)")
