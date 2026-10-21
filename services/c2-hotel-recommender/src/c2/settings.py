"""
All settings for Component 2 in one place.

Paths and switches come from environment variables, so nothing personal is written in the code:
    C2_DATA_DIR   folder with output_clean/ and models/   (the hotel_preprocessing folder)
    C2_EXTRA_DIR  folder with sl_hospitals_osm.csv, sl_places_osm.csv, hotel_urls.csv
    C1_MODE       dummy | file | http   (dummy = made-up review scores until Component 1 is ready)
    C1_FILE       JSON file with Component 1's scores (when C1_MODE=file)
    C1_URL        Component 1's address (when C1_MODE=http)
    OLLAMA_URL    where Ollama runs (for /recommendations/from-text)
    NLP_MODEL     the fine-tuned model's name in Ollama
"""
import os
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[2]          # services/c2-hotel-recommender
REPO_DIR = SERVICE_DIR.parents[1]                          # J26-DS-345


def _pick(env_name, *candidates):
    """The environment variable if set, otherwise the first folder that exists."""
    value = os.getenv(env_name)
    if value:
        return Path(value)
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


def data_dir():
    return _pick("C2_DATA_DIR", REPO_DIR / "data" / "c2" / "hotel_preprocessing",
                 r"D:\Research\hotel_preprocessing\hotel_preprocessing")


def extra_dir():
    return _pick("C2_EXTRA_DIR", REPO_DIR / "data" / "c2", r"D:\Research")


def c1_mode():
    return os.getenv("C1_MODE", "dummy").strip().lower()


def c1_file():
    return os.getenv("C1_FILE", "")


def c1_url():
    return os.getenv("C1_URL", "http://localhost:8001")


def ollama_url():
    return os.getenv("OLLAMA_URL", "http://localhost:11434")


def nlp_model():
    return os.getenv("NLP_MODEL", "hotel-nlp-v2")


def live_enabled():
    """Live Booking.com search allowed? (C2_LIVE=0 switches it off, e.g. with no internet)"""
    return os.getenv("C2_LIVE", "1").strip() not in ("0", "false", "no", "off")


def live_headless():
    """C2_LIVE_SHOW_BROWSER=1 opens a visible browser window, to watch the live search."""
    return os.getenv("C2_LIVE_SHOW_BROWSER", "0").strip() in ("0", "false", "no", "off", "")


# ---------------- rules and weights (change here, not in the code) ----------------
NEAR_HOSPITAL_KM = 5.0          # "near a hospital" = nearest hospital within this distance (straight line)
QUIET_KM_FROM_CENTRE = 2.0      # a hotel this far from the town centre counts as quiet (if no soundproofing listed)
OVERPRICED_PCT = 30             # price more than 30% above the fair price -> "Potentially Overpriced"
                                # (the price model's typical error is 26%, so smaller gaps are not shown)
WHAT_IF_MIN_RS, WHAT_IF_MIN_PCT = 1000, 5   # show "Rs X is for the pool" only above both (as in what_if_explanations.ipynb)
# The only facilities whose share of the price is shown. personal_fair_price.ipynb (section 4) tested all 14:
# only these 4 add money at >= 8 in 10 hotels AND are big enough to show for >= 1 in 10 prices.
PRICED_FACILITIES = ("pool", "bar", "breakfast", "beachfront")
FEW_REVIEWS = 5                 # fewer Booking.com reviews than this -> "this fair price is less certain"
SHAP_BACKGROUND = 20            # SHAP starts from 20 typical prices (one from the middle of each 5% price slice)
SHAP_CACHE_SIZE = 300           # explanations kept in memory (each takes about 5-20 seconds to make)
SEARCH_RADIUS_KM = (10, 25)     # towns that are not one of our 15 data towns: hotels within 10 km, else 25 km

IMPORTANCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}
SCORE_WEIGHTS = {               # how the final score is made (each part is 0..1)
    "wishes": 0.40,             # how many of the tourist's wishes the hotel really has
    "quality": 0.30,            # Component 1's trust score (or the Booking.com rating if C1 is missing)
    "value": 0.20,              # price compared with the fair price for this tourist
    "interests": 0.10,          # beach / nightlife / food / nature near the hotel
    "distance": 0.20,           # only for towns outside our data: closer to the asked town is better
}
MATCH_VALUE = {"yes": 1.0, "not sure": 0.5, "no": 0.0}

# Festival dates that raise prices (from the scraped event dates; Poya festivals move every year)
EVENT_WINDOWS = [               # (month, first day, last day)
    (12, 23, 26),               # Christmas
    (12, 30, 31), (1, 1, 1),    # New Year's Eve
    (4, 12, 15),                # Sinhala & Tamil New Year
    (5, 19, 21),                # Vesak 2027
    (6, 17, 19),                # Poson 2027
    (8, 6, 16),                 # Kandy Esala Perahera 2027
]

# Rough rainy months per area (for the free-cancellation advice only; first version)
RAINY_MONTHS = {
    "west_south": {5, 6, 7, 8, 9, 10, 11},
    "hills": {5, 6, 7, 10, 11, 12},
    "east": {11, 12, 1},
    "north_dry_zone": {10, 11, 12},
}
TOWN_AREA = {
    "Colombo": "west_south", "Negombo": "west_south", "Bentota": "west_south", "Galle": "west_south",
    "Unawatuna": "west_south", "Mirissa": "west_south", "Hikkaduwa": "west_south", "Weligama": "west_south",
    "Tangalle": "west_south", "Kalutara": "west_south", "Ahangama": "west_south", "Matara": "west_south",
    "Kandy": "hills", "Nuwara Eliya": "hills", "Ella": "hills", "Haputale": "hills", "Bandarawela": "hills",
    "Hatton": "hills", "Kitulgala": "hills",
    "Trincomalee": "east", "Arugam Bay": "east", "Pasikudah": "east", "Batticaloa": "east", "Nilaveli": "east",
    "Jaffna": "north_dry_zone", "Anuradhapura": "north_dry_zone", "Sigiriya": "north_dry_zone",
    "Dambulla": "north_dry_zone", "Polonnaruwa": "north_dry_zone", "Habarana": "north_dry_zone",
}

# Live Booking.com search (src/c2/live/search.py)
LIVE_MAX_NEW_HOTELS = 4         # hotel pages read per live search (each takes about 5-10 seconds)
LIVE_NEAR_KM = 10               # for a town outside our data: hotels further than this do not count as "in the town"
LIVE_TIMEOUT_MS = 30000         # per page
LIVE_MAX_SECONDS = 120          # the whole live search; hotels not reached by then are skipped
PHOTOS_PER_HOTEL = 8

PRICES_COLLECTED = "25-27 September 2026"
RATES_ATTRIBUTION = "Rates By Exchange Rate API (https://www.exchangerate-api.com)"
