"""
Stage 1b test: send each test sentence to a local LLM (Ollama), then run the
rule check, and score every field two ways:
    "Model only"    = what the model wrote
    "Model + rules" = what the system gives (after the rule check)

Needs only Python 3 (no extra packages) and Ollama running on this computer.

Usage (in PowerShell, inside the folder with these files):
    python run_stage1.py                          # uses qwen3:4b
    python run_stage1.py --model qwen3:1.7b       # try a smaller model
    python run_stage1.py --replay results_qwen3_4b.csv
          # re-score saved model answers with the current rules (no model needed, takes 1 second)
    python run_stage1.py --dry-run                # checks the script without a model
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- fixed word lists (same as output-format.md) ----
WISH_WORDS = {"quiet", "beach", "pool", "wifi", "breakfast", "parking", "view",
              "ac", "family_room", "spa", "nature"}
HEALTH_MUSTS = {"near_hospital", "low_steps", "wheelchair_access"}
MUST_WORDS = WISH_WORDS | {"budget"} | HEALTH_MUSTS
IMPORTANCE = {"high", "medium", "low"}
HEALTH_WORDS = {"heart", "mobility", "wheelchair", "breathing", "elderly"}
INTEREST_WORDS = {"beach", "nature", "wildlife", "culture", "religious", "surfing",
                  "hiking", "tea_country", "nightlife", "food"}
STYLES = {"solo", "couple", "family", "friends", "business"}

# health need -> must-have added by rules (never left to the model)
HEALTH_TO_MUST = {"heart": "near_hospital", "breathing": "near_hospital",
                  "mobility": "low_steps", "elderly": "low_steps",
                  "wheelchair": "wheelchair_access"}

FIELDS = ["trip.stay_in", "trip.want_to_visit", "trip.check_in", "trip.check_out",
          "trip.nights", "trip.dates_fixed", "trip.adults", "trip.children",
          "trip.travel_style", "trip.budget_total_lkr", "hotel.budget_max_lkr",
          "hotel.must", "hotel.wishes", "health", "interests"]
CODE_FIELDS = {"trip.check_out", "trip.nights"}   # the code calculates these; the model only copies what was said
SET_FIELDS = {"trip.stay_in", "trip.want_to_visit", "hotel.must", "health", "interests"}
PLACE_FIELDS = {"trip.stay_in", "trip.want_to_visit"}


# ---------------- calling the model ----------------
USE_THINK_FLAG = {"on": True}   # older Ollama versions reject "think"; we then stop sending it


def call_ollama(url, model, system_prompt, sentence, timeout=300):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Request: "{sentence}"\nJSON: /no_think'},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0, "num_predict": 600},
    }
    if not USE_THINK_FLAG["on"]:
        body.pop("think")
    for attempt in range(2):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))["message"]["content"]
        except urllib.error.HTTPError as e:
            if attempt == 0 and "think" in body:   # older Ollama: drop the think flag
                body.pop("think")
                USE_THINK_FLAG["on"] = False
                continue
            raise RuntimeError(f"Ollama error {e.code}: {e.read().decode('utf-8', 'ignore')}")
    raise RuntimeError("Ollama call failed")


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON found")
    return json.loads(text[start:end + 1])


# ---------------- step 1: clean (fix types, drop unknown words) ----------------
def to_number(x):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"\d[\d,]*(\.\d+)?", str(x))   # e.g. "Rs. 7,000" -> 7000
    return int(float(m.group(0).replace(",", ""))) if m else None


def to_date_text(x):
    if isinstance(x, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", x.strip()):
        return x.strip()
    return None


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str) and x.strip().lower() in ("true", "false"):
        return x.strip().lower() == "true"
    return None


def word_list(x, allowed, removed):
    out = []
    for w in (x if isinstance(x, list) else []):
        w = str(w).strip().lower().replace(" ", "_")
        if w in allowed:
            if w not in out:
                out.append(w)
        else:
            removed.append(w)
    return out


def clean(pred):
    """Fix types and drop words that are not in the fixed lists."""
    removed = []
    trip = pred.get("trip") or {}
    hotel = pred.get("hotel") or {}
    wishes_in = hotel.get("wishes") if isinstance(hotel.get("wishes"), dict) else {}
    wishes = {}
    for k, v in wishes_in.items():
        k2, v2 = str(k).strip().lower().replace(" ", "_"), str(v).strip().lower()
        if k2 in WISH_WORDS and v2 in IMPORTANCE:
            wishes[k2] = v2
        else:
            removed.append(f"{k}:{v}")
    style = str(trip.get("travel_style") or "").strip().lower()
    out = {
        "trip": {
            "stay_in": [str(p).strip() for p in (trip.get("stay_in") or []) if str(p).strip()],
            "want_to_visit": [str(p).strip() for p in (trip.get("want_to_visit") or []) if str(p).strip()],
            "check_in": to_date_text(trip.get("check_in")),
            "check_out": to_date_text(trip.get("check_out")),
            "nights": to_number(trip.get("nights")),
            "dates_fixed": to_bool(trip.get("dates_fixed")),
            "adults": to_number(trip.get("adults")),
            "children": to_number(trip.get("children")),
            "travel_style": style if style in STYLES else None,
            "budget_total_lkr": to_number(trip.get("budget_total_lkr")),
        },
        "hotel": {
            "budget_max_lkr": to_number(hotel.get("budget_max_lkr")),
            "must": word_list(hotel.get("must"), MUST_WORDS, removed),
            "wishes": wishes,
        },
        "health": word_list(pred.get("health"), HEALTH_WORDS, removed),
        "interests": word_list(pred.get("interests"), INTEREST_WORDS, removed),
    }
    return out, removed


# ---------------- step 2: rule check ----------------
# A "strong word" means the tourist made something a must-have.
STRONG = re.compile(r"\b(must|need|needs|needed|only|essential|required|require|requires|"
                    r"requirement|have to|has to|necessary|mandatory|compulsory|non-negotiable|"
                    r"can't do without|cannot do without|"
                    r"nead|neeed|neede|musst|reqired|requird|esential|essencial)\b", re.I)   # common misspellings

# A health need is kept only if the sentence has words that support it.
HEALTH_EVIDENCE = {
    "heart": r"\bheart|cardiac|bypass|pacemaker|angina|chest pain|\bhart (?:problem|surgery|patient|condition)|haert",
    "breathing": r"asthma|asthama|\bastma|\basma\b|breath|\blungs?\b|copd|wheez|inhaler|oxygen",
    "mobility": r"\bknees?\b|\bhips?\b|\blegs?\b|\bwalk|mobility|\bjoints?\b|arthritis|crutch|"
                r"walker|stairs|\bsteps\b|back pain|limp",
    "wheelchair": r"wheel\s?-?chair|weelchair|wheelchiar",
    "elderly": r"\b(elderly|eldery|elderley|senior|seniors|retired|retiree|pensioner|grand(mother|father|parents?|ma|pa)|old age)\b",
}
AGE_PATTERNS = [r"(\d{2,3})\s*-?\s*(?:years?|yrs?)\s*-?\s*old",
                r"\b(?:over|above|aged|age)\s+(\d{2,3})(?!\d|[,.]\d)",
                r"\b(\d{2,3})\s*\+",
                r"\b(?:i'm|i am|he's|he is|she's|she is|is)\s+(\d{2,3})(?!\d|[,.]\d|\s*(?:rs|lkr|rupees|usd|\$|k\b))",
                r"\bin (?:his|her|their|my|our) ([6-9])0s\b"]
HEALTH_MUST_EVIDENCE = {   # a health must-have is kept only with a health need or these words
    "near_hospital": ({"heart", "breathing"}, r"hospital|clinic|medical|doctor"),
    "low_steps": ({"mobility", "elderly"}, r"\bsteps\b|stairs|\blift\b|elevator|ground floor"),
    "wheelchair_access": ({"wheelchair"}, r"wheel\s?-?chair|accessib"),
}
# want_to_visit items made only of these words are not real place names
GENERIC = set("""the a an of and some old ancient ruins ruin history historical historic heritage culture
cultural site sites temple temples beach beaches sea airport station hotel hotels city town area nature
wildlife safari food nightlife surfing surf kite kitesurfing hiking hike trek tea estate estates factory
factories whale whales watching national park parks mountain mountains waterfall waterfalls village villages
market markets shopping restaurant restaurants place places spot spots sightseeing attractions nearby local
rainforest forest lake river""".split())


# Foreign money: the model copies the number as it was written; the code converts it to rupees.
# LKR for 1 unit of each currency.
# FIXED rates are used when scoring tests, so results can be repeated (frozen on 18-19 Oct 2026; sources:
# exchange-rates.org, poundsterlinglive.com, clock7.com, open.er-api.com). The live system (demo.py) uses
# today's rates from the internet instead - see live_rates().
CURRENCIES = ["USD", "EUR", "GBP", "INR", "AUD", "CNY", "RUB", "JPY"]
RATES_FIXED = {"USD": 331.20, "EUR": 372.87, "GBP": 441.72, "INR": 3.49,
               "AUD": 230.04, "CNY": 49.51, "RUB": 3.97, "JPY": 2.09}
RATES_TO_LKR = RATES_FIXED          # default used by apply_rules()
RATE_SOURCES = [   # free, no key, updated once a day
    ("Rates By Exchange Rate API (https://www.exchangerate-api.com)", "https://open.er-api.com/v6/latest/LKR", "er"),
    ("fawazahmed0 currency-api", "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/lkr.json", "fa"),
    ("fawazahmed0 currency-api (mirror)", "https://latest.currency-api.pages.dev/v1/currencies/lkr.json", "fa"),
]
RATES_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rates_cache.json")


def live_rates(cache_file=RATES_CACHE, max_age_hours=24, timeout=10):
    """Today's rates (LKR for 1 unit). Order: saved today -> internet -> last saved -> fixed rates.
    Returns (rates, where they came from)."""
    now = time.time()
    try:
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = None
    if cache and now - cache.get("fetched_at", 0) < max_age_hours * 3600:
        return cache["rates"], f"{cache['source']}, {cache['date']}"
    for name, url, kind in RATE_SOURCES:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            if kind == "er":
                if d.get("result") != "success":
                    continue
                raw, day = d["rates"], d.get("time_last_update_utc", "")[:16]
            else:
                raw, day = {k.upper(): v for k, v in d["lkr"].items()}, d.get("date", "")
            rates = {}
            for c in CURRENCIES:
                if raw.get(c):
                    r = round(1 / raw[c], 4)
                    # safety check: ignore a rate that is less than half or more than double the fixed one
                    rates[c] = r if 0.5 < r / RATES_FIXED[c] < 2 else RATES_FIXED[c]
            if "USD" not in rates:
                continue
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({"rates": rates, "source": name, "date": day, "fetched_at": now}, f, indent=1)
            except OSError:
                pass
            return rates, f"{name}, {day}"
        except Exception:
            continue
    if cache:
        return cache["rates"], f"{cache['source']}, {cache['date']} (saved earlier - no internet now)"
    return dict(RATES_FIXED), "fixed rates of 18-19 Oct 2026 (no internet)"


_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(k\b)?"
CURRENCY_PATTERNS = [   # (currency, regex); group 1 = amount, group 2 = optional "k". AUD before USD ("A$60").
    ("AUD", r"(?:a\$|au\$|\baud\b)\s*" + _NUM),
    ("AUD", _NUM + r"\s*(?:aud\b|australian dollars?\b)"),
    ("USD", r"(?:us\s?\$|(?<![a-z])\$|\busd\b)\s*" + _NUM),
    ("USD", _NUM + r"\s*(?:usd\b|us dollars?\b|dollars?\b|\$)"),
    ("EUR", r"(?:€|\beur\b|\beuros?\b)\s*" + _NUM),
    ("EUR", _NUM + r"\s*(?:€|eur\b|euros?\b)"),
    ("GBP", r"(?:£|\bgbp\b)\s*" + _NUM),
    ("GBP", _NUM + r"\s*(?:£|gbp\b|pounds?\b)"),
    ("INR", r"(?:₹|\binr\b)\s*" + _NUM),
    ("INR", _NUM + r"\s*(?:₹|inr\b|indian rupees?\b)"),
    ("CNY", r"(?:\bcny\b|\brmb\b)\s*" + _NUM),
    ("CNY", _NUM + r"\s*(?:cny\b|rmb\b|yuan\b)"),
    ("RUB", r"(?:₽|\brub\b)\s*" + _NUM),
    ("RUB", _NUM + r"\s*(?:₽|rub\b|roubles?\b|rubles?\b)"),
    ("JPY", r"\bjpy\s*" + _NUM),
    ("JPY", _NUM + r"\s*(?:jpy\b|yen\b)"),
]


def foreign_amounts(sentence):
    """[(currency, amount)] for every foreign-money amount written in the sentence."""
    found, taken = [], []
    for cur, pattern in CURRENCY_PATTERNS:
        for m in re.finditer(pattern, sentence, re.I):
            if any(m.start() < e and s0 < m.end() for s0, e in taken):
                continue                 # already read as another currency (e.g. "A$60" is not US dollars)
            taken.append((m.start(), m.end()))
            amount = float(m.group(1).replace(",", ""))
            if m.group(2):
                amount *= 1000
            if (cur, amount) not in found:
                found.append((cur, amount))
    return found

# Big rupee words: the model should write the full number; if it copied "1.5" from "1.5 lakhs", the code fixes it
BIG_UNITS = [(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?)\b", 100_000),
             (r"(\d+(?:\.\d+)?)\s*(?:million|mn)\b", 1_000_000)]

# Relative days: the code sets check_in for "tonight", "tomorrow" (the model only copies dates)
RELATIVE_DAYS = [(r"\bday after tomorrow\b", 2), (r"\btonight\b|\btoday\b", 0), (r"\btomorrow\b", 1)]
STAY_DAY_WORDS = r"\b(?:tonight|today|tomorrow night|for tomorrow|from tomorrow|day after tomorrow)\b"

# Sri Lankan towns and places, to fix spelling ("Kandyy" -> "Kandy"). Unknown names are kept as written.
TOWN_NAMES = """Colombo, Negombo, Katunayake, Kandy, Peradeniya, Ella, Nuwara Eliya, Haputale, Bandarawela, Badulla,
Hatton, Nallathanniya, Talawakele, Nanu Oya, Galle, Unawatuna, Hikkaduwa, Ambalangoda, Mirissa, Weligama, Ahangama,
Midigama, Koggala, Matara, Dikwella, Hiriketiya, Tangalle, Hambantota, Bentota, Induruwa, Beruwala, Aluthgama,
Kalutara, Wadduwa, Panadura, Mount Lavinia, Ahungalla, Kosgoda, Balapitiya, Tissamaharama, Kirinda, Kataragama, Yala,
Udawalawe, Embilipitiya, Ratnapura, Belihuloya, Sinharaja, Deniyaya, Kitulgala, Sigiriya, Dambulla, Habarana,
Polonnaruwa, Giritale, Anuradhapura, Mihintale, Kurunegala, Matale, Trincomalee, Nilaveli, Uppuveli, Pasikudah,
Kalkudah, Batticaloa, Arugam Bay, Pottuvil, Ampara, Jaffna, Mannar, Vavuniya, Kalpitiya, Puttalam, Chilaw, Wilpattu,
Kegalle, Gampola, Monaragala, Wellawaya, Welimada, Kandalama"""
TOWNS_SL = [t.strip() for t in TOWN_NAMES.replace("\n", " ").split(",")]
PLACE_TOWN = {   # places inside or next to a town (a hotel "at Galle Fort" is a hotel in Galle)
    "Galle Fort": "Galle", "Jungle Beach": "Unawatuna", "Temple of the Tooth": "Kandy", "Kandy Lake": "Kandy",
    "Peradeniya Gardens": "Kandy", "Nine Arch Bridge": "Ella", "Little Adam's Peak": "Ella", "Ella Rock": "Ella",
    "Ravana Falls": "Ella", "Gregory Lake": "Nuwara Eliya", "Hakgala Gardens": "Nuwara Eliya",
    "Lipton's Seat": "Haputale", "Coconut Tree Hill": "Mirissa", "Parrot Rock": "Mirissa",
    "Hummanaya Blowhole": "Tangalle", "Brief Garden": "Bentota", "Madu River": "Bentota",
    "Kalutara Bodhiya": "Kalutara", "Elephant Transit Home": "Udawalawe", "Sigiriya Rock": "Sigiriya",
    "Pidurangala": "Sigiriya", "Dambulla Cave Temple": "Dambulla", "Gal Vihara": "Polonnaruwa",
    "Ruwanwelisaya": "Anuradhapura", "Koneswaram Temple": "Trincomalee", "Marble Beach": "Trincomalee",
    "Pigeon Island": "Nilaveli", "Batticaloa Fort": "Batticaloa", "Muhudu Maha Viharaya": "Arugam Bay",
    "Nallur Kovil": "Jaffna", "Jaffna Fort": "Jaffna", "Casuarina Beach": "Jaffna", "Koggala Lake": "Koggala",
    "Gangaramaya Temple": "Colombo", "Galle Face Green": "Colombo", "Belilena Cave": "Kitulgala",
    "Dowa Temple": "Bandarawela", "Aluvihare": "Matale"}
PARKS = ["Horton Plains", "Minneriya", "Kaudulla", "Kumana", "Bundala", "Udawalawe National Park", "Adam's Peak",
         "Pinnawala", "Mulkirigala", "Muthurajawela", "Delft Island"]




def _key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def best_match(name, choices, cutoff=0.85):
    """Same name ignoring case/spaces, or a very close spelling. None if nothing is close."""
    keys = {_key(c): c for c in choices}
    k = _key(name)
    if k in keys:
        return keys[k]
    if len(k) < 4:
        return None
    m = difflib.get_close_matches(k, list(keys), n=1, cutoff=cutoff)
    return keys[m[0]] if m else None


def canon_town(name):
    """Town where the hotel should be: fixes spelling, drops 'area', maps 'Galle Fort' -> 'Galle'."""
    plain = re.sub(r"^(?:the)\s+|\s+(?:area|town|city|district|region|side)$", "", str(name).strip(), flags=re.I)
    return best_match(plain, TOWNS_SL) or PLACE_TOWN.get(best_match(plain, PLACE_TOWN) or "", None) or plain


def canon_place(name):
    """A place to visit with its usual spelling (unknown places are kept as written)."""
    return best_match(name, list(PLACE_TOWN) + PARKS + TOWNS_SL) or str(name).strip()


def has_old_age(sentence):
    if re.search(HEALTH_EVIDENCE["elderly"], sentence, re.I):
        return True
    for p in AGE_PATTERNS:
        for m in re.finditer(p, sentence, re.I):
            age = int(m.group(1))
            if age < 10:          # "in her 70s" -> 7
                age *= 10
            if age >= 60:
                return True
    return False


def health_supported(word, sentence):
    if word == "elderly":
        return has_old_age(sentence)
    return re.search(HEALTH_EVIDENCE[word], sentence, re.I) is not None


def to_date(text):
    try:
        return date.fromisoformat(text) if text else None
    except ValueError:
        return None          # e.g. "2026-11-31" does not exist


def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:       # 29 February
        return d.replace(year=d.year + n, day=28)


def norm_place(p):
    return re.sub(r"[^a-z0-9]", "", str(p).lower())


def apply_rules(pred, sentence, today_text, rates=None):
    """Rule check after the model. Returns (final output, list of what the rules changed).
    rates: LKR for 1 unit of each currency (default: the fixed rates used for tests)."""
    rates = rates or RATES_TO_LKR
    final = json.loads(json.dumps(pred))
    trip, hotel = final["trip"], final["hotel"]
    notes = []
    today = date.fromisoformat(today_text)

    # 0. "tonight", "tomorrow": the code knows today's date, the model does not always get it right
    target = None
    for pattern, days in RELATIVE_DAYS:
        if re.search(pattern, sentence, re.I):
            target = today + timedelta(days=days)
            break
    if target:
        ci0 = to_date(trip["check_in"])
        if (ci0 is None and re.search(STAY_DAY_WORDS, sentence, re.I)) or \
           (ci0 is not None and ci0 != target and abs((ci0 - target).days) <= 3):
            trip["check_in"] = target.isoformat()
            if ci0 is not None and trip["check_out"] and not trip["nights"]:
                co0 = to_date(trip["check_out"])
                trip["check_out"] = (target + (co0 - ci0)).isoformat() if co0 else None
            notes.append("check_in set from 'tonight/tomorrow'")

    # 1. Dates: real dates only, never in the past, code does the counting
    ci, co, n = to_date(trip["check_in"]), to_date(trip["check_out"]), trip["nights"]
    if trip["check_in"] and not ci:
        notes.append("bad check_in removed")
    if trip["check_out"] and not co:
        notes.append("bad check_out removed")
    if n is not None and not (0 < n <= 90):
        n = None
    if ci and ci < today:
        k = 1
        while add_years(ci, k) < today:
            k += 1
        ci = add_years(ci, k)
        co = add_years(co, k) if co else None
        notes.append("past date moved to next year")
    if ci and n:
        if co != ci + timedelta(days=n):
            co = ci + timedelta(days=n)
            notes.append("check_out calculated")
    elif ci and co:
        if co <= ci:
            co = add_years(co, 1) if add_years(co, 1) > ci else None
            notes.append("check_out moved after check_in")
        if co:
            n = (co - ci).days
            notes.append("nights calculated")
    trip["check_in"] = ci.isoformat() if ci else None
    trip["check_out"] = co.isoformat() if co else None
    trip["nights"] = n

    # 2. Health: keep a health need only if the sentence supports it
    kept = []
    for h in final["health"]:
        if health_supported(h, sentence):
            kept.append(h)
        else:
            notes.append(f"health '{h}' removed (not in sentence)")
    final["health"] = kept

    # 3. Health must-haves: keep only if a health need or the sentence supports them
    must = []
    for m in hotel["must"]:
        if m in HEALTH_MUSTS:
            needs, words = HEALTH_MUST_EVIDENCE[m]
            if not (needs & set(kept)) and not re.search(words, sentence, re.I):
                notes.append(f"must '{m}' removed (not asked)")
                continue
        must.append(m)

    # 4. Health needs always become must-haves
    for h in kept:
        if HEALTH_TO_MUST[h] not in must:
            must.append(HEALTH_TO_MUST[h])

    # 5a. Foreign money: if the model copied a dollar (or other) amount, convert it to rupees
    for cur, amount in foreign_amounts(sentence):
        for key, part in (("budget_max_lkr", hotel), ("budget_total_lkr", trip)):
            if part[key] is not None and abs(part[key] - amount) < 1:
                if cur in rates:
                    part[key] = int(round(amount * rates[cur]))
                    notes.append(f"{cur} {amount:g} converted to Rs. {part[key]:,} (1 {cur} = Rs. {rates[cur]:g})")
                else:
                    part[key] = None
                    notes.append(f"{cur} price removed (no exchange rate set)")

    # 5b. "1.5 lakhs" = Rs. 150,000: if the model copied only "1.5", the code writes the full number
    for pattern, mult in BIG_UNITS:
        for m in re.finditer(pattern, sentence, re.I):
            raw = float(m.group(1))
            full = int(round(raw * mult))
            for key, part in (("budget_max_lkr", hotel), ("budget_total_lkr", trip)):
                if part[key] is not None and part[key] != full and abs(part[key] - raw) < 1:
                    part[key] = full
                    notes.append(f"'{m.group(0)}' written as Rs. {full:,}")

    # 5. "budget" is a must-have exactly when a price per night is given
    if hotel["budget_max_lkr"] is None and "budget" in must:
        must.remove("budget")
        notes.append("'budget' removed (no price given)")
    if hotel["budget_max_lkr"] is not None and "budget" not in must:
        must.append("budget")

    # 6. No strong word (must, need, only ...) in the sentence -> hotel wishes cannot be must-haves
    if not STRONG.search(sentence):
        for m in [m for m in must if m in WISH_WORDS]:
            must.remove(m)
            hotel["wishes"].setdefault(m, "medium")
            notes.append(f"'{m}' moved from must to wish (no strong word)")
    hotel["must"] = must

    # 7. A must-have is not also a wish
    for m in must:
        hotel["wishes"].pop(m, None)

    # 8a. Spelling of towns and places ("Kandyy" -> "Kandy", "Galle Fort area" -> "Galle")
    fixed = []
    for p in trip["stay_in"]:
        t = canon_town(p)
        if t != p:
            notes.append(f"town '{p}' -> '{t}'")
        if t not in fixed:
            fixed.append(t)
    trip["stay_in"] = fixed
    visit = []
    for p in trip["want_to_visit"]:
        q = canon_place(p)
        if q != p:
            notes.append(f"place '{p}' -> '{q}'")
        visit.append(q)
    trip["want_to_visit"] = visit

    # 8. Places to visit: not the stay_in towns, not general words, no repeats
    stay = {norm_place(p) for p in trip["stay_in"]}
    places, seen = [], set()
    for p in trip["want_to_visit"]:
        key = norm_place(p)
        words = re.findall(r"[a-z]+", p.lower())
        if key in stay:
            notes.append(f"place '{p}' removed (same as stay_in)")
        elif words and all(w in GENERIC for w in words):
            notes.append(f"place '{p}' removed (not a place name)")
        elif key not in seen:
            places.append(p)
            seen.add(key)
    trip["want_to_visit"] = places
    return final, notes


def finalize_gold(gold):
    """The correct final answer: the correct model answer plus the fixed health and budget musts."""
    final = json.loads(json.dumps(gold))
    must = final["hotel"]["must"]
    for h in final["health"]:
        if HEALTH_TO_MUST[h] not in must:
            must.append(HEALTH_TO_MUST[h])
    if final["hotel"]["budget_max_lkr"] is not None and "budget" not in must:
        must.append("budget")
    for m in must:
        final["hotel"]["wishes"].pop(m, None)
    return final


# ---------------- scoring ----------------
def get(d, path):
    for p in path.split("."):
        d = d.get(p) if isinstance(d, dict) else None
    return d


def same(field, gold, pred):
    if field in PLACE_FIELDS:
        return {norm_place(x) for x in gold or []} == {norm_place(x) for x in pred or []}
    if field in SET_FIELDS:
        return set(gold or []) == set(pred or [])
    return gold == pred


def wrong_fields(gold, pred):
    out = []
    for f in FIELDS:
        g, p = get(gold, f), get(pred, f)
        if not same(f, g, p):
            out.append(f"{f}: correct={json.dumps(g, ensure_ascii=False)} got={json.dumps(p, ensure_ascii=False)}")
    return out


def load_replay(path):
    answers = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            answers[row["id"]] = row["model_output"]
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--prompt", default="stage1_prompt.txt")
    ap.add_argument("--tests", default="test_sentences.jsonl")
    ap.add_argument("--replay", help="re-score the model answers saved in this results CSV")
    ap.add_argument("--dry-run", action="store_true", help="use the correct answers instead of the model")
    ap.add_argument("--live-rates", action="store_true",
                    help="use today's exchange rates (the test answers are made with the fixed rates)")
    args = ap.parse_args()

    prompt_template = open(args.prompt, encoding="utf-8").read()
    rates, rates_info = (live_rates() if args.live_rates else (RATES_FIXED, "fixed rates of 18-19 Oct 2026"))
    tests = [json.loads(line) for line in open(args.tests, encoding="utf-8") if line.strip()]
    replay = load_replay(args.replay) if args.replay else None
    if args.dry_run:
        name = "dry_run"
    elif replay is not None:
        name = "replay_" + re.sub(r"[^\w.-]", "_", args.replay.rsplit(".", 1)[0])
    else:
        name = re.sub(r"[^\w.-]", "_", args.model)

    ok_model = {f: 0 for f in FIELDS}
    ok_rules = {f: 0 for f in FIELDS}
    all_model = all_rules = broken_json = 0
    must_missing_total = must_extra_total = must_safe = 0
    removed_total, seconds, rows, rule_counts = [], [], [], {}

    for t in tests:
        system_prompt = prompt_template.replace("{today}", t["today"])
        start = time.time()
        error = ""
        if args.dry_run:
            raw = json.dumps(t["gold"])
        elif replay is not None:
            if t["id"] not in replay:
                print(f"{t['id']} is not in {args.replay}")
                return
            raw = replay[t["id"]]
        else:
            try:
                raw = call_ollama(args.url, args.model, system_prompt, t["sentence"])
            except Exception as e:
                print(f"Could not reach the model: {e}")
                print("Is Ollama running, and did you run 'ollama pull " + args.model + "'?")
                return
        seconds.append(time.time() - start)

        try:
            pred, removed = clean(parse_json(raw))
        except Exception as e:
            try:  # one retry if the JSON was broken
                if args.dry_run or replay is not None:
                    raise
                raw = call_ollama(args.url, args.model, system_prompt, t["sentence"])
                pred, removed = clean(parse_json(raw))
            except Exception:
                broken_json += 1
                error = f"broken JSON: {e}"
                pred, removed = clean({})
        removed_total += removed

        final, notes = apply_rules(pred, t["sentence"], t["today"], rates)
        for nt in notes:
            key = re.sub(r"'[^']*'", "'…'", nt)
            rule_counts[key] = rule_counts.get(key, 0) + 1
        gold_final = finalize_gold(t["gold"])

        wrong_m = wrong_fields(t["gold"], pred)
        wrong_r = wrong_fields(gold_final, final)
        for f in FIELDS:
            ok_model[f] += same(f, get(t["gold"], f), get(pred, f))
            ok_rules[f] += same(f, get(gold_final, f), get(final, f))
        all_model += not wrong_m
        all_rules += not wrong_r

        need, got = set(gold_final["hotel"]["must"]), set(final["hotel"]["must"])
        missing, extra = sorted(need - got), sorted(got - need)
        must_missing_total += len(missing)
        must_extra_total += len(extra)
        must_safe += not missing

        status = "OK " if not wrong_r else "ERR"
        print(f"{t['id']} {status} {seconds[-1]:5.1f}s  {t['sentence'][:60]}...")
        for w in wrong_r:
            print("      " + w)
        if missing:
            print(f"      MISSING MUST-HAVES: {missing}")
        if extra:
            print(f"      EXTRA MUST-HAVES (too strict): {extra}")
        rows.append({"id": t["id"], "sentence": t["sentence"],
                     "wrong_model_only": len(wrong_m), "wrong_after_rules": len(wrong_r),
                     "wrong_details_after_rules": " | ".join(wrong_r),
                     "rule_changes": " | ".join(notes),
                     "missing_must": ",".join(missing), "extra_must": ",".join(extra),
                     "removed_words": ",".join(removed), "error": error,
                     "seconds": round(seconds[-1], 2),
                     "model_output": json.dumps(pred, ensure_ascii=False),
                     "final_output": json.dumps(final, ensure_ascii=False)})

    n = len(tests)
    print("\n================ RESULTS (" + name + ") ================")
    print(f"{'Field':24} {'Model only':>12} {'Model + rules':>15}")
    for f in FIELDS:
        label = f + (" *" if f in CODE_FIELDS else "")
        print(f"{label:24} {ok_model[f]:5d}/{n} {100*ok_model[f]/n:4.0f}%   {ok_rules[f]:5d}/{n} {100*ok_rules[f]/n:4.0f}%")
    print("-" * 56)
    tm, tr = sum(ok_model.values()), sum(ok_rules.values())
    print(f"{'Field accuracy (all)':24} {100*tm/(n*len(FIELDS)):10.1f}%   {100*tr/(n*len(FIELDS)):12.1f}%")
    print(f"{'Sentences fully right':24} {all_model:7d}/{n}   {all_rules:10d}/{n}")
    print("* the code calculates check_out and nights; the model only copies what was said")
    print()
    print(f"Must-haves never broken (system): {must_safe}/{n}   "
          f"missing must-haves: {must_missing_total}   extra must-haves (too strict): {must_extra_total}")
    print(f"Broken JSON: {broken_json}   Words removed by type check: {len(removed_total)}")
    print(f"Exchange rates used: {rates_info}")
    if rule_counts:
        print("What the rules changed:")
        for k, v in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
            print(f"   {v:3d} x {k}")
    if not args.dry_run and replay is None:
        print(f"Average time per sentence: {sum(seconds) / n:.1f} s")

    out_csv = f"results_{name}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Details saved to {out_csv}")


if __name__ == "__main__":
    main()
