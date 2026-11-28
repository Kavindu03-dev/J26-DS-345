# C2: Personalised hotel ranking and price intelligence

Owner: Hansada S.A.B. This service takes a tourist's request and returns ranked hotels. For each hotel it checks
the must-haves, the features, the fair price and where to book. It also sends what Component 3 needs.

```
tourist sentence
  -> Gap 1  NLP model (hotel-nlp-v2 in Ollama) + rule check  -> structured request (Output 1)
  -> hotels in the town (our 46 data towns, or hotels within 10-25 km of any other town)
  -> Gap 2  feature check: listing + Component 1 reviews     -> yes / no / not sure (and is it true NOW?)
  -> Gap 5  nearest hospital (OpenStreetMap), lift / steps, wheelchair access
  -> Gap 1  must-haves: a hotel that fails one is removed; "not sure" hotels come last and are marked
  -> Gap 3  fair price (price model) + what you pay for things you don't need or don't get
  -> score  wishes 40% + trusted rating 30% + value 20% + interests 10%  (weights in src/c2/settings.py)
  -> Gap 4  where to book: Booking.com vs Trip.com vs Agoda, same-deal rules (September data only)
  -> answer for Component 4 (team contract) + Output 2 for Component 3
```

## Component 1 is DUMMY for now

Component 1 (trusted review scores) is not ready yet, so the service makes **made-up** scores in the C1 contract
format (`src/c2/c1_client.py`, class `DummyC1`). Every answer says so: `c1_source` is
`"DUMMY - made-up values until Component 1 is ready"`, and the explanation says `[DUMMY review score]`.

When your friend has real scores, change only an environment variable. No code changes are needed:

| Your friend gives you | PowerShell |
|---|---|
| a JSON file (list of objects in `contracts/c1_aspect_scores.json` format) | `$env:C1_MODE="file"; $env:C1_FILE="D:\Research\c1_aspect_scores.json"` |
| a running service | `$env:C1_MODE="http"; $env:C1_URL="http://localhost:8001"` |

Run `python -m src.c2.c1_client --export dummy_c1_aspect_scores.json` to show your friend the exact format.
Hotel ids must be the Booking.com ids (`booking_hotel_id`, e.g. `cinnamon-red-colombo`).

The feature check reads C1's `verified_aspect_facts` sentences ("the pool was closed for repairs" means pool = no)
and the `wifi` aspect score. Later, C1 can send a clearer field such as
`feature_checks: {"pool": {"status": "contradicted", "reviews": 4, "last_seen": "2026-09"}}`.

## Run it on your laptop (PowerShell)

```
cd "C:\Users\USER\Desktop\Research Project Code\J26-DS-345\services\c2-hotel-recommender"
pip install -r requirements.txt
pytest
```

The data folders are found automatically on your laptop (`D:\Research\hotel_preprocessing\hotel_preprocessing`
and `D:\Research`). On another computer, set them first:

```
$env:C2_DATA_DIR="D:\Research\hotel_preprocessing\hotel_preprocessing"   # the folder with output_clean_v5\ etc.
$env:C2_EXTRA_DIR="D:\Research"        # sl_hospitals_osm.csv, sl_places_osm.csv, hotel_urls.csv
```

**Which data and price model.** The service uses the **November 2026 data** when its files are there, else the
September data:

| | November (default) | September |
|---|---|---|
| tables | `output_clean_v5\` | `output_clean\` |
| price model | `models_v5_lgbm\booking_v5_lgbm_*` (LightGBM, typical error 27.1%) | `models\booking_*` (gradient boosting, 26.1%) |
| hotels | 2,238 Booking.com hotels in 46 towns | 1,116 hotels in 15 towns |
| sites | Booking.com only | Booking.com, Agoda, Trip.com (where to book) |
| prices collected | 20-23 November 2026, 8 check-in dates | 25-27 September 2026, 20 check-in dates |

To go back to September: `$env:C2_DATA_VERSION="september"` (and `"november"` or nothing to come back).
In the November data a hotel belongs to the town whose centre is nearest (`town`, from `town_centres.csv`), or to
a town Booking.com listed it under if it is within 10 km of that town's centre. For small towns Booking.com also
lists hotels 15-50 km away, and those are not shown as "in" that town.

**Demo from a sentence** (Ollama must be running with `hotel-nlp-v2`):

```
python recommend.py "We are a family of 4, Kandy from 26 to 29 December, under Rs 25,000 a night, pool please"
python recommend.py                       # type several sentences, empty line to stop
python recommend.py --request experiments\example_request.json     # no Ollama needed
```

**As a service** (for Component 4):

```
uvicorn src.main:app --port 8002
```

Open http://localhost:8002/docs to try it in the browser.

| Endpoint | What it does |
|---|---|
| `GET /health` | is the service up |
| `POST /recommendations` | team contract request (destination, dates, budget_lkr, ...), plus optional extras: `must`, `wishes`, `not_needed`, `health`, `health_consent`, `nights`, `budget_total_lkr` |
| `POST /recommendations/from-text` | `{"text": "...", "health_consent": false}`: the tourist's own sentence goes through the NLP model first |
| `GET /hotels/{hotel_id}` | one hotel: feature checks, nearest hospital, C1 scores |

## What comes back

Each hotel has the contract fields (`hotel_id, name, rank, score, price{expected_price_lkr, price_status,
best_platform, offers}, explanation`) plus extras: `latitude, longitude, nights, total_hotel_cost_lkr,
budget_left_lkr, booking_url, nearest_hospital, must_check, feature_checks, price.what_if, price.breakdown,
price.explanation, price.good_value, price.price_limits, free_cancellation`.
`for_c3` is Output 2 for Component 3. Health needs are in it only when `health_consent` is true, as categories only.

Prices are **per night for all rooms, taxes included**. They were collected on 20-23 November 2026 for 8
check-in dates (September data: 25-27 September 2026, 20 check-in dates; each answer says which in
`prices_collected`). The service uses the collected date closest to the tourist's date in the year, then moves the
price to the tourist's dates with the price model. A price far from the hotel's own usual price is skipped.

## Good deal / Looks high: each town has its own limits

"Potentially Overpriced" and "Good value" use **the hotel's own town's limits** from `price_margins.csv`, not one
fixed ±30% for the whole country. The weekly cloud pipeline (`hotel-price-weekly`) makes this file again every
Sunday: for every price, ratio = price / fair price from a model that never saw the hotel; Good deal = the cheapest
1 in 10 of the town's ratios, Looks high = the dearest 1 in 10. Small towns borrow from their province:
limit = (n x town + 60 x province) / (n + 60), n = the town's hotels. Example (file built 26 Nov 2026): Kandy,
Good deal = 45% or more below the fair price, Looks high = 78% or more above.

Where the service looks for the file (first found wins):

1. `$env:C2_PRICE_MARGINS="D:\path\to\price_margins.csv"` (`off` = the fixed 30% rule)
2. `price_margins.csv` in the price model's folder (e.g. `models_v5_lgbm\`)
3. `D:\Research\hotel-price-weekly\models\current\price_margins.csv` (run `git pull` there to get the newest)

A town that is not in the file uses its province's, then the whole country's limits; no file at all = the old fixed
30% rule. Each answer says which was used (`price.price_limits`, and a line in `notes`). Restart `uvicorn` after
a new file.

## The hotel finder page (React)

Open **http://localhost:8002**: it goes to the hotel finder page (`/app`). The first, simpler page is still at `/chat`.

- Type the trip in one sentence, or click an example. The page shows what it understood, how many hotels were
  removed and why, and the best hotels.
- Each hotel shows its photos, must-haves and wishes, and a **fair-price gauge**: where the price sits against the
  fair price (good value / expected range / overpriced). A hollow diamond shows the same price against the fair
  price for most guests, when the tourist doesn't need some facilities.
- Three tabs: **Price and booking** (fair price, where the money goes, Booking.com vs Agoda vs Trip.com),
  **Why this price** (the SHAP waterfall, made when you click "Explain this price"), **Why this hotel**.
- When our data is not enough, the live Booking.com search starts by itself (untick the box to do it by hand).
- Light and dark mode follow the computer's setting. It works offline: fonts and code are in the built files.

The page is built from `ui/` (React + Vite) into `src/ui/app/`. The Python service serves the built files, so
**running it needs no Node.js**. To change the page, install Node.js 18 or newer once, then:

```
cd ui
npm install
npm run dev        # live-reloading page on http://localhost:5173 (uvicorn must run on port 8002)
npm run build      # writes the finished page to src\ui\app
```

## Where the fair price goes, and why it is this number (SHAP)

The same two ideas as `personal_fair_price.ipynb` (sections 4 and 5), now in every answer:

**1. Where the fair price goes (`price.breakdown`).** For each hotel, about how many rupees and what % of the fair
price is for the **pool, bar, breakfast and beachfront**, and what is left for everything else (the room, the town,
the hotel's class, the dates). It is the what-if: switch the facility off, ask the model again. Each part says what
we do with it: *you don't need it* (taken off your fair price), *reviews say it is not there* (taken off), *you
may use it* (stays), *too small to be sure* (under Rs 1,000 or 5%: not shown) or *the hotel doesn't have it*.
The parts add up to the fair price. Only these 4 facilities are priced: the notebook tested all 14, and for the
other 10 (spa, gym, air conditioning, airport shuttle, ...) the model can't tell their cost apart from the type
of hotel. Checked again with the November model (same test): pool, bar, breakfast and beachfront still pass
easily (they add money at 98-100% of the prices). Air conditioning (80%) and restaurant (95%) now pass too, but
they are not priced until `personal_fair_price.ipynb` is run on the November data.

**2. Why the fair price is this number (`GET /hotels/{hotel_id}/price-explanation`).** Exact SHAP in rupees over 11
groups (town, hotel class and reviews, room, pool, bar, beach, breakfast and meals, other facilities, your dates,
your group, booking terms): it starts from the average of 20 typical prices, and each group adds or takes away
rupees until it reaches the fair price exactly. Each answer has `price.explanation.url` (with `row_id`, `check_in`,
`nights`). It takes about 5-20 seconds, so it is made only when the tourist clicks **"Why this fair price?"**, and
then kept in memory. Two checks come with every answer: the parts add up to the fair price, and they are the same
as the textbook Shapley formula. Without the `shap` package the service uses the textbook formula (same numbers).

**The model that never saw the hotel.** A fair price must come from a model that had not seen the hotel. The
final model never saw the test hotels and the live hotels, so it explains those. But it learned the training
hotels (its price for them is close to the real price), so for those we use the cross-validation model that made
their fair price. The training notebook did not save these 3 models, so `src/c2/unseen_models.py` makes them
again exactly (same rows, same hotel groups, same settings). The check: November gives the same 22,562 fair prices
of the 1,660 training hotels to 0.000000% (September: 10,359 fair prices, 843 hotels). The same model also moves
the fair price to the tourist's dates.

```
python -m src.c2.unseen_models          # once, 3-10 minutes -> models_v5_lgbm\booking_v5_lgbm_cv_models.joblib
python -m src.c2.unseen_models --version september      # -> models\booking_cv_models.joblib  (never commit these)
```

Without that file, SHAP is shown only for test hotels and live hotels (the page says why for the others).

## Live Booking.com search and hotel photos

Most requests are answered from the CSV files. A **live search** runs only when the CSV data is not enough:
- fewer hotels than asked for (top_k) meet every must-have, or
- the town is not in our data, or our hotels are more than 10 km away (for example Tangalle or Hikkaduwa).

It opens **one** Booking.com search for the tourist's town, dates and group. The budget and must-haves become
Booking.com's own filters (pool, air conditioning, beachfront, breakfast, wheelchair access and so on), and only
hotels are shown. Then it reads the hotel pages of **at most 4 hotels that are not already in the CSV files**. The
reading rules are copied from the team scraper, and the conversion rules are copied from preprocessing.ipynb.
After that:
- **Saved:** the new hotels go into `output_clean_v5\live\*.csv` (September data: `output_clean\live\`) next to
  the research files (same columns; November hotels also get their own town, district and province).
  The next request gets them from the CSV, not from the web.
  - `hotels_booking_live.csv`: facts and facilities
  - `hotel_words_facilities_live.csv`: the full facility list
  - `prices_live.csv`: price for those dates, fair price and where to book
  - `hotel_photos.csv`: photos of every hotel (photos saved with the September data are used too)
  - `live_search_log.csv`: what was searched (no request details)
- **Fair price:** the price model gives the fair price. The new hotel was not in its training data.
- **Three sites (September data only):** Booking.com is live. Agoda and Trip.com are compared when the same hotel
  is in our September data (same name-matching rule as preprocessing). The November data is Booking.com only. Their price is moved to the tourist's dates with the price
  model and marked "estimate". Agoda and Trip.com are never scraped live.
- **Photos:** each recommended hotel shows its photos. They come from `hotel_photos.csv`, or are read once from
  the hotel's Booking.com page and saved.
- **Human check:** if Booking.com shows an "are you a robot?" check, the search stops and says so. It never
  tries to answer it.
- **Original files:** the research files are never changed, so the model results stay reproducible.

One-time setup on the laptop:

```
pip install -r requirements.txt
python -m playwright install chromium
```

| Setting | What it does |
|---|---|
| `$env:C2_LIVE="0"` | switch live search off (no internet at the venue) |
| `$env:C2_LIVE_SHOW_BROWSER="1"` | open a visible browser window, to watch it work |
| `LIVE_MAX_NEW_HOTELS` in settings.py | hotel pages per search (4, each about 5–10 s) |

In the chat page, the "Search Booking.com live when our data is not enough" box turns it on or off. From the
terminal, use `python recommend.py --live "..."`. Component 4 can call `POST /recommendations/live` when
`live_search.needed` is true in an answer.

## Files

```
src/main.py              the API
src/c2/settings.py       paths, rules and weights (change numbers here)
src/c2/data.py           loads the data once (about 3 seconds)
src/c2/c1_client.py      Component 1: DUMMY / file / http
src/c2/features.py       Gap 2 + 5: yes / no / not sure for each feature
src/c2/pricing.py        rooms, dates, fair price, what-if, links, free-cancellation advice
src/c2/ranker.py         the join: must-haves, score, explanation, where the fair price goes
src/c2/margins.py        Good deal / Looks high limits per town (price_margins.csv; else the fixed 30% rule)
src/c2/explain.py        SHAP: why the fair price is this number (exact, 11 groups, in rupees)
src/c2/unseen_models.py  makes the 3 cross-validation models again (the models that never saw each hotel)
src/c2/output.py         contract answer + Output 2 for Component 3
src/c2/live/             live Booking.com search: booking_pages.py (scraper's page readers), prep_rules.py
                         (preprocessing rules), to_rows.py (one hotel -> CSV rows), search.py (the flow)
src/ui/app/              the built hotel finder page (made from ui/ by npm run build)
src/ui/chat.html         the first, simpler chat page (http://localhost:8002/chat)
ui/src/                  the page's React code: App.jsx (searches, live search), components/Composer.jsx (the
                         question), Understood.jsx, TownResults.jsx, HotelCard.jsx, PriceGauge.jsx (fair-price
                         gauge), MoneySplit.jsx (where the fair price goes), PriceWhy.jsx (SHAP), styles.css (theme)
src/nlp/                 run_stage1.py + ft_prompt.txt (copied from the nlp_v2 kit) + understand.py
recommend.py             command-line demo
experiments/             must_break_check.py (Gap 1 proof) and the NLP test sentences
tests/                   run with pytest; uses a tiny made-up dataset, so it also runs in GitHub CI
```

## Gap 1 proof, first version (`python experiments\must_break_check.py`)

There are 32 test requests with must-haves where the town is in our data. For each one we looked at the top 5
hotels and counted how many break at least one must-have:

| Method | Hotels | Break a must-have |
|---|---|---|
| A. hotels in the town sorted by rating (like "top rated") | 155 | 82 (52.9%) |
| B. A + price filter | 144 | 17 (11.8%) |
| C. this system | 142 | 0 (0%) |

Feature checks use DUMMY C1 reviews, so only the budget and hospital numbers are final.

## Honest limits

- Component 1 scores are made up until Component 1 is ready.
- The prices are a snapshot (Nov 2026). Each hotel has prices for only some of the 8 dates, so the price for the
  tourist's dates is an estimate. A live system would check the sites at search time.
- We have hotels for 46 towns only. For other towns we search 10-25 km around the town, or say there are no hotels.
- The November data is Booking.com only, so "where to book" compares sites only with the September data.
- Hospital distance is a straight line, and the nearest hospital may not treat every condition. We say "near a
  hospital", never "safe". This is not medical advice.
- The price model's typical error is about 27% (September model: 26%), so a price is called "Potentially
  Overpriced" only when it is among the dearest 1 in 10 for its town compared with the fair price (Kandy: 78% or
  more above). The limits come from the weekly cloud model, which is retrained every Sunday from the same LightGBM
  recipe; this service's fair prices come from the laptop LightGBM model, so the limits are close but not exact.
- The money for each facility and the SHAP parts are the model's estimates, not the hotel's bill, so we say
  "about" and round to Rs 100. SHAP explains one room; a stay priced as several rooms is the sum of the rooms.
- The free-cancellation advice uses rough rainy months. It is a first version.