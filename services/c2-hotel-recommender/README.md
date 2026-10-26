# C2: Personalised hotel ranking and price intelligence

Owner: Hansada S.A.B. This service takes a tourist's request and returns ranked hotels. For each hotel it checks
the must-haves, the features, the fair price and where to book. It also sends what Component 3 needs.

```
tourist sentence
  -> Gap 1  NLP model (hotel-nlp-v2 in Ollama) + rule check  -> structured request (Output 1)
  -> hotels in the town (our 15 data towns, or hotels within 10-25 km of any other town)
  -> Gap 2  feature check: listing + Component 1 reviews     -> yes / no / not sure (and is it true NOW?)
  -> Gap 5  nearest hospital (OpenStreetMap), lift / steps, wheelchair access
  -> Gap 1  must-haves: a hotel that fails one is removed; "not sure" hotels come last and are marked
  -> Gap 3  fair price (price model) + what you pay for things you don't need or don't get
  -> score  wishes 40% + trusted rating 30% + value 20% + interests 10%  (weights in src/c2/settings.py)
  -> Gap 4  where to book: Booking.com vs Trip.com vs Agoda, same-deal rules
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
$env:C2_DATA_DIR="D:\Research\hotel_preprocessing\hotel_preprocessing"   # output_clean\ and models\
$env:C2_EXTRA_DIR="D:\Research"        # sl_hospitals_osm.csv, sl_places_osm.csv, hotel_urls.csv
```

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
price.explanation, free_cancellation`.
`for_c3` is Output 2 for Component 3. Health needs are in it only when `health_consent` is true, as categories only.

Prices are **per night for all rooms, taxes included**. They were collected on 25-27 September 2026 for 20
check-in dates. The service uses the collected date closest to the tourist's date in the year, then moves the
price to the tourist's dates with the price model. A price far from the hotel's own usual price is skipped.

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
of hotel.

**2. Why the fair price is this number (`GET /hotels/{hotel_id}/price-explanation`).** Exact SHAP in rupees over 11
groups (town, hotel class and reviews, room, pool, bar, beach, breakfast and meals, other facilities, your dates,
your group, booking terms): it starts from the average of 20 typical prices, and each group adds or takes away
rupees until it reaches the fair price exactly. Each answer has `price.explanation.url` (with `row_id`, `check_in`,
`nights`). It takes about 5-20 seconds, so it is made only when the tourist clicks **"Why this fair price?"**, and
then kept in memory. Two checks come with every answer: the parts add up to the fair price, and they are the same
as the textbook Shapley formula. Without the `shap` package the service uses the textbook formula (same numbers).

**The model that never saw the hotel.** A fair price must come from a model that had not seen the hotel. The
final model never saw the 222 test hotels and the live hotels, so it explains those. But it learned the 843
training hotels (its price for them is within about 3.5% of the real price), so for those we use the
cross-validation model that made their fair price. The training notebook did not save these 3 models, so
`src/c2/unseen_models.py` makes them again exactly (same rows, same hotel groups, same settings; the check gives
the same 10,359 fair prices to 0.000000%). The same model also moves the fair price to the tourist's dates.

```
python -m src.c2.unseen_models          # once, about 3-10 minutes -> models\booking_cv_models.joblib (do not commit)
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
- **Saved:** the new hotels go into `output_clean\live\*.csv` next to the research files (same columns).
  The next request gets them from the CSV, not from the web.
  - `hotels_booking_live.csv`: facts and facilities
  - `hotel_words_facilities_live.csv`: the full facility list
  - `prices_live.csv`: price for those dates, fair price and where to book
  - `hotel_photos.csv`: photos of every hotel, including September ones
  - `live_search_log.csv`: what was searched (no request details)
- **Fair price:** the price model gives the fair price. The new hotel was not in its training data.
- **Three sites:** Booking.com is live. Agoda and Trip.com are compared when the same hotel is in our September
  data (same name-matching rule as preprocessing). Their price is moved to the tourist's dates with the price
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
- The prices are a snapshot (Sep 2026). Each hotel has prices for only some of the 20 dates, so the price for the
  tourist's dates is an estimate. A live system would check the sites at search time.
- We have hotels for 15 towns only. For other towns we search 10-25 km around the town, or say there are no hotels.
- Hospital distance is a straight line, and the nearest hospital may not treat every condition. We say "near a
  hospital", never "safe". This is not medical advice.
- The price model's typical error is about 26%, so "Potentially Overpriced" is shown only above 30%.
- The money for each facility and the SHAP parts are the model's estimates, not the hotel's bill, so we say
  "about" and round to Rs 100. SHAP explains one room; a stay priced as several rooms is the sum of the rooms.
- The free-cancellation advice uses rough rainy months. It is a first version.
