# C4: Tourism agent (function calling)

Owner: Gajanayaka K.R.K.V. (IT23373884)

A fine-tuned small language model that decides which component to call
(C1 reviews, C2 hotels, C3 itinerary, or web search) and combines their
answers into one travel plan. This folder also contains mock versions of
C1-C3 so the agent can be built and tested before every real service is ready.

## Layout

```
src/
  config.py            settings (URLs, timeouts, data paths) - all overridable by env vars
  data_access.py       read-only loaders for C1's published data (hotels, profiles, attractions)
  hotel_directory.py   hotel name -> hotel_id lookup (users type names, contracts use IDs)
  geo.py               distance and drive-time helpers
  tools/clients.py     HTTP clients for C1, C2, C3 with uniform error codes
  mocks/
    c1_mock.py         GET  /hotels/{hotel_id}/aspect-scores  (serves C1's real profiles)
    c2_mock.py         POST /recommendations                  (real hotels + C1 scores, simulated prices)
    c3_mock.py         POST /plan-itinerary                   (real attractions, greedy route, mock restaurants)
    run_mocks.py       starts all three mocks
training/              Step 4 data preparation (see training/README.md)
schemas/               C4's own record of C3's real API (not a shared contract)
tests/                 pytest suite (contract checks, mocks, clients, hotel lookup)
```

## Mocks: what is real and what is simulated

| Mock | Real | Simulated | Port |
|------|------|-----------|------|
| C1 | All aspect scores, trust scores, verified facts (from `services/c1-review-engine/data`) | nothing | 9001 |
| C2 | Hotel names, locations, C1 scores used for ranking | Prices and platform offers (deterministic per hotel) | 9002 |
| C3 | Attraction names, categories, coordinates | Route (greedy, not TOPTW), restaurants | 9003 |

Every C2 and C3 mock response contains `"source": "c4-mock"` so mock output is never mistaken for real output.
Mock ports (900x) never clash with the real services (800x).

## Run

Windows PowerShell, from this folder (`services/c4-agent`):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

pytest                          # run all tests
python -m src.mocks.run_mocks   # start the three mocks, then open http://localhost:9002/docs
```

## Contract notes

The mocks are checked with jsonschema in the tests:

- C1 mock: `contracts/c1_aspect_scores.json` (C1's real service matches it).
- C2 mock: `contracts/c2_hotel_ranking.json` (draft; C2 not implemented yet).
- C3 mock: `schemas/c3_plan_itinerary.json`, C4's own record of C3's real `POST /plan-itinerary`.
  The shared `contracts/c3_itinerary.json` is an older draft (`POST /itineraries`) that C3 did not build.
  Updating the shared contracts is a team decision; C4 does not change files outside this folder.
