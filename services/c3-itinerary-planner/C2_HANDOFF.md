# Latest C2 handoff to C3

Confirmed against `services/c2-hotel-recommender/src/c2/output.py::to_c3` and
`src/main.py` at 6f69e10. POST /recommendations, /recommendations/from-text and
/recommendations/live each expose the following `for_c3` member:

```text
for_c3:
  trip: {stay_in[], want_to_visit[], check_in, check_out, nights, dates_fixed,
         adults, children, travel_style, budget_total_lkr}
  interests[]
  health: {consent_given, needs[] only when consent_given is true}
  recommended_hotels[]:
    {rank, hotel_id, name, town, stay_in, latitude, longitude,
     price_per_night_lkr, nights, total_hotel_cost_lkr, budget_left_lkr,
     price_label, book_on, booking_url, nearest_hospital_km, photo_url}
```

C2 copies its trip object; fields depend on upstream input and may be null or
absent. Dates may be null while nights remains usable. Hotel coordinates may be
null. The hotel array can be empty and can include multiple towns. C2 hospital
kilometers are straight-line distance, not C3 OSRM driving minutes.

POST /plan-itinerary additionally accepts:

```json
{"for_c3": {"trip": {}, "interests": [], "health": {"consent_given": false}, "recommended_hotels": []}, "selected_hotel_id": "explicit-id"}
```

This is a shape illustration, not a valid planning request. Choose exactly one
matching recommendation, with a nonempty name and finite numeric coordinates
within latitude/longitude ranges. No automatic first/rank-one hotel choice is
made. Unknown/missing IDs, duplicate matching IDs, empty lists and unavailable
coordinates return HTTP 422 before pipeline files are written.

District mapping uses project hotel city/district values: Ella → Badulla,
Kandy → Kandy, Colombo → Colombo, Galle/Galle Fort → Galle. Only the selected
hotel's town (or stay_in if town is absent) determines its district. Unknown
towns fail clearly; the full multi-town stay context is retained, but C3 still
plans from one selected hotel, not a multi-base hotel itinerary.

Valid ISO check-in/check-out dates determine planning days by date difference,
matching existing C3 behavior. Otherwise a positive integer **trip.nights** is
used as the same number of planning days; no checkout day is added. Missing or
invalid dates and nights fail; stays outside 1–14 planning days fail. Null
optional context is omitted. Dates, nights, party, travel style, stay context,
budget and interests are retained as metadata, with num_days_source indicating
which day-count rule was used. want_to_visit is preserved in the trip.

Each valid normalization generates `C3_` plus UUID4 hex, without global counters.
C3 supplies driving, day_start_hour=9, day_end_hour=18. Existing interest mapping
sets matching categories to 5 and others to 1; formulas are unchanged. Restaurant
preferences are a fresh copy of the existing C3 defaults, not C2 fields.

When consent is false, health needs are ignored (even if accidentally supplied).
With consent, supported heart/mobility/wheelchair/breathing/elderly values are
retained in context. Mobility/wheelchair mark reduced mobility; these and elderly
mark prefer_low_walking. This metadata does not alter optimizer/scoring logic.
Medical optimization stays off unless an explicit top-level C3 `medical_support`
override requests it; consent never activates it. Hotel price, booking, photo,
rank and hospital fields do not enter C3 scores or route-time metrics.

Native C3 payloads and the earlier selected_hotel C2 payload retain their existing
validation and behavior. Response shape and existing route freshness/locking
rules are unchanged.
