# Component 3 conversation demo

Requirements: Node.js 20.19+ or 22.12+, C3 on port 8003, OSRM on port 5000,
and the existing pipeline datasets/dependencies.

```powershell
cd D:\RP\J26-DS-345\services\c3-itinerary-planner\frontend
npm install
npm run dev
```

Open the Vite URL. Choose a **Demo Conversation Scenario**, explicitly select a
hotel, then click **Generate Optimized Itinerary**. No hotel is selected and no
request is sent on initial load. Changing scenario or hotel clears all results.
The flow represents Conversation → C2 Recommendations → Hotel Selection → C3
Planning → Results. Conversation text and hotel lists are predefined examples;
there is no live C4 chatbot or C2 API call.

`src/data/demoScenarios.js` contains actual C2-shaped `for_c3` examples, with two
hotel choices per area. Hotel names, IDs, towns and coordinates are copied from
`services/c1-review-engine/data/hotel_aspect_profiles.csv` (repository-relative).
Demo order is predefined, not a computed C2 ranking. Pricing, booking metadata,
photos and hospital distances are null because verified values are unavailable.
The shared three-night dates and party/interests are fictional traveler inputs,
not facts about hotels. No datasets were edited.

The client submits `{for_c3, selected_hotel_id}` to `/api/plan-itinerary`.
The C3 adapter generates the trip ID and uses its existing interest mapping and
restaurant defaults. Driving and 09:00–18:00 are explicitly C3 defaults. Health
needs are not shared in these examples; medical optimization remains off.
See [C2 handoff contract](../C2_HANDOFF.md) for the bridge rules.

The Vite proxy forwards `/api` to `http://127.0.0.1:8003` (localhost over IPv4),
with a five-minute pipeline timeout. `npm run build` writes ignored `dist/`;
production hosting needs an equivalent reverse proxy. Requests are serialized
by the existing backend lock.

All map, itinerary, metric and restaurant panels use the latest API response.
Day tabs update the selected route and details. Missing route metrics show N/A;
missing geometry leaves markers only. Optional restaurant detours are excluded
from route totals. ABSA aspects are shown only when present, never as invented
zeros. Medical indicators describe access, not safety or guaranteed response.
OpenStreetMap tiles and optional Google Fonts require internet access.
