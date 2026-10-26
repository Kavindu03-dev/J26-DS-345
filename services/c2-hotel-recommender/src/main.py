"""
Component 2 service: personalised hotel ranking and price intelligence.

    GET  /health                       is the service up?
    POST /recommendations              team contract (structured request from C4)        -> ranked hotels
    POST /recommendations/from-text    the tourist's sentence -> NLP (Ollama) -> ranking  -> ranked hotels + output for C3
    GET  /hotels/{hotel_id}            facts, feature checks and nearest hospital for one hotel
    POST /recommendations/live         not enough hotels in our data -> search Booking.com live, save, rank again
    GET  /hotels/{hotel_id}/photos     the hotel's photos (saved, or read once from Booking.com)
    GET  /hotels/{hotel_id}/price-explanation   why the fair price is this number (SHAP, in rupees; 5-20 s)
    GET  /app                          the hotel finder page (React, built into src/ui/app from the ui/ folder)
    GET  /chat                         the first, simpler chat page (src/ui/chat.html)

Run:  uvicorn src.main:app --port 8002     then open http://localhost:8002 (hotel finder) or /docs (API page)
"""
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .c2 import settings
from .c2.c1_client import make_c1
from .c2.data import DataMissing, get_data
from .c2.features import check_features
from .c2.output import contract_request_to_internal, plain, to_c3, to_contract
from .c2.ranker import recommend
from .c2.schemas import LiveRequest, RecommendationRequest, TextRequest

app = FastAPI(title="C2 Personalized Hotel Recommender",
              description="Hotel ranking with must-haves, feature checks, fair price and where to book. "
                          "Component 1 scores are DUMMY until Component 1 is ready (see c1_source in answers).")


@app.exception_handler(RequestValidationError)
async def explain_bad_request(request: Request, exc: RequestValidationError):
    """Same 422 answer as FastAPI, plus a hint in plain words."""
    errors = jsonable_encoder(exc.errors())
    missing = [e["loc"][-1] for e in errors if e.get("type") == "missing"]
    if request.url.path == "/recommendations" and "destination" in missing:
        hint = ("This endpoint needs structured fields (destination, check_in, ...). "
                "To send the tourist's sentence, use POST /recommendations/from-text with {\"text\": \"...\"}.")
    elif request.url.path == "/recommendations/from-text" and "text" in missing:
        hint = "Send the tourist's sentence as {\"text\": \"...\"}."
    else:
        hint = "Check the fields listed in 'detail'. Dates are YYYY-MM-DD."
    return JSONResponse(status_code=422, content={"detail": errors, "hint": hint})


def _data():
    try:
        return get_data()
    except DataMissing as e:
        raise HTTPException(status_code=503, detail=str(e))


UI_DIR = Path(__file__).with_name("ui")
APP_DIR = UI_DIR / "app"                      # made by "npm run build" in the ui/ folder
if (APP_DIR / "assets").is_dir():
    app.mount("/app/assets", StaticFiles(directory=APP_DIR / "assets"), name="app-assets")


@app.get("/", include_in_schema=False)
def home():
    """Opening http://localhost:8002 shows the hotel finder page (or the chat page if it is not built)."""
    return RedirectResponse("/app" if (APP_DIR / "index.html").exists() else "/chat")


@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
def app_page():
    """The hotel finder page (React): type a sentence, see the ranked hotels, the fair price and why."""
    index = APP_DIR / "index.html"
    if not index.exists():
        return RedirectResponse("/chat")
    return FileResponse(index, media_type="text/html")


@app.get("/chat", include_in_schema=False)
def chat_page():
    """Temporary demo page (src/ui/chat.html): type a sentence, see the ranked hotels."""
    return FileResponse(UI_DIR / "chat.html", media_type="text/html")


@app.get("/health")
def health():
    return {"service": "c2-hotel-recommender", "status": "ok", "c1_mode": settings.c1_mode()}


@app.post("/recommendations", summary="Ranked hotels from structured fields (for Component 4)")
def recommendations(body: RecommendationRequest):
    data = _data()
    request = contract_request_to_internal(body.model_dump())
    result = recommend(request, top_k=body.top_k, today=body.today, data=data)
    response = to_contract(result)
    response["for_c3"] = to_c3(request, result, body.health_consent)
    return plain(response)


@app.post("/recommendations/from-text", summary="Ranked hotels from the tourist's sentence (NLP + ranking)")
def recommendations_from_text(body: TextRequest):
    from .nlp.understand import understand      # only needed here (talks to Ollama)

    data = _data()
    try:
        request, notes, seconds, rate_info = understand(body.text, settings.nlp_model(), settings.ollama_url(),
                                                        today=body.today, live=body.live_rates)
    except Exception as e:  # Ollama not running, model missing ...
        raise HTTPException(status_code=503, detail=f"The NLP model could not be reached at {settings.ollama_url()} "
                                                    f"({type(e).__name__}: {str(e)[:200]}). Is Ollama running with "
                                                    f"'{settings.nlp_model()}'?")
    result = recommend(request, top_k=body.top_k, today=body.today, data=data)
    towns = [to_contract(result, i) for i in range(len(result["results"]))]
    return plain({
        "understood_request": request,                           # Output 1
        "nlp_notes": notes, "nlp_seconds": seconds,
        "exchange_rates": rate_info if any("converted" in n for n in notes) else None,
        "exchange_rates_attribution": settings.RATES_ATTRIBUTION if any("converted" in n for n in notes) else None,
        "ranked_hotels": towns[0]["ranked_hotels"] if towns else [],   # first town (team contract format)
        "towns": towns,                                          # every stay_in town
        "for_c3": to_c3(request, result, body.health_consent),   # Output 2
    })


@app.post("/recommendations/live", summary="Search Booking.com live for one town, save new hotels, rank again")
def recommendations_live(body: LiveRequest):
    from .c2.live.search import LiveUnavailable, run_live_search

    _data()
    if not settings.live_enabled():
        raise HTTPException(status_code=503, detail="Live search is switched off (C2_LIVE=0).")
    try:
        report = run_live_search(body.request, body.town, today=body.today)
    except LiveUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:                       # the live search must never break the answer
        import traceback
        traceback.print_exc()
        report = {"status": "failed", "message": f"The live search stopped ({type(e).__name__}: {str(e)[:160]}). "
                                                 "Showing hotels from our data only.", "new_hotels": [], "seconds": 0}
    request = dict(body.request, trip=dict(body.request.get("trip", {}), stay_in=[body.town]))
    result = recommend(request, top_k=body.top_k, today=body.today, data=get_data())
    town = to_contract(result)
    town["live_search"] = {"needed": False, "reason": "", "done": True}
    return plain({"town": town, "live_report": report, "for_c3": to_c3(request, result, body.health_consent)})


@app.get("/hotels/{hotel_id}/photos", summary="The hotel's photos (saved, or read once from its Booking.com page)")
def hotel_photos(hotel_id: str, fetch: bool = True):
    data = _data()
    if not fetch or not settings.live_enabled():
        return {"hotel_id": hotel_id, "photos": data.photos.get(hotel_id, []), "source": "saved"}
    from .c2.live.search import LiveUnavailable, fetch_photos
    try:
        photos, source = fetch_photos(hotel_id, data=data)
    except LiveUnavailable as e:
        return {"hotel_id": hotel_id, "photos": data.photos.get(hotel_id, []), "source": "saved", "note": str(e)}
    return {"hotel_id": hotel_id, "photos": photos, "source": source}


@app.get("/hotels/{hotel_id}/price-explanation",
         summary="Why is the fair price this number? SHAP in rupees, with the model that never saw the hotel")
def price_explanation(hotel_id: str, row_id: int | None = None, check_in: date | None = None, nights: int = 2,
                      adults: int = 2, children: int = 0, today: date | None = None):
    """Explains one room's fair price (a night). row_id is the price the ranking used (price.explanation.url);
    without it, the hotel's collected price closest to check_in for the group is used. Takes about 5-20 seconds
    the first time, then it is kept in memory."""
    from .c2.explain import explain_price
    from .c2.pricing import date_features, pick_stay

    data = _data()
    if hotel_id not in data.stay_index:
        raise HTTPException(status_code=404, detail="Unknown hotel_id, or no prices for it (use the Booking.com id).")
    if row_id is None:
        _, rows = pick_stay(data.stay_index[hotel_id], [(adults, children)], check_in)
        if not rows:
            raise HTTPException(status_code=404, detail="No collected price for this group.")
        row_id = int(rows[(adults, children)]["row_id"])
    elif row_id not in data.model_rows.index or data.model_rows.loc[[row_id]].booking_hotel_id.iloc[0] != hotel_id:
        raise HTTPException(status_code=404, detail="This row_id is not a price of this hotel.")
    changes = date_features(check_in, nights, today or date.today()) if check_in else {}
    return plain(explain_price(data, row_id, changes, check_in.isoformat() if check_in else None))


@app.get("/hotels/{hotel_id}", summary="One hotel: feature checks, nearest hospital, C1 scores")
def hotel_details(hotel_id: str):
    data = _data()
    if hotel_id not in data.hotels.index:
        raise HTTPException(status_code=404, detail="Unknown hotel_id (use the Booking.com id)")
    h = data.hotels.loc[hotel_id]
    c1 = make_c1(data.hotels)
    scores = c1.get(hotel_id)
    num = lambda x: None if x != x else float(x)      # NaN -> None
    return plain({"hotel_id": hotel_id, "name": h["hotel_name"], "town": h["destination_cluster"],
            "latitude": num(h["latitude"]), "longitude": num(h["longitude"]),
            "booking_rating": num(h["page_rating"]),
            "nearest_hospital": {"name": h["nearest_hospital"], "km": num(h["nearest_hospital_km"])},
            "feature_checks": check_features(h, scores), "review_scores_c1": scores, "c1_source": c1.source})
