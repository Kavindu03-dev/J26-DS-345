"""Mock of Component 1 (trust-aware review analysis).

Contract: contracts/c1_aspect_scores.json
Endpoint: GET /hotels/{hotel_id}/aspect-scores

This mock serves the REAL pre-computed profiles that Component 1 publishes in
services/c1-review-engine/data/hotel_aspect_profiles.json, so C4 can be
developed without running C1's full stack. Behaviour on unknown hotels (404)
matches the real service.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..data_access import load_aspect_profiles

app = FastAPI(
    title="C1 mock - aspect scores",
    description="Serves Component 1's published profiles. For C4 development only.",
    version="0.1.0",
)


def get_aspect_scores(hotel_id: str) -> dict:
    """Return the C1 profile for a hotel, or raise KeyError."""
    return load_aspect_profiles()[hotel_id]


@app.get("/health")
def health():
    return {"service": "c1-mock", "status": "ok", "indexed_hotels": len(load_aspect_profiles())}


@app.get("/hotels/{hotel_id}/aspect-scores")
def aspect_scores(hotel_id: str):
    try:
        return get_aspect_scores(hotel_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Hotel '{hotel_id}' not found in seed dataset.")
