"""Executor: runs the model's tool calls against C1, C2, C3 and web search.

Responsibilities (proposal: "tool executor validates calls against the schemas,
dispatches them over REST, handles errors"):
  1. strict schema validation of every call (no repair)
  2. translation to each component's real request (hotel names -> IDs, days -> dates)
  3. calling the component through ToolClients (mocks or real services)
  4. compacting the response so a small model's context stays short
  5. turning every failure into a structured error the model can react to
Tool outputs are data: they are passed back as JSON in a tool message and are
never executed or followed as instructions (NFR3).
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

from ..hotel_directory import HotelDirectory
from .clients import ErrorCode, ToolClients, ToolError
from .definitions import to_c2_request, to_c3_request, validate_call

WebSearch = Callable[[str, int], list[dict]]

_trip_ids = itertools.count(1)


def no_web_search(query: str, max_results: int) -> list[dict]:
    """Default web search: clearly labelled as not connected (no invented facts)."""
    return [{
        "title": "Web search not connected in this demo",
        "snippet": f"No live web results for: {query!r}. Check an official source such as sltda.gov.lk or railway.gov.lk.",
        "url": None,
        "source": "web (not connected)",
    }]


class ToolExecutor:
    def __init__(self, clients: ToolClients, directory: HotelDirectory | None = None,
                 web_search: WebSearch = no_web_search):
        self.clients = clients
        self.directory = directory or HotelDirectory()
        self.web_search = web_search

    # ------------------------------------------------------------------ helpers
    def resolve_hotel(self, ref: str) -> dict:
        """Hotel name or hotel_id -> {"hotel_id", "name", "latitude", "longitude", "district"}."""
        hotel = self.directory.get(ref)
        if hotel is None:
            matches = self.directory.find(ref, limit=3, min_score=0.35)  # low bar for suggestions only
            if not matches or matches[0].score < 0.75:
                hint = ", ".join(m.hotel.name for m in matches) or None
                raise ToolError(ErrorCode.NOT_FOUND, f"No hotel called '{ref}' in our data."
                                + (f" Did you mean: {hint}?" if hint else ""))
            hotel = matches[0].hotel
        return {"hotel_id": hotel.hotel_id, "name": hotel.name, "latitude": hotel.latitude,
                "longitude": hotel.longitude, "district": hotel.district}

    # ------------------------------------------------------------------ tools
    def get_review_insights(self, args: dict) -> dict:
        hotel = self.resolve_hotel(args["hotel"])
        r = self.clients.get_aspect_scores(hotel["hotel_id"])
        aspects = args.get("aspects") or list(r["aspect_scores"])
        return {
            "hotel": hotel["name"],
            "hotel_id": hotel["hotel_id"],
            "aspect_scores": {a: r["aspect_scores"].get(a) for a in aspects},
            "overall_trust_score": r["overall_trust_score"],
            "time_decay_rating": r["time_decay_rating"],
            "verified_facts": r.get("verified_aspect_facts", [])[:4],
            "source": "C1 trust-aware reviews",
        }

    def recommend_hotels(self, args: dict) -> dict:
        r = self.clients.recommend_hotels(to_c2_request(args))
        hotels = []
        for h in r["ranked_hotels"]:
            loc = self.directory.get(h["hotel_id"])
            hotels.append({
                "rank": h["rank"], "hotel_id": h["hotel_id"], "name": h["name"], "score": h["score"],
                "best_price_lkr": min(o["price_lkr"] for o in h["price"]["offers"]),
                "best_platform": h["price"]["best_platform"], "price_status": h["price"]["price_status"],
                "explanation": h["explanation"], "district": loc.district if loc else None,
            })
        out = {"destination": args["destination"], "hotels": hotels, "source": "C2 hotel recommender"}
        if r.get("source"):
            out["note"] = "simulated prices from the mock C2 service"
        return out

    def plan_itinerary(self, args: dict) -> dict:
        hotel = self.resolve_hotel(args["hotel"])
        r = self.clients.plan_itinerary(to_c3_request(args, hotel, trip_id=f"C4-{next(_trip_ids):05d}"))
        days = [{
            "day": d["day"],
            "stops": [{"time": s.get("arrival_time"), "name": s["name"]} for s in d["stops"]],
            "restaurants": [{"name": x["name"], "cuisine": x.get("cuisine"), "detour_min": x.get("detour_time_min")}
                            for x in d["recommended_restaurants"][:2]],
        } for d in r["days"]]
        return {"hotel": hotel["name"], "trip": r.get("trip"), "days": days, "source": "C3 itinerary planner"}

    def web_search_tool(self, args: dict) -> dict:
        return {"query": args["query"], "results": self.web_search(args["query"], int(args.get("max_results", 3))),
                "source": "web search (labelled as external)"}

    # ------------------------------------------------------------------ dispatch
    def execute(self, name: str, arguments: dict) -> dict:
        """Run one call. Always returns a dict: the result, or {"error": {...}}."""
        errors = validate_call(name, arguments)
        if errors:
            return ToolError(ErrorCode.INVALID_ARGUMENT, "; ".join(errors[:3])).to_dict()
        handler = {
            "get_review_insights": self.get_review_insights,
            "recommend_hotels": self.recommend_hotels,
            "plan_itinerary": self.plan_itinerary,
            "web_search": self.web_search_tool,
        }[name]
        try:
            return handler(arguments)
        except ToolError as e:
            return e.to_dict()


def tool_message(name: str, result: dict) -> dict:
    return {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)}


def assistant_calls_message(calls: list[dict[str, Any]]) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}
