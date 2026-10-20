"""Synthesiser: turns tool results into the final answer.

Deterministic and grounded: every hotel, price, score, review fact and stop in
the answer comes from a tool result, so the answer cannot invent hotels or
reviews (the hallucination risk the team's proposals all raise). Verified
review facts from C1 are quoted, and web results are labelled as external.
"""

from __future__ import annotations

ASPECT_LABEL = {"food": "Food", "staff": "Staff", "room": "Rooms", "cleanliness": "Cleanliness",
                "location": "Location", "wifi": "Wi-Fi"}


def _reviews(r: dict) -> list[str]:
    scores = ", ".join(f"{ASPECT_LABEL.get(a, a)} {v}/5" for a, v in r["aspect_scores"].items() if v is not None)
    out = [f"**{r['hotel']}**: {scores}. Trust score {r['overall_trust_score']}/10, recent rating {r['time_decay_rating']}/5."]
    for fact in r.get("verified_facts", [])[:2]:
        out.append(f'> "{fact}" *(verified guest review)*')
    return out


def _hotels(r: dict) -> list[str]:
    out = [f"**Hotels in {r['destination']}**" + (f" ({r['note']})" if r.get("note") else "") + ":"]
    for h in r["hotels"]:
        flag = " - listed above its fair price" if h["price_status"] == "Potentially Overpriced" else ""
        out.append(f"{h['rank']}. **{h['name']}** - LKR {h['best_price_lkr']:,.0f}/night on {h['best_platform']}{flag}. {h['explanation']}")
    if not r["hotels"]:
        out.append("No hotels matched those requirements. Try a wider budget or a nearby town.")
    return out


def _itinerary(r: dict) -> list[str]:
    out = [f"**Itinerary from {r['hotel']}:**"]
    for d in r["days"]:
        stops = "; ".join(f"{s['time']} {s['name']}" for s in d["stops"]) or "free day"
        food = ", ".join(x["name"] for x in d["restaurants"])
        out.append(f"- **Day {d['day']}:** {stops}" + (f". Eat at: {food}" if food else ""))
    return out


def _web(r: dict) -> list[str]:
    out = [f"**From the web** (external source, not verified by our team) for *{r['query']}*:"]
    for x in r["results"]:
        out.append(f"- {x['title']}: {x['snippet']}" + (f" ({x['url']})" if x.get("url") else ""))
    return out


RENDER = {"get_review_insights": _reviews, "recommend_hotels": _hotels, "plan_itinerary": _itinerary, "web_search": _web}


def render(results: list[tuple[str, dict]]) -> str:
    """results: [(tool_name, result_dict)] in call order."""
    parts: list[str] = []
    for name, res in results:
        if "error" in res:
            parts.append(f"Sorry, I couldn't complete that step: {res['error']['message']}")
        else:
            parts.extend(RENDER[name](res))
        parts.append("")
    return "\n".join(parts).strip()
