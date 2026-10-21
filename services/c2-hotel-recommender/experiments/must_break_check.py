"""
Gap 1 proof (first version): how often do the top hotels break the tourist's must-haves?

For every test request that has must-haves (the 46 NLP test sentences, using their correct understood
request), the top 5 hotels of three methods are checked:
    A. plain search      hotels in the town, sorted by Booking.com rating (like "top rated" on a booking site)
    B. price filter      A + only hotels under the per-night budget (like the price filter on a booking site)
    C. our system        must-haves checked one by one (listing + Component 1 reviews + hospital distance)

A must-have is "broken" when the hotel's price is above the budget, the nearest hospital is more than 5 km away,
or the feature check says "no" (not listed, or reviews say it is not there). "Not sure" is counted separately.

    python experiments/must_break_check.py            (needs the real data; see README.md)

Honest limits: feature checks use Component 1's scores, which are DUMMY until Component 1 is ready, so only the
budget and hospital numbers are final. A plain AI chat search is the next baseline to add.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.c2.c1_client import make_c1                       # noqa: E402
from src.c2.data import get_data                            # noqa: E402
from src.c2.features import check_features                  # noqa: E402
from src.c2.pricing import room_plan                        # noqa: E402
from src.c2.ranker import find_hotels, price_hotels, recommend   # noqa: E402
from src.nlp.run_stage1 import finalize_gold                # noqa: E402
from src.nlp.understand import find_not_needed              # noqa: E402

HERE = Path(__file__).resolve().parent
TOP = 5


def load_requests():
    out = []
    for name in ("test_sentences.jsonl", "hard_tests.jsonl"):
        for line in (HERE / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                g = finalize_gold(r["gold"])
                g["not_needed"] = find_not_needed(r["sentence"])
                out.append((r["id"], r.get("today", "2026-10-19"), g))
    return out


def must_result(m, check, price, budget):
    if m == "budget":
        return "yes" if price["price_per_night_lkr"] <= budget else "no"
    return check[m]["status"]


def main():
    data = get_data()
    c1 = make_c1(data.hotels)
    tally = {k: Counter() for k in ("A. plain search (by rating)", "B. + price filter", "C. our system")}
    by_must = {k: Counter() for k in tally}
    n_requests = 0
    for rid, today, req in load_requests():
        must = req["hotel"]["must"]
        if not must:
            continue
        trip = req["trip"]
        from datetime import date
        check_in = date.fromisoformat(trip["check_in"]) if trip["check_in"] else None
        budget = req["hotel"]["budget_max_lkr"]
        rooms, _ = room_plan(trip["adults"], trip["children"])
        for town in trip["stay_in"]:
            cand, _ = find_hotels(data, town, TOP)
            if not len(cand):
                continue
            n_requests += 1
            c1s = {h: c1.get(h) for h in cand.index}
            checks = {h: check_features(cand.loc[h], c1s[h]) for h in cand.index}
            prices = price_hotels(data, cand, rooms, check_in, trip["nights"], date.fromisoformat(today),
                                  req.get("not_needed", []), checks)
            priced = [h for h in cand.index if h in prices]
            by_rating = sorted(priced, key=lambda h: -(cand.loc[h, "page_rating"] if cand.loc[h, "page_rating"] ==
                                                       cand.loc[h, "page_rating"] else 0))
            lists = {
                "A. plain search (by rating)": by_rating[:TOP],
                "B. + price filter": [h for h in by_rating if not budget or
                                      prices[h]["price_per_night_lkr"] <= budget][:TOP],
            }
            ours = recommend({**req, "trip": {**trip, "stay_in": [town]}}, top_k=TOP, today=today, data=data, c1=c1)
            lists["C. our system"] = [h["hotel_id"] for h in ours["results"][0]["ranked_hotels"]]
            for method, hotels in lists.items():
                for h in hotels:
                    results = {m: must_result(m, checks[h], prices[h], budget) for m in must}
                    t = tally[method]
                    t["hotels"] += 1
                    t["breaks a must-have"] += any(v == "no" for v in results.values())
                    t["a must-have not sure"] += (not any(v == "no" for v in results.values()) and
                                                  any(v == "not sure" for v in results.values()))
                    for m, v in results.items():
                        by_must[method][(m, v)] += 1

    print(f"{n_requests} requests with must-haves (town in our data), top {TOP} hotels each\n")
    print(f"{'method':32s} {'hotels':>7s} {'break a must':>13s} {'not sure':>9s}")
    for method, t in tally.items():
        n = max(t["hotels"], 1)
        print(f"{method:32s} {t['hotels']:7d} {t['breaks a must-have']:6d} ({100 * t['breaks a must-have'] / n:4.1f}%)"
              f" {100 * t['a must-have not sure'] / n:8.1f}%")
    print("\nbreaks per must-have (hotels that break it / hotels checked for it):")
    musts = sorted({m for c in by_must.values() for (m, _) in c})
    for m in musts:
        row = []
        for method, c in by_must.items():
            checked = sum(v for (mm, _), v in c.items() if mm == m)
            row.append(f"{c[(m, 'no')]}/{checked}")
        print(f"  {m:18s} " + "   ".join(f"{method[:2]} {r:>7s}" for method, r in zip(by_must, row)))


if __name__ == "__main__":
    main()
