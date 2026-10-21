"""
Demo from the command line: a tourist's sentence -> understood request -> ranked hotels -> output for Component 3.

    python recommend.py "We are a family of 4, Kandy from 26 to 29 December, under Rs 20,000 a night, pool please"
    python recommend.py                          (type sentences one after another; empty line = stop)
    python recommend.py --request my_request.json  (skip the NLP: a request in Output 1 format)
    python recommend.py --json "..."             (print the full JSON answer)
    python recommend.py --live "..."             (search Booking.com live when our data is not enough)

Needs Ollama running with hotel-nlp-v2 (only for sentences), and the data folders (see README.md).
"""
import argparse
import json
import sys
import time

from src.c2 import settings
from src.c2.data import get_data
from src.c2.output import plain, to_c3, to_contract
from src.c2.ranker import recommend

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def show(request, result, consent):
    print("\n--- what the system understood (Output 1) ---")
    for part in ("trip", "hotel"):
        for k, v in request[part].items():
            print(f"  {k:18} {json.dumps(v, ensure_ascii=False)}")
    for k in ("health", "interests", "not_needed"):
        print(f"  {k:18} {json.dumps(request.get(k, []))}")
    print(f"\nComponent 1 scores: {result['c1_source']}")
    for r in result["results"]:
        print(f"\n=== {r['town']}: {r['hotels_considered']} hotels looked at, removed: {r['removed']}")
        if r["live_search"]["needed"]:
            print("  live search would help:", r["live_search"]["reason"], "(add --live)")
        for n in r["notes"]:
            print("  note:", n)
        for h in r["ranked_hotels"]:
            p = h["price"]
            total = f", {h['nights']} nights = Rs {h['total_hotel_cost_lkr']:,}" if h["total_hotel_cost_lkr"] else ""
            left = f", budget left Rs {h['budget_left_lkr']:,}" if h["budget_left_lkr"] is not None else ""
            print(f"\n {h['rank']}. {h['name']}  (score {h['score']:.2f})"
                  + ("  [found live on Booking.com]" if h["found_live"] else ""))
            print(f"    Rs {p['price_per_night_lkr']:,} a night on {p['best_platform']}{total}{left}")
            print(f"    {p['price_status']} | sites (check-in {p['seen_checkin']}): {p['offers_seen']}")
            print(f"    nearest hospital: {h['nearest_hospital']['km']} km ({h['nearest_hospital']['name']})")
            print(f"    why: {h['explanation']}")
            print(f"    book: {h['booking_url']}")
    for n in result["data_notes"]:
        print("data note:", n)
    c3 = to_c3(request, result, consent)
    print("\n--- sent to Component 3 (Output 2), first hotel ---")
    print(json.dumps(plain({**c3, "recommended_hotels": c3["recommended_hotels"][:1]}), indent=1,
                     ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence", nargs="?")
    ap.add_argument("--request", help="JSON file in Output 1 format (no NLP)")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--today", default=None)
    ap.add_argument("--consent", action="store_true", help="tourist agreed to share health needs with C3")
    ap.add_argument("--json", action="store_true", help="print the full JSON answer")
    ap.add_argument("--fixed-rates", action="store_true")
    ap.add_argument("--live", action="store_true", help="search Booking.com live when our data is not enough")
    args = ap.parse_args()

    t = time.time()
    get_data()
    print(f"data loaded in {time.time() - t:.1f} s | Component 1 mode: {settings.c1_mode()}")

    def run(request):
        result = recommend(request, top_k=args.top, today=args.today)
        if args.live:
            from src.c2.live.search import run_live_search
            for r in result["results"]:
                if r["live_search"]["needed"]:
                    print(f"\n{r['live_search']['reason']} Searching Booking.com live for {r['town']} ...")
                    report = run_live_search(request, r["town"], today=args.today)
                    print(report["message"], f"({report.get('seconds', '?')} s)")
            result = recommend(request, top_k=args.top, today=args.today)
        if args.json:
            print(json.dumps(plain({"understood_request": request, "towns": [
                to_contract(result, i) for i in range(len(result["results"]))],
                "for_c3": to_c3(request, result, args.consent)}), indent=1, ensure_ascii=False))
        else:
            show(request, result, args.consent)

    if args.request:
        with open(args.request, encoding="utf-8") as f:
            run(json.load(f))
        return

    from src.nlp.understand import understand

    def one(sentence):
        request, notes, seconds, rate_info = understand(sentence, settings.nlp_model(), settings.ollama_url(),
                                                        today=args.today, live=not args.fixed_rates)
        print(f"(understood in {seconds} s; rules: {notes or 'no changes'})")
        if any("converted" in n for n in notes):
            print(f"(exchange rates: {rate_info}; {settings.RATES_ATTRIBUTION})")
        run(request)

    if args.sentence:
        one(args.sentence)
        return
    while True:
        try:
            s = input("\nTourist: ").strip()
        except EOFError:
            break
        if not s:
            break
        one(s)


if __name__ == "__main__":
    main()
