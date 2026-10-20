"""Build the tourism test set files from evaluation/testset_items.py.

    python -m evaluation.build_testset

Checks every gold call against the tool schemas (src/tools/definitions.py),
checks that every named hotel exists in Component 1's data, fills in TOP
(the hotel a multi-step question must continue with) using the mock C2
service, and writes:
    evaluation/data/tourism_testset_v1.jsonl   used by the evaluation code
    evaluation/data/tourism_testset_v1.csv     for reading / reviewing in Excel
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.hotel_directory import HotelDirectory
from src.mocks.c2_mock import RecommendationRequest, rank_hotels
from src.tools.definitions import TOOLS_VERSION, to_c2_request, validate_call

from .testset_items import ITEMS, TOP, alt, opt

OUT_DIR = Path(__file__).resolve().parent / "data"
JSONL = OUT_DIR / "tourism_testset_v1.jsonl"
CSV = OUT_DIR / "tourism_testset_v1.csv"

# Arguments that cannot be judged exactly; allowed even when the gold does not list them.
FREE_ARGS = {"recommend_hotels": {"interests", "top_k"}, "web_search": {"query", "max_results"}}


class TestsetError(ValueError):
    pass


def gold_arg(value) -> dict:
    optional = isinstance(value, opt)
    v = value.value if optional else value
    accept = list(v) if isinstance(v, alt) else [v]
    return {"accept": accept, "optional": optional}


def concrete(arguments: dict) -> dict:
    """The first accepted value of every argument (used for validation and mock calls)."""
    return {k: spec["accept"][0] for k, spec in arguments.items()}


def build_call(name: str, raw_args: dict, top: dict | None) -> dict:
    args = {}
    for k, v in raw_args.items():
        if v == TOP:
            if top is None:
                raise TestsetError("TOP used in a step that has no recommend_hotels before it")
            v = alt(top["name"], top["hotel_id"])
        args[k] = gold_arg(v)
    return {"name": name, "arguments": args, "free": sorted(FREE_ARGS.get(name, set()) - set(args))}


def tool_message(call: dict) -> list[dict]:
    """Assistant tool-call message + tool result for a gold step (multi-step context)."""
    if call["name"] != "recommend_hotels":
        raise TestsetError("only recommend_hotels may precede another step in v1")
    args = {k: v for k, v in concrete(call["arguments"]).items() if not call["arguments"][k]["optional"]}
    result = rank_hotels(RecommendationRequest(**to_c2_request(args)))
    return [result]


def build() -> list[dict]:
    directory = HotelDirectory()
    records, seen = [], set()
    for it in ITEMS:
        if it["id"] in seen:
            raise TestsetError(f"duplicate id {it['id']}")
        seen.add(it["id"])
        steps, after, top = [], [], None
        for i, step in enumerate(it["steps"]):
            calls = [build_call(name, raw, top) for name, raw in step]
            for c in calls:
                args = concrete(c["arguments"])
                if c["name"] == "web_search":
                    args["query"] = "placeholder"
                errors = validate_call(c["name"], args)
                if errors:
                    raise TestsetError(f"{it['id']} step {i + 1} {c['name']}: {errors}")
                name = c["arguments"].get("hotel", {}).get("accept", [None])[0]
                if name and not (top and name == top["name"]):  # a hotel the user named: must exist
                    match = directory.find(name, limit=1)
                    if not match or match[0].score < 0.8:
                        raise TestsetError(f"{it['id']}: hotel '{name}' not found in C1 data")
            steps.append({"calls": calls})
            if i < len(it["steps"]) - 1:
                if len(calls) != 1:
                    raise TestsetError(f"{it['id']}: only single-call steps may be followed by another step")
                (result,) = tool_message(calls[0])
                if not result["ranked_hotels"]:
                    raise TestsetError(f"{it['id']}: mock C2 returned no hotels for step {i + 1}")
                top = result["ranked_hotels"][0]
                gold_args = concrete({k: v for k, v in calls[0]["arguments"].items() if not v["optional"]})
                after.append([
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": calls[0]["name"], "arguments": gold_args}}]},
                    {"role": "tool", "name": calls[0]["name"], "content": json.dumps(result, ensure_ascii=False)},
                ])
        records.append({
            "id": it["id"],
            "category": it["category"],
            "messages": it["history"] + [{"role": "user", "content": it["user"]}],
            "steps": steps,
            "context_after_step": after,
            "notes": it["notes"],
            "tools_version": TOOLS_VERSION,
        })
    return records


def readable(steps: list[dict]) -> str:
    if not steps:
        return "NO TOOL CALL"
    parts = []
    for i, s in enumerate(steps, 1):
        calls = []
        for c in s["calls"]:
            args = []
            for k, spec in c["arguments"].items():
                val = " | ".join(json.dumps(a, ensure_ascii=False) for a in spec["accept"])
                args.append(f"{k}={val}" + (" (optional)" if spec["optional"] else ""))
            calls.append(f"{c['name']}({', '.join(args)})")
        parts.append(f"Step {i}: " + " + ".join(calls))
    return "\n".join(parts)


def main() -> None:
    records = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "conversation_before", "user_message", "expected_answer", "notes"])
        for r in records:
            history = "\n".join(f"{m['role']}: {m['content']}" for m in r["messages"][:-1])
            w.writerow([r["id"], r["category"], history, r["messages"][-1]["content"], readable(r["steps"]), r["notes"]])
    counts = Counter(r["category"] for r in records)
    print(f"Wrote {len(records)} items to {JSONL.name} and {CSV.name}")
    for k, v in counts.items():
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
