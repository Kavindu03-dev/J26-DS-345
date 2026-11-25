"""Export the offline demo's scenarios by running the REAL C4 agent code.

    python -m demo-ui.tools.export_scenarios      (run from services/c4-agent)
    -> demo-ui/src/scenarios.json

The router (the fine-tuned model) is replaced by a *scripted* router that emits
the test set's correct tool calls, so the demo needs no GPU. Everything after
the router is the real code: strict parsing, schema validation, hotel-name
resolution, the C1-C3 mock services (C1 uses Component 1's published data),
and the synthesiser that writes the answer from the tool results.

The decoding-modes view uses illustrative model outputs (clearly labelled in
the UI); they are scored with the real parser, validator and scoring code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # services/c4-agent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from evaluation.decoding import MODES, PROMPT_JSON_INSTRUCTION, REASON_INSTRUCTION, strip_reasoning  # noqa: E402
from evaluation.scoring import score_tourism_step  # noqa: E402
from src.agent import Agent  # noqa: E402
from src.mocks.c1_mock import app as c1  # noqa: E402
from src.mocks.c2_mock import app as c2  # noqa: E402
from src.mocks.c3_mock import app as c3  # noqa: E402
from src.tools.clients import ToolClients  # noqa: E402
from src.tools.definitions import SYSTEM_PROMPT, TOOLS, TOOLS_VERSION, validate_call  # noqa: E402
from src.tools.executor import ToolExecutor  # noqa: E402
from training.fc_format import parse_tool_calls  # noqa: E402

OUT = ROOT / "demo-ui" / "src" / "scenarios.json"
TESTSET = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "evaluation/data/tourism_testset_v1.jsonl", encoding="utf-8")}


def call_text(calls: list[dict]) -> str:
    return "\n".join(f'<tool_call>\n{json.dumps(c, ensure_ascii=False)}\n</tool_call>' for c in calls)


def concrete(gold_call: dict) -> dict:
    """Gold call (accept lists / optional) -> one concrete call: first accepted value, optional args left out."""
    return {"name": gold_call["name"],
            "arguments": {k: v["accept"][0] for k, v in gold_call["arguments"].items() if not v["optional"]}}


class ScriptedRouter:
    """Stands in for the fine-tuned model: returns the scripted output for each round, then a plain reply."""

    def __init__(self, rounds: list[str], final_text: str = ""):
        self.rounds, self.final_text, self.i = rounds, final_text, 0

    def __call__(self, messages, tools, mode, system_prompt):
        text = self.rounds[self.i] if self.i < len(self.rounds) else self.final_text
        self.i += 1
        return {"text": text, "latency_s": 0.0}


def run_turn(executor, history, user, rounds, final_text=""):
    r = Agent(ScriptedRouter(rounds, final_text), executor, mode="grammar").respond(history, user)
    trace = []
    for e in r.trace:
        trace.append({
            "round": e["round"],
            "model_output": e["model_output"],
            "status": e["status"],
            "calls": [{"name": c["name"], "arguments": c["arguments"], "schema_errors": validate_call(c["name"], c["arguments"])}
                      for c in (e["calls"] or [])],
            "results": e.get("results", []),
        })
    return {"user": user, "answer": r.answer, "trace": trace}


def from_testset(executor, item_id, final_text=""):
    it = TESTSET[item_id]
    user = it["messages"][-1]["content"]
    steps = [[concrete(c) for c in step["calls"]] for step in it["steps"]]
    for step in steps:  # web_search arguments are "free" in the gold answer; the query is the user's question
        for c in step:
            if c["name"] == "web_search" and "query" not in c["arguments"]:
                c["arguments"] = {"query": user.rstrip("?"), "max_results": 3}
    rounds = [call_text(step) for step in steps]
    history = it["messages"][:-1]
    return it, run_turn(executor, history, user, rounds, final_text)


SCENARIOS = [
    # (id, test item, group, title, scripted reply when no tool is needed)
    ("T01", "Hotels (C2)", "Hotel search with budget, family and hospital", ""),
    ("S01", "Reviews (C1)", "Review insights for one hotel", ""),
    ("S20", "Itinerary (C3)", "3-day plan from a hotel", ""),
    ("M01", "Multi-step (C2 → C3)", "Pick a hotel, then plan the trip", ""),
    ("M06", "Parallel calls (C1 × 2)", "Compare two hotels' reviews", ""),
    ("W01", "Web search", "Information no team tool has", ""),
    ("C01", "Clarify", "A required detail is missing",
     "Sure! Which town or area in Sri Lanka would you like to stay in? If you also tell me your dates, budget and how many people are travelling, I can find better matches."),
    ("X01", "Decline", "Off-topic request",
     "Sorry, I can't write essays. I'm a travel assistant for Sri Lanka, so I can help with hotels, reviews, trip plans and travel information."),
    ("D03", "Direct answer", "Greeting, no tool needed",
     "Hi! I can find hotels, show what reviews say, plan day-by-day trips and look up travel information for Sri Lanka. Where are you thinking of going?"),
]


# How each mode works: what is added to the prompt, whether XGrammar is on, and what happened in the example.
HOW = {
    "free": {"instruction": None, "grammar": False,
             "generation": "The model writes freely, one token at a time. Nothing checks the format while it writes.",
             "why": "It forgot the closing } and </tool_call>, so the call cannot be parsed and is never run."},
    "prompt_json": {"instruction": PROMPT_JSON_INSTRUCTION, "grammar": False,
                    "generation": "The model is asked to follow the format, but nothing enforces it while it writes.",
                    "why": "The format is right, but it wrote adults = 4 instead of 2 adults + 2 children."},
    "grammar": {"instruction": None, "grammar": True,
                "generation": "Same prompt as (a). XGrammar blocks every token that would break the call: only real tool names and schema-valid arguments are allowed.",
                "why": "The format is guaranteed, but the forced call dropped children and the hospital: valid but wrong (the constraint tax)."},
    "reason_grammar": {"instruction": REASON_INSTRUCTION, "grammar": True,
                       "generation": "The model first writes one free \"Reasoning:\" line. The grammar switches on only at <tool_call>, then forces a valid call.",
                       "why": "Thinking first kept every detail, and the grammar kept the format valid: valid and correct."},
}


def illustrative_modes():
    """Same question, four decoding modes. Outputs are illustrative; scoring uses the real code."""
    item = TESTSET["T01"]
    gold = item["steps"][0]["calls"]
    full = '{"name": "recommend_hotels", "arguments": {"destination": "Ella", "budget_max_lkr": 25000, "adults": 2, "children": 2, "nearby": ["hospital"]}}'
    outputs = {
        "free": '<tool_call>\n{"name": "recommend_hotels", "arguments": {"destination": "Ella", "budget_max_lkr": 25000, "adults": 2, "children": 2, "nearby": ["hospital"]}\n',
        "prompt_json": '<tool_call>\n{"name": "recommend_hotels", "arguments": {"destination": "Ella", "budget_max_lkr": 25000, "adults": 4, "nearby": ["hospital"]}}\n</tool_call>',
        "grammar": '<tool_call>\n{"name": "recommend_hotels", "arguments": {"destination": "Ella", "budget_max_lkr": 25000, "adults": 2}}\n</tool_call>',
        "reason_grammar": f'Reasoning: hotel search in Ella for 2 adults and 2 children, max LKR 25,000, near a hospital.\n<tool_call>\n{full}\n</tool_call>',
    }
    rows = []
    for mode in MODES:
        text = strip_reasoning(outputs[mode]) if mode == "reason_grammar" else outputs[mode]
        parsed = parse_tool_calls(text)
        score = score_tourism_step(parsed, gold, validate_call)
        rows.append({"mode": mode, "raw": outputs[mode], "parsed": parsed, **score, **HOW[mode]})
    return {"question": item["messages"][-1]["content"], "gold": gold, "rows": rows, "system_prompt": SYSTEM_PROMPT}


def mock_web_search(query: str, max_results: int) -> list[dict]:
    """Labelled MOCK web results for the offline demo (no live search API is connected)."""
    return [
        {"title": "Sri Lanka Railways \u2013 train timetable (mock result)",
         "snippet": "Several trains run daily between Colombo Fort and Kandy; the journey takes about 2.5\u20133 hours. Check the official timetable before travelling.",
         "url": "https://railway.gov.lk", "source": "mock web result"},
        {"title": "Travel guide \u2013 Colombo to Kandy by train (mock result)",
         "snippet": "Reserved seats in 1st and 2nd class can be booked in advance; 3rd class is unreserved. Morning trains are the most popular.",
         "url": None, "source": "mock web result"},
        {"title": "Tourist forum \u2013 tips for the Kandy train (mock result)",
         "snippet": "Arrive early at Colombo Fort station, especially if you plan to travel in unreserved 3rd class.",
         "url": None, "source": "mock web result"},
    ][:max_results]


def main():
    executor = ToolExecutor(ToolClients(c1=TestClient(c1), c2=TestClient(c2), c3=TestClient(c3)), web_search=mock_web_search)
    scenarios = []
    for sid, group, title, reply in SCENARIOS:
        it, turn = from_testset(executor, sid, reply)
        scenarios.append({"id": sid, "group": group, "title": title, "category": it["category"], "turns": [turn]})

    # a real multi-turn conversation (follow-ups refer to earlier turns)
    hist, turns = [], []
    script = [
        ("Suggest hotels in Kandy near a hospital",
         [call_text([{"name": "recommend_hotels", "arguments": {"destination": "Kandy", "nearby": ["hospital"]}}])]),
        ("How is the food at Kandy Hills?",
         [call_text([{"name": "get_review_insights", "arguments": {"hotel": "Kandy Hills", "aspects": ["food"]}}])]),
        ("Plan 2 easy days from there, my father can't walk much",
         [call_text([{"name": "plan_itinerary", "arguments": {"hotel": "Kandy Hills", "days": 2, "mobility_needs": True}}])]),
    ]
    for user, rounds in script:
        t = run_turn(executor, hist, user, rounds)
        turns.append(t)
        hist = hist + [{"role": "user", "content": user}, {"role": "assistant", "content": t["answer"]}]
    scenarios.insert(4, {"id": "CONV", "group": "Multi-turn chat", "title": "Follow-up questions remember context",
                         "category": "multi_turn", "turns": turns})

    data = {
        "tools_version": TOOLS_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "tools": [{"name": t["function"]["name"], "description": t["function"]["description"],
                   "required": t["function"]["parameters"].get("required", []),
                   "params": list(t["function"]["parameters"]["properties"])} for t in TOOLS],
        "scenarios": scenarios,
        "modes": illustrative_modes(),
        "testset_size": len(TESTSET),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({len(scenarios)} scenarios)")


if __name__ == "__main__":
    main()
