"""Demo logic without Gradio or a GPU."""

import json

from fastapi.testclient import TestClient

from evaluation.build_testset import JSONL
from evaluation.run_eval import load_jsonl
from src.agent import Agent
from src.demo.app import compare_modes, gold_markdown, history_markdown, mode_card, run_chat_turn
from src.mocks.c1_mock import app as c1_app
from src.mocks.c2_mock import app as c2_app
from src.mocks.c3_mock import app as c3_app
from src.tools.clients import ToolClients
from src.tools.executor import ToolExecutor

TESTSET = load_jsonl(JSONL)


def test_side_by_side_shows_the_constraint_tax_case():
    item = next(t for t in TESTSET if t["id"] == "S01")
    good = '<tool_call>\n{"name": "get_review_insights", "arguments": {"hotel": "Orchard House Galle", "aspects": ["cleanliness"]}}\n</tool_call>'
    wrong_but_valid = good.replace("cleanliness", "food")
    broken = good[:-15]

    def llm(messages, tools, mode, sp):
        text = {"free": broken, "prompt_json": good, "grammar": wrong_but_valid, "reason_grammar": "Reasoning: reviews.\n" + good}[mode]
        return {"text": text, "latency_s": 0.3}

    rows = {r["mode"]: r for r in compare_modes(llm, item)}
    assert rows["free"]["malformed"] and not rows["free"]["correct"]
    assert rows["grammar"]["valid"] and not rows["grammar"]["correct"]
    assert rows["reason_grammar"]["correct"] and rows["prompt_json"]["correct"]
    assert "valid but WRONG" in mode_card(rows["grammar"]) and "malformed" in mode_card(rows["free"])
    assert "get_review_insights" in gold_markdown(item)


def test_chat_turn_updates_history_and_reports_latency():
    ex = ToolExecutor(ToolClients(c1=TestClient(c1_app), c2=TestClient(c2_app), c3=TestClient(c3_app)))
    llm = lambda m, t, mode, sp: {"text": "Hello! How can I help?", "latency_s": 0.25}
    hist, info, trace = run_chat_turn(Agent(llm, ex), [], "Hi")
    assert hist[-1] == {"role": "assistant", "content": "Hello! How can I help?"}
    assert "0.25 s" in info and trace["rounds"][0]["status"] == "answer"
    assert "**You:** Hi" in history_markdown(hist)


def test_demo_notebook_carries_the_current_code():
    """Fails if C4 code changed without `python -m src.demo.build_demo_notebook` (C1 data is not checked,
    so Component 1 updating its data never breaks C4's tests)."""
    from src.demo import build_demo_notebook as b

    src = "".join(json.loads(b.OUT.read_text(encoding="utf-8"))["cells"][5]["source"])
    committed = json.loads(src[len("FILES = "):src.index("\n")])
    current = b.bundle()
    for path in b.CODE_FILES + b.EMPTY_INITS:
        assert committed[path] == current[path], f"{path} changed: rebuild the demo notebook"
