"""Evaluation pipeline without a GPU: a fake generator replays scripted outputs."""

import json

import pytest

from evaluation import run_eval
from evaluation.build_testset import JSONL
from evaluation.decoding import MODES, build_messages, strip_reasoning, structural_tag
from evaluation.scoring import (bfcl_call_ok, bfcl_tools, call_matches_gold, holm, mcnemar_exact,
                                score_bfcl, step_correct, values_equal)

TOURISM = run_eval.load_jsonl(JSONL)
BFCL = run_eval.load_jsonl(JSONL.with_name("bfcl_v4_subset.jsonl"))


def gold_text(calls):
    """Render gold calls the way Qwen2.5 writes them (first accepted value of each argument)."""
    out = []
    for c in calls:
        args = {k: v["accept"][0] for k, v in c["arguments"].items() if not v["optional"]}
        if c["name"] == "web_search":
            args["query"] = "something"
        out.append('<tool_call>\n' + json.dumps({"name": c["name"], "arguments": args}) + '\n</tool_call>')
    return "\n".join(out)


class OracleGen:
    """Returns the gold answer for every step; with `break_mode` set, that mode returns broken JSON."""

    def __init__(self, break_mode=None):
        self.break_mode = break_mode
        self.lookup = {}
        for it in TOURISM:
            for i, step in enumerate(it["steps"] or [{"calls": []}]):
                msgs = it["messages"] + [m for ctx in it["context_after_step"][:i] for m in ctx]
                self.lookup[json.dumps(msgs, sort_keys=True)] = gold_text(step["calls"]) or "Happy to help!"

    def __call__(self, messages, tools, mode, system_prompt):
        text = self.lookup[json.dumps(messages, sort_keys=True)]
        if mode == self.break_mode and text.startswith("<tool_call>"):
            text = text.replace("}\n</tool_call>", "\n</tool_call>", 1)  # drop a brace -> malformed
        if mode == "reason_grammar":
            text = "Reasoning: the user needs this.\n" + text
        return {"text": text, "latency_s": 0.5, "new_tokens": 10}


def test_oracle_scores_100_percent_in_every_mode(tmp_path):
    rows = run_eval.run(OracleGen(), "oracle", {"tourism": TOURISM}, tmp_path, log_every=10**9)
    assert len(rows) == 150 * len(MODES)
    res = run_eval.analyse(rows)
    for mode in MODES:
        s = res["summaries"]["oracle"]["tourism"][mode]
        assert (s["accuracy"], s["validity"], s["malformed"]) == (100.0, 100.0, 0.0), mode


def test_broken_mode_loses_validity_and_accuracy_and_is_resumable(tmp_path):
    gen = OracleGen(break_mode="free")
    run_eval.run(gen, "m", {"tourism": TOURISM[:30]}, tmp_path, log_every=10**9)
    rows = run_eval.run(gen, "m", {"tourism": TOURISM[:30]}, tmp_path, log_every=10**9)  # resume: no duplicates
    assert len(rows) == 30 * len(MODES)
    res = run_eval.analyse(rows)
    free, grammar = res["summaries"]["m"]["tourism"]["free"], res["summaries"]["m"]["tourism"]["grammar"]
    assert free["malformed"] > 0 and free["accuracy"] < grammar["accuracy"]
    assert res["constraint_tax_pp"]["m|tourism|grammar"] < 0  # grammar did better here: negative tax
    table = run_eval.write_results(res, tmp_path)
    assert "constraint_tax_pp" in table.read_text()


def test_values_equal_is_lenient_but_not_sloppy():
    assert values_equal(25000, 25000.0) and values_equal("Orchard House, Galle", "orchard house galle")
    assert values_equal(["beach", "hospital"], ["hospital", "beach"])
    assert not values_equal(["beach"], ["beach", "hospital"]) and not values_equal(True, 1)


def test_gold_matching_rules():
    gold = {"name": "recommend_hotels", "free": ["interests", "top_k"], "arguments": {
        "destination": {"accept": ["Ella"], "optional": False},
        "travel_purpose": {"accept": ["family"], "optional": True}}}
    ok = {"name": "recommend_hotels", "arguments": {"destination": "ella", "top_k": 3}}
    assert call_matches_gold(ok, gold)
    assert not call_matches_gold({"name": "recommend_hotels", "arguments": {"destination": "Ella", "travel_purpose": "business"}}, gold)
    assert not call_matches_gold({"name": "recommend_hotels", "arguments": {"destination": "Ella", "adults": 2}}, gold)  # hallucinated
    assert not step_correct([ok, ok], [gold])  # extra call
    assert not step_correct(None, [gold])


def test_bfcl_subset_and_checker():
    assert len(BFCL) == 300
    simple = next(r for r in BFCL if r["category"] == "simple_python")
    (name, params), = simple["ground_truth"][0].items()
    args = {p: v[0] for p, v in params.items() if v and v[0] != ""}
    assert score_bfcl([{"name": name, "arguments": args}], simple, simple["function"])["correct"]
    assert not score_bfcl([], simple, simple["function"])["correct"]
    irr = next(r for r in BFCL if r["category"] == "irrelevance")
    assert score_bfcl([], irr, irr["function"])["correct"]
    assert all(t["function"]["parameters"].get("type") == "object" for t in bfcl_tools(simple["function"]))


def test_bfcl_string_normalisation_and_optional_params():
    fn = [{"name": "f", "parameters": {"type": "dict", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}}}]
    gold = {"f": {"city": ["New York"], "unit": ["", "celsius"]}}
    assert bfcl_call_ok({"name": "f", "arguments": {"city": "new york"}}, gold, fn)
    assert not bfcl_call_ok({"name": "f", "arguments": {"city": "Boston"}}, gold, fn)
    assert not bfcl_call_ok({"name": "f", "arguments": {"city": "New York", "zip": "1"}}, gold, fn)


def test_stats():
    assert mcnemar_exact([True] * 10, [True] * 10)["p_value"] == 1.0
    r = mcnemar_exact([True] * 20 + [False] * 5, [False] * 20 + [False] * 5)
    assert r["a_only"] == 20 and r["p_value"] < 0.001
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"]["p_holm"] == pytest.approx(0.03) and not h["b"]["reject"]


def test_mode_prompts_and_grammar():
    msgs = [{"role": "user", "content": "hi"}]
    assert build_messages(msgs, "free") == msgs
    assert "ONLY the tool call" in build_messages(msgs, "prompt_json")[0]["content"]
    assert build_messages(msgs, "reason_grammar", "SYS")[0]["content"].startswith("SYS")
    assert strip_reasoning("Reasoning: x\n<tool_call>{}</tool_call>") == "<tool_call>{}</tool_call>"
    st = structural_tag(bfcl_tools(BFCL[0]["function"]))
    assert st["format"]["triggers"] == ["<tool_call>"] and st["format"]["tags"][0]["content"]["json_schema"]["additionalProperties"] is False


def test_eval_notebook_is_up_to_date_and_valid_python():
    import ast

    from evaluation import build_eval_notebook

    committed = json.loads(build_eval_notebook.OUT.read_text(encoding="utf-8"))
    built = build_eval_notebook.build()
    assert committed["cells"] == built["cells"], "run: python -m evaluation.build_eval_notebook"
    for cell in built["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.startswith("%%writefile"):
            name, body = src.split("\n", 1)
            if not name.endswith(".py"):
                continue
            src = body
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!"))
        ast.parse(src)
