"""Run the decoding-mode experiment and write predictions + results (Steps 7-8).

Normally run from notebooks/02_evaluate.ipynb on a Colab T4. Predictions are
appended to predictions.jsonl as they are produced, so a disconnected run
continues where it stopped.

Outputs in --out-dir:
    predictions.jsonl   one row per (dataset, item, mode): raw text, parsed calls, scores, latency
    results.json        all summaries, constraint tax, McNemar tests, hypothesis checks
    results_table.csv   the main table for the slides
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

try:  # repo layout (services/c4-agent)
    from evaluation.decoding import MODES, strip_reasoning
    from evaluation.scoring import bfcl_tools, bootstrap_ci, holm, mcnemar_exact, score_bfcl, score_tourism_step, summarize
    from src.tools.definitions import SYSTEM_PROMPT, TOOLS, validate_call
    from training.fc_format import parse_tool_calls
except ImportError:  # flat layout in Colab (files written by the notebook)
    from decoding import MODES, strip_reasoning
    from fc_format import parse_tool_calls
    from scoring import bfcl_tools, bootstrap_ci, holm, mcnemar_exact, score_bfcl, score_tourism_step, summarize
    from tool_defs import SYSTEM_PROMPT, TOOLS, validate_call


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _parse(text: str, mode: str) -> list[dict] | None:
    return parse_tool_calls(strip_reasoning(text) if mode == "reason_grammar" else text)


def eval_tourism_item(gen: Callable, item: dict, mode: str) -> dict:
    texts, steps, latency = [], [], None
    for i, step in enumerate(item["steps"] or [{"calls": []}]):
        messages = item["messages"] + [m for ctx in item["context_after_step"][:i] for m in ctx]
        out = gen(messages, TOOLS, mode, SYSTEM_PROMPT)
        latency = out["latency_s"] if latency is None else latency  # tool-decision latency = first step
        parsed = _parse(out["text"], mode)
        texts.append(out["text"])
        steps.append(dict(score_tourism_step(parsed, step["calls"], validate_call), parsed=parsed))
    return {
        "texts": texts,
        "parsed": [s["parsed"] for s in steps],
        "valid": all(s["valid"] for s in steps),
        "correct": all(s["correct"] for s in steps),
        "malformed": any(s["malformed"] for s in steps),
        "step1_correct": steps[0]["correct"],
        "latency_s": latency,
    }


def eval_bfcl_item(gen: Callable, item: dict, mode: str) -> dict:
    out = gen(item["messages"], bfcl_tools(item["function"]), mode, None)
    parsed = _parse(out["text"], mode)
    s = score_bfcl(parsed, item, item["function"])
    return {"texts": [out["text"]], "parsed": [parsed], **s, "latency_s": out["latency_s"]}


def run(gen: Callable, model_label: str, datasets: dict[str, list[dict]], out_dir: str | Path,
        modes: list[str] = MODES, log_every: int = 25) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_file = out / "predictions.jsonl"
    done = {(r["model"], r["dataset"], r["id"], r["mode"]) for r in load_jsonl(pred_file)}
    total = sum(len(v) for v in datasets.values()) * len(modes)
    count = len([k for k in done if k[0] == model_label])
    with open(pred_file, "a", encoding="utf-8") as f:
        for ds_name, items in datasets.items():
            for item in items:
                for mode in modes:
                    key = (model_label, ds_name, item["id"], mode)
                    if key in done:
                        continue
                    res = eval_tourism_item(gen, item, mode) if ds_name == "tourism" else eval_bfcl_item(gen, item, mode)
                    row = {"model": model_label, "dataset": ds_name, "id": item["id"], "category": item["category"], "mode": mode, **res}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    count += 1
                    if count % log_every == 0:
                        print(f"[{count}/{total}] {ds_name} {item['id']} {mode} correct={row['correct']}")
    return load_jsonl(pred_file)


def analyse(rows: list[dict]) -> dict:
    """Summaries, constraint tax, paired tests and hypothesis checks from predictions."""
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["dataset"], r["mode"])].append(r)
        by[(r["model"], f"{r['dataset']}:{r['category']}", r["mode"])].append(r)
    summaries = {}
    for (model, ds, mode), rs in by.items():
        s = summarize(rs)
        s["accuracy_ci95"] = bootstrap_ci([bool(r["correct"]) for r in rs])
        summaries.setdefault(model, {}).setdefault(ds, {})[mode] = s

    tests, taxes = {}, {}
    for model, per_ds in summaries.items():  # constraint tax for datasets AND categories
        for ds, modes_here in per_ds.items():
            if "free" in modes_here:
                for mode in modes_here:
                    if mode != "free":
                        taxes[f"{model}|{ds}|{mode}"] = round(modes_here["free"]["accuracy"] - modes_here[mode]["accuracy"], 1)
    for model, per_ds in summaries.items():
        for ds in [d for d in per_ds if ":" not in d]:
            modes_here = per_ds[ds]
            if "free" not in modes_here:
                continue
            free_rows = {r["id"]: r for r in by[(model, ds, "free")]}
            for mode in modes_here:
                if mode == "free":
                    continue
                other = {r["id"]: r for r in by[(model, ds, mode)]}
                ids = sorted(set(free_rows) & set(other))
                tests[f"{model}|{ds}|free_vs_{mode}|accuracy"] = mcnemar_exact(
                    [bool(free_rows[i]["correct"]) for i in ids], [bool(other[i]["correct"]) for i in ids])
                tests[f"{model}|{ds}|free_vs_{mode}|validity"] = mcnemar_exact(
                    [bool(free_rows[i]["valid"]) for i in ids], [bool(other[i]["valid"]) for i in ids])
    adjusted = holm({k: v["p_value"] for k, v in tests.items()})
    for k in tests:
        tests[k].update(adjusted[k])

    hypotheses = {}
    for model, per_ds in summaries.items():
        for ds in [d for d in per_ds if ":" not in d]:
            m = per_ds[ds]
            if not {"free", "grammar"} <= m.keys():
                continue
            acc = tests.get(f"{model}|{ds}|free_vs_grammar|accuracy", {})
            val = tests.get(f"{model}|{ds}|free_vs_grammar|validity", {})
            hypotheses[f"{model}|{ds}"] = {
                "H1_constraint_tax_pp": taxes.get(f"{model}|{ds}|grammar"),
                "H1_supported": bool(acc.get("reject")) and m["free"]["accuracy"] > m["grammar"]["accuracy"],
                "H2_validity_up": m["grammar"]["validity"] - m["free"]["validity"],
                "H2_supported": m["grammar"]["validity"] >= m["free"]["validity"]
                and not (acc.get("reject") and m["grammar"]["accuracy"] > m["free"]["accuracy"]),
                "H3_best_mode": max(m, key=lambda k: m[k]["accuracy"]),
                "H3_supported": max(m, key=lambda k: m[k]["accuracy"]) == "reason_grammar",
                "validity_test_p_holm": val.get("p_holm"),
                "accuracy_test_p_holm": acc.get("p_holm"),
            }
    return {"summaries": summaries, "constraint_tax_pp": taxes, "mcnemar": tests, "hypotheses": hypotheses}


def write_results(results: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = out / "results_table.csv"
    with open(table, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "mode", "n", "validity_%", "accuracy_%", "accuracy_95%_CI", "wrong_but_valid_%",
                    "malformed_%", "constraint_tax_pp", "latency_median_s", "latency_p95_s"])
        for model, per_ds in results["summaries"].items():
            for ds in sorted(per_ds):
                for mode in MODES:
                    s = per_ds[ds].get(mode)
                    if not s:
                        continue
                    tax = 0.0 if mode == "free" else results["constraint_tax_pp"].get(f"{model}|{ds}|{mode}", "")
                    w.writerow([model, ds, mode, s["n"], s["validity"], s["accuracy"], f"{s['accuracy_ci95'][0]}-{s['accuracy_ci95'][1]}",
                                s["wrong_but_valid"], s["malformed"], tax, s["latency_median_s"], s["latency_p95_s"]])
    return table
