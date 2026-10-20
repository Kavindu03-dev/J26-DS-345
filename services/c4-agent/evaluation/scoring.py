"""Scoring for the decoding-mode experiment (Steps 7-8). Pure Python, no GPU needed.

Per response we record:
  parsed    - the calls extracted by the strict, repair-free parser (None = malformed)
  valid     - the response is well-formed AND every call passes its tool's JSON schema
              (a plain-text answer with no call counts as valid)
  correct   - the response matches the gold answer (see below)

Aggregates per (model, mode, dataset):
  validity           % valid
  accuracy           % correct
  wrong_but_valid    % valid but not correct      <- the "constraint tax" failure mode
  malformed          % unparseable
  constraint_tax     accuracy(free) - accuracy(mode), in percentage points
McNemar's exact test compares two modes on the SAME items (paired), with
Holm correction across the comparisons.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Iterable

# ----------------------------------------------------------------------------- tourism set


def _norm_str(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def values_equal(pred: Any, gold: Any) -> bool:
    """Lenient type comparison: 3 == 3.0, strings case/punctuation-insensitive, lists as sets."""
    if isinstance(gold, bool) or isinstance(pred, bool):
        return isinstance(gold, bool) and isinstance(pred, bool) and gold == pred
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return math.isclose(float(pred), float(gold), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(gold, str) and isinstance(pred, str):
        return _norm_str(pred) == _norm_str(gold)
    if isinstance(gold, list) and isinstance(pred, list):
        if len(gold) != len(pred):
            return False
        remaining = list(pred)
        for g in gold:
            hit = next((i for i, p in enumerate(remaining) if values_equal(p, g)), None)
            if hit is None:
                return False
            remaining.pop(hit)
        return True
    if isinstance(gold, dict) and isinstance(pred, dict):
        return gold.keys() == pred.keys() and all(values_equal(pred[k], gold[k]) for k in gold)
    return False


def call_matches_gold(pred: dict, gold: dict) -> bool:
    """One predicted call vs one gold call spec ({"name", "arguments": {arg: {accept, optional}}, "free"})."""
    if pred["name"] != gold["name"]:
        return False
    args = pred["arguments"]
    for name, spec in gold["arguments"].items():
        if name not in args:
            if not spec["optional"]:
                return False
            continue
        if not any(values_equal(args[name], a) for a in spec["accept"]):
            return False
    extra = set(args) - set(gold["arguments"]) - set(gold.get("free", []))
    return not extra


def step_correct(pred_calls: list[dict] | None, gold_calls: list[dict]) -> bool:
    """Exact set match: same number of calls, each matched to a distinct gold call."""
    if pred_calls is None or len(pred_calls) != len(gold_calls):
        return False
    remaining = list(gold_calls)
    for p in pred_calls:
        hit = next((i for i, g in enumerate(remaining) if call_matches_gold(p, g)), None)
        if hit is None:
            return False
        remaining.pop(hit)
    return True


def response_valid(pred_calls: list[dict] | None, validate: Callable[[str, dict], list[str]]) -> bool:
    if pred_calls is None:
        return False
    return all(not validate(c["name"], c["arguments"]) for c in pred_calls)


def score_tourism_step(pred_calls: list[dict] | None, gold_calls: list[dict], validate) -> dict:
    """gold_calls == [] means the correct behaviour is no call."""
    return {
        "malformed": pred_calls is None,
        "valid": response_valid(pred_calls, validate),
        "correct": (pred_calls == []) if not gold_calls else step_correct(pred_calls, gold_calls),
        "called": bool(pred_calls),
    }


# ----------------------------------------------------------------------------- BFCL

_BFCL_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": None}


def _convert_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                mapped = _BFCL_TYPES.get(v, v)
                if mapped is not None:
                    out[k] = mapped
            else:
                out[k] = _convert_schema(v)
        return out
    if isinstance(node, list):
        return [_convert_schema(x) for x in node]
    return node


def bfcl_tools(functions: list[dict]) -> list[dict]:
    """BFCL function docs -> OpenAI-style tools (BFCL uses 'dict'/'float'/'tuple' type names)."""
    return [
        {
            "type": "function",
            "function": {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": _convert_schema(f.get("parameters", {"type": "object", "properties": {}})),
            },
        }
        for f in functions
    ]


def _bfcl_standardize(s: str) -> str:
    # Same normalisation idea as the official checker: case and common punctuation ignored.
    return re.sub(r"[ ,./\-_*^]", "", s.lower()).replace("'", '"')


def _bfcl_value_ok(pred: Any, accepted: list) -> bool:
    for a in accepted:
        if a == "":
            continue
        if isinstance(a, str) and isinstance(pred, str):
            if _bfcl_standardize(pred) == _bfcl_standardize(a):
                return True
        elif isinstance(a, bool) or isinstance(pred, bool):
            if isinstance(a, bool) and isinstance(pred, bool) and a == pred:
                return True
        elif isinstance(a, (int, float)) and isinstance(pred, (int, float)):
            if math.isclose(float(pred), float(a), rel_tol=1e-9, abs_tol=1e-9):
                return True
        elif isinstance(a, list) and isinstance(pred, list):
            if len(a) == len(pred) and all(_bfcl_value_ok(p, [x]) for p, x in zip(pred, a)):
                return True
        elif isinstance(a, dict) and isinstance(pred, dict):
            if a.keys() == pred.keys() and all(
                _bfcl_value_ok(pred[k], v if isinstance(v, list) else [v]) for k, v in a.items()
            ):
                return True
    return False


def bfcl_call_ok(pred: dict, gold: dict, functions: list[dict]) -> bool:
    """AST-style check of one call against one BFCL ground-truth entry {func_name: {param: [accepted]}}."""
    (gname, gparams), = gold.items()
    if pred["name"] != gname:
        return False
    fn = next((f for f in functions if f["name"] == gname), None)
    if fn is None:
        return False
    known = set(fn.get("parameters", {}).get("properties", {}))
    args = pred["arguments"]
    if set(args) - known or set(args) - set(gparams):
        return False  # hallucinated or unexpected parameter
    for p, accepted in gparams.items():
        if p not in args:
            if "" not in accepted:
                return False  # required parameter missing
        elif not _bfcl_value_ok(args[p], accepted):
            return False
    return True


def score_bfcl(pred_calls: list[dict] | None, item: dict, validate_against: list[dict]) -> dict:
    """item: {"category", "function", "ground_truth" (absent for irrelevance)}."""
    import jsonschema

    tools = bfcl_tools(item["function"])
    schemas = {t["function"]["name"]: dict(t["function"]["parameters"], additionalProperties=False) for t in tools}

    def _validate(name, args):
        if name not in schemas:
            return ["unknown tool"]
        return [e.message for e in jsonschema.Draft202012Validator(schemas[name]).iter_errors(args)]

    valid = response_valid(pred_calls, _validate)
    if item["category"].endswith("irrelevance"):
        correct = pred_calls == []
    else:
        gold = item["ground_truth"]
        correct = pred_calls is not None and len(pred_calls) == len(gold)
        if correct:
            remaining = list(gold)
            for p in pred_calls:
                hit = next((i for i, g in enumerate(remaining) if bfcl_call_ok(p, g, item["function"])), None)
                if hit is None:
                    correct = False
                    break
                remaining.pop(hit)
    return {"malformed": pred_calls is None, "valid": valid, "correct": correct, "called": bool(pred_calls)}


# ----------------------------------------------------------------------------- aggregation & stats


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)
    if not n:
        return {"n": 0}
    pct = lambda k: round(100 * sum(bool(r[k]) for r in rows) / n, 1)
    lat = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    return {
        "n": n,
        "validity": pct("valid"),
        "accuracy": pct("correct"),
        "wrong_but_valid": round(100 * sum(r["valid"] and not r["correct"] for r in rows) / n, 1),
        "malformed": pct("malformed"),
        "latency_median_s": round(lat[len(lat) // 2], 3) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 3) if lat else None,
    }


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact (binomial) McNemar test on paired correctness. b10 = a right & b wrong."""
    from scipy.stats import binomtest

    assert len(a) == len(b)
    b10 = sum(x and not y for x, y in zip(a, b))
    b01 = sum(y and not x for x, y in zip(a, b))
    n = b10 + b01
    p = 1.0 if n == 0 else float(binomtest(b10, n, 0.5, alternative="two-sided").pvalue)
    return {"a_only": b10, "b_only": b01, "p_value": p}


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni adjusted p-values and reject decisions."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": float(p), "p_holm": float(adj), "reject": bool(adj < alpha)}
    return out


def bootstrap_ci(values: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI for a proportion (in %)."""
    import random

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(100 * sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    return (round(stats[int(0.025 * n_boot)], 1), round(stats[int(0.975 * n_boot) - 1], 1))


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
