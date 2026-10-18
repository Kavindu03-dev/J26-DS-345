"""Prepare the fine-tuning dataset (Step 4).

Pipeline (all steps are logged in stats.json):
  1. load xLAM-60k (Hugging Face, needs your access token) or a local copy
  2. convert to chat format and drop malformed rows (no repair)
  3. drop duplicate queries and anything overlapping the BFCL test set
  4. sample N tool-call examples (seeded, reproducible)
  5. add no-call examples so they are >= 15% of the final set
  6. split train / validation (stratified by kind) and write JSONL

Usage (from services/c4-agent, after `pip install -r requirements-training.txt`):

    # Windows PowerShell - your Hugging Face token, only for this terminal
    $env:HF_TOKEN = "hf_..."
    python -m training.prepare_data

    # options
    python -m training.prepare_data --n-tool-call 10000 --no-call-share 0.15 --seed 42
    python -m training.prepare_data --input path\\to\\xlam_function_calling_60k.json
    python -m training.prepare_data --push-to-hub <hf-username>/c4-fc-sft

Output (default: <repo>/data/c4_fc/, which Git ignores):
    train.jsonl, val.jsonl, stats.json, preview.md
Each JSONL line: {"id", "kind", "source_id", "tools": "<json>", "messages": "<json>"}.
tools and messages are stored as JSON strings so every row has the same
columns (Hugging Face datasets cannot hold free-form nested arguments);
the training script decodes them before applying the chat template.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .decontaminate import Decontaminator
from .no_call import make_conversational, make_irrelevance
from .xlam_convert import ConversionError, convert_row

XLAM_DATASET = "Salesforce/xlam-function-calling-60k"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data" / "c4_fc"
CONVERSATIONAL_SHARE_OF_NO_CALL = 0.2  # 80% irrelevance, 20% conversational


def load_xlam(input_path: str | None) -> list[dict]:
    if input_path:
        text = Path(input_path).read_text(encoding="utf-8")
        if text.lstrip().startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    from datasets import load_dataset  # imported lazily: only needed for the download

    token = os.getenv("HF_TOKEN")
    ds = load_dataset(XLAM_DATASET, split="train", token=token)
    return [dict(r) for r in ds]


def build(rows: list[dict], n_tool_call: int, no_call_share: float, seed: int, decon: Decontaminator) -> tuple[list[dict], dict]:
    stats: Counter = Counter(raw_rows=len(rows))
    rejected: Counter = Counter()
    converted, seen = [], set()
    for row in rows:
        try:
            ex = convert_row(row)
        except ConversionError as e:
            rejected[str(e).split(" ")[0]] += 1
            stats["dropped_malformed"] += 1
            continue
        key = " ".join(ex["messages"][0]["content"].lower().split())
        if key in seen:
            stats["dropped_duplicate_query"] += 1
            continue
        seen.add(key)
        if decon.is_contaminated(ex["messages"][0]["content"]):
            stats["dropped_bfcl_overlap"] += 1
            continue
        converted.append(ex)
    stats["clean_tool_call_pool"] = len(converted)

    rng = random.Random(seed)
    rng.shuffle(converted)
    tool_calls = converted[:n_tool_call]
    if len(tool_calls) < n_tool_call:
        print(f"[warn] only {len(tool_calls)} clean tool-call examples available (asked for {n_tool_call})")

    # ceil, so the no-call share is never below the requested minimum
    n_no_call = math.ceil(len(tool_calls) * no_call_share / (1 - no_call_share))
    n_conv = round(n_no_call * CONVERSATIONAL_SHARE_OF_NO_CALL)
    # Irrelevance queries come from the pool NOT used for tool-call training when
    # possible, so the model never sees the same query with both answers.
    spare = converted[n_tool_call:] or tool_calls
    no_calls = make_irrelevance(spare, converted, n_no_call - n_conv, rng)
    no_calls += make_conversational(tool_calls, n_no_call - len(no_calls), rng)

    examples = tool_calls + no_calls
    rng.shuffle(examples)
    stats.update({f"rejected_{k}": v for k, v in rejected.items()})
    return examples, dict(stats)


def split(examples: list[dict], val_share: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified by kind, and grouped by user message.

    Examples with the same user message (e.g. repeated "Thanks!" turns) always
    land on the same side, so no validation question is ever seen in training.
    """
    rng = random.Random(seed + 1)
    groups: dict[tuple[str, str], list[dict]] = {}
    for ex in examples:
        key = (ex["kind"], " ".join(ex["messages"][0]["content"].lower().split()))
        groups.setdefault(key, []).append(ex)
    by_kind: dict[str, list[list[dict]]] = {}
    for (kind, _), items in groups.items():
        by_kind.setdefault(kind, []).append(items)
    train, val = [], []
    for kind_groups in by_kind.values():
        rng.shuffle(kind_groups)
        total = sum(len(g) for g in kind_groups)
        target = round(total * val_share) if len(kind_groups) > 1 else 0
        taken = 0
        for g in kind_groups:
            if taken < target:
                val += g
                taken += len(g)
            else:
                train += g
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def to_record(ex: dict, idx: int, prefix: str) -> dict:
    return {
        "id": f"{prefix}-{idx:06d}",
        "kind": ex["kind"],
        "source_id": ex["source_id"],
        "tools": json.dumps(ex["tools"], ensure_ascii=False),
        "messages": json.dumps(ex["messages"], ensure_ascii=False),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_preview(path: Path, records: list[dict], per_kind: int = 2) -> None:
    lines = ["# Training data preview", ""]
    shown: Counter = Counter()
    for r in records:
        if shown[r["kind"]] >= per_kind:
            continue
        shown[r["kind"]] += 1
        msgs = json.loads(r["messages"])
        tools = [t["function"]["name"] for t in json.loads(r["tools"])]
        lines += [f"## {r['id']} ({r['kind']})", "", f"**Tools available:** {', '.join(tools)}", "",
                  f"**User:** {msgs[0]['content']}", ""]
        a = msgs[1]
        if a.get("tool_calls"):
            calls = [json.dumps(c["function"], ensure_ascii=False) for c in a["tool_calls"]]
            lines += ["**Assistant calls:**", "", "```json", *calls, "```", ""]
        else:
            lines += [f"**Assistant says:** {a['content']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="local xLAM .json/.jsonl instead of downloading")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--n-tool-call", type=int, default=10000)
    p.add_argument("--no-call-share", type=float, default=0.15)
    p.add_argument("--val-share", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push-to-hub", metavar="REPO_ID", help="also upload as a PRIVATE HF dataset")
    args = p.parse_args(argv)

    decon = Decontaminator()
    rows = load_xlam(args.input)
    examples, stats = build(rows, args.n_tool_call, args.no_call_share, args.seed, decon)
    train, val = split(examples, args.val_share, args.seed)
    train_r = [to_record(e, i, "train") for i, e in enumerate(train)]
    val_r = [to_record(e, i, "val") for i, e in enumerate(val)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train_r)
    write_jsonl(out / "val.jsonl", val_r)
    write_preview(out / "preview.md", train_r)

    kinds = Counter(e["kind"] for e in examples)
    no_call = sum(v for k, v in kinds.items() if k.startswith("no_call"))
    stats.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": args.input or XLAM_DATASET,
            "seed": args.seed,
            "bfcl_reference_items": decon.reference_count,
            "bfcl_gorilla_commit": decon.gorilla_commit,
            "final_examples": len(examples),
            "train": len(train_r),
            "val": len(val_r),
            "kinds": dict(kinds),
            "no_call_share": round(no_call / len(examples), 4) if examples else 0,
        }
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {out / 'train.jsonl'} and {out / 'val.jsonl'}")

    if args.push_to_hub:
        from datasets import Dataset, DatasetDict

        DatasetDict({"train": Dataset.from_list(train_r), "validation": Dataset.from_list(val_r)}).push_to_hub(
            args.push_to_hub, private=True, token=os.getenv("HF_TOKEN")
        )
        print(f"Uploaded to https://huggingface.co/datasets/{args.push_to_hub} (private)")
    return stats


if __name__ == "__main__":
    main()
