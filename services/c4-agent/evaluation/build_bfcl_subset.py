"""Build the BFCL v4 subset used for Layer-1 evaluation in the progress review.

    python -m evaluation.build_bfcl_subset <gorilla>/berkeley-function-call-leaderboard/bfcl_eval/data

Takes a seeded random sample of N items from each single-turn category used in
the proposal (simple, multiple, irrelevance), joins the ground truth, and writes
evaluation/data/bfcl_v4_subset.jsonl. The full categories are run for the thesis.
BFCL data: Berkeley Function Calling Leaderboard (Gorilla project), Apache-2.0.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

CATEGORIES = {"simple_python": 100, "multiple": 100, "irrelevance": 100}
SEED = 42
OUT = Path(__file__).resolve().parent / "data" / "bfcl_v4_subset.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(data_dir: str) -> None:
    root = Path(data_dir)
    try:
        commit = subprocess.run(["git", "-C", data_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    rows = []
    for cat, n in CATEGORIES.items():
        items = load(root / f"BFCL_v4_{cat}.json")
        answers = {}
        answer_file = root / "possible_answer" / f"BFCL_v4_{cat}.json"
        if answer_file.exists():
            answers = {a["id"]: a["ground_truth"] for a in load(answer_file)}
        for it in sorted(random.Random(SEED).sample(items, n), key=lambda r: r["id"]):
            rows.append({
                "id": it["id"],
                "category": cat,
                "messages": it["question"][0],
                "function": it["function"],
                **({"ground_truth": answers[it["id"]]} if it["id"] in answers else {}),
                "source_commit": commit,
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} BFCL items to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
