"""Build the BFCL reference file used to decontaminate the training data.

We must not train on questions that appear in the BFCL test set, or the BFCL
scores (Layer 1 of the evaluation) would be inflated. This script extracts the
user questions and function names of every single-turn BFCL v4 test item and
saves them in bfcl_v4_reference.json, which prepare_data.py reads.

Run once (the output file is committed, so you normally do not need to):

    git clone --depth 1 --filter=blob:none --sparse https://github.com/ShishirPatil/gorilla.git
    cd gorilla && git sparse-checkout set berkeley-function-call-leaderboard/bfcl_eval/data && cd ..
    python -m training.build_bfcl_reference gorilla/berkeley-function-call-leaderboard/bfcl_eval/data

BFCL data: Berkeley Function Calling Leaderboard (Gorilla project), Apache-2.0.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Single-turn categories used in our evaluation (proposal: single-turn + irrelevance).
CATEGORIES = [
    "simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance",
]
OUT = Path(__file__).with_name("bfcl_v4_reference.json")


def main(data_dir: str) -> None:
    items = []
    for cat in CATEGORIES:
        path = Path(data_dir) / f"BFCL_v4_{cat}.json"
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                turns = row["question"][0] if row["question"] else []
                text = " ".join(m["content"] for m in turns if m.get("role") == "user")
                items.append(
                    {
                        "id": row["id"],
                        "question": text,
                        "functions": sorted({fn["name"] for fn in row.get("function", [])}),
                    }
                )
    try:
        commit = subprocess.run(
            ["git", "-C", data_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    payload = {
        "source": "BFCL v4 (github.com/ShishirPatil/gorilla), Apache-2.0",
        "gorilla_commit": commit,
        "categories": CATEGORIES,
        "count": len(items),
        "items": items,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"Wrote {len(items)} BFCL items to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
