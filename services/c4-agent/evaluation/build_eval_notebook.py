"""Generate notebooks/02_evaluate_decoding.ipynb from the tested source files.

    python -m evaluation.build_eval_notebook
"""

from __future__ import annotations

import json
from pathlib import Path

from training.build_notebook import code, md

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_evaluate_decoding.ipynb"

FILES = {  # file name in Colab -> source in the repo
    "fc_format.py": ROOT / "training" / "fc_format.py",
    "tool_defs.py": ROOT / "src" / "tools" / "definitions.py",
    "scoring.py": ROOT / "evaluation" / "scoring.py",
    "decoding.py": ROOT / "evaluation" / "decoding.py",
    "run_eval.py": ROOT / "evaluation" / "run_eval.py",
    "tourism_testset_v1.jsonl": ROOT / "evaluation" / "data" / "tourism_testset_v1.jsonl",
    "bfcl_v4_subset.jsonl": ROOT / "evaluation" / "data" / "bfcl_v4_subset.jsonl",
}


def build() -> dict:
    cells = [
        md("""
# C4 - Steps 7-8: decoding-mode experiment

Runs one model under the **four decoding modes** of the study on
- **Layer 2:** the tourism test set v1.1 (150 questions, 7 categories)
- **Layer 1:** a BFCL v4 subset (100 simple + 100 multiple + 100 irrelevance)

and writes predictions, the results table, the constraint tax and McNemar tests (Holm-corrected)
to Google Drive. Generated from `services/c4-agent/evaluation/` - edit the `.py` files, then run
`python -m evaluation.build_eval_notebook`.

| Mode | What it is |
|---|---|
| `free` | native tool template, no constraint |
| `prompt_json` | + instruction to answer only with valid calls or a short answer |
| `grammar` | XGrammar: every `<tool_call>` must be valid JSON for one of the tools |
| `reason_grammar` | one free reasoning line first, then the same grammar ("reason free, constrain late") |

**How to run:** T4 GPU runtime, set the first cell, `Runtime -> Run all`. Start with `QUICK_TEST = True`
(5 items per dataset, about 3 minutes). Interrupted runs continue where they stopped.
"""),
        code("""
# ==== Settings: the only cell you change ====
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "Kavindu03dev/c4-qwen2.5-0.5b-fc-lora"   # None = evaluate the base model (not fine-tuned)
QUICK_TEST = True                                    # True = 5 items per dataset to check the setup

LABEL = BASE_MODEL.split("/")[-1].lower().replace("-instruct", "") + ("-ft" if ADAPTER else "-base")
OUT_DIR = f"/content/drive/MyDrive/tourism-agent/results/{LABEL}" + ("-quick" if QUICK_TEST else "")
print("evaluating:", LABEL, "->", OUT_DIR)
"""),
        code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install -U "transformers>=4.46" "peft>=0.13" "accelerate>=1.0" xgrammar jsonschema scipy
"""),
        code("""
import os
from google.colab import drive, userdata
from huggingface_hub import login
drive.mount("/content/drive")
login(token=userdata.get("HF_TOKEN").strip())
"""),
        md("### Evaluation code and test data (copied from the repo by `build_eval_notebook.py`)"),
    ]
    for name, src in FILES.items():
        cells.append(code(f"%%writefile {name}\n" + src.read_text(encoding="utf-8")))
    cells += [
        md("### Load the model and try one question in every mode"),
        code("""
from decoding import Generator, MODES
from run_eval import load_jsonl
from tool_defs import SYSTEM_PROMPT, TOOLS

gen = Generator(BASE_MODEL, ADAPTER)
tourism = load_jsonl("tourism_testset_v1.jsonl")
bfcl = load_jsonl("bfcl_v4_subset.jsonl")
q = next(t for t in tourism if t["id"] == "T01")
print("Q:", q["messages"][-1]["content"])
for mode in MODES:
    out = gen(q["messages"], TOOLS, mode, SYSTEM_PROMPT)
    print(f"\\n--- {mode} ({out['latency_s']} s)\\n{out['text']}")
"""),
        md("### Run the experiment"),
        code("""
import random
from run_eval import run

def pick(items, n):  # quick test: a few items from every category
    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    return [it for cat in by_cat.values() for it in cat[:n]]

datasets = {"tourism": pick(tourism, 1) if QUICK_TEST else tourism,
            "bfcl": pick(bfcl, 2) if QUICK_TEST else bfcl}
print({k: len(v) for k, v in datasets.items()}, "items x", len(MODES), "modes")
rows = run(gen, LABEL, datasets, OUT_DIR)
print("done:", len(rows), "responses")
"""),
        md("### Results"),
        code("""
import json, pandas as pd
from run_eval import analyse, write_results

results = analyse(rows)
table = write_results(results, OUT_DIR)
df = pd.read_csv(table)
display(df[df.dataset.isin(["tourism", "bfcl"])])
print("\\nHypotheses:")
print(json.dumps(results["hypotheses"], indent=2))
"""),
        md("### Per-category accuracy (tourism)"),
        code("""
cat = df[df.dataset.str.startswith("tourism:")].pivot(index="dataset", columns="mode", values="accuracy_%")
display(cat[[m for m in MODES if m in cat.columns]])
"""),
        md("### Demo material: questions where free decoding and grammar decoding disagree"),
        code("""
by = {}
for r in rows:
    by.setdefault((r["dataset"], r["id"]), {})[r["mode"]] = r
shown = 0
for (ds, i), m in by.items():
    if "free" in m and "grammar" in m and m["free"]["correct"] != m["grammar"]["correct"]:
        print(f"\\n=== {ds} {i} | free correct={m['free']['correct']} | grammar correct={m['grammar']['correct']}")
        print("FREE:   ", m["free"]["texts"][0][:300])
        print("GRAMMAR:", m["grammar"]["texts"][0][:300])
        shown += 1
    if shown >= 8:
        break
print(f"\\nFiles saved in {OUT_DIR}: predictions.jsonl, results.json, results_table.csv")
"""),
    ]
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
