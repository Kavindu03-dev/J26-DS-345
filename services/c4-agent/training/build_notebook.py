"""Generate notebooks/01_finetune_qlora.ipynb from the tested source files.

The notebook embeds fc_format.py and train_qlora.py with %%writefile, so Colab
runs exactly the code that the tests cover (the repo is private, so Colab
cannot clone it). Re-run after changing either file:

    python -m training.build_notebook
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks" / "01_finetune_qlora.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def build() -> dict:
    fc_format = (HERE / "fc_format.py").read_text(encoding="utf-8")
    train_qlora = (HERE / "train_qlora.py").read_text(encoding="utf-8")
    cells = [
        md("""
# C4 - Step 5: QLoRA fine-tuning for function calling

Fine-tunes one model on the prepared dataset (`Kavindu03dev/c4-fc-sft`) with the **same QLoRA
settings for every model** in the study. Generated from `services/c4-agent/training/` - do not edit
the code cells here; change the `.py` files and run `python -m training.build_notebook`.

**How to run**
1. `Runtime -> Change runtime type -> T4 GPU`
2. Set `MODEL` in the first cell (start with 0.5B; set `SMOKE_TEST = True` the very first time)
3. `Runtime -> Run all`

| Model | Approx. time on a free T4 |
|---|---|
| Qwen2.5-0.5B-Instruct | about 40-60 min |
| Qwen2.5-1.5B-Instruct | about 1.5-2.5 h |

Checkpoints are saved to Google Drive every 100 steps. If Colab disconnects, just **Run all** again:
training resumes from the last checkpoint.
"""),
        code("""
# ==== Settings: the only cell you change ====
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # or "Qwen/Qwen2.5-0.5B-Instruct" (fp16 re-run)
HF_USER = "Kavindu03dev"
SMOKE_TEST = False                       # True = 200 examples, about 5 minutes, to check the setup
PRECISION = "auto"                       # auto = bf16 on L4/A100, fp16 on T4; "fp32" if fp16 gives NaN loss

short = MODEL.split("/")[-1].lower().replace("-instruct", "")
OUTPUT_DIR = (f"/content/drive/MyDrive/tourism-agent/checkpoints/{short}" + ("" if PRECISION == "auto" else f"-{PRECISION}")
              + ("-smoke" if SMOKE_TEST else ""))
HUB_ADAPTER_ID = None if SMOKE_TEST else f"{HF_USER}/c4-{short}-fc-lora"
print("model:", MODEL, "| output:", OUTPUT_DIR, "| upload to:", HUB_ADAPTER_ID)
"""),
        code("""
# GPU check and libraries
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install -U "transformers>=4.46" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" "datasets>=3.0" wandb
"""),
        code("""
# Google Drive and logins (tokens come from Colab Secrets, never typed here)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # less GPU memory waste
from google.colab import drive, userdata
from huggingface_hub import login

drive.mount("/content/drive")
login(token=userdata.get("HF_TOKEN").strip())
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY").strip()
os.environ["WANDB_PROJECT"] = "c4-function-calling"
"""),
        md("### Training code (copied from the repo by `build_notebook.py`)"),
        code("%%writefile fc_format.py\n" + fc_format),
        code("%%writefile train_qlora.py\n" + train_qlora),
        md("""
### Check the data with the real tokenizer
Shows exactly what the model is trained on (the answer part) and confirms that almost every
example fits the chat template and length limit.
"""),
        code("""
from transformers import AutoTokenizer
from fc_format import IGNORE_INDEX, build_features, decode_record
from train_qlora import TrainConfig, load_records

tok = AutoTokenizer.from_pretrained(MODEL)
rows = load_records(TrainConfig().dataset, "train", 300)
feats = [build_features(tok, *decode_record(r), TrainConfig().max_length) for r in rows]
kept = [f for f in feats if f is not None]
print(f"kept {len(kept)}/{len(rows)} examples ({100 * len(kept) / len(rows):.1f}%)")

f = kept[0]
answer = tok.decode([i for i, l in zip(f["input_ids"], f["labels"]) if l != IGNORE_INDEX])
print("\\n===== FULL TEXT THE MODEL SEES =====\\n" + tok.decode(f["input_ids"]))
print("\\n===== PART IT IS TRAINED TO WRITE =====\\n" + answer)
assert len(kept) / len(rows) > 0.95, "Too many examples skipped - check the chat template"
"""),
        md("### Train"),
        code("""
from train_qlora import TrainConfig, train

cfg = TrainConfig(
    model=MODEL,
    output_dir=OUTPUT_DIR,
    hub_adapter_id=HUB_ADAPTER_ID,
    max_train_examples=200 if SMOKE_TEST else None,
    eval_steps=10 if SMOKE_TEST else 100,
    save_steps=10 if SMOKE_TEST else 100,
    quick_eval_n=10 if SMOKE_TEST else 50,
    precision=PRECISION,
    run_name=short + ("-smoke" if SMOKE_TEST else "-qlora-r16"),
)
summary = train(cfg)
"""),
        md("""
### Result
Copy the printed summary (or `train_summary.json` in the Drive folder) back to Claude.
Also take a screenshot of the loss curves in Weights & Biases for the slides.
"""),
        code("""
import json
print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
samples = json.load(open(f"{OUTPUT_DIR}/quick_eval_samples.json"))
for s in samples[:5]:
    print("\\n---", s["kind"], "correct" if s["correct"] else "WRONG", "\\n", s["output"])
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
