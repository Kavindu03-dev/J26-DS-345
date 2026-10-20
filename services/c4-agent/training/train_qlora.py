"""QLoRA fine-tuning for function calling (Step 5).

The SAME settings are used for every model in the study (proposal, Table 4:
"QLoRA ... held identical for all models"); only --model changes.

Runs on a free Colab T4 (16 GB). Checkpoints go to Google Drive every
--save-steps steps, and a restarted run resumes from the last checkpoint, so a
Colab disconnect costs at most a few minutes of work.

Usage (normally via notebooks/01_finetune_qlora.ipynb):
    python -m training.train_qlora --model Qwen/Qwen2.5-0.5B-Instruct \
        --output-dir /content/drive/MyDrive/tourism-agent/checkpoints/qwen2.5-0.5b \
        --hub-adapter-id <user>/c4-qwen2.5-0.5b-fc-lora
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # works both as "training.train_qlora" and as a standalone file in Colab
    from .fc_format import IGNORE_INDEX, build_features, calls_match, decode_record, gold_calls, parse_tool_calls
except ImportError:
    from fc_format import IGNORE_INDEX, build_features, calls_match, decode_record, gold_calls, parse_tool_calls

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset: str = "Kavindu03dev/c4-fc-sft"
    output_dir: str = "/content/drive/MyDrive/tourism-agent/checkpoints/qwen2.5-0.5b"
    hub_adapter_id: str | None = None
    # --- identical for every model in the study ---
    max_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 1.0
    # 1 x 16: same effective batch of 16, but Qwen's 152k-word vocabulary makes
    # the loss step too large for a T4 at 4 examples per micro-batch.
    per_device_batch: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    seed: int = 42
    # --- speed only (do not change results) ---
    gradient_checkpointing: str = "auto"  # auto = on when eager attention is used (T4), else off; "true" / "false"
    eval_subset: int = 200  # validation examples used for the progress checks; final eval uses all
    precision: str = "auto"  # auto = bf16 on GPUs with native bf16, else fp16; or force "bf16" / "fp16" / "fp32"
    # fp16 can overflow to NaN with Qwen2.5 on a T4; "fp32" is slower but numerically safe there
    attention: str = "auto"  # auto = "sdpa" on compute capability >= 8, "eager" on older GPUs (T4)
    optim: str = "adamw_torch"  # full-precision AdamW; LoRA has only ~1-2% trainable params, so 8-bit gains nothing
    # --- logging / checkpointing ---
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    wandb_project: str = "c4-function-calling"
    run_name: str | None = None
    quick_eval_n: int = 50
    max_train_examples: int | None = None  # small number = smoke test
    lora_targets: list[str] = field(default_factory=lambda: list(LORA_TARGETS))


def load_records(dataset: str, split: str, limit: int | None = None) -> list[dict]:
    """Rows from the private Hub dataset (or a local JSONL path)."""
    if dataset.endswith(".jsonl"):
        path = Path(dataset)
        if split != "train":
            path = path.with_name(path.name.replace("train", "val"))
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        from datasets import load_dataset

        rows = [dict(r) for r in load_dataset(dataset, split=split)]
    return rows[:limit] if limit else rows


def tokenize_all(tokenizer, records: list[dict], max_length: int) -> tuple[list[dict], dict]:
    feats, skipped = [], 0
    for rec in records:
        messages, tools = decode_record(rec)
        f = build_features(tokenizer, messages, tools, max_length)
        if f is None:
            skipped += 1
        else:
            feats.append(f)
    lengths = sorted(len(f["input_ids"]) for f in feats) or [0]
    stats = {
        "examples": len(records),
        "kept": len(feats),
        "skipped": skipped,
        "tokens_p50": lengths[len(lengths) // 2],
        "tokens_p95": lengths[int(len(lengths) * 0.95)],
        "tokens_max": lengths[-1],
        "tokens_total": sum(lengths),
    }
    return feats, stats


def compatible_training_args(params: set[str], kwargs: dict, n_train: int, cfg: "TrainConfig") -> dict:
    """Adapt TrainingArguments keywords to the installed transformers version.

    transformers renames or removes options between major versions (e.g.
    warmup_ratio was removed in v5, evaluation_strategy became eval_strategy).
    The training itself must stay identical, so renamed options are converted
    to their equivalent and anything else that is unknown is reported.
    """
    kwargs = dict(kwargs)
    if "warmup_ratio" in kwargs and "warmup_ratio" not in params:
        steps_per_epoch = math.ceil(n_train / (cfg.per_device_batch * cfg.grad_accum))
        total_steps = math.ceil(steps_per_epoch * cfg.epochs)
        kwargs["warmup_steps"] = max(1, round(kwargs.pop("warmup_ratio") * total_steps))
    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    unknown = sorted(k for k in kwargs if k not in params)
    for k in unknown:
        print(f"[compat] this transformers version has no '{k}' option; skipped")
        kwargs.pop(k)
    return kwargs


def finite_loss(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def loss_report(log_history: list[dict]) -> dict:
    """First/last logged training loss and how many logged values were NaN/inf (or a hidden 0.0)."""
    losses = [h["loss"] for h in log_history if "loss" in h]
    evals = [h["eval_loss"] for h in log_history if "eval_loss" in h]
    bad = [x for x in losses if not finite_loss(x) or x == 0.0]
    return {
        "logged_losses": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "bad_losses": len(bad),
        "eval_loss_first": evals[0] if evals else None,
        "eval_loss_last": evals[-1] if evals else None,
    }


def lora_b_norm(model) -> float:
    """LoRA B matrices start at zero; a trained adapter must have moved them."""
    total = 0.0
    for n, p in model.named_parameters():
        if "lora_B" in n:
            total += float(p.detach().float().norm()) ** 2
    return total ** 0.5


class NanGuard:
    """Stop training early when the loss is not a real number, instead of wasting hours of GPU.

    With fp16 the Trainer's GradScaler silently skips every update whose gradients overflow,
    so a broken run still "finishes" with an untrained adapter. Two bad logs in a row = stop.
    """

    def __init__(self, patience: int = 2):
        self.patience, self.bad = patience, 0

    def check(self, logs: dict) -> bool:
        """Returns True when training must stop."""
        if "loss" not in logs:
            return False
        self.bad = self.bad + 1 if (not finite_loss(logs["loss"]) or logs["loss"] == 0.0) else 0
        return self.bad >= self.patience


NAN_MESSAGE = (
    "Training loss is NaN/inf (or a hidden 0.0) - the weights are not being updated. "
    "On a T4 this is fp16 overflow: set precision='fp32' (slower, safe), "
    "or use an L4/A100 runtime where bf16 is native."
)


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (width - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], IGNORE_INDEX) for b in batch]),
        }


def quick_eval(model, tokenizer, records: list[dict], n: int) -> dict:
    """Sanity check only (greedy, free decoding): does the model produce correct calls?

    The real study evaluation (4 decoding modes, BFCL, tourism set) is Step 7-8.
    """
    import torch

    model.eval()
    correct, malformed, rows = 0, 0, []
    for rec in records[:n]:
        messages, tools = decode_record(rec)
        prompt = tokenizer.apply_chat_template(messages[:-1], tools=tools, add_generation_prompt=True, tokenize=False)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=384, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred, gold = parse_tool_calls(text), gold_calls(messages)
        ok = calls_match(pred, gold)
        correct += ok
        malformed += pred is None
        rows.append({"id": rec.get("id"), "kind": rec.get("kind"), "correct": ok, "output": text[:500]})
    n = max(1, len(rows))
    return {"n": len(rows), "exact_match": round(correct / n, 4), "malformed": round(malformed / n, 4), "samples": rows}


def _dtype_kwarg(dtype) -> dict:
    """transformers >= 4.56 calls it dtype; older versions call it torch_dtype."""
    import transformers
    from packaging.version import Version

    return {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}


def train(cfg: TrainConfig) -> dict:
    # Reduce GPU memory fragmentation; must be set before CUDA starts.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Native bf16 needs compute capability >= 8 (A100, L4, ...). torch.cuda.is_bf16_supported()
    # also returns True on a T4 (sm75) because bf16 can be *emulated* there, which is slow,
    # so check the hardware directly. T4 -> fp16 tensor cores.
    bf16 = cfg.precision == "bf16" or (
        cfg.precision == "auto" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    )
    fp32 = cfg.precision == "fp32"
    dtype = torch.bfloat16 if bf16 else (torch.float32 if fp32 else torch.float16)
    # SDPA's fused fp16 kernels can hit "illegal memory access" on pre-Ampere GPUs (T4);
    # standard ("eager") attention computes the same thing and is stable there.
    modern_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    attention = cfg.attention if cfg.attention != "auto" else ("sdpa" if modern_gpu else "eager")
    # Eager attention stores full attention matrices; checkpointing keeps memory safe (same maths, slower).
    gc = str(cfg.gradient_checkpointing).lower()
    grad_ckpt = attention == "eager" if gc == "auto" else gc in ("1", "true", "yes")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_records(cfg.dataset, "train", cfg.max_train_examples)
    val_rows = load_records(cfg.dataset, "validation" if not cfg.dataset.endswith(".jsonl") else "val")
    train_feats, train_stats = tokenize_all(tokenizer, train_rows, cfg.max_length)
    val_feats, val_stats = tokenize_all(tokenizer, val_rows, cfg.max_length)
    print("train data:", train_stats)
    print("val data:  ", val_stats)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        **_dtype_kwarg(dtype),
        device_map={"": 0},
        attn_implementation=attention,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    for p in model.parameters():  # fp16 AMP needs fp32 trainable weights
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    run_name = cfg.run_name or f"{cfg.model.split('/')[-1]}-qlora-r{cfg.lora_r}"

    arg_kwargs = dict(
        output_dir=str(out),
        run_name=run_name,
        per_device_train_batch_size=cfg.per_device_batch,
        per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        fp16=not bf16 and not fp32,
        bf16=bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        optim=cfg.optim,
        group_by_length=True,
        report_to=["wandb"] if use_wandb else [],
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        logging_nan_inf_filter=False,  # show NaN honestly instead of replacing it (NanGuard needs it)
    )
    params = set(inspect.signature(TrainingArguments.__init__).parameters)
    args = TrainingArguments(**compatible_training_args(params, arg_kwargs, len(train_feats), cfg))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_feats,
        eval_dataset=val_feats[: cfg.eval_subset] if cfg.eval_subset else val_feats,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    from transformers import TrainerCallback

    guard = NanGuard()

    class _NanStop(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if guard.check(logs or {}):
                print("\n!!! " + NAN_MESSAGE)
                control.should_training_stop = True
                state.nan_stopped = True

    trainer.add_callback(_NanStop())

    resume = get_last_checkpoint(str(out)) if any(out.glob("checkpoint-*")) else None
    if resume:
        print(f"Resuming from {resume}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    minutes = round((time.time() - t0) / 60, 1)
    losses = loss_report(trainer.state.log_history)
    b_norm = lora_b_norm(model)
    if getattr(trainer.state, "nan_stopped", False) or b_norm == 0.0:
        print(json.dumps({"loss_report": losses, "lora_B_norm": b_norm}, indent=2))
        raise RuntimeError(NAN_MESSAGE + " Nothing was saved or uploaded.")
    final_eval = trainer.evaluate(eval_dataset=val_feats)  # all validation examples

    adapter_dir = out / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    check = quick_eval(model, tokenizer, val_rows, cfg.quick_eval_n) if cfg.quick_eval_n else None
    summary = {
        "config": asdict(cfg),
        "run_name": run_name,
        "precision": "bf16" if bf16 else ("fp32" if fp32 else "fp16"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "attention": attention,
        "gradient_checkpointing": grad_ckpt,
        "optimizer": cfg.optim,
        "trainable_params": trainable,
        "total_params": total,
        "train_data": train_stats,
        "val_data": val_stats,
        "train_minutes_this_session": minutes,
        "train_tokens_per_second": round(train_stats["tokens_total"] * cfg.epochs / max(1.0, result.metrics.get("train_runtime", 1.0)), 1),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "train_loss": result.training_loss,
        "loss_report": losses,
        "lora_B_norm": round(b_norm, 4),  # 0 would mean the adapter never trained
        "final_eval_loss": final_eval.get("eval_loss"),
        "global_steps": result.global_step,
        "quick_eval": {k: v for k, v in (check or {}).items() if k != "samples"},
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if check:
        (out / "quick_eval_samples.json").write_text(json.dumps(check["samples"], indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))

    if cfg.hub_adapter_id:
        model.push_to_hub(cfg.hub_adapter_id, private=True)
        tokenizer.push_to_hub(cfg.hub_adapter_id, private=True)
        print(f"Adapter uploaded to https://huggingface.co/{cfg.hub_adapter_id} (private)")
    return summary


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in TrainConfig.__dataclass_fields__.values():
        if f.name == "lora_targets":
            continue
        default = f.default
        typ = {int: int, float: float}.get(type(default), str)
        if isinstance(default, bool):
            typ = lambda v: str(v).lower() in ("1", "true", "yes")
        if f.name in ("max_train_examples",):
            typ = int
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=typ, default=default)
    return TrainConfig(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
