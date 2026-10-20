"""Checks that need no GPU: CLI parsing, notebook generation and code sync."""

import ast
import json

from training import build_notebook
from training.train_qlora import LORA_TARGETS, TrainConfig, compatible_training_args, parse_args


def test_defaults_match_the_study_settings():
    cfg = TrainConfig()
    assert (cfg.lora_r, cfg.lora_alpha, cfg.learning_rate, cfg.seed) == (16, 32, 2e-4, 42)
    assert cfg.per_device_batch * cfg.grad_accum == 16
    assert cfg.lora_targets == LORA_TARGETS


def test_cli_overrides_only_what_is_given():
    cfg = parse_args(["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--max-train-examples", "200"])
    assert cfg.model == "Qwen/Qwen2.5-1.5B-Instruct" and cfg.max_train_examples == 200
    assert cfg.lora_r == TrainConfig().lora_r
    assert parse_args(["--gradient-checkpointing", "true"]).gradient_checkpointing == "true"
    assert TrainConfig().gradient_checkpointing == "auto"


def test_committed_notebook_is_up_to_date():
    """Fails if fc_format.py / train_qlora.py changed without rebuilding the notebook."""
    committed = json.loads(build_notebook.OUT.read_text(encoding="utf-8"))
    assert committed["cells"] == build_notebook.build()["cells"]


def test_notebook_code_cells_are_valid_python():
    nb = build_notebook.build()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.startswith("%%writefile"):
            src = src.split("\n", 1)[1]
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!"))
        ast.parse(src)


ARGS = {"output_dir": "x", "warmup_ratio": 0.03, "eval_strategy": "steps", "learning_rate": 2e-4}


def test_new_transformers_without_warmup_ratio_gets_equivalent_warmup_steps():
    params = {"output_dir", "warmup_steps", "eval_strategy", "learning_rate"}
    out = compatible_training_args(params, ARGS, n_train=11168, cfg=TrainConfig())
    # 11168 / 16 = 698 steps; 3% of 698 = 21 warm-up steps
    assert out == {"output_dir": "x", "warmup_steps": 21, "eval_strategy": "steps", "learning_rate": 2e-4}


def test_old_transformers_keeps_ratio_and_renames_eval_strategy():
    params = {"output_dir", "warmup_ratio", "evaluation_strategy", "learning_rate"}
    out = compatible_training_args(params, ARGS, n_train=100, cfg=TrainConfig())
    assert out == {"output_dir": "x", "warmup_ratio": 0.03, "evaluation_strategy": "steps", "learning_rate": 2e-4}


def test_unknown_options_are_reported_and_skipped(capsys):
    out = compatible_training_args({"output_dir"}, {"output_dir": "x", "group_by_length": True}, 10, TrainConfig())
    assert out == {"output_dir": "x"}
    assert "group_by_length" in capsys.readouterr().out


def test_nan_guard_stops_after_two_bad_logs():
    from training.train_qlora import NanGuard

    g = NanGuard()
    assert not g.check({"loss": 1.2})
    assert not g.check({"loss": float("nan")})
    assert not g.check({"loss": 0.9})  # one bad log is tolerated
    assert not g.check({"eval_loss": 0.5})  # eval logs are ignored
    assert not g.check({"loss": float("inf")})
    assert g.check({"loss": 0.0})  # 0.0 is how a hidden NaN shows up


def test_loss_report_counts_bad_values():
    from training.train_qlora import loss_report

    r = loss_report([{"loss": 2.0}, {"eval_loss": 1.5}, {"loss": float("nan")}, {"loss": 0.4}, {"eval_loss": 0.6}])
    assert (r["logged_losses"], r["loss_first"], r["loss_last"], r["bad_losses"]) == (3, 2.0, 0.4, 1)
    assert (r["eval_loss_first"], r["eval_loss_last"]) == (1.5, 0.6)
