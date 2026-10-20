"""The four decoding modes of the study (proposal, Table 4).

  free           (a) native tool template, unconstrained greedy decoding
  prompt_json    (b) native template + an explicit instruction to answer only with
                     well-formed tool calls or a short plain answer; no constraint
  grammar        (c) hard grammar with XGrammar: free text is allowed, but as soon
                     as the model opens <tool_call>, the rest of the call is forced
                     to be {"name": <one of the tools>, "arguments": <valid JSON for
                     that tool's schema>}  -> "call OR plain answer"
  reason_grammar (d) "reason free, constrain late": the model is asked to write one
                     short reasoning line first (unconstrained), then the same
                     grammar as (c) applies to the call itself

All modes use greedy decoding (temperature 0) so runs are reproducible.
The prompt and grammar builders are pure Python and unit-tested; generate()
needs torch/transformers/xgrammar and runs on the GPU in Colab.
"""

from __future__ import annotations

import json
import time
from typing import Any

MODES = ["free", "prompt_json", "grammar", "reason_grammar"]

PROMPT_JSON_INSTRUCTION = (
    "Answer in exactly one of two ways. (1) If a tool is needed, output ONLY the tool call(s), each as "
    '<tool_call>\n{"name": <tool name>, "arguments": <JSON object>}\n</tool_call>, with valid JSON and only '
    "arguments defined in the tool's schema. (2) If no tool fits or none is needed, reply with a short plain answer "
    "and no tool call."
)
REASON_INSTRUCTION = (
    "First write one short line starting with 'Reasoning:' that says which tool (if any) is needed and why. "
    "Then give the tool call(s) or the answer."
)

CALL_BEGIN = '<tool_call>\n{"name": "'
CALL_TRIGGER = "<tool_call>"


def build_messages(messages: list[dict], mode: str, system_prompt: str | None = None) -> list[dict]:
    """Add the mode's instructions as a system message (merged with any existing one)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    parts = [system_prompt] if system_prompt else []
    if mode == "prompt_json":
        parts.append(PROMPT_JSON_INSTRUCTION)
    if mode == "reason_grammar":
        parts.append(REASON_INSTRUCTION)
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        parts.insert(0, msgs.pop(0)["content"])
    return ([{"role": "system", "content": "\n\n".join(parts)}] if parts else []) + msgs


def structural_tag(tools: list[dict]) -> dict:
    """XGrammar structural tag: free text, but every <tool_call> must be a valid call to a known tool.

    Matches the Qwen2.5 tool-call format exactly:
        <tool_call>\\n{"name": "NAME", "arguments": ARGS}\\n</tool_call>
    """
    tags = []
    for t in tools:
        fn = t["function"]
        schema = dict(fn.get("parameters") or {"type": "object", "properties": {}})
        schema.setdefault("type", "object")
        schema["additionalProperties"] = False
        tags.append(
            {
                "type": "tag",
                "begin": f'{CALL_BEGIN}{fn["name"]}", "arguments": ',
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}\n</tool_call>",
            }
        )
    return {
        "type": "structural_tag",
        "format": {"type": "triggered_tags", "triggers": [CALL_TRIGGER], "tags": tags, "at_least_one": False, "stop_after_first": False},
    }


def legacy_structural_tag_items(tools: list[dict]) -> list[dict]:
    """Same grammar for older xgrammar versions (compile_structural_tag(tags, triggers))."""
    return [
        {
            "begin": f'{CALL_BEGIN}{t["function"]["name"]}", "arguments": ',
            "schema": json.dumps(dict(t["function"].get("parameters") or {"type": "object", "properties": {}}, additionalProperties=False)),
            "end": "}\n</tool_call>",
        }
        for t in tools
    ]


def strip_reasoning(text: str) -> str:
    """Remove the leading 'Reasoning:' line of mode (d) before parsing calls / answers."""
    lines = text.strip().splitlines()
    while lines and lines[0].strip().lower().startswith("reasoning:"):
        lines.pop(0)
    return "\n".join(lines).strip()


class Generator:
    """Greedy generation with optional XGrammar constraint. Load once, call many times."""

    def __init__(self, base_model: str, adapter: str | None, max_new_tokens: int = 384):
        import torch
        import xgrammar as xgr
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch, self.xgr = torch, xgr
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU in this runtime. In Colab: Runtime -> Change runtime type -> T4 GPU -> Save, then Run all. "
                "If Colab says 'Cannot connect to GPU backend', this account's free GPU time is used up for now."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        modern = torch.cuda.get_device_capability()[0] >= 8
        dtype = torch.bfloat16 if modern else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, device_map={"": 0}, attn_implementation="sdpa" if modern else "eager"
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens
        vocab = AutoConfig.from_pretrained(base_model).vocab_size
        self.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=vocab))
        self._grammar_cache: dict[str, Any] = {}

    def _compiled(self, tools: list[dict]):
        key = json.dumps(tools, sort_keys=True)
        if key not in self._grammar_cache:
            try:
                g = self.compiler.compile_structural_tag(json.dumps(structural_tag(tools)))
            except Exception:  # older xgrammar
                items = [self.xgr.StructuralTagItem(**i) for i in legacy_structural_tag_items(tools)]
                g = self.compiler.compile_structural_tag(items, [CALL_TRIGGER])
            self._grammar_cache[key] = g
        return self._grammar_cache[key]

    def __call__(self, messages: list[dict], tools: list[dict], mode: str, system_prompt: str | None = None) -> dict:
        torch = self.torch
        msgs = build_messages(messages, mode, system_prompt)
        prompt = self.tokenizer.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        kwargs = {}
        if mode in ("grammar", "reason_grammar"):
            from xgrammar.contrib.hf import LogitsProcessor

            kwargs["logits_processor"] = [LogitsProcessor(self._compiled(tools))]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens + (64 if mode == "reason_grammar" else 0),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        new = out[0][ids["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return {"text": text, "latency_s": round(latency, 4), "new_tokens": int(new.shape[0])}
