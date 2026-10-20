"""Turning chat examples into training features, and model output back into calls.

Kept free of torch/transformers imports so it can be unit-tested anywhere.
Used by train_qlora.py now and by the decoding-mode evaluation (Step 7) later.
"""

from __future__ import annotations

import json
import re
from typing import Any

IGNORE_INDEX = -100  # label value the loss ignores


def decode_record(record: dict) -> tuple[list[dict], list[dict]]:
    """A JSONL / Hub row -> (messages, tools). tools and messages are stored as JSON strings."""
    messages = record["messages"]
    tools = record["tools"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    if isinstance(tools, str):
        tools = json.loads(tools)
    return messages, tools


def build_features(tokenizer: Any, messages: list[dict], tools: list[dict], max_length: int) -> dict | None:
    """Tokenise one example so that ONLY the assistant's answer is trained on.

    The prompt (system text, tool definitions, user message) gets label -100,
    so the loss covers just the tool call or the text reply, including the
    end-of-turn token that teaches the model to stop.

    Returns None when the example cannot be rendered by this model's chat
    template (e.g. Llama-3.2's template rejects parallel calls) or is too long.
    """
    try:
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tools=tools, add_generation_prompt=True, tokenize=False
        )
        full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    except Exception:  # template refuses this example
        return None
    if not full.startswith(prompt):
        return None
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if not answer_ids:
        return None
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


_QWEN_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict] | None:
    """Model output -> list of {"name", "arguments"}; [] for a plain answer; None if a call is malformed.

    Strict, repair-free parsing as the proposal requires: anything that looks
    like a call but is not valid JSON with a name and an arguments object counts
    as malformed (None). Formats understood:
      Qwen2.5:  <tool_call>{"name": ..., "arguments": {...}}</tool_call> (repeated)
      Llama-3.2: {"name": ..., "parameters": {...}} as the whole message
    """
    text = text.strip()
    blocks = _QWEN_CALL.findall(text)
    if blocks:
        calls = []
        for b in blocks:
            call = _to_call(b)
            if call is None:
                return None
            calls.append(call)
        return calls
    if "<tool_call>" in text:
        return None  # opened a call but never closed it
    if text.startswith("{") and '"name"' in text:
        call = _to_call(text)
        return [call] if call is not None else None
    return []


def _to_call(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):  # some models emit arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": obj["name"], "arguments": args}


def gold_calls(messages: list[dict]) -> list[dict]:
    """The reference calls in a training example ([] for a no-call example)."""
    calls = messages[-1].get("tool_calls") or []
    return [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls]


def calls_match(predicted: list[dict] | None, gold: list[dict]) -> bool:
    """Exact match of the set of calls (order-insensitive; argument values compared as JSON)."""
    if predicted is None:
        return False
    key = lambda c: json.dumps({"n": c["name"], "a": c["arguments"]}, sort_keys=True)
    return sorted(map(key, predicted)) == sorted(map(key, gold))
