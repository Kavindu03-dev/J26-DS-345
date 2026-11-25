"""Render one prompt with tools and print the raw completion (README ground rule 2)."""

import argparse
from typing import Any

import httpx

from c4.common.config import get_settings
from c4.common.schemas import EXPERIMENT_TOOLS, load_tools, to_openai_tools

MESSAGES = [{"role": "user", "content": "How clean is the hotel with id NE-001?"}]


def render_prompt(model: str) -> str:
    from transformers import AutoTokenizer  # only installed in the gpu group

    token = get_settings().hf_token.get_secret_value() or None
    tokenizer = AutoTokenizer.from_pretrained(model, token=token)
    return tokenizer.apply_chat_template(
        MESSAGES,
        tools=to_openai_tools(load_tools(EXPERIMENT_TOOLS)),
        add_generation_prompt=True,
        tokenize=False,
    )


def build_request(model: str, prompt: str, max_tokens: int = 256) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        # The chat template already wrote BOS and the other special tokens.
        "add_special_tokens": False,
        # Keep special tokens so the raw output is what the model really produced.
        "skip_special_tokens": False,
    }


def generate(model: str, base_url: str | None = None) -> str:
    url = (base_url or get_settings().vllm_base_url).rstrip("/") + "/completions"
    reply = httpx.post(url, json=build_request(model, render_prompt(model)), timeout=120)
    reply.raise_for_status()
    return reply.json()["choices"][0]["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    print(generate(parser.parse_args().model))
