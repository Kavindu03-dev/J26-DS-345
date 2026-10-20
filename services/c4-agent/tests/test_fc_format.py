"""Tests for training/fc_format.py (label masking and strict call parsing).

The tokenizer here is a stand-in: character-level ids plus a Jinja template
with the same structure as Qwen2.5-Instruct's tool template. The Colab
notebook repeats the masking check with the real Qwen tokenizer.
"""

import json

import jinja2
import pytest

from training.fc_format import (
    IGNORE_INDEX, build_features, calls_match, decode_record, gold_calls, parse_tool_calls,
)

QWEN_LIKE_TEMPLATE = """
{%- if tools %}<|im_start|>system
You are a helpful assistant.

# Tools
<tools>{% for t in tools %}
{{ t | tojson }}{% endfor %}
</tools><|im_end|>
{% endif %}
{%- for m in messages %}
{%- if m.role == 'user' or (m.role == 'assistant' and not m.tool_calls) %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% elif m.role == 'assistant' %}<|im_start|>assistant{% if m.content %}
{{ m.content }}{% endif %}{% for c in m.tool_calls %}
<tool_call>
{"name": "{{ c.function.name }}", "arguments": {{ c.function.arguments | tojson }}}
</tool_call>{% endfor %}<|im_end|>
{% endif %}{% endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


class FakeTokenizer:
    def __init__(self, template=QWEN_LIKE_TEMPLATE):
        env = jinja2.Environment()
        env.policies["json.dumps_kwargs"] = {"sort_keys": False}
        self.t = env.from_string(template)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        return self.t.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
CALL = [{"role": "user", "content": "Weather in Kandy?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Kandy"}}}]}]
TEXT = [{"role": "user", "content": "Thanks!"}, {"role": "assistant", "content": "You're welcome!"}]


def _answer(f):
    return "".join(chr(i) for i, l in zip(f["input_ids"], f["labels"]) if l != IGNORE_INDEX)


def test_only_the_tool_call_is_trained_on():
    f = build_features(FakeTokenizer(), CALL, TOOLS, 4096)
    answer = _answer(f)
    assert '"name": "get_weather"' in answer and answer.rstrip().endswith("<|im_end|>")
    assert "Weather in Kandy?" not in answer and "# Tools" not in answer
    assert len(f["input_ids"]) == len(f["labels"]) == len(f["attention_mask"])


def test_only_the_text_reply_is_trained_on():
    answer = _answer(build_features(FakeTokenizer(), TEXT, TOOLS, 4096))
    assert answer.startswith("You're welcome!") and "Thanks!" not in answer


def test_too_long_examples_are_skipped():
    assert build_features(FakeTokenizer(), CALL, TOOLS, 50) is None


def test_template_that_rejects_an_example_is_skipped():
    class Refusing(FakeTokenizer):
        def apply_chat_template(self, *a, **k):
            raise ValueError("only single tool-calls supported")
    assert build_features(Refusing(), CALL, TOOLS, 4096) is None


def test_decode_record_accepts_json_strings():
    msgs, tools = decode_record({"messages": json.dumps(CALL), "tools": json.dumps(TOOLS)})
    assert msgs == CALL and tools == TOOLS


@pytest.mark.parametrize("text, expected", [
    ('<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>', [{"name": "f", "arguments": {"x": 1}}]),
    ('<tool_call>{"name": "f", "arguments": {}}</tool_call>\n<tool_call>{"name": "g", "arguments": {"y": "a"}}</tool_call>',
     [{"name": "f", "arguments": {}}, {"name": "g", "arguments": {"y": "a"}}]),
    ('{"name": "f", "parameters": {"x": 2}}', [{"name": "f", "arguments": {"x": 2}}]),          # Llama-3.2 style
    ('<tool_call>{"name": "f", "arguments": "{\\"x\\": 3}"}</tool_call>', [{"name": "f", "arguments": {"x": 3}}]),
    ("Sorry, none of my tools can do that.", []),                                                 # plain answer
    ('<tool_call>{"name": "f", "arguments": {"x": 1}</tool_call>', None),                         # broken JSON
    ('<tool_call>{"name": "f", "arguments": {"x": 1}}', None),                                    # never closed
    ('<tool_call>{"arguments": {"x": 1}}</tool_call>', None),                                     # no name
])
def test_strict_parsing(text, expected):
    assert parse_tool_calls(text) == expected


def test_matching_ignores_call_order_but_not_values():
    gold = gold_calls(CALL)
    assert calls_match([{"name": "get_weather", "arguments": {"city": "Kandy"}}], gold)
    assert not calls_match([{"name": "get_weather", "arguments": {"city": "Colombo"}}], gold)
    assert not calls_match(None, gold)
    assert calls_match([], gold_calls(TEXT))
    two = [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"k": 1}}]
    assert calls_match(list(reversed(two)), two)
