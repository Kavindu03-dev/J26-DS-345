import httpx
import pytest

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_server_answers(vllm_server):
    reply = httpx.get(f"{vllm_server}/models", timeout=10)
    assert reply.status_code == 200
    assert MODEL in [m["id"] for m in reply.json()["data"]]


def test_prompt_contains_the_tools():
    from scripts.smoke_generate import render_prompt

    prompt = render_prompt(MODEL)
    assert "get_review_insights" in prompt
    assert "search_knowledge_base" not in prompt  # Tool 5 is not in the experiment
    assert prompt.endswith("<|im_start|>assistant\n")


def test_raw_completion_is_not_empty(vllm_server):
    from scripts.smoke_generate import generate

    text = generate(MODEL, base_url=vllm_server)
    print(text)
    assert text.strip()
