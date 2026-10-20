"""Agent loop (router -> executor -> synthesiser) with a scripted model and the mock services."""

import json

import pytest
from fastapi.testclient import TestClient

from src.agent import MALFORMED_REPLY, Agent
from src.mocks.c1_mock import app as c1_app
from src.mocks.c2_mock import app as c2_app
from src.mocks.c3_mock import app as c3_app
from src.tools.clients import ToolClients
from src.tools.executor import ToolExecutor


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


class ScriptedLLM:
    """Plays back one output per round; records what it was shown."""

    def __init__(self, *outputs):
        self.outputs, self.seen = list(outputs), []

    def __call__(self, messages, tools, mode, system_prompt):
        self.seen.append(messages)
        return {"text": self.outputs.pop(0), "latency_s": 0.4}


@pytest.fixture
def executor():
    return ToolExecutor(ToolClients(c1=TestClient(c1_app), c2=TestClient(c2_app), c3=TestClient(c3_app)))


def test_review_question_quotes_verified_facts(executor):
    llm = ScriptedLLM(call("get_review_insights", hotel="Orchard House Galle", aspects=["cleanliness"]), "done")
    r = Agent(llm, executor).respond([], "How clean is Orchard House Galle?")
    assert "Orchard House Galle" in r.answer and "Cleanliness" in r.answer and "verified guest review" in r.answer
    assert r.decision_latency_s == 0.4 and r.trace[0]["status"] == "called_tools"


def test_two_step_hotel_then_itinerary_uses_the_tool_result(executor):
    rounds = []

    def llm(messages, tools, mode, sp):
        rounds.append(messages)
        if len(rounds) == 1:
            return {"text": call("recommend_hotels", destination="Ella", top_k=1), "latency_s": 0.4}
        if len(rounds) == 2:  # the model reads the ranking and continues with the top hotel
            top = json.loads(messages[-1]["content"])["hotels"][0]["name"]
            return {"text": call("plan_itinerary", hotel=top, days=2, interests=["nature"]), "latency_s": 0.3}
        return {"text": "Here is your plan.", "latency_s": 0.2}

    r = Agent(llm, executor).respond([], "Find a hotel in Ella and plan 2 days of nature.")
    assert [n for n, _ in r.tool_results] == ["recommend_hotels", "plan_itinerary"]
    assert "Hotels in Ella" in r.answer and "Day 1" in r.answer and "Day 2" in r.answer
    assert rounds[1][-1]["role"] == "tool"  # tool output went back to the model as data


def test_direct_answer_and_decline_need_no_tools(executor):
    r = Agent(ScriptedLLM("You're welcome!"), executor).respond([], "Thanks!")
    assert r.answer == "You're welcome!" and not r.tool_results


def test_malformed_call_is_never_executed(executor):
    r = Agent(ScriptedLLM('<tool_call>{"name": "recommend_hotels", "arguments": {</tool_call>'), executor).respond([], "hotels in Ella")
    assert r.answer == MALFORMED_REPLY and r.trace[0]["status"] == "malformed" and not r.tool_results


def test_schema_invalid_call_returns_an_error_not_a_crash(executor):
    llm = ScriptedLLM(call("recommend_hotels", destination="Ella", nearby=["airport"]), "ok")
    r = Agent(llm, executor).respond([], "hotels near the airport in Ella")
    assert "error" in r.trace[0]["results"][0]["result"] and "Sorry" in r.answer


def test_unknown_hotel_suggests_alternatives(executor):
    llm = ScriptedLLM(call("get_review_insights", hotel="Orchad Hous"), "ok")
    r = Agent(llm, executor).respond([], "reviews of Orchad Hous")
    err = r.trace[0]["results"][0]["result"]
    if "error" in err:
        assert "Did you mean" in err["error"]["message"] and "Orchard House" in err["error"]["message"]
    else:
        assert err["hotel"] == "Orchard House Galle"


def test_web_search_is_labelled_external(executor):
    r = Agent(ScriptedLLM(call("web_search", query="train Colombo to Kandy"), "ok"), executor).respond([], "train times?")
    assert "external source" in r.answer


def test_repeated_identical_calls_stop_the_loop(executor):
    c = call("get_review_insights", hotel="Kandy Hills")
    r = Agent(ScriptedLLM(c, c, c), executor).respond([], "reviews of Kandy Hills")
    assert len(r.tool_results) == 1 and r.trace[-1]["status"] == "repeated_calls"
