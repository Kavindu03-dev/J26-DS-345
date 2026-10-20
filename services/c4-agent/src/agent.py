"""LLM orchestrator for the tourism agent.

Receives a natural-language tourist request, decides which tools to call
(C1 aspect scores, C2 hotel ranking, C3 itinerary, web search), and combines the
results into one answer. The tool callers live in src/tools/ and the fake
endpoints used for testing live in src/mocks/.

    router       the fine-tuned model, under one of the four decoding modes, decides
                 which tool(s) to call - or answers directly / declines
    executor     src/tools/executor.py validates and runs each call
    synthesiser  src/synthesizer.py writes the final answer from the tool results only

The loop allows several rounds, so the model can e.g. call recommend_hotels,
see the ranking, then call plan_itinerary on the top hotel (Tool 3 depends on Tool 2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .synthesizer import render
from .tools.definitions import SYSTEM_PROMPT, TOOLS
from .tools.executor import ToolExecutor, assistant_calls_message, tool_message

try:
    from training.fc_format import parse_tool_calls
except ImportError:  # flat layout (Colab demo)
    from fc_format import parse_tool_calls

# llm(messages, tools, mode, system_prompt) -> {"text", "latency_s", ...}  (evaluation.decoding.Generator)
LLM = Callable[[list[dict], list[dict], str, str], dict]

MAX_ROUNDS = 3
MALFORMED_REPLY = "Sorry, I couldn't work out the right action for that. Could you rephrase it?"


@dataclass
class TurnResult:
    answer: str
    trace: list[dict] = field(default_factory=list)  # one entry per model round
    tool_results: list[tuple[str, dict]] = field(default_factory=list)
    decision_latency_s: float | None = None  # first-round model latency (proposal: <= 2 s)
    total_latency_s: float = 0.0


class Agent:
    def __init__(self, llm: LLM, executor: ToolExecutor, mode: str = "grammar", max_rounds: int = MAX_ROUNDS):
        self.llm, self.executor, self.mode, self.max_rounds = llm, executor, mode, max_rounds

    def respond(self, history: list[dict], user_message: str) -> TurnResult:
        """history: previous user/assistant turns (plain text). Returns the answer and a full trace."""
        t0 = time.perf_counter()
        messages = history + [{"role": "user", "content": user_message}]
        result = TurnResult(answer="")
        for round_no in range(1, self.max_rounds + 1):
            out = self.llm(messages, TOOLS, self.mode, SYSTEM_PROMPT)
            if result.decision_latency_s is None:
                result.decision_latency_s = out["latency_s"]
            text = out["text"]
            if self.mode == "reason_grammar":
                lines = [l for l in text.strip().splitlines() if not l.lower().startswith("reasoning:")]
                text = "\n".join(lines).strip()
            calls = parse_tool_calls(text)
            entry = {"round": round_no, "model_output": out["text"], "latency_s": out["latency_s"], "calls": calls}
            result.trace.append(entry)

            if calls is None:  # malformed: never executed, never repaired
                entry["status"] = "malformed"
                result.answer = render(result.tool_results) if result.tool_results else MALFORMED_REPLY
                break
            if not calls:  # the model answers or declines
                entry["status"] = "answer"
                result.answer = render(result.tool_results) if result.tool_results else text
                break

            # run the calls; repeated identical calls are not run twice
            done = {(n, repr(sorted(a.items()))) for n, a in ((c, r.get("_args", {})) for c, r in result.tool_results)}
            new_calls = [c for c in calls if (c["name"], repr(sorted(c["arguments"].items()))) not in done]
            if not new_calls:
                entry["status"] = "repeated_calls"
                result.answer = render(result.tool_results)
                break
            messages = messages + [assistant_calls_message(new_calls)]
            entry["results"] = []
            for c in new_calls:
                res = self.executor.execute(c["name"], c["arguments"])
                entry["results"].append({"name": c["name"], "result": res})
                result.tool_results.append((c["name"], dict(res, _args=c["arguments"])))
                messages.append(tool_message(c["name"], res))
            entry["status"] = "called_tools"
        else:
            result.answer = render(result.tool_results)

        result.tool_results = [(n, {k: v for k, v in r.items() if k != "_args"}) for n, r in result.tool_results]
        if result.tool_results and result.answer == "":
            result.answer = render(result.tool_results)
        result.total_latency_s = round(time.perf_counter() - t0, 3)
        return result
