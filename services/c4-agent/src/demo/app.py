"""Demo screen (Gradio) for the progress review.

Tab 1 - Agent chat:   type a travel request; the agent calls C1-C3 (mocks or real
                      services) and answers; the trace shows every tool call.
Tab 2 - Decoding modes side by side:  the SAME question under all four decoding
                      modes, showing raw output, validity, correctness against the
                      gold answer and latency. This is the proof of concept for
                      the constraint-tax research question.

The logic (run_chat_turn, compare_modes, formatting) is plain Python and tested;
build_ui() only wires it into Gradio.
"""

from __future__ import annotations

import json
from typing import Callable

from ..agent import Agent
from ..tools.definitions import SYSTEM_PROMPT, TOOLS, validate_call

try:
    from evaluation.decoding import MODES, strip_reasoning
    from evaluation.scoring import score_tourism_step
    from training.fc_format import parse_tool_calls
except ImportError:  # pragma: no cover
    from decoding import MODES, strip_reasoning
    from fc_format import parse_tool_calls
    from scoring import score_tourism_step

MODE_LABEL = {
    "free": "(a) Free",
    "prompt_json": "(b) Prompt-only JSON",
    "grammar": "(c) Hard grammar",
    "reason_grammar": "(d) Reason, then grammar",
}


def run_chat_turn(agent: Agent, history: list[dict], message: str) -> tuple[list[dict], str, dict]:
    """One chat turn -> (new history, latency line, trace for display)."""
    r = agent.respond(history, message)
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": r.answer}]
    info = f"Tool decision: **{r.decision_latency_s:.2f} s** - whole turn: {r.total_latency_s:.2f} s - mode: {MODE_LABEL[agent.mode]}"
    return history, info, {"rounds": r.trace}


def history_markdown(history: list[dict]) -> str:
    out = []
    for m in history:
        who = "**You:**" if m["role"] == "user" else "**Agent:**"
        out.append(f"{who} {m['content']}")
    return "\n\n---\n\n".join(out) or "*Ask about hotels, reviews, itineraries or travel information in Sri Lanka.*"


def compare_modes(llm: Callable, item: dict | None, question: str | None = None) -> list[dict]:
    """Run one question (a test-set item, or free text) under every mode; score against gold if known."""
    messages = item["messages"] if item else [{"role": "user", "content": question}]
    gold = item["steps"][0]["calls"] if item and item["steps"] else ([] if item else None)
    rows = []
    for mode in MODES:
        out = llm(messages, TOOLS, mode, SYSTEM_PROMPT)
        text = strip_reasoning(out["text"]) if mode == "reason_grammar" else out["text"]
        parsed = parse_tool_calls(text)
        row = {"mode": mode, "raw": out["text"], "parsed": parsed, "latency_s": out["latency_s"],
               "malformed": parsed is None,
               "valid": parsed is not None and all(not validate_call(c["name"], c["arguments"]) for c in parsed)}
        row["correct"] = score_tourism_step(parsed, gold, validate_call)["correct"] if gold is not None else None
        rows.append(row)
    return rows


def mode_card(row: dict) -> str:
    badges = ["❌ malformed" if row["malformed"] else ("✅ schema-valid" if row["valid"] else "⚠️ schema-invalid")]
    if row["correct"] is not None:
        badges.append("✅ correct" if row["correct"] else ("🟠 valid but WRONG" if row["valid"] else "❌ wrong"))
    calls = json.dumps(row["parsed"], indent=1, ensure_ascii=False) if row["parsed"] else ("(no tool call)" if row["parsed"] == [] else "(could not parse)")
    return (f"### {MODE_LABEL[row['mode']]}\n{' | '.join(badges)}  \n⏱ {row['latency_s']:.2f} s\n\n"
            f"**Raw output**\n```\n{row['raw'][:600]}\n```\n**Parsed call(s)**\n```json\n{calls}\n```")


def gold_markdown(item: dict | None) -> str:
    if not item:
        return "*Free-text question: no gold answer, so correctness is not scored.*"
    if not item["steps"]:
        return "**Expected:** no tool call (answer directly / decline)."
    parts = []
    for c in item["steps"][0]["calls"]:
        args = ", ".join(f"{k}={' | '.join(json.dumps(a, ensure_ascii=False) for a in v['accept'])}" + (" (optional)" if v["optional"] else "")
                         for k, v in c["arguments"].items())
        parts.append(f"`{c['name']}({args})`")
    return "**Expected (step 1):** " + " + ".join(parts)


def build_ui(llm: Callable, executor, testset: list[dict]):
    import gradio as gr

    items = {f"{t['id']} - {t['messages'][-1]['content'][:80]}": t for t in testset}
    showcase = [k for k in items if k.split(" ")[0] in ("T01", "T04", "S01", "M01", "M06", "W01", "D01", "X01", "T10", "S21")]

    with gr.Blocks(title="C4 Tourism Agent - J26-DS-345") as demo:
        gr.Markdown("# Sri Lanka tourism agent (Component 4)\nFine-tuned small model + function calling over C1 reviews, C2 hotels, C3 itineraries.")
        with gr.Tab("Agent chat"):
            history = gr.State([])
            mode = gr.Dropdown(choices=[(MODE_LABEL[m], m) for m in MODES], value="grammar", label="Decoding mode")
            conversation = gr.Markdown(history_markdown([]))
            latency = gr.Markdown("")
            with gr.Row():
                msg = gr.Textbox(placeholder="e.g. Plan a 2-day family trip to Ella with a hotel under 25k a night", scale=5, show_label=False)
                send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("New conversation")
            trace = gr.JSON(label="Trace: model output, tool calls and tool results")

            def on_send(message, hist, m):
                if not message.strip():
                    return hist, history_markdown(hist), "", {}, ""
                new_hist, info, tr = run_chat_turn(Agent(llm, executor, mode=m), hist, message)
                return new_hist, history_markdown(new_hist), info, tr, ""

            send.click(on_send, [msg, history, mode], [history, conversation, latency, trace, msg])
            msg.submit(on_send, [msg, history, mode], [history, conversation, latency, trace, msg])
            clear.click(lambda: ([], history_markdown([]), "", {}), None, [history, conversation, latency, trace])

        with gr.Tab("Decoding modes side by side"):
            gr.Markdown("The **same fine-tuned model** answers the same question under the four decoding modes of the study.")
            pick = gr.Dropdown(choices=showcase + [k for k in items if k not in showcase], value=showcase[0] if showcase else None,
                               label="Question from the tourism test set")
            custom = gr.Textbox(label="...or type your own question (overrides the list)")
            run_btn = gr.Button("Run all four modes", variant="primary")
            expected = gr.Markdown(gold_markdown(items.get(showcase[0])) if showcase else "")
            with gr.Row():
                cards = [gr.Markdown() for _ in MODES]

            def on_compare(choice, text):
                item = None if text.strip() else items.get(choice)
                rows = compare_modes(llm, item, text.strip() or None)
                return [gold_markdown(item)] + [mode_card(r) for r in rows]

            run_btn.click(on_compare, [pick, custom], [expected] + cards)
            pick.change(lambda c: gold_markdown(items.get(c)), pick, expected)
    return demo
