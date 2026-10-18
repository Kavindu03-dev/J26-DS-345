"""Build "no tool call" training examples.

xLAM only contains examples where the right answer IS a tool call. A model
trained on it alone learns to call a tool for everything, which fails BFCL's
irrelevance category and our tourism "direct answer" and "decline" categories.
The proposal therefore adds at least 15% no-call examples, of two kinds:

1. irrelevance - a real xLAM query paired with the tools of a DIFFERENT,
   unrelated xLAM row. None of the tools can answer it, so the assistant
   replies in text without calling anything.
2. conversational - greetings, thanks, capability questions and off-topic
   requests, with tools available but not needed.

The templates are generic on purpose. Tourism wording is kept out of training
so the tourism test set (Layer 2) stays a fair, unseen evaluation.
"""

from __future__ import annotations

import random
import re

_WORD = re.compile(r"[a-z]{4,}")
_STOP = {
    "what", "with", "from", "that", "this", "have", "give", "find", "show", "list", "please",
    "about", "information", "details", "data", "using", "based", "provide", "need", "want",
    "know", "like", "given", "fetch", "retrieve", "returns", "return", "default", "specified",
    "string", "number", "value", "list", "type", "name", "also", "them", "their", "which",
}

IRRELEVANCE_REPLIES = [
    "None of the available tools can help with this request, so I can't complete it with a tool call.",
    "I don't have a tool that can do that. The tools I can use cover different tasks.",
    "That request is outside what my available tools support, so I won't call any of them.",
    "I can't answer this with the tools provided. None of them match what you're asking for.",
    "Sorry, none of my current tools are suitable for this request.",
]

# (user message, assistant reply) pairs. Generic, not tourism-specific.
CONVERSATIONAL = [
    ("Hi there!", "Hello! How can I help you today?"),
    ("Good morning", "Good morning! What can I do for you?"),
    ("Thanks, that's all I needed.", "You're welcome! Let me know if you need anything else."),
    ("Thank you so much!", "Happy to help!"),
    ("Great, that works for me.", "Glad to hear it. Anything else I can help with?"),
    ("What can you do?", "I can answer questions and use the tools available to me to look things up or take actions for you. What would you like to do?"),
    ("Who are you?", "I'm an AI assistant that can use tools to help with your requests."),
    ("Can you write my university essay for me?", "I can't write an assignment for you to submit as your own, but I'm happy to help you plan it or explain the topic."),
    ("Tell me a joke.", "Why did the developer go broke? Because he used up all his cache."),
    ("Ok", "Okay! Let me know if there's anything else."),
    ("Never mind.", "No problem. I'm here if you need anything."),
    ("Can you hack into my neighbour's wifi?", "I can't help with accessing networks you don't own or have permission to use."),
    ("What's 2 + 2?", "2 + 2 = 4."),
    ("How are you?", "I'm doing well, thanks for asking! How can I help?"),
    ("Bye!", "Goodbye! Have a great day."),
    ("Can you repeat that more simply?", "Sure. Could you tell me which part you'd like me to explain more simply?"),
]


def keywords(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def tool_keywords(tools: list[dict]) -> set[str]:
    words: set[str] = set()
    for t in tools:
        fn = t["function"]
        words |= keywords(fn["name"].replace("_", " ").replace(".", " ") + " " + fn["description"])
        words |= keywords(" ".join(fn["parameters"]["properties"]))
    return words


def make_irrelevance(
    query_pool: list[dict], tool_pool: list[dict], n: int, rng: random.Random, max_tries: int = 50
) -> list[dict]:
    """Pair queries from query_pool with tools of an unrelated example from tool_pool."""
    out = []
    if not query_pool or len(tool_pool) < 2:
        return out
    # Each query is used at most once (sampling without replacement) while possible.
    order = rng.sample(query_pool, len(query_pool))
    for i in range(n):
        q_ex = order[i % len(order)]
        query = q_ex["messages"][0]["content"]
        q_words = keywords(query)
        q_tools = {t["function"]["name"] for t in q_ex["tools"]}
        for _ in range(max_tries):
            t_ex = rng.choice(tool_pool)
            names = {t["function"]["name"] for t in t_ex["tools"]}
            if names & q_tools:
                continue
            if q_words & tool_keywords(t_ex["tools"]):
                continue
            out.append(
                {
                    "source_id": f"irrel:{q_ex['source_id']}|{t_ex['source_id']}",
                    "kind": "no_call_irrelevance",
                    "tools": t_ex["tools"],
                    "messages": [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": rng.choice(IRRELEVANCE_REPLIES)},
                    ],
                }
            )
            break
    return out


def make_conversational(examples: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Conversational turns with a random real tool set available."""
    out = []
    for i in range(n):
        user, reply = CONVERSATIONAL[i % len(CONVERSATIONAL)]
        tools = rng.choice(examples)["tools"]
        out.append(
            {
                "source_id": f"conv:{i}",
                "kind": "no_call_conversational",
                "tools": tools,
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
            }
        )
    return out
