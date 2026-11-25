"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import DATA from "../data/scenarios.json";

const TOOL_META = {
  get_review_insights: { comp: "C1", label: "Review engine", tone: "c1" },
  recommend_hotels: { comp: "C2", label: "Hotel recommender", tone: "c2" },
  plan_itinerary: { comp: "C3", label: "Itinerary planner", tone: "c3" },
  web_search: { comp: "Web", label: "Web search (unverified)", tone: "web" },
};
const MODE_LABEL = {
  free: "(a) Free",
  prompt_json: "(b) Prompt-only JSON",
  grammar: "(c) Hard grammar",
  reason_grammar: "(d) Reason, then grammar",
};
const MODE_HINT = {
  free: "no constraint",
  prompt_json: "asked, not enforced",
  grammar: "XGrammar forces valid calls",
  reason_grammar: "think first, constrain late",
};

/* ---------- tiny markdown: **bold**, *italic*, lists, quotes ---------- */
function inline(text, key) {
  const parts = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m, i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const t = m[0];
    parts.push(t.startsWith("**") ? <strong key={`${key}-${i++}`}>{t.slice(2, -2)}</strong> : <em key={`${key}-${i++}`}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function Markdown({ text }) {
  const blocks = [];
  let list = null;
  text.split("\n").forEach((line, idx) => {
    const ol = line.match(/^(\d+)\.\s+(.*)/);
    const ul = line.match(/^-\s+(.*)/);
    if (ol || ul) {
      const kind = ol ? "ol" : "ul";
      if (!list || list.kind !== kind) { list = { kind, items: [] }; blocks.push(list); }
      list.items.push(ol ? ol[2] : ul[1]);
      return;
    }
    list = null;
    if (!line.trim()) return;
    if (line.startsWith("> ")) blocks.push({ kind: "quote", text: line.slice(2), idx });
    else blocks.push({ kind: "p", text: line, idx });
  });
  return (
    <div className="md">
      {blocks.map((b, i) => {
        if (b.kind === "ol" || b.kind === "ul") {
          const Tag = b.kind;
          return <Tag key={i}>{b.items.map((t, j) => <li key={j}>{inline(t, `${i}-${j}`)}</li>)}</Tag>;
        }
        if (b.kind === "quote") return <blockquote key={i}>{inline(b.text, i)}</blockquote>;
        return <p key={i}>{inline(b.text, i)}</p>;
      })}
    </div>
  );
}

/* ---------- icons (inline SVG) ---------- */
const Icon = {
  pin: (p) => (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...p}><path d="M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21z" /><circle cx="12" cy="9.5" r="2.5" /></svg>),
  send: (p) => (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...p}><path d="M5 12h14M13 6l6 6-6 6" /></svg>),
  tool: (p) => (<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...p}><path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.5-.5-.5-2.5z" /></svg>),
  check: (p) => (<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...p}><path d="M5 12l5 5L20 7" /></svg>),
  plus: (p) => (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" {...p}><path d="M12 5v14M5 12h14" /></svg>),
  chev: ({ open }) => (<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" style={{ transform: open ? "rotate(90deg)" : "none", transition: "transform .15s" }}><path d="M9 6l6 6-6 6" /></svg>),
};

/* ---------- tool call card ---------- */
function ToolCard({ call, result, round }) {
  const [open, setOpen] = useState(false);
  const meta = TOOL_META[call.name] || { comp: "?", label: call.name, tone: "web" };
  const ok = call.schema_errors.length === 0;
  const err = result && result.error;
  const args = Object.entries(call.arguments).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  return (
    <div className="toolcard">
      <button className="toolhead" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={`chip ${meta.tone}`}>{meta.comp}</span>
        <span className="toolname">Round {round} · {meta.label}</span>
        <code className="toolargs">{call.name}({args})</code>
        <span className={`status ${ok && !err ? "ok" : "bad"}`}>{ok && !err ? <><Icon.check /> schema-valid</> : "error"}</span>
        <Icon.chev open={open} />
      </button>
      {open && (
        <div className="toolbody">
          <div className="kv"><span>Strict check</span><span>{ok ? "passed — every argument matches the tool schema" : call.schema_errors.join("; ")}</span></div>
          <div className="kv"><span>Tool result</span><span>{err ? err.message : "returned as data (never run as instructions)"}</span></div>
          <pre>{JSON.stringify(result, null, 1)}</pre>
        </div>
      )}
    </div>
  );
}

/* ---------- one assistant turn, revealed step by step ---------- */
function AssistantTurn({ turn, animate, onDone }) {
  const steps = useMemo(() => {
    const s = [];
    turn.trace.forEach((e) => {
      (e.calls || []).forEach((c, i) => s.push({ kind: "call", call: c, result: (e.results || [])[i]?.result, round: e.round }));
    });
    s.push({ kind: "answer" });
    return s;
  }, [turn]);
  const [shown, setShown] = useState(animate ? 0 : steps.length);
  const [chars, setChars] = useState(animate ? 0 : turn.answer.length);

  useEffect(() => {
    if (!animate) return;
    if (shown < steps.length - 1) { const t = setTimeout(() => setShown(shown + 1), 650); return () => clearTimeout(t); }
    if (shown === steps.length - 1 && chars < turn.answer.length) {
      const t = setTimeout(() => setChars(Math.min(turn.answer.length, chars + 9)), 12); return () => clearTimeout(t);
    }
    if (chars >= turn.answer.length) onDone && onDone();
  }, [shown, chars, animate]);


  const thinking = animate && shown < steps.length - 1;
  const answerVisible = shown >= steps.length - 1;
  return (
    <div className="aturn">
      <div className="avatar"><Icon.pin /></div>
      <div className="abody">
        {steps.filter((s) => s.kind === "call").slice(0, Math.max(0, shown)).map((s, i) => <ToolCard key={i} {...s} />)}
        {thinking && <div className="thinking"><span /><span /><span /> {shown === 0 ? "Choosing a tool…" : "Calling the next tool…"}</div>}
        {answerVisible && <div className="answer"><Markdown text={turn.answer.slice(0, chars)} /></div>}
      </div>
    </div>
  );
}

/* ---------- chat view ---------- */
function Chat({ mode, apiRef, onBusy }) {
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState([]); // queued follow-up turns of a multi-turn scenario
  const [note, setNote] = useState("");
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, busy]);
  useEffect(() => { onBusy(busy); }, [busy]);

  const play = (turn, rest = []) => {
    setNote("");
    setMessages((m) => [...m, { role: "user", text: turn.user }, { role: "assistant", turn, animate: true }]);
    setBusy(true);
    setPending(rest);
  };

  const start = (sc) => { if (busy) return; play(sc.turns[0], sc.turns.slice(1)); };
  apiRef.current = { start, reset: () => { if (!busy) { setMessages([]); setPending([]); } } };

  const allTurns = useMemo(() => DATA.scenarios.flatMap((s) => s.turns.map((t, i) => ({ t, rest: s.turns.slice(i + 1) }))), []);
  const words = (s) => new Set(s.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter((w) => w.length > 2));

  const send = (e) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    if (pending.length && pending[0].user.toLowerCase() === q.toLowerCase()) { play(pending[0], pending.slice(1)); return; }
    const qw = words(q);
    let best = null, bestScore = 0;
    allTurns.forEach((x) => {
      const tw = words(x.t.user);
      const inter = [...qw].filter((w) => tw.has(w)).length;
      const score = inter / Math.max(1, Math.min(qw.size, tw.size));
      if (score > bestScore) { bestScore = score; best = x; }
    });
    if (best && bestScore >= 0.6) { play(best.t, best.rest); return; }
    setMessages((m) => [...m, { role: "user", text: q }, { role: "note" }]);
    setNote("");
  };

  return (
    <div className="chatwrap">
      <div className="chatscroll">
        {messages.length === 0 && (
          <div className="welcome">
            <div className="wicon"><Icon.pin width="26" height="26" /></div>
            <h1>Where would you like to go in Sri Lanka?</h1>
            <p>The agent picks the right component — <b>C1</b> reviews, <b>C2</b> hotels, <b>C3</b> itineraries or web search — checks every call, and answers only from the tool results.</p>
            <div className="starters">
              {DATA.scenarios.slice(0, 6).map((s) => (
                <button key={s.id} className="starter" onClick={() => start(s)}>
                  <span className="sgroup">{s.group}</span>
                  <span>{s.turns[0].user}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => {
          if (m.role === "user") return <div key={i} className="uturn"><div className="ububble">{m.text}</div></div>;
          if (m.role === "note") return (
            <div key={i} className="aturn"><div className="avatar"><Icon.pin /></div>
              <div className="abody"><div className="answer offline">This offline demo replays the example requests only — the live model runs in the Colab demo. Try one of the examples on the left, or type one of them.</div></div></div>
          );
          return <AssistantTurn key={i} turn={m.turn} animate={m.animate} onDone={() => { setBusy(false); m.animate = false; }} />;
        })}
        <div ref={endRef} />
      </div>

      <div className="composer">
        {pending.length > 0 && !busy && (
          <div className="followups"><span>Follow-up:</span>
            <button className="chipbtn" onClick={() => play(pending[0], pending.slice(1))}>{pending[0].user}</button>
          </div>
        )}
        <form onSubmit={send} className="inputrow">
          <label htmlFor="msg" className="sr">Message</label>
          <input id="msg" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about hotels, reviews or a trip plan in Sri Lanka…" disabled={busy} />
          <button type="submit" aria-label="Send" disabled={busy || !input.trim()}><Icon.send /></button>
        </form>
        <div className="foot">Mode: <b>{MODE_LABEL[mode]}</b> · Tools v{DATA.tools_version} · Router scripted for this offline demo; tool results and answers are real outputs of the C4 code</div>
      </div>
    </div>
  );
}


/* ---------- decoding modes view ---------- */
function Modes() {
  const d = DATA.modes;
  const gold = d.gold.map((c) => `${c.name}(${Object.entries(c.arguments).map(([k, v]) => `${k}=${v.accept.map((a) => JSON.stringify(a)).join(" | ")}${v.optional ? " (optional)" : ""}`).join(", ")})`).join(" + ");
  return (
    <div className="page">
      <div className="pagehead">
        <h1>Decoding modes side by side</h1>
        <p>The same fine-tuned model answers the same question under the four decoding modes. Forcing valid JSON guarantees the format — but is the call still correct? That gap is the <b>constraint tax</b>.</p>
      </div>
      <div className="qbox">
        <div className="qlabel">Question · tourism test set item T01</div>
        <div className="qtext">{d.question}</div>
        <div className="qlabel">Expected call</div>
        <code className="gold">{gold}</code>
      </div>
      <div className="promptbox">
        <div className="qlabel">What every mode sends to the model</div>
        <div className="pparts">
          <div className="ppart"><span className="pnum">1</span><div><b>System prompt</b><p>{d.system_prompt}</p></div></div>
          <div className="ppart"><span className="pnum">2</span><div><b>Tools</b><p>The 4 tool schemas (C1 reviews, C2 hotels, C3 itinerary, web search), added by the model's chat template.</p></div></div>
          <div className="ppart"><span className="pnum">3</span><div><b>User message</b><p>{d.question}</p></div></div>
        </div>
        <p className="pnote">The modes differ in only two things: an <b>extra instruction</b> added to the prompt, and whether the <b>grammar</b> controls the writing.</p>
      </div>
      <div className="modegrid">
        {d.rows.map((r) => {
          const fmt = r.malformed ? ["bad", "Malformed"] : r.valid ? ["ok", "Schema-valid"] : ["warn", "Schema-invalid"];
          const cor = r.malformed ? ["bad", "Not executed"] : r.correct ? ["ok", "Correct"] : r.valid ? ["warn", "Valid but wrong"] : ["bad", "Wrong"];
          return (
            <div key={r.mode} className={`modecard ${r.correct ? "win" : ""}`}>
              <div className="mtitle">{MODE_LABEL[r.mode]}</div>
              <div className="mhint">{MODE_HINT[r.mode]}</div>
              <div className="step"><span className="snum">1</span><div><b>Prompt</b>
                {r.instruction ? <p className="instr">+ “{r.instruction}”</p> : <p className="muted">No extra instruction.</p>}</div></div>
              <div className="step"><span className="snum">2</span><div><b>Generation</b> <span className={`gchip ${r.grammar ? "on" : "off"}`}>{r.grammar ? "Grammar ON" : "Grammar OFF"}</span>
                <p className="muted">{r.generation}</p></div></div>
              <div className="step"><span className="snum">3</span><div><b>Model output</b></div></div>
              <pre className="raw">{r.raw}</pre>
              <div className="badges"><span className={`badge ${fmt[0]}`}>{fmt[1]}</span><span className={`badge ${cor[0]}`}>{cor[1]}</span></div>
              <p className="why">{r.why}</p>
            </div>
          );
        })}
      </div>
      <div className="takeaway">
        <b>Why this matters:</b> (a) and (c) use the same prompt, so any accuracy difference between them comes from the grammar alone. That difference is the constraint tax, which this research measures on fine-tuned models, and (d) tests whether reasoning first wins the accuracy back.
      </div>
      <p className="disclaimer">Illustrative model outputs, scored by the real C4 parser, schema validator and scorer. Measured results for all 16 configurations (4 models × 4 modes) come from the evaluation run.</p>
    </div>
  );
}

/* ---------- how it works view ---------- */
function HowItWorks() {
  const flow = [
    ["Tourist", "Plain-English request, multi-turn"],
    ["Router", "Fine-tuned small LLM (QLoRA) picks the tool and arguments under 1 of 4 decoding modes"],
    ["Executor", "Strict, repair-free check · hotel name → ID · calls C1 / C2 / C3 / web · structured errors"],
    ["Components", "C1 reviews · C2 hotels · C3 itinerary · web search (labelled unverified)"],
    ["Synthesiser", "Writes the answer only from tool results — no invented facts"],
  ];
  const stats = [["4", "small LLMs: Qwen2.5-0.5B / 1.5B / 3B, Llama-3.2-1B"], ["4", "decoding modes → 16 configurations"], ["11,168", "training examples (xLAM + 15% no-call, BFCL-decontaminated)"], [String(DATA.testset_size), "tourism test questions + 300 BFCL items"]];
  return (
    <div className="page">
      <div className="pagehead">
        <h1>How the agent works</h1>
        <p>Component 4 is the orchestrator of the Agentic Tourism Intelligence Framework. The model decides <i>what</i> to call; rule-based code checks it, runs it and writes the answer.</p>
      </div>
      <div className="flow">
        {flow.map(([t, d], i) => (
          <React.Fragment key={t}>
            <div className="fstep"><div className="fnum">{i + 1}</div><div className="ft">{t}</div><div className="fd">{d}</div></div>
            {i < flow.length - 1 && <div className="farrow" aria-hidden="true">→</div>}
          </React.Fragment>
        ))}
      </div>
      <div className="stats">{stats.map(([n, l]) => <div key={l} className="stat"><div className="sn">{n}</div><div className="sl">{l}</div></div>)}</div>
      <div className="toolsgrid">
        {DATA.tools.map((t) => (
          <div key={t.name} className="tooldef">
            <div className="tdhead"><span className={`chip ${TOOL_META[t.name].tone}`}>{TOOL_META[t.name].comp}</span><code>{t.name}</code></div>
            <p>{t.description}</p>
            <div className="params">{t.params.map((p) => <span key={p} className={t.required.includes(p) ? "req" : ""}>{p}</span>)}</div>
          </div>
        ))}
      </div>
      <p className="disclaimer">Required arguments are highlighted. Tool schemas are versioned (v{DATA.tools_version}); a new tool is added by schema, without retraining.</p>
    </div>
  );
}

/* ---------- app shell ---------- */
export default function App() {
  const [tab, setTab] = useState("chat");
  const [mode, setMode] = useState("grammar");
  const [busy, setBusy] = useState(false);
  const chatApi = useRef(null);
  const groups = DATA.scenarios;

  return (
    <div className="app">
      <header className="top">
        <div className="brand"><div className="logo"><Icon.pin /></div>
          <div><div className="bt">Sri Lanka Tourism Agent</div><div className="bs">Component 4 · J26-DS-345 · Gajanayaka K.R.K.V</div></div>
        </div>
        <nav className="tabs" role="tablist" aria-label="View">
          {[["chat", "Chat"], ["modes", "Decoding modes"], ["how", "How it works"]].map(([k, l]) => (
            <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
          ))}
        </nav>
        <span className="offbadge">Offline demo · scripted router · real C4 outputs</span>
      </header>

      <div className="main">
        <aside className="side">
          <button className="newchat" onClick={() => { setTab("chat"); chatApi.current?.reset(); }}><Icon.plus /> New conversation</button>
          <div className="sect">Examples</div>
          {groups.map((s) => (
            <button key={s.id} className="ex" disabled={busy} onClick={() => { setTab("chat"); chatApi.current?.start(s); }}>
              <span className="exg">{s.group}</span><span className="ext">{s.title}</span>
            </button>
          ))}
          <div className="sect">Decoding mode</div>
          <div className="modes">
            {Object.keys(MODE_LABEL).map((m) => (
              <button key={m} className={`modebtn ${mode === m ? "sel" : ""}`} aria-pressed={mode === m} onClick={() => setMode(m)}>
                <span>{MODE_LABEL[m]}</span><small>{MODE_HINT[m]}</small>
              </button>
            ))}
          </div>
          <div className="services">
            <div className="sect">Connected (mock services)</div>
            {[["c1", "C1 Review engine"], ["c2", "C2 Hotel recommender"], ["c3", "C3 Itinerary planner"], ["web", "Web search · labelled external"]].map(([t, l]) => (
              <div key={t} className="svc"><span className={`dot ${t}`} />{l}</div>
            ))}
          </div>
        </aside>
        <main className="content">
          <div style={{ display: tab === "chat" ? "contents" : "none" }}><Chat mode={mode} apiRef={chatApi} onBusy={setBusy} /></div>
          {tab === "modes" && <Modes />}
          {tab === "how" && <HowItWorks />}
        </main>
      </div>
    </div>
  );
}
