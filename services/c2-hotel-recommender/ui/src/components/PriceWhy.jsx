// Why the fair price is this number: the SHAP explanation in rupees, as a waterfall.
import { useState } from "react";
import { getPriceExplanation } from "../api.js";
import { rs10, rs100 } from "../format.js";
import { ErrorBox, Waiting } from "./Bits.jsx";

const saved = {};       // url -> explanation (kept while the page is open)

export default function PriceWhy({ e }) {
  const [state, setState] = useState(() => (e?.url && saved[e.url] ? { done: saved[e.url] } : {}));
  if (!e) return <p className="muted">No explanation for this price.</p>;
  if (!e.available) return <p className="muted">{e.reason}</p>;

  async function explain() {
    setState({ started: Date.now() });
    try {
      const b = await getPriceExplanation(e.url);
      if (!b.available) { setState({ error: { detail: b.reason } }); return; }
      saved[e.url] = b;
      setState({ done: b });
    } catch (error) {
      setState({ error });
    }
  }

  if (state.done) return <Waterfall b={state.done} />;
  return (
    <div className="why-price">
      <p>Starting from an average hotel price, see how much the town, the hotel's class, the room, each facility and
        your dates add or take away, in rupees. The parts add up to the fair price.</p>
      {state.started ? <Waiting started={state.started} text="Working it out (2,048 combinations of facts)" />
        : <button type="button" className="btn" onClick={explain}>Explain this price</button>}
      <p className="muted small">For {e.for}. Takes a few seconds. Method: SHAP, with {e.model}.</p>
      {state.error && <ErrorBox error={state.error} />}
    </div>
  );
}

function Waterfall({ b }) {
  let total = b.start_lkr, hi = b.start_lkr;
  const steps = b.parts.map((p) => {
    const from = total;
    total += p.lkr;
    hi = Math.max(hi, from, total);
    return { p, from, to: total };
  });
  const at = (v) => `${((100 * v) / hi).toFixed(2)}%`;
  const ok = b.checks.parts_add_up_to_the_fair_price;
  return (
    <div className="why-price">
      <div className="wf" role="table" aria-label="How each group of facts moves the price">
        <Row label="Average hotel price" sub="20 typical prices" left={0} width={b.start_lkr} cls="total"
             amount={rs100(b.start_lkr)} at={at} />
        {steps.map(({ p, from, to }) => (
          <Row key={p.group} label={p.group} sub={p.what_it_is} left={Math.min(from, to)} width={Math.abs(p.lkr)}
               cls={p.lkr >= 0 ? "up" : "down"} at={at}
               amount={(Math.abs(p.lkr) < 50 ? "" : p.lkr >= 0 ? "+" : "−") + rs100(Math.abs(p.lkr))} />
        ))}
        <Row label="Fair price" sub={b.explains} left={0} width={b.fair_price_lkr} cls="total end" at={at}
             amount={rs10(b.fair_price_lkr)} />
      </div>
      <div className="wf-text">{b.sentences.map((s) => <p key={s}>{s}</p>)}</div>
      <p className="muted small">Blue pushes the price up, red pulls it down. Model: {b.model}. Method: {b.method}.
        The parts add up to the fair price: {ok ? "yes" : "no"}
        {b.checks.same_as_textbook_shapley_formula ? ", and they are the same as the textbook Shapley formula" : ""}.
        {b.from_cache ? " Saved earlier." : ` Made in ${b.seconds} s.`}</p>
    </div>
  );
}

function Row({ label, sub, left, width, cls, amount, at }) {
  return (
    <div className={`wf-row ${cls}`} role="row">
      <div className="wf-label" role="cell">{label}{sub && <small>{sub}</small>}</div>
      <div className="wf-track" role="cell"><span style={{ left: at(left), width: at(width) }} /></div>
      <div className="wf-amt" role="cell">{amount}</div>
    </div>
  );
}

