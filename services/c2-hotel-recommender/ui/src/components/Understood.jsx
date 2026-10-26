import { feature, guests, HEALTH, IMPORTANCE, niceDate, plural, rs, STYLE } from "../format.js";

function Field({ k, children, wide }) {
  return (
    <div className={`field${wide ? " wide" : ""}`}>
      <div className="k">{k}</div>
      <div className="v">{children}</div>
    </div>
  );
}
const None = () => <span className="none">not said</span>;
const Chips = ({ items }) => (items.length ? <div className="chips">{items}</div> : <None />);

export default function Understood({ answer }) {
  const req = answer.understood_request, t = req.trip, h = req.hotel;
  let dates = <None />;
  if (t.check_in) {
    dates = <>{niceDate(t.check_in, !t.check_out)}{t.check_out ? ` to ${niceDate(t.check_out)}` : ""}
      {t.nights ? ` (${plural(t.nights, "night")})` : ""}
      {t.dates_fixed === false && <span className="none"> dates may change</span>}</>;
  } else if (t.nights) dates = `${plural(t.nights, "night")}, no dates`;
  const who = guests(t.adults, t.children);
  const budget = [h.budget_max_lkr ? `${rs(h.budget_max_lkr)} a night` : null,
                  t.budget_total_lkr ? `${rs(t.budget_total_lkr)} for the whole trip` : null].filter(Boolean);
  const musts = h.must.map((m) => <span key={m} className="chip must">
    {m === "budget" ? `Under ${rs(h.budget_max_lkr)}` : feature(m)}</span>);
  const wishes = Object.entries(h.wishes).map(([k, v]) => <span key={k} className={`chip wish-${v}`}>
    {feature(k)}: {IMPORTANCE[v] || v}</span>);
  const notNeeded = (req.not_needed || []).map((k) => <span key={k} className="chip not">{feature(k)}</span>);
  const health = req.health.map((x) => <span key={x} className="chip">{HEALTH[x] || x}</span>);
  const interests = req.interests.map((x) => <span key={x} className="chip">{x.replace(/_/g, " ")}</span>);
  const fixes = answer.nlp_notes.filter((n) => !n.startsWith("not needed"));

  return (
    <section className="ticket" aria-label="What I understood">
      <div className="stub">
        <b>What I understood</b>
        <span>from your sentence, in {answer.nlp_seconds} s</span>
      </div>
      <div className="ticket-body">
        <Field k="Stay in">{t.stay_in.length ? t.stay_in.join(", ") : <None />}</Field>
        <Field k="Dates">{dates}</Field>
        <Field k="Who">{who || <None />}{t.travel_style && <span className="none"> {STYLE[t.travel_style]}</span>}</Field>
        <Field k="Budget">{budget.length ? budget.map((b) => <div key={b}>{b}</div>) : <None />}</Field>
        <Field k="Must have" wide><Chips items={musts} /></Field>
        <Field k="Wishes"><Chips items={wishes} /></Field>
        {notNeeded.length > 0 && <Field k="Doesn't need"><Chips items={notNeeded} /></Field>}
        <Field k="Health"><Chips items={health} /></Field>
        <Field k="Interests"><Chips items={interests} /></Field>
        {t.want_to_visit?.length > 0 && <Field k="Wants to visit">{t.want_to_visit.join(", ")}</Field>}
      </div>
      <div className="ticket-foot">
        {fixes.length ? `The rule check fixed: ${fixes.join("; ")}.` : "The rule check changed nothing."}
        {answer.exchange_rates && <> Exchange rates: {answer.exchange_rates}.{" "}
          <a href="https://www.exchangerate-api.com" target="_blank" rel="noopener noreferrer">Rates By Exchange Rate API</a>.</>}
      </div>
    </section>
  );
}
