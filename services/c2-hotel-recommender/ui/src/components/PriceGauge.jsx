// The fair-price gauge: where this price sits against the fair price (good value / expected range / overpriced).
const LOW = -75, HIGH = 75, LIMIT = 30;               // the system's rule: 30% below = good value, 30% above = overpriced
const pos = (x) => ((Math.max(LOW, Math.min(HIGH, x)) - LOW) / (HIGH - LOW)) * 100;
const pct = (x) => `${Math.round(Math.abs(x))}%`;

export function verdict(p) {
  const you = p.price_vs_your_fair_pct, all = p.price_vs_fair_pct;
  const forYou = p.what_if.length > 0 ? " for you" : "";
  if (you == null) return { kind: "none", title: "No fair price", detail: "We have no fair price for this hotel." };
  const side = (x) => (x >= 0 ? `about ${pct(x)} above` : `about ${pct(x)} below`);
  if (p.price_status === "Potentially Overpriced") {
    if (all != null && all <= LIMIT) {
      return { kind: "over", title: "Overpriced for you",
               detail: `Fair for most guests (${side(all)}), but ${side(you)} the fair price for you.` };
    }
    return { kind: "over", title: "Potentially overpriced", detail: `${side(you)} the fair price${forYou}.` };
  }
  if (you <= -LIMIT) return { kind: "good", title: "Good value", detail: `${side(you)} the fair price${forYou}.` };
  return { kind: "ok", title: "Within the expected price range", detail: `${side(you)} the fair price${forYou}.` };
}

export default function PriceGauge({ p, v }) {
  const you = p.price_vs_your_fair_pct, all = p.price_vs_fair_pct;
  if (you == null) return <div className="gauge-wrap"><div className="verdict none">{v.title}</div></div>;
  const ghost = p.what_if.length > 0 && all != null && Math.abs(all - you) >= 2;
  const out = you > HIGH ? "right" : you < LOW ? "left" : null;
  return (
    <div className={`gauge-wrap ${v.kind}`}>
      <div className="verdict"><b>{v.title}</b> <span>{v.detail}</span></div>
      <div className="gauge" role="img" aria-label={`${v.title}: ${v.detail}`}>
        <span className="zone good" /><span className="zone ok" /><span className="zone over" />
        <span className="tick" style={{ left: "50%" }} />
        {ghost && <span className="mark ghost" style={{ left: `${pos(all)}%` }} title="Price against the fair price for most guests" />}
        <span className={`mark${out ? " out-" + out : ""}`} style={{ left: `${pos(you)}%` }}
              title={`Price against the fair price${ghost ? " for you" : ""}`} />
      </div>
      <div className="gauge-labels" aria-hidden="true">
        <span>Good value<small> 30% or more below</small></span>
        <span>Fair price</span>
        <span>Overpriced<small> over 30% above</small></span>
      </div>
      {ghost && (
        <p className="gauge-key">
          <span className="key"><span className="key-mark" />this price against the fair price for you</span>
          <span className="key"><span className="key-mark ghost" />against the fair price for most guests</span>
        </p>
      )}
    </div>
  );
}
