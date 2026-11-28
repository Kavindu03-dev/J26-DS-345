// The fair-price gauge: where this price sits against the fair price (good value / expected range / overpriced).
// The limits are the hotel's own town's (price.price_limits, from price_margins.csv): the cheapest / dearest
// 1 in 10 prices against the fair price there. Without the file the service sends the fixed 30% rule.
// The gauge uses a ratio scale: from 2.5 times cheaper (60% below) to 2.5 times dearer (150% above) the fair price.
const LOW = -60, HIGH = 150, SPAN = Math.log(2.5);
const pos = (x) => 50 + 50 * Math.max(-1, Math.min(1, Math.log(Math.max(1 + x / 100, 0.01)) / SPAN));
const pct = (x) => `${Math.round(Math.abs(x))}%`;

export function limits(p) {
  const l = p.price_limits || {};
  return { lo: l.good_deal_below_pct ?? 30, hi: l.looks_high_above_pct ?? 30, place: l.place || null };
}

export function verdict(p) {
  const you = p.price_vs_your_fair_pct, all = p.price_vs_fair_pct;
  const forYou = p.what_if.length > 0 ? " for you" : "";
  const { lo, hi, place } = limits(p);
  if (you == null) return { kind: "none", title: "No fair price", detail: "We have no fair price for this hotel." };
  const side = (x) => (x >= 0 ? `about ${pct(x)} above` : `about ${pct(x)} below`);
  const highNote = place ? ` In ${place}, the dearest 1 in 10 prices are ${pct(hi)} or more above it.` : "";
  const lowNote = place ? ` In ${place}, the cheapest 1 in 10 prices are ${pct(lo)} or more below it.` : "";
  if (p.price_status === "Potentially Overpriced") {
    if (all != null && all <= hi) {
      return { kind: "over", title: "Overpriced for you",
               detail: `Fair for most guests (${side(all)}), but ${side(you)} the fair price for you.${highNote}` };
    }
    return { kind: "over", title: "Potentially overpriced", detail: `${side(you)} the fair price${forYou}.${highNote}` };
  }
  if (p.good_value ?? you <= -lo) {
    return { kind: "good", title: "Good value", detail: `${side(you)} the fair price${forYou}.${lowNote}` };
  }
  return { kind: "ok", title: "Within the expected price range", detail: `${side(you)} the fair price${forYou}.` };
}

export default function PriceGauge({ p, v }) {
  const you = p.price_vs_your_fair_pct, all = p.price_vs_fair_pct;
  if (you == null) return <div className="gauge-wrap"><div className="verdict none">{v.title}</div></div>;
  const { lo, hi } = limits(p);
  const a = pos(-lo), b = pos(hi);                     // where the good-value and overpriced zones start
  const columns = { gridTemplateColumns: `${a}% ${b - a}% ${100 - b}%` };
  const ghost = p.what_if.length > 0 && all != null && Math.abs(all - you) >= 2;
  const out = you > HIGH ? "right" : you < LOW ? "left" : null;
  return (
    <div className={`gauge-wrap ${v.kind}`}>
      <div className="verdict"><b>{v.title}</b> <span>{v.detail}</span></div>
      <div className="gauge" style={columns} role="img" aria-label={`${v.title}: ${v.detail}`}>
        <span className="zone good" /><span className="zone ok" /><span className="zone over" />
        <span className="tick" style={{ left: "50%" }} />
        {ghost && <span className="mark ghost" style={{ left: `${pos(all)}%` }} title="Price against the fair price for most guests" />}
        <span className={`mark${out ? " out-" + out : ""}`} style={{ left: `${pos(you)}%` }}
              title={`Price against the fair price${ghost ? " for you" : ""}`} />
      </div>
      <div className="gauge-labels" style={columns} aria-hidden="true">
        <span>Good value<small> {pct(lo)} or more below</small></span>
        <span>Fair price</span>
        <span>Overpriced<small> over {pct(hi)} above</small></span>
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