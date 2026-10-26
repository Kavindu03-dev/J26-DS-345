// Where the fair price goes: pool, bar, breakfast, beachfront and everything else (the what-if of the price model).
import { FACILITY, rs } from "../format.js";

export default function MoneySplit({ b }) {
  if (!b) return null;
  const shown = b.parts.filter((p) => p.lkr != null);
  if (!shown.length) {
    const small = b.parts.filter((p) => p.status === "too_small").map((p) => FACILITY[p.facility].toLowerCase());
    return (
      <div className="split">
        <div className="row-label">Where the fair price goes</div>
        <p>None of the 4 facilities we can price (pool, bar, breakfast, beachfront) is a big part of this price, so all
          of the {rs(b.fair_price_lkr)} is for the room, the town, the hotel's class and the dates.
          {small.length > 0 && ` This hotel has ${small.join(" and ")}, but it is less than Rs 1,000 or 5% of the price, too small to be sure.`}</p>
      </div>
    );
  }
  const label = shown.map((p) => `${FACILITY[p.facility]} ${Math.round(p.pct_of_fair)}%`)
    .concat([`everything else ${Math.round(b.everything_else_pct)}%`]).join(", ");
  return (
    <div className="split">
      <div className="row-label">Where the fair price goes: {rs(b.fair_price_lkr)} {b.per}</div>
      <div className="split-bar" role="img" aria-label={label}>
        <span className="seg rest" style={{ flexGrow: Math.max(b.everything_else_lkr, 1) }} />
        {shown.map((p) => (
          <span key={p.facility} className={`seg seg-${p.facility}${p.taken_off ? " off" : ""}`}
                style={{ flexGrow: Math.max(p.lkr, 1) }} title={`${FACILITY[p.facility]}: ${rs(p.lkr)}`} />
        ))}
      </div>
      <ul className="split-list">
        {b.parts.filter((p) => p.has_it).map((p) => (
          <li key={p.facility}>
            <i className={`swatch ${p.lkr != null ? `seg-${p.facility}${p.taken_off ? " off" : ""}` : "rest"}`} />
            <span>{FACILITY[p.facility]}: {p.text}</span>
            <b>{p.lkr != null ? `${rs(p.lkr)} (${Math.round(p.pct_of_fair)}%)` : "–"}</b>
          </li>
        ))}
        <li>
          <i className="swatch rest" />
          <span>Everything else (the room, the town, the hotel's class, the dates)</span>
          <b>{rs(b.everything_else_lkr)} ({Math.round(b.everything_else_pct)}%)</b>
        </li>
      </ul>
      {b.your_fair_price_lkr !== b.fair_price_lkr && (
        <p>Striped parts are taken off your fair price (about {rs(b.your_fair_price_lkr)} for you).</p>
      )}
      <p className="muted small" title={b.note}>Amounts are the price model's estimate (what-if), rounded to Rs 100.</p>
    </div>
  );
}
