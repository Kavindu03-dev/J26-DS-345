import { useRef, useState } from "react";
import { Gem } from "./Bits.jsx";

const EXAMPLES = [
  ["Family in Ella, knee pain", "We are a family of 2 adults and 1 child. We want to stay in Ella from 20 to 23 December, but dates are not fixed. We have Rs. 100,000 for the whole trip and the hotel must be under Rs. 15,000 a night. Quiet place is very important, pool would be nice. My father has knee pain."],
  ["Heart patient in Galle", "My father is joining us. He has heart problems, so we need a hotel near a hospital in Galle for 3 nights from 4 December."],
  ["4 friends, $60 a night", "Hotel in Ella under $60 per night for 2 nights from 15 November. Me and my 3 friends love hiking."],
  ["6 friends, no breakfast", "Group of 6 friends, Mirissa, 2 nights from 5 January, about 3 lakhs total, we don't need breakfast but AC is a must"],
  ["Tonight in Negombo", "Need a room in Negombo for tonight only, near the airport, max Rs. 15,000."],
  ["Tangalle, searched live", "Tangalle beach resort for my husband and me, 20 to 23 January. Breakfast is very important."],
  ["Overpriced hotels", "Hotel in Galle for 2 nights from 15 January for my wife and me. We really want a beachfront hotel and we'd like a spa."],
  ["Overpriced only for you", "We are a couple going to Trincomalee for 2 nights from 15 January. A spa is very important and we would like a view. We don't need a pool, a bar or breakfast."],
];

export default function Composer({ onSend, busy, options, setOptions, health, searches, currentId, onPick }) {
  const [text, setText] = useState("");
  const box = useRef(null);

  function send(t = text) {
    const clean = t.trim();
    if (!clean || busy) return;
    onSend(clean);
    setText("");
  }

  return (
    <header className={`hero${searches.length ? " compact" : ""}`}>
      <div className="hero-in">
        <div className="topline">
          <div className="brand"><Gem /> <span>Hotel finder</span><small>Sri Lanka, Component 2</small></div>
          <Status health={health} />
        </div>
        <h1>{searches.length ? "Another trip?" : "Tell me about your trip in one sentence."}</h1>
        <p className="lede">The town, your dates, who is coming, your budget, what the hotel must have and what would
          be nice. Spelling mistakes, dollars and lakhs are fine.</p>
        <form className="ask" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <label htmlFor="trip" className="sr">Your trip</label>
          <textarea id="trip" ref={box} rows={2} value={text} disabled={busy}
                    placeholder="Hotel in Kandy for 2 nights from 5 December, under Rs 15,000, a pool please"
                    onChange={(e) => { setText(e.target.value); e.target.style.height = "auto";
                                       e.target.style.height = e.target.scrollHeight + "px"; }}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
          <button type="submit" className="go" disabled={busy || !text.trim()}>{busy ? "Finding…" : "Find hotels"}</button>
        </form>
        <div className="examples" aria-label="Example requests">
          {EXAMPLES.map(([title, sentence]) => (
            <button key={title} type="button" className="example" disabled={busy} title={sentence}
                    onClick={() => send(sentence)}>{title}</button>
          ))}
        </div>
        <div className="options">
          <label><input type="checkbox" checked={options.live}
                        onChange={(e) => setOptions({ ...options, live: e.target.checked })} />
            Search Booking.com live when our data is not enough</label>
          <label><input type="checkbox" checked={options.consent}
                        onChange={(e) => setOptions({ ...options, consent: e.target.checked })} />
            Share health needs with Component 3</label>
          <label>Show
            <select value={options.topK} onChange={(e) => setOptions({ ...options, topK: Number(e.target.value) })}>
              {[3, 5, 8, 10].map((n) => <option key={n}>{n}</option>)}
            </select> hotels</label>
        </div>
        {searches.length > 1 && (
          <nav className="recent" aria-label="Your searches">
            <span>Your searches:</span>
            {searches.map((s) => (
              <button key={s.id} type="button" className={s.id === currentId ? "on" : ""} onClick={() => onPick(s.id)}
                      title={s.text}>{s.text.length > 38 ? s.text.slice(0, 36) + "…" : s.text}</button>
            ))}
          </nav>
        )}
      </div>
    </header>
  );
}

function Status({ health }) {
  if (!health) return <div className="status"><span className="pill">Checking the service…</span></div>;
  if (!health.ok) return <div className="status"><span className="pill bad">Service not reachable</span></div>;
  return (
    <div className="status">
      <span className="pill ok">Service running</span>
      {health.c1_mode === "dummy"
        ? <span className="pill warn">Review scores are made up until Component 1 is ready</span>
        : <span className="pill ok">Review scores from Component 1 ({health.c1_mode})</span>}
    </div>
  );
}
