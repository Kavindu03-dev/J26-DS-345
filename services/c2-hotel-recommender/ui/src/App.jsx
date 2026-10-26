import { useEffect, useRef, useState } from "react";
import { findHotels, getHealth, oneAtATime, searchLive } from "./api.js";
import Composer from "./components/Composer.jsx";
import Understood from "./components/Understood.jsx";
import TownResults from "./components/TownResults.jsx";
import { Waiting, ErrorBox } from "./components/Bits.jsx";

let nextId = 1;

export default function App() {
  const [health, setHealth] = useState(null);
  const [searches, setSearches] = useState([]);           // every search in this visit, newest last
  const [currentId, setCurrentId] = useState(null);
  const [options, setOptions] = useState({ consent: false, live: true, topK: 5 });
  const [live, setLive] = useState({});                    // "searchId:town" -> live search state
  const liveStarted = useRef(new Set());
  const resultsRef = useRef(null);

  useEffect(() => {
    getHealth().then((h) => setHealth({ ok: true, ...h })).catch(() => setHealth({ ok: false }));
  }, []);

  const update = (id, change) => setSearches((all) => all.map((s) => (s.id === id ? { ...s, ...change(s) } : s)));

  async function run(text) {
    const id = nextId++;
    setSearches((all) => [...all, { id, text, status: "loading", started: Date.now() }]);
    setCurrentId(id);
    setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    try {
      const answer = await findHotels(text, options.topK, options.consent);
      update(id, () => ({ status: "done", answer, options: { ...options } }));
    } catch (error) {
      update(id, () => ({ status: "error", error }));
    }
  }

  function runLive(search, townIndex) {
    const town = search.answer.towns[townIndex];
    const key = `${search.id}:${town.town}`;
    if (liveStarted.current.has(key)) return;
    liveStarted.current.add(key);
    setLive((l) => ({ ...l, [key]: { state: "waiting", started: Date.now() } }));
    oneAtATime(async () => {
      setLive((l) => ({ ...l, [key]: { state: "running", started: Date.now() } }));
      try {
        const body = await searchLive(search.answer.understood_request, town.town, search.options.topK,
                                      search.options.consent);
        const report = body.live_report;
        if (report.status !== "ok") {
          setLive((l) => ({ ...l, [key]: { state: "stopped", message: report.message } }));
          return;
        }
        update(search.id, (s) => {
          const towns = s.answer.towns.map((t, i) => (i === townIndex ? body.town : t));
          return { answer: { ...s.answer, towns, for_c3: body.for_c3 } };
        });
        setLive((l) => ({ ...l, [key]: { state: "done", message: report.message, seconds: report.seconds } }));
      } catch (error) {
        setLive((l) => ({ ...l, [key]: { state: "failed", message: error.detail } }));
      }
    });
  }

  const current = searches.find((s) => s.id === currentId);

  return (
    <div className="page">
      <Composer onSend={run} busy={current?.status === "loading"} options={options} setOptions={setOptions}
                health={health} searches={searches} currentId={currentId} onPick={setCurrentId} />
      <main className="results" ref={resultsRef} aria-live="polite">
        {!current && <Intro />}
        {current?.status === "loading" && (
          <div className="panel">
            <p className="you-said">{current.text}</p>
            <Waiting started={current.started}
                     text="Reading your request with the language model (about 10 seconds, longer the first time)" />
          </div>
        )}
        {current?.status === "error" && (
          <div className="panel">
            <p className="you-said">{current.text}</p>
            <ErrorBox error={current.error} />
          </div>
        )}
        {current?.status === "done" && (
          <Answer search={current} live={live} onLive={(i) => runLive(current, i)} />
        )}
      </main>
      <footer className="foot">
        Component 2 of J26-DS-345: personalised hotel ranking and price intelligence.
        Prices were collected on Booking.com, Agoda and Trip.com in September 2026; hotels found live are newer.
        {" "}<a href="/docs">API page</a>, <a href="/chat">old chat page</a>.
      </footer>
    </div>
  );
}

function Intro() {
  return (
    <section className="intro">
      <h2>What you get back</h2>
      <ul>
        <li><b>Only hotels that have your must-haves.</b> A hotel we can't confirm is shown last and marked.</li>
        <li><b>A fair price for you.</b> What the hotel should cost, minus what it charges for things you don't need.</li>
        <li><b>Why the price is what it is,</b> in rupees, and where the same deal is cheapest.</li>
      </ul>
    </section>
  );
}

function Answer({ search, live, onLive }) {
  const a = search.answer;
  const [tab, setTab] = useState(0);
  const town = a.towns[Math.min(tab, a.towns.length - 1)];
  const key = town ? `${search.id}:${town.town}` : null;

  useEffect(() => {          // the live search starts by itself when the box is ticked
    if (!search.options.live) return;
    a.towns.forEach((t, i) => { if (t.live_search?.needed) onLive(i); });
  }, [search.id]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <p className="you-said">{search.text}</p>
      <Understood answer={a} />
      {!a.towns.length && (
        <div className="panel"><p>I could not find a town in your sentence. Which town do you want to stay in?</p></div>
      )}
      {a.towns.length > 1 && (
        <div className="town-tabs" role="tablist" aria-label="Towns">
          {a.towns.map((t, i) => (
            <button key={t.town} role="tab" aria-selected={i === tab} className={i === tab ? "on" : ""}
                    onClick={() => setTab(i)}>
              {t.town} <span>{t.ranked_hotels.length}</span>
            </button>
          ))}
        </div>
      )}
      {town && (
        <TownResults key={key} town={town} request={a.understood_request} live={live[key]}
                     onLive={() => onLive(a.towns.indexOf(town))} />
      )}
      <details className="raw">
        <summary>Data sent to Component 3</summary>
        <pre>{JSON.stringify(a.for_c3, null, 2)}</pre>
      </details>
      <details className="raw">
        <summary>Full answer (JSON)</summary>
        <pre>{JSON.stringify(a, null, 2)}</pre>
      </details>
    </>
  );
}
