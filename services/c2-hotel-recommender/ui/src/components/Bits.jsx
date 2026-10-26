import { useEffect, useState } from "react";

export function Seconds({ started }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  return <span className="secs">{Math.max(0, Math.round((now - started) / 1000))} s</span>;
}

export function Waiting({ text, started }) {
  return (
    <div className="waiting" role="status">
      <span className="gem-spin" aria-hidden="true" />
      <span>{text}{started ? <> <Seconds started={started} /></> : null}</span>
    </div>
  );
}

export function ErrorBox({ error }) {
  const detail = error?.detail || "Something went wrong.";
  let what = "The search did not work.";
  if (error?.status === 503 && /NLP|Ollama/.test(detail)) what = "The language model is not running. Start Ollama (with hotel-nlp-v2), then send again.";
  else if (error?.status === 503) what = "The hotel data was not found. Check C2_DATA_DIR, then restart the service.";
  else if (error?.status === 0) what = "The service is not reachable.";
  return (
    <div className="error" role="alert">
      <b>{what}</b>
      <p>{detail}</p>
    </div>
  );
}

const MARK = { yes: "✓", no: "✗", "not sure": "?" };
export function Check({ label, status, why, must }) {
  const cls = status === "not sure" ? "unsure" : status;
  return (
    <span className={`check ${cls}${must ? " must" : ""}`} title={why || ""}>
      <i aria-hidden="true">{MARK[status] || "?"}</i>
      <span>{label}</span>
      <span className="sr">({status})</span>
    </span>
  );
}

export function Gem({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="gem">
      <path d="M16 2 28 12 16 30 4 12Z" fill="currentColor" />
      <path d="M4 12h24M16 2l-5 10 5 18 5-18-5-10" fill="none" stroke="var(--gem-line)" strokeWidth="1.5" />
    </svg>
  );
}
