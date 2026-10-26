import { useEffect, useRef, useState } from "react";
import { getPhotos, oneAtATime } from "../api.js";
import { feature, niceDate, rs } from "../format.js";
import { Check } from "./Bits.jsx";
import PriceGauge, { verdict } from "./PriceGauge.jsx";
import MoneySplit from "./MoneySplit.jsx";
import PriceWhy from "./PriceWhy.jsx";

const TABS = [<>Price<span className="wide-only"> and booking</span></>, "Why this price", "Why this hotel"];

export default function HotelCard({ h, request, asked }) {
  const [tab, setTab] = useState(0);
  const p = h.price;
  const budget = request.hotel.budget_max_lkr;
  const v = verdict(p);

  const musts = Object.entries(h.must_check).map(([m, s]) => {
    let label = m === "budget" ? `Under ${rs(budget)}` : feature(m);
    if (m === "near_hospital" && h.nearest_hospital.km != null) label += ` (${h.nearest_hospital.km} km)`;
    if (s === "not sure") label += ": not confirmed";
    const why = m === "budget" ? `${rs(p.price_per_night_lkr)} a night` : (h.feature_checks[m] || {}).why;
    return <Check key={m} label={label} status={s} why={why} must />;
  });
  const wishes = Object.entries(h.feature_checks).filter(([k]) => !(k in h.must_check)).map(([k, f]) => {
    let label = feature(k);
    if (f.reviews === "contradicted") label += f.listing === "yes" ? ": listed, but reviews say no" : ": reviews say no";
    else if (f.reviews === "confirmed") label += ": reviews agree";
    else if (f.status === "not sure") label += ": not sure";
    else if (f.status === "no") label += ": no";
    return <Check key={k} label={label} status={f.status} why={f.why} />;
  });

  return (
    <article className={`hotel${h.musts_confirmed ? "" : " unsure"}`} aria-label={`${h.rank}. ${h.name}`}>
      <Photos h={h} />
      <div className="hotel-body">
        <div className="hotel-head">
          <span className="rank" aria-hidden="true">{h.rank}</span>
          <div className="who">
            <h3>{h.name}</h3>
            <p className="where">
              {h.town}{h.km_from_town != null ? `, ${h.km_from_town} km from ${asked} centre` : ""}. Match {Math.round(h.score * 100)}%.
            </p>
            <div className="badges">
              {h.found_live && <span className="badge live">Found live on Booking.com{h.found_live.date ? `, ${niceDate(h.found_live.date)}` : ""}</span>}
              {!h.musts_confirmed && <span className="badge unsure">A must-have is not confirmed</span>}
            </div>
          </div>
          <div className="price">
            <div className="n">{rs(p.price_per_night_lkr)}<small> a night</small></div>
            {h.nights ? <div className="t">{h.nights} night{h.nights > 1 ? "s" : ""}: {rs(h.total_hotel_cost_lkr)}</div> : null}
            {h.budget_left_lkr != null && <div className="t">Budget left: {rs(h.budget_left_lkr)}</div>}
          </div>
        </div>

        {(musts.length > 0 || wishes.length > 0) && (
          <div className="checks-row">
            {musts.length > 0 && <div><div className="row-label">Must-haves</div><div className="checks">{musts}</div></div>}
            {wishes.length > 0 && <div><div className="row-label">Wishes</div><div className="checks">{wishes}</div></div>}
          </div>
        )}

        <PriceGauge p={p} v={v} />

        <div className="tabs" role="tablist" aria-label={`More about ${h.name}`}>
          {TABS.map((t, i) => (
            <button key={i} role="tab" aria-selected={tab === i} className={tab === i ? "on" : ""}
                    onClick={() => setTab(i)}>{t}</button>
          ))}
        </div>
        <div className="tab-panel" role="tabpanel">
          {tab === 0 && <PriceAndBooking h={h} p={p} request={request} />}
          {tab === 1 && <PriceWhy e={p.explanation} />}
          {tab === 2 && <WhyHotel h={h} />}
        </div>
      </div>
    </article>
  );
}

function PriceAndBooking({ h, p, request }) {
  const fair = p.your_expected_price_lkr || p.expected_price_lkr;
  const forYou = p.what_if.length > 0;
  const exactDates = h.found_live && p.seen_checkin === request.trip.check_in;
  const estimate = (o) => h.found_live && o.platform !== "Booking.com";
  return (
    <div className="pb">
      <div>
        {fair ? <p className="fair">Fair price {forYou ? "for you" : "for this hotel"}: about <b>{rs(fair)}</b> a night
          {forYou && p.expected_price_lkr ? <span className="muted"> (for most guests: about {rs(p.expected_price_lkr)})</span> : null}.</p>
          : <p className="muted">No fair price for this hotel.</p>}
        {p.caution && <p className="caution">{p.caution}</p>}
        {p.what_if.map((m) => <p key={m} className="whatif">{m}</p>)}
        <MoneySplit b={p.breakdown} />
      </div>
      <div className="book">
        <div className="row-label">Where to book</div>
        <ul className="sites">
          {p.offers.map((o) => (
            <li key={o.platform} className={o.platform === p.best_platform ? "best" : ""}>
              <span>{o.platform}{o.platform === "Agoda" ? " (from)" : ""}{estimate(o) ? " (estimate)" : ""}
                {o.platform === p.best_platform ? <b> best</b> : null}</span>
              <span className="amt">{rs(o.price_lkr)}</span>
            </li>
          ))}
        </ul>
        <p className="muted">{exactDates ? "The Booking.com price is live for your dates."
          : `Prices seen for check-in ${niceDate(p.seen_checkin)}, moved to your dates by the price model.`}</p>
        {p.where_to_book?.advice && <p>{p.where_to_book.advice}</p>}
        {h.booking_url && <a className="btn" href={h.booking_url} target="_blank" rel="noopener noreferrer">Open on {p.best_platform}</a>}
        {Object.keys(h.links || {}).length > 1 && (
          <p className="other-links">Also on: {Object.entries(h.links).filter(([site]) => site !== p.best_platform)
            .map(([site, url], i) => <span key={site}>{i ? ", " : ""}<a href={url} target="_blank" rel="noopener noreferrer">{site}</a></span>)}</p>
        )}
      </div>
    </div>
  );
}

function WhyHotel({ h }) {
  const c1 = h.review_scores_c1 || {};
  const dummy = String(c1.source || "").includes("DUMMY");
  return (
    <div className="why-hotel">
      <p>{h.explanation}</p>
      <dl>
        <div><dt>Nearest hospital</dt><dd>{h.nearest_hospital.name || "unknown"} ({h.nearest_hospital.km ?? "?"} km, straight line)</dd></div>
        {c1.overall_trust_score != null && (
          <div><dt>Review score</dt><dd>{c1.overall_trust_score}/10{dummy ? " (made up until Component 1 is ready)" : ""}</dd></div>
        )}
        <div><dt>Free cancellation</dt><dd>{h.free_cancellation.advice}</dd></div>
      </dl>
    </div>
  );
}

const photoCache = {};
function Photos({ h }) {
  const [photos, setPhotos] = useState(h.photos?.length ? h.photos : photoCache[h.hotel_id]);
  const asked = useRef(false);
  useEffect(() => {
    if (photos || asked.current) return;
    asked.current = true;
    oneAtATime(() => getPhotos(h.hotel_id))
      .then((b) => { photoCache[h.hotel_id] = b.photos || []; setPhotos(photoCache[h.hotel_id]); })
      .catch(() => setPhotos([]));
  }, [h.hotel_id, photos]);
  if (!photos) return <div className="photos empty">Loading photos…</div>;
  if (!photos.length) return <div className="photos empty">No photos found</div>;
  return (
    <div className="photos">
      {photos.slice(0, 4).map((ph, i) => (
        <a key={ph.photo_url} href={ph.photo_url} target="_blank" rel="noopener noreferrer">
          <img src={i ? ph.thumb_url || ph.photo_url : ph.photo_url} alt={`${h.name}, photo ${i + 1}`} loading="lazy"
               referrerPolicy="no-referrer" />
        </a>
      ))}
    </div>
  );
}
