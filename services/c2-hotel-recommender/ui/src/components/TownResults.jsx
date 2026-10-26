import { removedReason } from "../format.js";
import { Waiting } from "./Bits.jsx";
import HotelCard from "./HotelCard.jsx";

export default function TownResults({ town, request, live, onLive }) {
  const shown = town.ranked_hotels.length;
  const removed = Object.entries(town.removed).filter(([, n]) => n > 0);
  const needed = town.live_search?.needed;

  return (
    <section className="town" aria-label={`Hotels in ${town.town}`}>
      <div className="town-head">
        <h2>{town.town}</h2>
        <ol className="funnel">
          <li><strong>{town.hotels_considered}</strong> hotels looked at</li>
          <li><strong>{town.with_price}</strong> with a price for your group</li>
          <li><strong>{town.meet_all_musts}</strong> meet every must-have
            {town.musts_not_sure ? <span className="muted"> (+{town.musts_not_sure} not confirmed)</span> : null}</li>
          <li><strong>{shown}</strong> shown</li>
        </ol>
        {removed.length > 0 && (
          <ul className="removed">
            {removed.map(([k, n]) => <li key={k}>{n} removed: {removedReason(k)}</li>)}
          </ul>
        )}
      </div>

      {live?.state === "done" && <p className="note live-done">{live.message} ({live.seconds} s)</p>}
      {town.notes.map((n) => <p key={n} className="note">{n}</p>)}
      {needed && <LiveStep town={town} live={live} onLive={onLive} />}

      {!shown && !needed && <div className="panel"><p>No hotel here meets everything you asked for.</p></div>}
      <div className="hotels">
        {town.ranked_hotels.map((h) => <HotelCard key={h.hotel_id} h={h} request={request} asked={town.town} />)}
      </div>
    </section>
  );
}

function LiveStep({ town, live, onLive }) {
  if (live?.state === "waiting") return <div className="live-step"><Waiting text="Waiting for the photo search to finish…" /></div>;
  if (live?.state === "running") {
    return (
      <div className="live-step">
        <Waiting started={live.started}
                 text={`Searching Booking.com live for hotels in ${town.town}: one search page and a few hotel pages, about 30–90 s`} />
      </div>
    );
  }
  if (live?.state === "stopped" || live?.state === "failed") {
    return <div className="live-step"><b>The live search stopped.</b><p>{live.message}</p></div>;
  }
  return (
    <div className="live-step">
      <b>Not enough hotels in our data.</b>
      <p>{town.live_search.reason}</p>
      <button type="button" className="btn" onClick={onLive}>Search Booking.com now</button>
    </div>
  );
}
