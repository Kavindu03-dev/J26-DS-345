import { formatNumber, isHotel, isNumber, orderedStops, poiCount } from '../lib/display.js';

export default function DayItinerary({ day }) {
  const stops = orderedStops(day);
  return (
    <section className="card itinerary-panel" aria-labelledby="itinerary-heading">
      <div className="section-heading"><div><span className="eyebrow">THE DAILY PLAN</span><h2 id="itinerary-heading">Day {day.day} itinerary</h2></div>
        <span className="count-badge">{poiCount(day)} attractions</span></div>
      <ol className="timeline">
        {stops.map((stop, index) => {
          const hotel = isHotel(stop);
          return <li key={`${stop.stop_order}-${stop.poi_id}`} className={hotel ? 'timeline-stop hotel-stop' : 'timeline-stop'}>
            <span className="stop-number" aria-label={`Stop ${stop.stop_order}`}>{stop.stop_order}</span>
            <div className="stop-content"><div className="stop-title"><h3>{stop.name}</h3><time>{stop.arrival_time ?? 'N/A'}</time></div>
              {hotel ? <span className="stop-caption">{index === 0 ? 'Start at hotel' : 'Return to hotel'}</span>
                : <div className="stop-caption"><span>{formatNumber(stop.visit_duration_min, 0)} min visit</span><span>Preference: {formatNumber(stop.preference_score)}</span></div>}
              {isNumber(stop.emergency_hospital_travel_time_min) && <div className="hospital-access">
                Hospital access: {formatNumber(stop.emergency_hospital_travel_time_min)} min
                {stop.nearest_emergency_hospital_name && <span>{stop.nearest_emergency_hospital_name}</span>}
              </div>}
            </div>
          </li>;
        })}
      </ol>
      {!stops.length && <p className="muted">No stops returned for this day.</p>}
    </section>
  );
}
