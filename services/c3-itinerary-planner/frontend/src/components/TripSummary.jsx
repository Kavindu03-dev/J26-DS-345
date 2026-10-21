import { formatNumber, isNumber, orderedStops, poiCount, sumRouteMetric } from '../lib/display.js';

export default function TripSummary({ result, day }) {
  const days = result.days;
  const stops = orderedStops(day);
  const medical = result.medical_support;
  const dayMetrics = [
    ['Road distance', formatNumber(day.route?.distance_km), 'km'],
    ['Driving time', formatNumber(day.route?.duration_min), 'min'],
    ['Attractions', poiCount(day), 'POIs'],
    ['Start → return', `${stops[0]?.arrival_time ?? 'N/A'} → ${stops.at(-1)?.arrival_time ?? 'N/A'}`, ''],
  ];
  return (
    <>
      <section className="trip-overview" aria-label="Whole trip summary">
        <div className="hotel-overview"><span className="eyebrow">YOUR BASE</span><h2>{result.hotel.name}</h2>
          <span className="muted">{result.hotel.district} · {days.length} {days.length === 1 ? 'day' : 'days'} · Driving</span></div>
        <div className="trip-totals">
          <div><strong>{formatNumber(sumRouteMetric(days, 'distance_km'))}<small> km</small></strong><span>Total road distance</span></div>
          <div><strong>{formatNumber(sumRouteMetric(days, 'duration_min'))}<small> min</small></strong><span>Total driving time</span></div>
          <div><strong>{days.reduce((total, item) => total + poiCount(item), 0)}</strong><span>Attractions visited</span></div>
        </div>
      </section>
      <div className="day-metrics" aria-label={`Day ${day.day} summary`}>
        {dayMetrics.map(([label, value, unit]) => <div className="metric" key={label}>
          <span>{label}</span><strong>{value}<small> {unit}</small></strong>
        </div>)}
      </div>
      <div className="medical-summary">
        <span className="medical-cross" aria-hidden="true">+</span>
        <span><strong>Medical-access-aware mode</strong><span className="medical-state">Medical access: {medical?.required
          ? (medical.priority === 'high' ? 'High priority' : 'Normal priority') : 'Off'}</span></span>
        {medical?.required && <div className="medical-stats">
          {[
            ['Average', medical.average_hospital_access_min],
            ['Best', medical.best_hospital_access_min],
            ['Worst', medical.worst_hospital_access_min],
          ].filter(([, value]) => isNumber(value)).map(([label, value]) => <span key={label}>{label}: <strong>{formatNumber(value)} min</strong></span>)}
        </div>}
      </div>
    </>
  );
}
