import { formatNumber, isNumber } from '../lib/display.js';

export default function RestaurantDetails({ day }) {
  const restaurants = [...(day.recommended_restaurants ?? [])].sort((a, b) => a.final_day_rank - b.final_day_rank);
  return (
    <section className="card restaurant-panel" aria-labelledby="restaurants-heading">
      <div className="section-heading"><div><span className="eyebrow">ALONG YOUR ROUTE</span><h2 id="restaurants-heading">Restaurant recommendations</h2></div>
        <span className="count-badge">{restaurants.length} places</span></div>
      <p className="section-note">Ranked suggestions near this day's road route. Visits are optional and are not included in route totals.</p>
      <div className="restaurant-list">
        {restaurants.map((restaurant) => {
          const hasAbsa = restaurant.has_absa_data === true;
          const aspects = hasAbsa ? ['food', 'service', 'ambience', 'value', 'location']
            .filter((aspect) => isNumber(restaurant[`${aspect}_score`])) : [];
          const hasWeightedScore = hasAbsa && isNumber(restaurant.weighted_absa_score);
          return <article className="restaurant" key={`${day.day}-${restaurant.osm_id}`}>
            <span className="restaurant-rank">{restaurant.final_day_rank}</span>
            <div className="restaurant-title">
              <h3>{restaurant.name || 'Unnamed restaurant'}</h3>
              <p title={restaurant.cuisine?.replaceAll(';', ' · ').replaceAll('_', ' ')}>
                {restaurant.cuisine?.replaceAll(';', ' · ').replaceAll('_', ' ') || 'Cuisine unavailable'}
              </p>
            </div>
            <div className="restaurant-meta">
              <span className="detour-badge">+{formatNumber(restaurant.detour_time_min)} min detour</span>
              <p className="ranking-mode">{hasAbsa ? 'ABSA + Cuisine + Detour' : 'Cuisine + Detour'}</p>
            </div>
            <div className="restaurant-score"><strong>{formatNumber(restaurant.final_restaurant_score, 3)}</strong><span>Final score</span></div>
            {(aspects.length > 0 || hasWeightedScore) && <dl className="aspect-scores">
              {aspects.map((aspect) => <div key={aspect}>
                <dt>{aspect}</dt><dd>{formatNumber(restaurant[`${aspect}_score`], 2)}</dd>
              </div>)}
              {hasWeightedScore && <div className="weighted-score"><dt>Weighted ABSA</dt><dd>{formatNumber(restaurant.weighted_absa_score, 3)}</dd></div>}
            </dl>}
          </article>;
        })}
      </div>
      {!restaurants.length && <p className="muted">No restaurant recommendations returned for this day.</p>}
    </section>
  );
}
