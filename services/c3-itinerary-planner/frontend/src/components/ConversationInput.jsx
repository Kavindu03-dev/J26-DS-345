const INTEREST_LABELS = {
  nature: 'Nature', wildlife: 'Wildlife', tea_country: 'Tea country',
  beach: 'Coast', culture: 'Culture', religious: 'Religious sites',
  surfing: 'Surfing', hiking: 'Hiking', nightlife: 'City experiences', food: 'Food',
};
const money = (value) => typeof value === 'number' && Number.isFinite(value)
  ? `LKR ${value.toLocaleString('en-US')}` : 'Not supplied';

export default function ConversationInput({ scenarios, scenario, selectedHotelId, loading, onScenario, onHotel, onGenerate }) {
  const { trip, interests, recommended_hotels: hotels } = scenario.for_c3;
  const selectedHotel = hotels.find((hotel) => hotel.hotel_id === selectedHotelId);
  return <aside className="control-panel card conversation-panel" aria-labelledby="conversation-heading">
    <div className="scenario-selector">
      <label htmlFor="demo-scenario">Demo Conversation Scenario</label>
      <select id="demo-scenario" value={scenario.id} onChange={onScenario} disabled={loading}>
        {scenarios.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select>
    </div>
    <div className="control-heading">
      <span className="eyebrow">01 / CONVERSATION CONTEXT</span>
      <h2 id="conversation-heading">A trip starts with a conversation</h2>
      <p>Predefined examples of the intended C4 → C2 → C3 flow. No live chat or C2 API is connected.</p>
    </div>
    <div className="conversation-message"><span>User</span><p>“{scenario.conversation}”</p></div>
    <div className="conversation-message assistant-message"><span>Assistant · demo</span><p>Review these example Component 2 recommendations and select your hotel.</p></div>
    <section className="hotel-recommendations" aria-labelledby="hotel-choices-heading">
      <h3 id="hotel-choices-heading">Component 2 Recommendations</h3>
      <p className="section-note">Project hotel records in demo order. Prices and hospital distances are unavailable.</p>
      {hotels.map((hotel) => <article className={`hotel-choice ${selectedHotelId === hotel.hotel_id ? 'selected' : ''}`} key={hotel.hotel_id}>
        <h4>{hotel.name}</h4><p className="hotel-town">Town: {hotel.town}</p>
        <dl className="hotel-prices">
          <div><dt>Price / night</dt><dd>{money(hotel.price_per_night_lkr)}</dd></div>
          <div><dt>Total cost</dt><dd>{money(hotel.total_hotel_cost_lkr)}</dd></div>
          <div><dt>Budget left</dt><dd>{money(hotel.budget_left_lkr)}</dd></div>
        </dl>
        <button type="button" className="select-hotel" aria-pressed={selectedHotelId === hotel.hotel_id} disabled={loading} onClick={() => onHotel(hotel.hotel_id)}>
          {selectedHotelId === hotel.hotel_id ? 'Selected Hotel' : 'Select Hotel'}<span className="sr-only">: {hotel.name}</span>
        </button>
      </article>)}
    </section>
    {selectedHotel ? <section className="planning-context" aria-labelledby="planning-context-heading">
      <span className="eyebrow">02 / COMPONENT 3 PLANNING</span>
      <h3 id="planning-context-heading">Selected Hotel</h3><p className="selected-hotel-name">{selectedHotel.name}</p>
      <p className="context-intro">Your selected hotel will be used to generate the optimized itinerary.</p>
      <dl className="input-facts">
        <div><dt>Dates</dt><dd>{trip.check_in} – {trip.check_out}</dd></div>
        <div><dt>Stay</dt><dd>{trip.nights} nights · {trip.travel_style}</dd></div>
        <div><dt>Party</dt><dd>{trip.adults} adults · {trip.children} child</dd></div>
      </dl>
      <div className="interest-chips">{interests.map((interest) => <span key={interest}>{INTEREST_LABELS[interest] ?? interest}</span>)}</div>
      <p className="planning-defaults"><strong>C3 defaults:</strong> Driving | 09:00–18:00</p>
      <p className="medical-context">Health needs: not shared.<br />Medical-access-aware optimization: Off.</p>
    </section> : <p className="selection-prompt">Select one hotel to review the C3 planning context.</p>}
    <button className="generate-button" disabled={loading || !selectedHotel} type="button" onClick={onGenerate}>
      {loading ? 'Generating itinerary…' : 'Generate Optimized Itinerary'}<span aria-hidden="true">{loading ? '· · ·' : '↗'}</span>
    </button>
  </aside>;
}
