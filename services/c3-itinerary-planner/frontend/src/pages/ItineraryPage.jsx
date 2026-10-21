import { useState } from 'react';
import { planItinerary } from '../api/c3.js';
import { DEMO_SCENARIOS, createScenarioRequest } from '../data/demoScenarios.js';
import ConversationInput from '../components/ConversationInput.jsx';
import DayTabs from '../components/DayTabs.jsx';
import TripSummary from '../components/TripSummary.jsx';
import ItineraryMap from '../components/ItineraryMap.jsx';
import DayItinerary from '../components/DayItinerary.jsx';
import RestaurantDetails from '../components/RestaurantDetails.jsx';

function RouteSymbol({ large = false }) {
  return <svg className={large ? 'route-symbol large' : 'route-symbol'} viewBox="0 0 40 40" fill="none" aria-hidden="true">
    <path d="M10 11h15a7 7 0 0 1 0 14H15a5 5 0 0 0 0 10h14" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
    <circle cx="10" cy="11" r="4" fill="currentColor" /><circle cx="29" cy="35" r="4" fill="currentColor" />
  </svg>;
}

export default function ItineraryPage() {
  const [scenarioId, setScenarioId] = useState(DEMO_SCENARIOS[0].id);
  const [selectedHotelId, setSelectedHotelId] = useState(null);
  const [result, setResult] = useState(null);
  const [selectedDay, setSelectedDay] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const day = result?.days.find((item) => item.day === selectedDay);
  const scenario = DEMO_SCENARIOS.find((item) => item.id === scenarioId);

  function selectScenario(event) {
    setScenarioId(event.target.value);
    setSelectedHotelId(null);
    setResult(null);
    setSelectedDay(null);
    setError('');
  }

  function selectHotel(hotelId) {
    setSelectedHotelId(hotelId);
    setResult(null);
    setSelectedDay(null);
    setError('');
  }

  async function generate() {
    if (loading || !selectedHotelId) return;
    setError('');
    setResult(null);
    setSelectedDay(null);
    setLoading(true);
    try {
      const response = await planItinerary(createScenarioRequest(scenario, selectedHotelId));
      setResult(response);
      setSelectedDay(response.days[0]?.day ?? null);
    } catch (failure) {
      setError(failure.message || 'Unable to generate an itinerary. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <header className="site-header"><a className="brand" href="#main"><span className="brand-mark"><RouteSymbol /></span><span>Route & <strong>Discover</strong></span></a>
        <div className="header-context"><span className="status-dot" />Component 03<span className="header-divider" /><span>PP1 Research Demo</span></div>
      </header>
      <main id="main" className="page-shell">
        <div className="page-intro"><div><span className="eyebrow">PERSONALIZED JOURNEYS · SRI LANKA</span>
          <h1>Dynamic Spatial-Temporal<br className="desktop-break" /> Itinerary Planner</h1>
          <p>Route-Based Restaurant Recommendation</p></div>
          <div className="intro-note"><span className="note-line" />Your interests. A considered route.<br />A new way to explore.</div>
        </div>
        <nav className="architecture-flow" aria-label="Integration flow">
          <span>Conversation</span><span>C2 Recommendations</span><span>Hotel Selection</span><span>C3 Planning</span><span>Itinerary Results</span>
        </nav>
        <div className="workspace">
          <ConversationInput scenarios={DEMO_SCENARIOS} scenario={scenario} selectedHotelId={selectedHotelId}
            loading={loading} onScenario={selectScenario} onHotel={selectHotel} onGenerate={generate} />
          <div className="results" aria-busy={loading}>
            <div className="results-heading"><span className="eyebrow">02 / YOUR JOURNEY</span><span className={`result-state ${result ? 'ready' : ''}`}>{loading ? 'Planning in progress' : result ? 'Itinerary generated' : 'Ready to plan'}</span></div>
            {error && <div className="error-banner" role="alert"><strong>Could not generate itinerary</strong><pre>{error}</pre></div>}
            <div className="sr-only" role="status" aria-live="polite">{loading ? 'Generating your itinerary. This may take a minute.' : result ? `Itinerary generated with ${result.days.length} days.` : ''}</div>
            {result && day ? <>
              <DayTabs days={result.days} selectedDay={selectedDay} onSelect={setSelectedDay} />
              <TripSummary result={result} day={day} />
              <ItineraryMap hotel={result.hotel} day={day} />
              <div className="detail-panels"><DayItinerary day={day} /><RestaurantDetails day={day} /></div>
              <p className="response-reference">Trip reference: {result.trip_id}</p>
            </> : <section className="empty-journey card">
              <div className="empty-graphic"><RouteSymbol large /></div>
              <span className="eyebrow">{loading ? 'FINDING YOUR WAY' : 'A JOURNEY SHAPED AROUND YOU'}</span>
              <h2>{loading ? 'Building your itinerary' : result ? 'No itinerary days returned' : 'Your next journey starts here'}</h2>
              <p>{loading ? 'Optimizing attractions, tracing road routes and ranking nearby restaurants. This may take a minute.' : result ? 'No itinerary days were found for the supplied Component 2 input.' : 'Generate a day-by-day plan from the Component 2 input, with real road routes and places to eat along the way.'}</p>
              {!loading && !result && <div className="empty-steps"><span><b>01</b> Choose a hotel</span><span><b>02</b> Optimize attractions</span><span><b>03</b> Explore the route</span></div>}
              {loading && <div className="loading-track"><span /></div>}
            </section>}
          </div>
        </div>
        <footer className="page-footer"><span>COMPONENT 03 · SPATIAL-TEMPORAL ITINERARY OPTIMIZATION</span><span>Sri Lanka Tourism Research · PP1</span></footer>
      </main>
    </>
  );
}
