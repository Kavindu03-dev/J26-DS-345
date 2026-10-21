export async function planItinerary(payload) {
  let response;
  try {
    response = await fetch('/api/plan-itinerary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error('Unable to reach the itinerary service. Check the backend connection and try again.');
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if (Array.isArray(data?.detail)) {
      throw new Error(data.detail.map((issue) => `${issue.loc.join('.')}: ${issue.msg}`).join('; '));
    }
    throw new Error(typeof data?.detail === 'string'
      ? data.detail
      : `The itinerary service could not complete the request (HTTP ${response.status}). Check that C3 and OSRM are running.`);
  }
  if (data?.status !== 'success' || !Array.isArray(data.days) || !data.hotel) {
    throw new Error('The itinerary service returned an unexpected response.');
  }
  return data;
}
