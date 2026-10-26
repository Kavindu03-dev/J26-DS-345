// Talking to the Component 2 service (same address as this page).

async function call(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw { status: 0, detail: "The page could not reach the service. Is uvicorn still running?" };
  }
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw { status: res.status, detail: describe(body), body };
  return body;
}

function describe(body) {
  if (!body) return "No answer from the service.";
  if (typeof body.detail === "string") return body.detail;
  return body.hint || JSON.stringify(body.detail);
}

const post = (url, data) => call(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify(data) });

export const getHealth = () => call("/health");
export const findHotels = (text, topK, consent) =>
  post("/recommendations/from-text", { text, top_k: topK, health_consent: consent });
export const searchLive = (request, town, topK, consent) =>
  post("/recommendations/live", { request, town, top_k: topK, health_consent: consent });
export const getPhotos = (hotelId) => call(`/hotels/${encodeURIComponent(hotelId)}/photos`);
export const getPriceExplanation = (url) => call(url);

// One slow job at a time: the service uses one browser for live searches and photos.
let queue = Promise.resolve();
export function oneAtATime(job) {
  const run = queue.then(job, job);
  queue = run.catch(() => {});
  return run;
}
