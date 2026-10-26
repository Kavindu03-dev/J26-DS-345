// Small helpers for numbers, dates and the words the page uses.

export const rs = (n) => (n == null ? "" : "Rs " + Math.round(n).toLocaleString("en-US"));
export const rs10 = (n) => rs(Math.round(n / 10) * 10);
export const rs100 = (n) => (Math.abs(n) < 50 ? "under Rs 100" : "Rs " + (Math.round(n / 100) * 100).toLocaleString("en-US"));

export function niceDate(d, withYear = true) {
  if (!d) return "";
  const date = new Date(String(d).slice(0, 10) + "T00:00:00");
  if (isNaN(date)) return String(d);
  return date.toLocaleDateString("en-GB", withYear ? { day: "numeric", month: "short", year: "numeric" }
    : { day: "numeric", month: "short" });
}

export const plural = (n, word, many) => `${n} ${n === 1 ? word : many || word + "s"}`;

export const FEATURE = {
  quiet: "Quiet", beach: "Beach", pool: "Pool", wifi: "WiFi", breakfast: "Breakfast", parking: "Parking",
  view: "A view", ac: "Air conditioning", family_room: "Family room", spa: "Spa", nature: "Nature or garden",
  near_hospital: "Near a hospital", low_steps: "Few steps or a lift", wheelchair_access: "Wheelchair access",
  budget: "Budget", beachfront: "Beachfront", bar: "Bar",
};
export const feature = (k) => FEATURE[k] || String(k).replace(/_/g, " ");

export const IMPORTANCE = { high: "really want", medium: "would like", low: "nice to have" };
export const HEALTH = { heart: "heart", mobility: "walking difficulty", wheelchair: "wheelchair",
  breathing: "breathing", elderly: "elderly traveller" };
export const STYLE = { solo: "solo", couple: "couple", family: "family", friends: "friends", business: "business" };
export const FACILITY = { pool: "Pool", bar: "Bar", breakfast: "Breakfast", beachfront: "Beachfront" };

export function guests(adults, children) {
  return [adults != null ? plural(adults, "adult") : null,
          children ? plural(children, "child", "children") : null].filter(Boolean).join(" + ");
}

export function removedReason(k) {
  const m = k.match(/must-have '(.+)' not met/);
  if (!m) return k;
  return m[1] === "budget" ? "over your budget" : `no ${feature(m[1]).toLowerCase()} (must-have)`;
}
