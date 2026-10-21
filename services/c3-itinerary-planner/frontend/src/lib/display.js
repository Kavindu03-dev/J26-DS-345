export const isNumber = (value) => typeof value === 'number' && Number.isFinite(value);
export const formatNumber = (value, digits = 1) => isNumber(value)
  ? value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
  : 'N/A';
export const isHotel = (stop) => stop.poi_id === 'HOTEL';
export const orderedStops = (day) => [...(day?.stops ?? [])].sort((a, b) => a.stop_order - b.stop_order);
export const poiCount = (day) => (day?.stops ?? []).filter((stop) => !isHotel(stop)).length;
export function sumRouteMetric(days, key) {
  return days.length && days.every((day) => isNumber(day.route?.[key]))
    ? days.reduce((total, day) => total + day.route[key], 0)
    : null;
}
