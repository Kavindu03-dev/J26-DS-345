import { useEffect, useMemo } from 'react';
import { divIcon, latLngBounds } from 'leaflet';
import { MapContainer, Marker, Polyline, Popup, TileLayer, useMap } from 'react-leaflet';
import { formatNumber, isHotel, isNumber, orderedStops } from '../lib/display.js';

const hotelIcon = divIcon({ className: 'map-marker hotel-marker', html: '<span>H</span>', iconSize: [34, 34], iconAnchor: [17, 17] });
const restaurantIcon = divIcon({ className: 'map-marker restaurant-marker', html: '<span>R</span>', iconSize: [28, 28], iconAnchor: [14, 14] });
const attractionIcon = (order) => divIcon({ className: 'map-marker attraction-marker', html: `<span>${Number(order)}</span>`, iconSize: [30, 30], iconAnchor: [15, 15] });
const validPoint = (item) => isNumber(item.latitude) && isNumber(item.longitude);

function FitBounds({ points }) {
  const map = useMap();
  useEffect(() => {
    const fit = () => {
      map.invalidateSize({ pan: false });
      if (points.length) map.fitBounds(latLngBounds(points), {
        padding: [35, 35], maxZoom: 15, animate: false,
      });
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [map, points]);
  return null;
}

export default function ItineraryMap({ hotel, day }) {
  const stops = useMemo(() => orderedStops(day).filter((stop) => !isHotel(stop) && validPoint(stop)), [day]);
  const restaurants = useMemo(() => (day.recommended_restaurants ?? []).filter(validPoint), [day]);
  const line = useMemo(() => (day.route?.geometry ?? [])
    .filter((pair) => Array.isArray(pair) && pair.length >= 2 && pair.every(isNumber))
    .map(([longitude, latitude]) => [latitude, longitude]), [day]);
  const bounds = useMemo(() => [
    ...line, ...(validPoint(hotel) ? [[hotel.latitude, hotel.longitude]] : []),
    ...stops.map((stop) => [stop.latitude, stop.longitude]),
    ...restaurants.map((restaurant) => [restaurant.latitude, restaurant.longitude]),
  ], [line, hotel, stops, restaurants]);
  return (
    <section className="card map-panel" aria-labelledby="map-heading">
      <div className="section-heading"><div><span className="eyebrow">EXPLORE THE DAY</span><h2 id="map-heading">Day {day.day} road route</h2></div>
        <span className="route-badge">{line.length > 1 ? 'OSRM road route' : 'Route unavailable'}</span></div>
      <div className="map-frame">
        <MapContainer center={[hotel.latitude, hotel.longitude]} zoom={12} scrollWheelZoom={false} className="itinerary-map" aria-label={`Day ${day.day} itinerary map`}>
          <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
          <FitBounds points={bounds} />
          {line.length > 1 && <Polyline positions={line} pathOptions={{ color: '#21675a', weight: 5, opacity: 0.9 }} />}
          {validPoint(hotel) && <Marker position={[hotel.latitude, hotel.longitude]} icon={hotelIcon}><Popup><strong>{hotel.name}</strong><br />Hotel · daily start and return</Popup></Marker>}
          {stops.map((stop) => <Marker key={`${stop.stop_order}-${stop.poi_id}`} position={[stop.latitude, stop.longitude]} icon={attractionIcon(stop.stop_order)}>
            <Popup><strong>{stop.stop_order}. {stop.name}</strong><br />Arrival: {stop.arrival_time ?? 'N/A'}<br />Visit: {formatNumber(stop.visit_duration_min, 0)} min</Popup>
          </Marker>)}
          {restaurants.map((restaurant) => <Marker key={restaurant.osm_id} position={[restaurant.latitude, restaurant.longitude]} icon={restaurantIcon}>
            <Popup><strong>#{restaurant.final_day_rank} {restaurant.name || 'Unnamed restaurant'}</strong><br />Detour: {formatNumber(restaurant.detour_time_min)} min<br />Score: {formatNumber(restaurant.final_restaurant_score, 3)}</Popup>
          </Marker>)}
        </MapContainer>
      </div>
      <div className="map-footer"><div className="map-legend"><span><i className="legend-hotel" />Hotel</span><span><i className="legend-attraction" />Attraction</span><span><i className="legend-restaurant" />Restaurant</span><span><i className="legend-route" />Road route</span></div><span className="muted">Day {day.day} · Driving</span></div>
      {!line.length && <p className="route-warning" role="status">Road geometry is unavailable for this day. Markers show the returned locations; no route has been estimated.</p>}
    </section>
  );
}
