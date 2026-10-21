"""Distances: town points, hotels near a town, nearest hospital to each hotel."""
import re

import numpy as np
import pandas as pd

EARTH_KM = 6371.0

# Towns that OpenStreetMap does not have under this name. Approximate points, typed by hand.
MANUAL_POINTS = {
    "Hiriketiya": (5.963, 80.709),
    "Panadura": (6.713, 79.904),
    "Mount Lavinia": (6.838, 79.863),
    "Sinharaja": (6.440, 80.421),       # Kudawa entrance
    "Kitulgala": (6.998, 80.408),       # OSM spells it Kithulgala
    "Wilpattu": (8.350, 80.070),        # Hunuwilagama entrance
    "Kirinda": (6.215, 81.333),         # Kirinda near Yala (OSM's first "Kirinda" is a different village)
    "Yala": (6.300, 81.380),            # hotels for Yala are along the Tissa - Palatupana road
}
PLACE_RANK = {"city": 0, "town": 1, "suburb": 2, "village": 3, "locality": 4, "quarter": 5,
              "neighbourhood": 6, "hamlet": 7}


def km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km (haversine). Works on numbers or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def _key(name):
    return re.sub(r"[^a-z]", "", str(name).lower())


class TownFinder:
    """Finds a point for a town name: our 15 data towns, then OpenStreetMap places, then the manual list."""

    def __init__(self, places: pd.DataFrame, hotels: pd.DataFrame):
        self.points = {}
        if places is not None and len(places):
            p = places.copy()
            p["rank"] = p["place"].map(PLACE_RANK).fillna(9)
            p = p.sort_values("rank")
            for col in ("name", "name_local"):
                if col in p:
                    for name, lat, lon in zip(p[col], p["latitude"], p["longitude"]):
                        k = _key(name)
                        if k and k not in self.points:
                            self.points[k] = (float(lat), float(lon), "OpenStreetMap")
        for name, (lat, lon) in MANUAL_POINTS.items():
            self.points[_key(name)] = (lat, lon, "typed by hand (approximate)")
        # our data towns: the middle of their hotels is used only if OpenStreetMap does not have the town
        self.data_towns = sorted(hotels["destination_cluster"].dropna().unique()) if len(hotels) else []
        for town, g in hotels.groupby("destination_cluster"):
            self.points.setdefault(_key(town), (float(g.latitude.median()), float(g.longitude.median()),
                                                "middle of our hotels"))

    def data_town(self, name):
        k = _key(name)
        for t in self.data_towns:
            if _key(t) == k:
                return t
        return None

    def point(self, name):
        return self.points.get(_key(name))


def nearest_hospitals(hotels: pd.DataFrame, hospitals: pd.DataFrame):
    """For every hotel: the nearest hospital's name and distance in km (straight line)."""
    if hospitals is None or not len(hospitals) or not len(hotels):
        return pd.DataFrame({"nearest_hospital": None, "nearest_hospital_km": np.nan}, index=hotels.index)
    hl = hospitals[["latitude", "longitude"]].to_numpy(float)
    names = hospitals["name"].fillna("Hospital").to_numpy()
    out_name, out_km = [], []
    for lat, lon in zip(hotels["latitude"].to_numpy(float), hotels["longitude"].to_numpy(float)):
        if np.isnan(lat) or np.isnan(lon):
            out_name.append(None)
            out_km.append(np.nan)
            continue
        d = km(lat, lon, hl[:, 0], hl[:, 1])
        i = int(np.argmin(d))
        out_name.append(names[i])
        out_km.append(round(float(d[i]), 2))
    return pd.DataFrame({"nearest_hospital": out_name, "nearest_hospital_km": out_km}, index=hotels.index)
