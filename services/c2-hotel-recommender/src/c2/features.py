"""
Gap 2 + Gap 5: is each hotel feature true, false or not sure - and is it true NOW?

Step 1. What the listing says (Booking.com facts and facility list).      -> yes / no / unknown
Step 2. What trusted reviews say (Component 1: facts and aspect scores).  -> confirmed / contradicted / silent
Step 3. Put them together:
        reviews contradict      -> "no"        ("listed, but reviews say the pool was closed for repairs")
        reviews confirm         -> "yes"       ("listed, and reviews agree")
        reviews silent          -> the listing ("yes: listed, not checked by reviews" / "no" / "not sure")

near_hospital is measured, not claimed: distance to the nearest OpenStreetMap hospital.
"""
import math
import re

from . import settings

FEATURE_WORDS = ["quiet", "beach", "pool", "wifi", "breakfast", "parking", "view", "ac", "family_room", "spa",
                 "nature", "near_hospital", "low_steps", "wheelchair_access"]
NICE_NAME = {"quiet": "quiet", "beach": "on or near the beach", "pool": "pool", "wifi": "WiFi",
             "breakfast": "breakfast", "parking": "parking", "view": "a view", "ac": "air conditioning",
             "family_room": "family rooms", "spa": "spa", "nature": "nature / garden",
             "near_hospital": "near a hospital", "low_steps": "few steps / lift",
             "wheelchair_access": "wheelchair access"}


def _v(h, col):
    try:
        x = float(h.get(col))
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _any(h, *cols):
    return any((_v(h, c) or 0) >= 1 for c in cols)


def listing_evidence(h):
    """What the Booking.com listing says about each feature: {feature: (yes/no/unknown, reason)}."""
    complete = (_v(h, "details_complete") or 0) == 1      # full facility list collected -> "not listed" means no
    absent = ("no", "not listed") if complete else ("unknown", "facility list not complete")
    out = {}

    def simple(feature, cols, label):
        out[feature] = ("yes", f"{label} listed") if _any(h, *cols) else absent

    simple("pool", ["has_pool", "n_pools"], "pool")
    simple("beach", ["is_beachfront", "has_private_beach", "fac_beachfront", "fac_private_beach_area"], "beachfront")
    simple("spa", ["has_spa", "fac_spa_and_wellness_center", "fac_spa_facilities"], "spa")
    simple("ac", ["rooms_air_conditioning", "fac_air_conditioning", "fac_room_air_conditioning"], "air conditioning")
    simple("family_room", ["has_family_rooms"], "family rooms")
    simple("view", ["rooms_sea_view", "fac_sea_view", "fac_room_sea_view", "fac_mountain_view",
                    "fac_room_mountain_view", "fac_room_lake_view", "fac_room_river_view", "fac_landmark_view",
                    "fac_room_landmark_view", "fac_view", "fac_room_view"], "rooms with a view")

    if _any(h, "fac_internet_no_internet_access_available"):
        out["wifi"] = ("no", "listing says no internet")
    else:
        simple("wifi", ["has_free_wifi", "fac_free_wifi"], "free WiFi")
    if _any(h, "fac_parking_no_parking_available"):
        out["parking"] = ("no", "listing says no parking")
    else:
        simple("parking", ["has_free_parking", "fac_free_parking", "fac_parking", "fac_private_parking"], "parking")

    bf = _v(h, "offers_breakfast_rate")
    out["breakfast"] = (("yes", "breakfast rates offered") if bf == 1 else
                        ("no", "no breakfast rate offered") if bf == 0 else ("unknown", "breakfast not known"))

    town_km = _v(h, "km_from_town_centre")
    if _any(h, "rooms_soundproofing", "fac_soundproof_rooms", "fac_soundproofing", "fac_room_soundproofing"):
        out["quiet"] = ("yes", "soundproof rooms listed")
    elif town_km is not None and town_km >= settings.QUIET_KM_FROM_CENTRE:
        out["quiet"] = ("yes", f"{town_km:.1f} km from the town centre")
    else:
        out["quiet"] = ("unknown", "no soundproofing listed, close to the town centre")

    if _any(h, "has_garden", "fac_garden", "fac_garden_view", "fac_room_garden_view", "fac_mountain_view",
            "fac_room_mountain_view"):
        out["nature"] = ("yes", "garden or mountain view listed")
    elif town_km is not None and town_km >= 3:
        out["nature"] = ("yes", f"outside the town ({town_km:.1f} km from the centre)")
    else:
        out["nature"] = ("unknown", "no garden listed")

    if _any(h, "fac_elevator", "fac_upper_floors_accessible_by_elevator",
            "fac_room_upper_floors_accessible_by_elevator", "fac_entire_unit_located_on_ground_floor",
            "fac_room_entire_unit_located_on_ground_floor"):
        out["low_steps"] = ("yes", "lift or ground-floor rooms listed")
    elif _any(h, "fac_upper_floors_accessible_by_stairs_only", "fac_room_upper_floors_accessible_by_stairs_only"):
        out["low_steps"] = ("no", "upper floors by stairs only")
    else:
        out["low_steps"] = ("unknown", "no lift or ground-floor information")

    if _any(h, "fac_wheelchair_accessible", "fac_entire_unit_wheelchair_accessible",
            "fac_room_entire_unit_wheelchair_accessible", "has_disabled_facilities",
            "fac_facilities_for_disabled_guests"):
        out["wheelchair_access"] = ("yes", "wheelchair access listed")
    else:
        out["wheelchair_access"] = absent if complete else ("unknown", "facility list not complete")

    hosp_km = _v(h, "nearest_hospital_km")
    if hosp_km is None:
        out["near_hospital"] = ("unknown", "hospital distance not known")
    elif hosp_km <= settings.NEAR_HOSPITAL_KM:
        out["near_hospital"] = ("yes", f"{hosp_km:.1f} km to {h.get('nearest_hospital')}")
    else:
        out["near_hospital"] = ("no", f"nearest hospital {hosp_km:.1f} km away")
    return out


# ---- what trusted reviews say (Component 1) ----
FACT_TOPIC = {
    "pool": r"\bpool\b", "wifi": r"wi-?fi|internet", "view": r"\bview\b", "breakfast": r"breakfast",
    "ac": r"air ?con|\ba/c\b", "low_steps": r"\blift\b|elevator|stairs|\bsteps\b",
    "wheelchair_access": r"wheelchair|\bramp", "beach": r"\bbeach\b", "parking": r"parking", "spa": r"\bspa\b",
    "family_room": r"family room", "nature": r"garden|nature|birds",
    "quiet": r"nois|quiet|peaceful|\bloud\b|traffic",
}
BAD = re.compile(r"closed|not work|n't work|did not work|broken|out of order|under repair|blocked|no longer|"
                 r"not available|\bweak\b|\bpoor\b|\bslow\b|dirty|nois|\bloud\b|traffic|many steps|steep|no lift|"
                 r"not suitable|had to use the stairs", re.I)
GOOD = re.compile(r"clean|great|good|fast|strong|lovely|beautiful|amazing|excellent|well kept|worked well|quiet|"
                  r"peaceful|fresh|easy access|ground floor", re.I)


def review_evidence(c1):
    """{feature: (confirmed/contradicted, reason)} from Component 1's facts and aspect scores."""
    out = {}
    if not c1:
        return out
    for fact in c1.get("verified_aspect_facts") or []:
        for feature, topic in FACT_TOPIC.items():
            if re.search(topic, fact, re.I):
                if BAD.search(fact):
                    out[feature] = ("contradicted", fact)
                elif GOOD.search(fact) and feature not in out:
                    out[feature] = ("confirmed", fact)
    wifi = (c1.get("aspect_scores") or {}).get("wifi")
    if wifi is not None and "wifi" not in out:
        if wifi < 2.5:
            out["wifi"] = ("contradicted", f"guests rate the WiFi {wifi:.1f}/5")
        elif wifi >= 4.0:
            out["wifi"] = ("confirmed", f"guests rate the WiFi {wifi:.1f}/5")
    return out


def check_features(h, c1):
    """Final answer for every feature: {feature: {"status": yes/no/not sure, "listing": .., "reviews": .., "why": ..}}."""
    listing, reviews = listing_evidence(h), review_evidence(c1)
    result = {}
    for f in FEATURE_WORDS:
        l_status, l_reason = listing.get(f, ("unknown", ""))
        r_status, r_reason = reviews.get(f, (None, None))
        if f == "near_hospital":                       # measured, reviews do not change it
            status = {"yes": "yes", "no": "no"}.get(l_status, "not sure")
            why = l_reason
        elif r_status == "contradicted":
            status = "no"
            why = (f"listed, but reviews say: {r_reason}" if l_status == "yes" else f"reviews say: {r_reason}")
        elif r_status == "confirmed":
            status = "yes"
            why = f"{l_reason}, and reviews agree" if l_status == "yes" else f"reviews say: {r_reason}"
        else:
            status = {"yes": "yes", "no": "no"}.get(l_status, "not sure")
            why = l_reason + (", not checked by reviews" if l_status == "yes" else "")
        result[f] = {"status": status, "listing": l_status, "reviews": r_status or "no review information",
                     "why": why}
    return result
