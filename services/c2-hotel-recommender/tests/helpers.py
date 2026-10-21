"""Test helpers: a tiny made-up dataset (same columns as the real files) and requests in Output 1 format."""
import pandas as pd

DATES = ["2026-12-09", "2027-01-23", "2027-03-09"]
GROUPS = [(1, 0), (2, 0), (2, 1), (2, 2)]

#            id         name             town     lat     lon    rating pool wifi ac beach km  brk  lift price
HOTELS = [("k-cheap", "Cheap Kandy Inn", "Kandy", 7.2930, 80.6350, 8.0, 0, 1, 1, 0, 0.5, 1, 0, 6000),
          ("k-pool", "Kandy Pool Hotel", "Kandy", 7.2950, 80.6360, 8.8, 1, 1, 1, 0, 1.0, 1, 1, 9000),
          ("k-closed", "Kandy Closed Pool", "Kandy", 7.2960, 80.6370, 8.6, 1, 1, 1, 0, 2.5, 1, 1, 8500),
          ("k-far", "Hill Top Far Away", "Kandy", 7.4000, 80.7500, 9.2, 1, 1, 1, 0, 14.0, 0, 0, 9500),
          ("k-lux", "Kandy Luxury", "Kandy", 7.2940, 80.6340, 9.5, 1, 1, 1, 0, 0.8, 1, 1, 45000),
          ("k-nowifi", "No Wifi Lodge", "Kandy", 7.2935, 80.6355, 7.5, 0, 0, 0, 0, 0.3, 0, 0, 4000),
          ("g-beach", "Galle Beach Hotel", "Galle", 6.0300, 80.2150, 8.4, 1, 1, 1, 1, 0.5, 1, 1, 30000),
          ("g-town", "Galle Town Rooms", "Galle", 6.0330, 80.2160, 7.9, 0, 1, 1, 0, 0.2, 0, 0, 7000)]


def make_files(root):
    clean, models = root / "data" / "output_clean", root / "data" / "models"
    clean.mkdir(parents=True)
    models.mkdir(parents=True)
    hotels, rows, preds, wtb = [], [], [], []
    for (hid, name, town, lat, lon, rating, pool, wifi, ac, beach, km_c, brk, lift, price) in HOTELS:
        hotels.append({"booking_hotel_id": hid, "hotel_name": name, "destination_cluster": town, "latitude": lat,
                       "longitude": lon, "page_rating": rating, "subscore_staff": rating, "subscore_comfort": rating,
                       "subscore_cleanliness": rating, "subscore_location": rating, "subscore_free_wifi": rating,
                       "has_pool": pool, "n_pools": pool, "has_free_wifi": wifi, "rooms_air_conditioning": ac,
                       "is_beachfront": beach, "km_from_town_centre": km_c, "offers_breakfast_rate": brk,
                       "has_family_rooms": 1, "details_complete": 1, "has_bar": 0, "has_restaurant": 1})
        for d in DATES:
            for a, c in GROUPS:
                rid = len(rows)
                p = price * (1 + 0.3 * c)
                rows.append({"row_id": rid, "source_ota": "booking", "is_extra_date": 0, "is_sold_out": 0,
                             "booking_hotel_id": hid, "hotel_name": name, "destination_clean": town,
                             "checkin_date": d, "num_adults": a, "num_children": c, "price_incl_tax_lkr_night": p,
                             "meal_plan": "breakfast" if brk else "none", "free_cancellation": 1,
                             "room_tier_name": "standard",
                             # columns a price model can use (tests/test_price_explanation.py trains a tiny one)
                             "hotel_group_id": hid, "split": "train" if hid == "k-far" else "test",
                             "has_pool": pool, "n_pools": pool, "has_bar": 0, "is_beachfront": beach,
                             "breakfast_included": brk, "meal_level": brk, "stars": 5 if price > 20000 else 3,
                             "lead_time_days": 60, "checkin_month": int(d[5:7]), "checkin_weekday": 2,
                             "stay_has_weekend_night": 0, "is_event_date": 0})
                fair = p * (0.6 if hid == "k-lux" else 1.0)          # the luxury hotel is overpriced
                preds.append({"row_id": rid, "fair_price_lkr": fair, "price_vs_fair_pct": 100 * (p / fair - 1)})
                guests = f"{a} adult{'s' if a > 1 else ''}" + (f" + {c} child{'ren' if c > 1 else ''}" if c else "")
                cheaper = hid == "k-pool"
                wtb.append({"hotel": name, "town": town, "check-in": d, "guests": guests, "Booking.com": round(p),
                            "Trip.com": round(p * 0.9) if cheaper else None, "Agoda (from)": None, "notes": "",
                            "advice": "Book on Trip.com - about Rs 900 per night cheaper." if cheaper
                            else "Book on Booking.com.",
                            "recommended site": "Trip.com" if cheaper else "Booking.com"})
    pd.DataFrame(hotels).to_csv(clean / "hotels_booking.csv", index=False)
    pd.DataFrame(rows).to_csv(clean / "model_table.csv", index=False)
    pd.DataFrame(preds).to_csv(models / "booking_predictions.csv", index=False)
    pd.DataFrame(wtb).to_csv(clean / "where_to_book.csv", index=False)
    pd.DataFrame([{"source_ota": "tripcom", "hotel_id_on_source": "123", "booking_hotel_id": "k-pool",
                   "status": "matched", "score": 1.0}]).to_csv(clean / "hotel_matches.csv", index=False)
    extra = root / "extra"
    extra.mkdir()
    pd.DataFrame([{"osm_id": "n1", "name": "Kandy General Hospital", "latitude": 7.2950, "longitude": 80.6340,
                   "emergency": "yes"},
                  {"osm_id": "n2", "name": "Galle Hospital", "latitude": 6.0600, "longitude": 80.2500,
                   "emergency": "yes"}]).to_csv(extra / "sl_hospitals_osm.csv", index=False)
    pd.DataFrame([{"osm_id": "p1", "name": "Kandy", "name_local": "", "place": "city", "latitude": 7.2931,
                   "longitude": 80.6350},
                  {"osm_id": "p2", "name": "Peradeniya", "name_local": "", "place": "suburb", "latitude": 7.2642,
                   "longitude": 80.5931}]).to_csv(extra / "sl_places_osm.csv", index=False)
    pd.DataFrame([{"source_ota": "booking", "hotel_id_on_source": h[0], "hotel_name": h[1],
                   "url": f"https://www.booking.com/hotel/lk/{h[0]}.html"} for h in HOTELS] +
                 [{"source_ota": "tripcom", "hotel_id_on_source": "123", "hotel_name": "Kandy Pool Hotel",
                   "url": "https://www.trip.com/hotels/detail/?hotelId=123"}]).to_csv(extra / "hotel_urls.csv",
                                                                                      index=False)
    return root / "data", extra



def make_request(town="Kandy", ci="2027-01-20", nights=2, adults=2, children=0, bmax=None, must=(), wishes=None,
            health=(), not_needed=(), total=None):
    must = list(must)
    if bmax is not None and "budget" not in must:
        must.insert(0, "budget")
    return {"trip": {"stay_in": [town], "want_to_visit": [], "check_in": ci, "check_out": None, "nights": nights,
                     "dates_fixed": True, "adults": adults, "children": children, "travel_style": None,
                     "budget_total_lkr": total},
            "hotel": {"budget_max_lkr": bmax, "must": must, "wishes": wishes or {}},
            "health": list(health), "interests": [], "not_needed": list(not_needed)}
