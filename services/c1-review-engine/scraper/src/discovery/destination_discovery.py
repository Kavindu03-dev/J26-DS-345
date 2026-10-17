"""
Destination & Attraction Discovery Module.
Exports Sri Lanka's Top 50 Tourism Landmarks & Cultural Sites for Recommender,
Itinerary Planning, and RAG Assistant components.
"""

import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DESTINATIONS_SEED_FILE, DESTINATION_ENTITY_FIELDS
from src.discovery.geo_seeds import TOP_SRI_LANKA_DESTINATIONS

def initialize_destination_seeds():
    """
    Saves top Sri Lankan attractions into structured CSV format.
    """
    print("--- Initializing Top Sri Lankan Destination Seeds ---")
    destinations = []
    
    for dest in TOP_SRI_LANKA_DESTINATIONS:
        dest_record = {
            "destination_id": dest["id"],
            "destination_name": dest["name"],
            "destination_url": f"https://www.google.com/maps/search/{dest['search_query'].replace(' ', '+')}",
            "category": dest["category"],
            "latitude": dest["latitude"],
            "longitude": dest["longitude"],
            "city": dest["city"],
            "district": dest["district"],
            "province": dest["province"],
            "description": f"Key Sri Lankan tourism destination: {dest['name']} in {dest['city']}, {dest['district']} District.",
            "platform_overall_rating": 4.7, # High default landmark rating
            "platform_total_reviews_count": 2500
        }
        destinations.append(dest_record)
        
    with open(DESTINATIONS_SEED_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DESTINATION_ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(destinations)
        
    print(f"[OK] Seeded {len(destinations)} destination landmarks to {DESTINATIONS_SEED_FILE}")
    return destinations

if __name__ == "__main__":
    initialize_destination_seeds()
