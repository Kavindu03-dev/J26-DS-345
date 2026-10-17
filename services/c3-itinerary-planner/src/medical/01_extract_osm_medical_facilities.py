from pathlib import Path
import os

import osmium
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OSM_FILE = Path(
    os.getenv(
        "C3_OSM_PBF",
        r"D:\Component3_OSRM\sri-lanka-261001.osm.pbf",
    )
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "medical_facilities.csv"
)


# ============================================================
# OSM MEDICAL FACILITY HANDLER
# ============================================================

class MedicalFacilityHandler(osmium.SimpleHandler):

    def __init__(self):
        super().__init__()
        self.facilities = []

    @staticmethod
    def is_medical_facility(tags):
        amenity = tags.get("amenity")
        return amenity in {
            "hospital",
            "clinic",
        }

    @staticmethod
    def extract_tags(tags):
        return {
            "name": tags.get("name"),
            "name_en": tags.get("name:en"),
            "amenity": tags.get("amenity"),
            "healthcare": tags.get("healthcare"),
            "emergency": tags.get("emergency"),
            "wheelchair": tags.get("wheelchair"),
            "operator": tags.get("operator"),
            "phone": (
                tags.get("phone")
                or tags.get("contact:phone")
            ),
            "website": (
                tags.get("website")
                or tags.get("contact:website")
            ),
            "opening_hours": tags.get("opening_hours"),
            "addr_housenumber": tags.get("addr:housenumber"),
            "addr_street": tags.get("addr:street"),
            "addr_city": tags.get("addr:city"),
            "addr_district": tags.get("addr:district"),
        }

    def node(self, node):
        if not self.is_medical_facility(node.tags):
            return

        if not node.location.valid():
            return

        record = {
            "facility_id": f"node/{node.id}",
            "osm_id": f"node/{node.id}",
            "osm_type": "node",
            **self.extract_tags(node.tags),
            "latitude": node.location.lat,
            "longitude": node.location.lon,
        }

        self.facilities.append(record)

    def way(self, way):
        if not self.is_medical_facility(way.tags):
            return

        coordinates = []

        for node in way.nodes:
            if node.location.valid():
                coordinates.append(
                    (
                        node.location.lat,
                        node.location.lon,
                    )
                )

        if not coordinates:
            return

        latitude = (
            sum(x[0] for x in coordinates)
            / len(coordinates)
        )

        longitude = (
            sum(x[1] for x in coordinates)
            / len(coordinates)
        )

        record = {
            "facility_id": f"way/{way.id}",
            "osm_id": f"way/{way.id}",
            "osm_type": "way",
            **self.extract_tags(way.tags),
            "latitude": latitude,
            "longitude": longitude,
        }

        self.facilities.append(record)


# ============================================================
# EXTRACTION
# ============================================================

print("=" * 70)
print("COMPONENT 3 - OSM MEDICAL FACILITY EXTRACTION")
print("=" * 70)

print(f"\nOSM source:\n{OSM_FILE}")

if not OSM_FILE.exists():
    raise FileNotFoundError(
        f"OSM PBF file not found:\n{OSM_FILE}\n\n"
        "Set the C3_OSM_PBF environment variable "
        "if the file is stored elsewhere."
    )

handler = MedicalFacilityHandler()

print("\nReading Sri Lanka OSM nodes and ways...")

handler.apply_file(
    str(OSM_FILE),
    locations=True,
)

facilities = pd.DataFrame(
    handler.facilities
)

if facilities.empty:
    raise RuntimeError(
        "No hospitals or clinics were extracted."
    )


# ============================================================
# COORDINATE VALIDATION
# ============================================================

valid_coordinates = (
    facilities["latitude"].between(
        5.5,
        10.2,
    )
    &
    facilities["longitude"].between(
        79.0,
        82.2,
    )
)

print(
    "\nValid Sri Lanka coordinates:",
    int(valid_coordinates.sum()),
)

print(
    "Invalid coordinates:",
    int((~valid_coordinates).sum()),
)

facilities = (
    facilities[
        valid_coordinates
    ]
    .copy()
)

facilities = (
    facilities
    .drop_duplicates(
        subset=["osm_id"]
    )
    .copy()
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MEDICAL FACILITY EXTRACTION RESULTS")
print("=" * 70)

print(
    "Total facilities:",
    len(facilities),
)

print("\nFacility types:")

print(
    facilities["amenity"]
    .value_counts(
        dropna=False
    )
)

print(
    "\nFacilities with names:",
    int(
        facilities["name"]
        .notna()
        .sum()
    ),
)

print(
    "Facilities with emergency tag:",
    int(
        facilities["emergency"]
        .notna()
        .sum()
    ),
)

print(
    "Facilities with wheelchair tag:",
    int(
        facilities["wheelchair"]
        .notna()
        .sum()
    ),
)

print(
    "Facilities with opening hours:",
    int(
        facilities["opening_hours"]
        .notna()
        .sum()
    ),
)

print("\nSample:")

sample_columns = [
    "facility_id",
    "name",
    "amenity",
    "emergency",
    "wheelchair",
    "addr_city",
    "latitude",
    "longitude",
]

print(
    facilities[
        sample_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

facilities.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\nOutput saved:")
print(OUTPUT_FILE)

print("\nStatus: SUCCESS")