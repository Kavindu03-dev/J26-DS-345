"""Settings for the C4 agent service.

Every value can be overridden with an environment variable (see .env.example
at the repo root). Defaults are chosen so that the service works out of the
box on a developer laptop with the mock services from src/mocks running.
"""

from __future__ import annotations

import os
from pathlib import Path

# services/c4-agent/
SERVICE_ROOT = Path(__file__).resolve().parents[1]
# repo root (J26-DS-345/)
REPO_ROOT = SERVICE_ROOT.parents[1]

# Shared API contracts agreed by the four components.
CONTRACTS_DIR = Path(os.getenv("CONTRACTS_DIR", REPO_ROOT / "contracts"))

# Component 1 publishes its processed hotel data in the repo. C4 reads it
# (read-only) for the hotel directory and the mock services, so no data is
# copied into this folder.
C1_DATA_DIR = Path(
    os.getenv("C1_DATA_DIR", REPO_ROOT / "services" / "c1-review-engine" / "data")
)
HOTELS_SEED_FILE = C1_DATA_DIR / "hotels_seed.csv"
HOTEL_PROFILES_FILE = C1_DATA_DIR / "hotel_aspect_profiles.json"
DESTINATIONS_FILE = C1_DATA_DIR / "destinations_seed.csv"

# Where C4 reaches the other components. The defaults point at the mock
# services (ports 9001-9003) so they never clash with the real services
# (ports 8001-8003). docker-compose overrides these with the real URLs.
C1_URL = os.getenv("C1_URL", "http://localhost:9001")
C2_URL = os.getenv("C2_URL", "http://localhost:9002")
C3_URL = os.getenv("C3_URL", "http://localhost:9003")

# Per-tool HTTP timeouts in seconds. C3 runs an optimisation pipeline, so it
# gets more time than the database-backed C1 and C2 lookups.
C1_TIMEOUT_S = float(os.getenv("C1_TIMEOUT_S", "5"))
C2_TIMEOUT_S = float(os.getenv("C2_TIMEOUT_S", "10"))
C3_TIMEOUT_S = float(os.getenv("C3_TIMEOUT_S", "60"))

MOCK_PORTS = {"c1": 9001, "c2": 9002, "c3": 9003}
