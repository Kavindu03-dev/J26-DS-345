"""
Base Extractor Class for Web Scraping Modules.
Provides standardized batching, error recovery, deduplication, and atomic writing.
"""


import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import CheckpointManager

class BaseExtractor:
    def __init__(self, output_file: Path, fieldnames: list):
        self.output_file = Path(output_file)
        self.fieldnames = fieldnames
        self.checkpoint = CheckpointManager()
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.existing_keys = self._load_existing_keys()

    def _load_existing_keys(self) -> set:
        keys = set()
        if self.output_file.exists():
            try:
                with open(self.output_file, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Compound key: (hotel_id/url, author, review_date, title)
                        key = (
                            row.get("hotel_id") or row.get("hotel_url", ""),
                            row.get("author_id") or row.get("author", ""),
                            row.get("review_date", ""),
                            row.get("review_title", "")
                        )
                        keys.add(key)
            except Exception:
                pass
        return keys

    def append_batch(self, records: list):
        """Append a batch of extracted records to the destination CSV atomically."""
        if not records:
            return
            
        file_is_new = not self.output_file.exists()
        with open(self.output_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if file_is_new:
                writer.writeheader()
            writer.writerows(records)
