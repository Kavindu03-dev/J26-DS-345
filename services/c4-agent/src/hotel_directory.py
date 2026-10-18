"""Hotel directory: turns what a tourist types into hotel IDs.

Tourists say "How clean is Orchard House Galle?", not "HTL_GAL_ORCHARDH_2FE6".
The C1 contract only accepts hotel IDs, so C4 resolves names itself using the
hotel seed file published by Component 1. This keeps the shared contract
unchanged.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .data_access import Hotel, load_hotels

_STOPWORDS = {"hotel", "the", "resort", "and", "&", "by", "at", "of", "sri", "lanka"}


def normalise(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in normalise(text).split() if t not in _STOPWORDS}


@dataclass(frozen=True)
class HotelMatch:
    hotel: Hotel
    score: float  # 0.0-1.0, higher is a closer match


class HotelDirectory:
    def __init__(self, hotels: dict[str, Hotel] | None = None):
        self._hotels = hotels if hotels is not None else load_hotels()
        self._by_norm_name: dict[str, list[Hotel]] = {}
        for h in self._hotels.values():
            self._by_norm_name.setdefault(normalise(h.name), []).append(h)

    def __len__(self) -> int:
        return len(self._hotels)

    def get(self, hotel_id: str) -> Hotel | None:
        return self._hotels.get(hotel_id)

    def find(self, name: str, limit: int = 5, min_score: float = 0.55) -> list[HotelMatch]:
        """Best matches for a hotel name, best first.

        An exact (normalised) name match scores 1.0. Otherwise the score mixes
        character similarity with how many of the query's words appear in the
        hotel name, so "Orchard House" still finds "Orchard House Galle".
        """
        query = normalise(name)
        if not query:
            return []
        exact = self._by_norm_name.get(query)
        if exact:
            return [HotelMatch(h, 1.0) for h in exact[:limit]]

        q_tokens = _tokens(name)
        matches = []
        for h in self._hotels.values():
            h_norm = normalise(h.name)
            char_sim = difflib.SequenceMatcher(None, query, h_norm).ratio()
            h_tokens = _tokens(h.name)
            overlap = len(q_tokens & h_tokens) / len(q_tokens) if q_tokens else 0.0
            score = 0.5 * char_sim + 0.5 * overlap
            if score >= min_score:
                matches.append(HotelMatch(h, round(score, 3)))
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]

    def in_destination(self, destination: str) -> list[Hotel]:
        """Hotels whose city, district or province matches the destination."""
        d = normalise(destination)
        if not d:
            return []
        for field in ("city", "district", "province"):
            found = [h for h in self._hotels.values() if normalise(getattr(h, field)) == d]
            if found:
                return found
        # Partial match, e.g. "Galle Fort area" -> city "Galle Fort".
        partial = []
        for h in self._hotels.values():
            city = normalise(h.city)
            if city and (d in city or city in d):
                partial.append(h)
        return partial

    def known_destinations(self) -> list[str]:
        return sorted({h.city for h in self._hotels.values()} | {h.district for h in self._hotels.values()})
