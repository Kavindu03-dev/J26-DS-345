"""Remove training examples that overlap with the BFCL test set.

Rule (a common contamination check, as used for large-model benchmarks):
an example is removed if its user query shares any 13-word sequence with a
BFCL question. Queries shorter than 13 words are compared as a whole, so a
short query is removed only on an exact (normalised) match.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

N = 13
REFERENCE_FILE = Path(__file__).with_name("bfcl_v4_reference.json")


def normalise_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def ngrams(words: list[str], n: int = N) -> set[tuple[str, ...]]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class Decontaminator:
    def __init__(self, reference_file: Path = REFERENCE_FILE):
        with open(reference_file, encoding="utf-8") as f:
            ref = json.load(f)
        self.reference_count = ref["count"]
        self.gorilla_commit = ref.get("gorilla_commit", "unknown")
        self._grams: set[tuple[str, ...]] = set()
        self._short: set[tuple[str, ...]] = set()
        for item in ref["items"]:
            words = normalise_words(item["question"])
            if len(words) < N:
                self._short.add(tuple(words))
            # Index every n-gram length a short query could match exactly.
            self._grams |= ngrams(words)

    def is_contaminated(self, query: str) -> bool:
        words = normalise_words(query)
        if not words:
            return False
        if len(words) < N:
            return tuple(words) in self._short
        return not self._grams.isdisjoint(ngrams(words))
