"""JSONL logging with PII scrubbing (ground rule 10).

Emails, phone numbers, values under personal-data keys, and names that are introduced as
such ("my name is ...", "I'm Kasun", "Mr. Perera") are scrubbed. Regexes cannot recognise
an arbitrary bare name, so callers that know a user's name must pass it in `known_names`.
"""

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EMAIL, PHONE, NAME = "[EMAIL]", "[PHONE]", "[NAME]"

# Plain "name" is deliberately absent: it holds tool and hotel names.
PII_KEYS = {
    **dict.fromkeys(("email", "email_address", "user_email"), EMAIL),
    **dict.fromkeys(("phone", "phone_number", "mobile", "telephone", "whatsapp"), PHONE),
    **dict.fromkeys(
        (
            "user_name",
            "username",
            "full_name",
            "first_name",
            "last_name",
            "surname",
            "guest_name",
            "customer_name",
            "contact_name",
            "traveller_name",
            "traveler_name",
        ),
        NAME,
    ),
}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# Sri Lankan formats are grouped explicitly so dates and LKR amounts are never matched.
_PHONE_RE = re.compile(
    r"""(?<![\w.+-])(?:
        (?:\+|00)94[\s-]?\(?\d{2}\)?[\s-]?\d{3}[\s-]?\d{4}   # +94 77 123 4567
      | 94\d{9}                                              # 94771234567
      | \(?0\d{2}\)?[\s-]?\d{3}[\s-]?\d{4}                   # 077-123 4567
      | 0\d{3}[\s-]\d{3}[\s-]\d{3}                           # 0771 234 567
      | 0\d{2}[\s-]\d[\s-]?\d{3}[\s-]?\d{3}                  # 011 2 345 678
      | \+\d{1,3}(?:[\s.-]?\(?\d{2,5}\)?){2,4}               # other international
    )(?!\w)""",
    re.VERBOSE,
)

_STOP = (
    r"and|i|we|my|our|but|from|with|for|to|the|a|an|is|are|am|in|at|on|of|"
    r"looking|planning|want|need|visiting|travelling|traveling|please"
)
_WORD = rf"(?!(?:{_STOP})\b)[^\W\d_][\w'’-]*"
_CAP = r"[A-Z][\w'’-]+"
_NAME_RES = [
    # Case-insensitive: the phrase itself says that a name follows.
    re.compile(
        rf"\b((?:my|his|her|their)\s+(?:full\s+)?name(?:\s+is|'s|:)\s+|"
        rf"under\s+the\s+name(?:\s+of)?\s+)({_WORD}(?:[ \t]+{_WORD}){{0,2}})",
        re.IGNORECASE,
    ),
    # Case-sensitive: only capitalised words, so "I'm looking for..." is kept.
    re.compile(rf"\b((?:[Ii]['’]m|[Ii] am|[Cc]all me)\s+)({_CAP}(?:[ \t]+{_CAP}){{0,2}})"),
    re.compile(rf"\b((?:Mrs|Mr|Ms|Miss|Dr|Prof)\.?\s+)({_CAP}(?:[ \t]+{_CAP}){{0,2}})"),
]


def scrub_text(text: str, known_names: Iterable[str] = ()) -> str:
    text = _EMAIL_RE.sub(EMAIL, text)
    text = _PHONE_RE.sub(PHONE, text)
    for name in sorted(filter(None, known_names), key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", NAME, text, flags=re.IGNORECASE)
    for pattern in _NAME_RES:
        text = pattern.sub(lambda m: m.group(1) + NAME, text)
    return text


def scrub(value: Any, known_names: Iterable[str] = ()) -> Any:
    """Return a scrubbed copy of any JSON-like value. Never mutates the input."""
    known_names = tuple(known_names)
    if isinstance(value, str):
        return scrub_text(value, known_names)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            placeholder = PII_KEYS.get(str(key).lower())
            out[scrub_text(str(key), known_names)] = (
                placeholder if placeholder and item is not None else scrub(item, known_names)
            )
        return out
    if isinstance(value, (list, tuple)):
        return [scrub(item, known_names) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub_text(str(value), known_names)


class JsonlLogger:
    """Append-only JSONL log. Every record is scrubbed before it reaches the disk."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def log(self, event: str, *, known_names: Iterable[str] = (), **fields: Any) -> dict[str, Any]:
        record = {
            **scrub(fields, known_names),
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
