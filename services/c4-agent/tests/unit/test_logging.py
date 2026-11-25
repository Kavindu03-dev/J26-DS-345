import json

import pytest

from c4.common.ids import new_request_id, new_run_id
from c4.common.logging import JsonlLogger, scrub, scrub_text


@pytest.mark.parametrize(
    "email", ["kasun@example.com", "k.perera+trip@mail.example.co.uk", "a_b-c@x-y.lk"]
)
def test_email_scrubbed(email):
    assert scrub_text(f"write to {email}, please") == "write to [EMAIL], please"


@pytest.mark.parametrize(
    "phone",
    [
        "0771234567",
        "077 123 4567",
        "077-1234567",
        "077-123-4567",
        "0771 234 567",
        "011 2 345 678",
        "0112345678",
        "(077) 123 4567",
        "+94771234567",
        "+94 77 123 4567",
        "0094 77 123 4567",
        "94771234567",
        "+44 20 7946 0958",
        "+1 (202) 555-0143",
    ],
)
def test_phone_scrubbed(phone):
    assert scrub_text(f"call {phone}.") == "call [PHONE]."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My name is Kasun Perera and I want tea", "My name is [NAME] and I want tea"),
        ("my name is kasun, hello", "my name is [NAME], hello"),
        ("I'm Kasun Perera. Find a hotel", "I'm [NAME]. Find a hotel"),
        ("I am Nimali from Galle", "I am [NAME] from Galle"),
        ("Call me Nimali", "Call me [NAME]"),
        ("Book for Mr. Perera today", "Book for Mr. [NAME] today"),
        ("under the name of Silva", "under the name of [NAME]"),
    ],
)
def test_introduced_name_scrubbed(text, expected):
    assert scrub_text(text) == expected


def test_known_name_scrubbed_anywhere():
    text = "The room for kasun perera and Kasun is ready"
    out = scrub_text(text, known_names=["Kasun", "Kasun Perera"])
    assert out == "The room for [NAME] and [NAME] is ready"


@pytest.mark.parametrize(
    "text",
    [
        "Budget LKR 15000-30000 for 3 nights from 2026-12-20",
        "dates 08-12-2026 10:30 to 2026-12-23T10:00:00Z",
        "I'm looking for a hotel in Nuwara Eliya",
        "I am planning 3 days in Kandy",
        "hotel NE-001 costs 0.95 of 1234567890",
        "Temple of the Tooth, Kandy 7.2936 80.6413",
    ],
)
def test_useful_text_kept(text):
    assert scrub_text(text) == text


def test_ids_survive_scrubbing():
    for _ in range(500):
        for value in (new_request_id(), new_run_id()):
            assert scrub_text(value) == value


def test_pii_keys_scrubbed_but_tool_name_kept():
    record = {
        "name": "recommend_hotels",
        "guest_name": "Kasun",
        "Phone": 94771234567,
        "email": None,
        "calls": [{"name": "plan_itinerary", "arguments": {"note": "mail a@b.lk"}}],
    }
    before = json.dumps(record)
    assert scrub(record) == {
        "name": "recommend_hotels",
        "guest_name": "[NAME]",
        "Phone": "[PHONE]",
        "email": None,
        "calls": [{"name": "plan_itinerary", "arguments": {"note": "mail [EMAIL]"}}],
    }
    assert json.dumps(record) == before


def test_sample_log_line_has_no_pii(tmp_path):
    path = tmp_path / "logs" / "agent.jsonl"
    logger = JsonlLogger(path)
    request_id = new_request_id()
    logger.log(
        "chat",
        known_names=["Nimali"],
        request_id=request_id,
        message="My name is Kasun Perera, kasun@example.com, 077 123 4567. Nimali joins.",
        tool_calls=[{"name": "recommend_hotels", "arguments": {"budget_max_lkr": 30000}}],
        latency_ms=12.5,
        error=ValueError("failed for kasun@example.com"),
    )
    logger.log("chat", message="සිංහල text")

    raw = path.read_text(encoding="utf-8")
    for secret in ("Kasun", "Perera", "Nimali", "kasun@example.com", "077 123 4567"):
        assert secret not in raw
    first, second = [json.loads(line) for line in raw.splitlines()]
    assert first["event"] == "chat"
    assert first["request_id"] == request_id
    assert first["latency_ms"] == 12.5
    assert first["tool_calls"][0] == {
        "name": "recommend_hotels",
        "arguments": {"budget_max_lkr": 30000},
    }
    assert first["timestamp"]
    assert second["message"] == "සිංහල text"
