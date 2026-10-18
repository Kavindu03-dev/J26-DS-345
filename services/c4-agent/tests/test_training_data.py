"""Tests for the Step 4 data pipeline, using a small file in xLAM's exact format
(tests/fixtures/xlam_sample.json: synthetic rows plus deliberately broken ones)."""

import json
import random

import pytest

from training.decontaminate import Decontaminator
from training.no_call import make_irrelevance, tool_keywords, keywords
from training.prepare_data import build, main, split
from training.xlam_convert import ConversionError, convert_row, map_type

FIXTURE = "tests/fixtures/xlam_sample.json"


@pytest.fixture(scope="module")
def rows():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def decon():
    return Decontaminator()


@pytest.mark.parametrize(
    "xlam_type, schema, optional",
    [
        ("str", {"type": "string"}, False),
        ("int, optional", {"type": "integer"}, True),
        ("float", {"type": "number"}, False),
        ("bool", {"type": "boolean"}, False),
        ("List[int]", {"type": "array", "items": {"type": "integer"}}, False),
        ("List[Dict[str, Any]]", {"type": "array", "items": {"type": "object"}}, False),
        ("Dict[str, Any], optional", {"type": "object"}, True),
        ("Optional[str]", {"type": "string"}, True),
        ("SomethingOdd", {"type": "string"}, False),
    ],
)
def test_type_mapping(xlam_type, schema, optional):
    assert map_type(xlam_type) == (schema, optional)


def test_convert_row_produces_chat_format(rows):
    ex = convert_row(rows[2])  # t3ma
    fn = ex["tools"][0]["function"]
    assert ex["tools"][0]["type"] == "function"
    assert fn["parameters"]["type"] == "object"
    assert fn["parameters"]["properties"]["time_period"]["type"] == "integer"
    # every t3ma parameter has a default, so none is required
    assert fn["parameters"]["required"] == []
    user, assistant = ex["messages"]
    assert user == {"role": "user", "content": rows[2]["query"]}
    call = assistant["tool_calls"][0]["function"]
    assert call == {"name": "t3ma", "arguments": {"symbol": "ETH/BTC", "interval": "1h", "time_period": 14}}


def test_parallel_calls_are_kept(rows):
    assert convert_row(rows[0])["kind"] == "parallel_tool_call"


def test_required_params_detected(rows):
    fn = convert_row(rows[3])["tools"][0]["function"]  # get_weather
    assert fn["parameters"]["required"] == ["city"]


@pytest.mark.parametrize("i", [7, 8, 9])
def test_malformed_rows_are_rejected_not_repaired(rows, i):
    with pytest.raises(ConversionError):
        convert_row(rows[i])


def test_bfcl_question_is_detected(decon, rows):
    assert decon.reference_count > 3000
    assert decon.is_contaminated(rows[11]["query"])
    assert not decon.is_contaminated("Find me three recipes that use chickpeas.")


def test_long_query_sharing_13_words_is_detected(decon):
    with open("training/bfcl_v4_reference.json", encoding="utf-8") as f:
        long_q = next(i["question"] for i in json.load(f)["items"] if len(i["question"].split()) > 25)
    words = long_q.split()
    assert decon.is_contaminated("Please help me. " + " ".join(words[3:17]) + " thanks")


def test_irrelevance_examples_have_no_overlap_with_tools(rows):
    pool = [convert_row(r) for r in rows if r["id"] in (1, 2, 3, 4, 5, 6)]
    out = make_irrelevance(pool, pool, 20, random.Random(0))
    assert out
    for ex in out:
        assert "tool_calls" not in ex["messages"][1]
        assert not keywords(ex["messages"][0]["content"]) & tool_keywords(ex["tools"])


def test_build_counts_and_no_call_share(rows, decon):
    examples, stats = build(rows, n_tool_call=8, no_call_share=0.15, seed=1, decon=decon)
    assert stats["dropped_malformed"] == 3
    assert stats["dropped_duplicate_query"] == 1
    assert stats["dropped_bfcl_overlap"] == 1
    assert stats["clean_tool_call_pool"] == 9
    no_call = [e for e in examples if e["kind"].startswith("no_call")]
    assert len(no_call) / len(examples) >= 0.15


def test_build_is_reproducible(rows, decon):
    a, _ = build(rows, 8, 0.15, 7, decon)
    b, _ = build(rows, 8, 0.15, 7, decon)
    assert a == b


def test_end_to_end_writes_files(tmp_path):
    stats = main(["--input", FIXTURE, "--out", str(tmp_path), "--n-tool-call", "8", "--val-share", "0.2"])
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    val = [json.loads(l) for l in (tmp_path / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) + len(val) == stats["final_examples"]
    for rec in train + val:
        assert set(rec) == {"id", "kind", "source_id", "tools", "messages"}
        assert isinstance(json.loads(rec["tools"]), list)
        assert json.loads(rec["messages"])[0]["role"] == "user"
    assert (tmp_path / "preview.md").exists()
    assert json.loads((tmp_path / "stats.json").read_text())["bfcl_gorilla_commit"]


def test_no_question_is_in_both_train_and_val(rows, decon):
    examples, _ = build(rows, 8, 0.5, 3, decon)
    examples += [dict(e, source_id=f"copy{i}") for i, e in enumerate(examples)]  # force repeats
    train, val = split(examples, 0.3, 3)
    q = lambda e: " ".join(e["messages"][0]["content"].lower().split())
    assert val and not {q(e) for e in train} & {q(e) for e in val}
    assert len(train) + len(val) == len(examples)
