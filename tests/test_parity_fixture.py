"""Parity tier 1 in CI: real published url-extraction records rescore to the verdicts stored in them.

The fixture (`tests/fixtures/paper_urls_published.jsonl`, built by `examples/paper-urls/parity/make_fixture.py`)
holds the smallest published record of each outcome per source file, with its label row and the tool-call limit
its job ran with. `distill rescore`'s code path must reproduce the stored per-field verification, `attempt_complete`
and `accepted` exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_distill.credentials import scan_text
from agent_distill.rescore import rescore
from agent_distill.task import load_task

FIXTURE = Path(__file__).parent / "fixtures" / "paper_urls_published.jsonl"
TASK = load_task("examples/paper-urls")
ENTRIES = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def test_fixture_covers_every_outcome_and_both_limits():
    outcomes = {entry["source"]["outcome"] for entry in ENTRIES}
    assert outcomes == {"passed", "wrong_field", "incomplete", "no_answer", "tool_call_limit"}
    assert {entry["source"]["max_tool_calls"] for entry in ENTRIES} == {15, 20}
    assert not scan_text(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"{e['source']['file']}:{e['source']['outcome']}")
def test_published_record_rescores_to_its_stored_verdict(entry):
    record, row, source = entry["record"], entry["row"], entry["source"]
    (item,) = rescore([record], TASK, {str(row["paper_id"]): row}, name=source["file"])
    scores, stored = item.result.scores, record["verification"]
    fields = {key: value for key, value in item.result.metrics.items() if key.endswith("_correct")}
    expected = {f"url_correctness/{name}_correct": float(ok) for name, ok in stored["fields"].items() if ok is not None}
    assert fields == expected
    assert (scores["protocol"] == 1.0) is bool(record["attempt_complete"])
    assert item.result.passed is bool(record["accepted"])
    assert item.episode.harness_id == "opencode@1.2.4"
