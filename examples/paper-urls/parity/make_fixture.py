"""Build the tier-1 parity fixture: a few real published url-extraction records with their stored verdicts.

Downloads a handful of published record files at pinned revisions, scores every record with url-extraction's
scorer (a checkout given with `--url-extraction`, as in `tier1.py`) and with agent-distill's rubrics, and keeps
the smallest records of each outcome (passed, wrong field, incomplete, no answer, tool-call limit) for which both
scorers agree with each other and with the verdicts stored in the record (`verification.fields`, `accepted`,
`attempt_complete`). The records are written through the redacting record writer, with the fields the reader does
not use (`trajectory` beyond its system message, `tool_events` when raw messages exist) dropped to keep the
fixture small, next to their label rows. `tests/test_parity_fixture.py` rescores the fixture in CI.

    uv run --extra hub --with httpx python examples/paper-urls/parity/make_fixture.py \
        --url-extraction ../url-extraction --output tests/fixtures/paper_urls_published.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# (repo, revision, file): evaluations and teacher baselines ran with 15 tool calls, trace generation with 20.
SOURCES = (
    ("nielsr/qwen35-9b-url-agent-eval", "bf5cbc681ff922f5445cc5b0d6c7f0eaa0f02569", "sft-grpo/test.jsonl"),
    (
        "nielsr/paper-url-opencode-traces-kimi",
        "f44c5cd7b10a5ad2776a1b0ba2974740d7d900f2",
        "teacher-test-kimi-k31.jsonl",
    ),
    ("nielsr/paper-url-opencode-traces", "4f87953b235ff9b0d5bd23b723ff51ade95e7eb5", "train.jsonl"),
    ("nielsr/inkling-2-mini-url-agent-eval", "571cff5becebf382e2b72cf1c0b8f79a6f63005c", "base/test.jsonl"),
    ("nielsr/inkling-2-mini-url-agent-eval", "571cff5becebf382e2b72cf1c0b8f79a6f63005c", "base/test_hard.jsonl"),
)
MAX_RECORD_BYTES = 40_000  # per file and outcome, keep the smallest record if it is below this...
# ...and for an outcome no file has below it (e.g. the tool-call limit), the smallest one overall.


def outcome(ours: dict[str, Any]) -> str:
    if ours["answer"] is None:
        return "no_answer"
    if ours["reason"] and "limit" in ours["reason"]:
        return "tool_call_limit"
    if not ours["complete"]:
        return "incomplete"
    return "passed" if ours["passed"] else "wrong_field"


def slim(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    record["trajectory"] = [m for m in record.get("trajectory") or [] if m.get("role") == "system"]
    if record.get("opencode_messages"):
        record.pop("tool_events", None)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url-extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/paper_urls_published.jsonl"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.url_extraction.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from huggingface_hub import hf_hub_download
    from tier1 import agent_distill_view, compare, load_rows, url_extraction_view

    from agent_distill.credentials import record_line
    from agent_distill.rescore import read_jsonl
    from agent_distill.task import load_task, task_module

    here = Path(__file__).resolve().parent.parent
    task = load_task(here)
    rubrics = task_module(here, "rubrics")
    rows = load_rows()
    chosen: list[tuple[int, str, dict[str, Any]]] = []
    for repo, revision, name in SOURCES:
        limit = task.record_limits(name).max_tool_calls
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))
        candidates: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        stats = {"records": 0, "stored_mismatch": 0, "scorer_mismatch": 0}
        for record in read_jsonl(path):
            if "verification" not in record or not record.get("opencode_messages"):
                continue
            stats["records"] += 1
            row = rows[int(record["paper_id"])]
            theirs = url_extraction_view(record, row, limit)
            ours = agent_distill_view(record, row, limit, task, rubrics)
            if compare(theirs, ours):
                stats["scorer_mismatch"] += 1
                continue
            stored = {
                "fields": (record.get("verification") or {}).get("fields"),
                "complete": record.get("attempt_complete"),
                "passed": record.get("accepted"),
            }
            if stored != {"fields": ours["fields"], "complete": ours["complete"], "passed": ours["passed"]}:
                stats["stored_mismatch"] += 1
                continue
            item = slim(record)
            candidates.setdefault(outcome(ours), []).append((len(json.dumps(item)), item))
        print(f"{repo}:{name} limit {limit}: {stats}, outcomes {sorted(candidates)}")
        for kind, items in candidates.items():
            size, item = min(items, key=lambda pair: pair[0])
            source = {"repo": repo, "revision": revision, "file": name, "max_tool_calls": limit, "outcome": kind}
            chosen.append((size, kind, {"source": source, "row": rows[int(item["paper_id"])], "record": item}))
    kept = [entry for size, _, entry in chosen if size <= MAX_RECORD_BYTES]
    for kind in sorted({kind for _, kind, _ in chosen} - {entry["source"]["outcome"] for entry in kept}):
        kept.append(min((pair for pair in chosen if pair[1] == kind), key=lambda pair: pair[0])[2])
    lines = [record_line(entry) for entry in kept]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {len(lines)} records, {args.output.stat().st_size // 1024} KiB, to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
