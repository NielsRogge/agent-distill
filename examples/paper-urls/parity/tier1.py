"""Parity tier 1: agent-distill's paper-URL rubrics vs url-extraction's scorer on every published record.

For each episode record of the published url-extraction datasets, both implementations score the same stored
episode against the same label row:

- url-extraction (`harness.report.rescore_record` logic, `harness.scoring.score_episode` / `rollout_reward`)
  from a checkout given with `--url-extraction`;
- agent-distill: `opencode_v1_reader.episode_from_record` + the example's `RubricSet` (eval mode for
  verification/completeness/passed, train mode for the GRPO reward).

Compared per episode: the parsed answer, per-field verification, completeness and its reason, the success
criterion (`passed` vs `all_scoreable_correct and attempt_complete`), and the GRPO reward (protocol weight 0.5,
tool-efficiency bonus, shortcut guard). The only documented difference: a row with no verifiable field is
unscorable here (reward None) where url-extraction's reward is 0.0; such episodes are counted separately.

    uv run --extra hub --with httpx python examples/paper-urls/parity/tier1.py \
        --url-extraction ../url-extraction --output parity-tier1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PUBLISHED = {
    "nielsr/qwen35-4b-url-agent-eval-v2": None,
    "nielsr/qwen35-4b-url-agent-eval": None,
    "nielsr/qwen35-9b-url-agent-eval": None,
    "nielsr/inkling-2-mini-url-agent-eval": None,
    "nielsr/paper-url-opencode-traces": None,
    "nielsr/paper-url-opencode-traces-kimi": None,
    "nielsr/inkling-2-mini-url-agent-traces": None,
}
LABELS = (
    "nielsr/paper-url-extraction-v1",
    "nielsr/paper-url-extraction-v1-validated",
    "nielsr/paper-url-extraction-v2",  # test_hard
)
PROTOCOL_WEIGHT = 0.5


def max_tool_calls_for(name: str) -> int:
    """The limit the episodes ran with: teacher trace generation used 20, evaluation and baselines 15."""
    stem = name.rsplit("/", 1)[-1]
    return 20 if stem.startswith(("train", "distill")) else 15


def load_rows() -> dict[int, dict[str, Any]]:
    """paper_id -> row; v1 labels first, later datasets only for papers earlier ones lack (e.g. test_hard)."""
    from datasets import get_dataset_split_names, load_dataset

    rows: dict[int, dict[str, Any]] = {}
    for repo in LABELS:
        for split in get_dataset_split_names(repo):
            for row in load_dataset(repo, split=split):
                rows.setdefault(int(row["paper_id"]), dict(row))
    return rows


def url_extraction_view(record: dict[str, Any], row: dict[str, Any], limit: int) -> dict[str, Any]:
    """What url-extraction's rescoring computes for one record (mirrors `harness.report.rescore_record`)."""
    from harness.opencode import findings_from_messages, tool_events
    from harness.scoring import rollout_reward, score_episode
    from harness.task import ToolEvent, URLFindings
    from pydantic import ValidationError

    findings = None
    if record.get("findings"):
        try:
            findings = URLFindings.model_validate(record["findings"])
        except ValidationError:
            findings = None
    if findings is None and record.get("opencode_messages"):
        findings = findings_from_messages(record["opencode_messages"])
    if record.get("opencode_messages"):
        events = tool_events(record["opencode_messages"])
    else:
        events = [ToolEvent.model_validate(event) for event in record.get("tool_events") or []]
    score = score_episode(row, findings, events, max_tool_calls=limit)
    reward = rollout_reward(row, findings, events, protocol_weight=PROTOCOL_WEIGHT, max_tool_calls=limit)
    verification = score["verification"]
    return {
        "answer": findings.model_dump(mode="json") if findings else None,
        "fields": verification["fields"],
        "complete": score["attempt_complete"],
        "reason": score["incomplete_reason"],
        "passed": bool(verification["all_scoreable_correct"]) and bool(score["attempt_complete"]),
        "reward": reward,
        "num_scoreable": verification["num_scoreable"],
        "tool_calls": len(events),
    }


def agent_distill_view(
    record: dict[str, Any], row: dict[str, Any], limit: int, task: Any, rubrics: Any
) -> dict[str, Any]:
    from agent_distill.harnesses.opencode_v1_reader import episode_from_record
    from agent_distill.rubrics import episode_context
    from agent_distill.task import Limits

    episode = episode_from_record(record, answer_model=task.answer, task=task.name, row_id=str(row["paper_id"]))
    context = episode_context(episode, row, limits=Limits(max_tool_calls=limit), task=task)
    evaluation = task.rubrics.score(context, mode="eval")
    train = task.rubrics.score(context, mode="train")
    return {
        "answer": episode.answer.model_dump(mode="json") if episode.answer is not None else None,
        "fields": rubrics.verify_fields(row, episode.answer),
        "complete": evaluation.scores["protocol"] == 1.0,
        "reason": rubrics.incomplete_reason(row, episode.answer, episode.tool_events, limit),
        "passed": evaluation.passed,
        "reward": train.reward,
        "tool_calls": episode.tool_calls,
    }


def compare(theirs: dict[str, Any], ours: dict[str, Any]) -> list[str]:
    differences = [
        f"{key}: url-extraction {theirs[key]!r} vs agent-distill {ours[key]!r}"
        for key in ("answer", "fields", "complete", "reason", "passed", "tool_calls")
        if theirs[key] != ours[key]
    ]
    if ours["reward"] is None:
        if not (theirs["num_scoreable"] == 0 and theirs["reward"] == 0.0):
            differences.append(f"reward: url-extraction {theirs['reward']} vs agent-distill None")
    elif not math.isclose(theirs["reward"], ours["reward"], abs_tol=1e-12):
        differences.append(f"reward: url-extraction {theirs['reward']} vs agent-distill {ours['reward']}")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url-extraction", type=Path, required=True, help="Checkout of NielsRogge/url-extraction")
    parser.add_argument("--repos", nargs="*", default=list(PUBLISHED))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.url_extraction.resolve()))
    from huggingface_hub import HfApi, hf_hub_download

    from agent_distill.rescore import read_jsonl, repo_record_files
    from agent_distill.task import load_task, task_module

    here = Path(__file__).resolve().parent.parent
    task = load_task(here)
    rubrics = task_module(here, "rubrics")
    rows = load_rows()
    api = HfApi()
    report: dict[str, Any] = {"files": {}, "totals": {"episodes": 0, "differences": 0, "unscorable": 0}}
    for repo in args.repos:
        revision = api.dataset_info(repo).sha
        for name in repo_record_files(repo, revision):
            started = time.monotonic()
            path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))
            records = [
                r for r in read_jsonl(path) if "verification" in r and ("opencode_messages" in r or "tool_events" in r)
            ]
            if not records:
                continue
            limit = max_tool_calls_for(name)
            entry = {"revision": revision, "max_tool_calls": limit, "episodes": len(records), "unscorable": 0,
                     "passed": 0, "differences": []}  # fmt: skip
            for record in records:
                row = rows[int(record["paper_id"])]
                theirs = url_extraction_view(record, row, limit)
                ours = agent_distill_view(record, row, limit, task, rubrics)
                entry["passed"] += ours["passed"]
                entry["unscorable"] += ours["reward"] is None
                notes = compare(theirs, ours)
                if notes:
                    entry["differences"].append({"paper_id": record["paper_id"], "attempt": record.get("attempt"),
                                                 "label": record.get("evaluation_model"), "notes": notes})  # fmt: skip
            report["files"][f"{repo}:{name}"] = entry
            for key in ("episodes", "unscorable"):
                report["totals"][key] += entry[key]
            report["totals"]["differences"] += len(entry["differences"])
            print(
                f"{repo}:{name}  {entry['episodes']} episodes (limit {limit}), passed {entry['passed']}, "
                f"unscorable {entry['unscorable']}, differences {len(entry['differences'])}"
                f"  [{time.monotonic() - started:.0f}s]",
                flush=True,
            )
            for difference in entry["differences"][:5]:
                print(f"    {difference}", flush=True)
    totals = report["totals"]
    print(
        f"\nTOTAL {totals['episodes']} episodes, {totals['differences']} differing, {totals['unscorable']} unscorable"
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    return 1 if totals["differences"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
