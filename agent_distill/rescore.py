"""Rescore stored episode records with a task's rubric set (`distill rescore`, DESIGN.md §7).

Reads episode records from a Hub dataset (or local JSONL files), converts each to an `Episode` with the reader for
its harness log, scores it with the task's rubrics, and prints a summary per file and model label. Records written
by url-extraction (raw OpenCode 1.2.4 messages) are read with `opencode_v1_reader`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_distill.episode import Episode
from agent_distill.rubrics import RubricResult, episode_context
from agent_distill.task import Limits, Task, load_task


@dataclass
class Rescored:
    record: dict[str, Any]
    episode: Episode
    result: RubricResult


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_rows(task: Task, datasets: Iterable[str] | None = None, revision: str | None = None) -> dict[str, dict]:
    """Every row of every split of the label datasets, keyed by the task's row id (later datasets win)."""
    from datasets import get_dataset_split_names, load_dataset

    rows: dict[str, dict] = {}
    for repo in list(datasets or [task.dataset.repo]):
        for split in get_dataset_split_names(repo, revision=revision):
            for row in load_dataset(repo, split=split, revision=revision):
                rows[task.row_id(dict(row))] = dict(row)
    return rows


def record_to_episode(record: dict[str, Any], task: Task, row_id: str) -> Episode:
    if record.get("version") == 2 and "harness" in record:
        from agent_distill.records import episode_from_record_v2

        return episode_from_record_v2(record, task.answer)
    if "opencode_messages" in record or "tool_events" in record:
        from agent_distill.harnesses.opencode_v1_reader import episode_from_record

        return episode_from_record(record, answer_model=task.answer, task=task.name, row_id=row_id)
    raise ValueError("unknown record format (no opencode_messages / tool_events)")


def rescore(
    records: Iterable[dict[str, Any]],
    task: Task,
    rows: dict[str, dict],
    *,
    limits: Limits | None = None,
    mode: str = "eval",
) -> list[Rescored]:
    limits = limits or task.limits
    out: list[Rescored] = []
    for record in records:
        row_id = str(record["row_id"]) if record.get("version") == 2 else task.row_id(record)
        row = rows.get(row_id)
        if row is None:
            raise KeyError(f"no label row for {row_id!r}")
        episode = record_to_episode(record, task, row_id)
        result = task.rubrics.score(episode_context(episode, row, limits=limits, task=task), mode=mode)  # type: ignore[arg-type]
        out.append(Rescored(record, episode, result))
    return out


def summarize(items: list[Rescored]) -> dict[str, dict[str, Any]]:
    """Per model label: episodes, pass rate, mean reward, mean of each rubric score, sums of rubric metrics."""
    groups: dict[str, list[Rescored]] = defaultdict(list)
    for item in items:
        record = item.record
        groups[str(record.get("label") or record.get("evaluation_model") or "teacher")].append(item)
    summary: dict[str, dict[str, Any]] = {}
    for label, group in sorted(groups.items()):
        rewards = [item.result.reward for item in group if item.result.reward is not None]
        scores: dict[str, list[float]] = defaultdict(list)
        metrics: dict[str, float] = defaultdict(float)
        for item in group:
            for name, value in item.result.scores.items():
                if value is not None:
                    scores[name].append(value)
            for name, value in item.result.metrics.items():
                metrics[name] += value
        summary[label] = {
            "episodes": len(group),
            "passed": sum(item.result.passed for item in group),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "scores": {name: sum(values) / len(values) for name, values in sorted(scores.items())},
            "metric_sums": dict(sorted(metrics.items())),
            "tool_calls": sum(item.episode.tool_calls for item in group) / len(group),
        }
    return summary


def repo_record_files(repo: str, revision: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(
        name
        for name in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
        if name.endswith(".jsonl") and not name.startswith("sessions/") and "difficulty/" not in name
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distill rescore", description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, help="Task folder or task.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Dataset repo holding episode records")
    source.add_argument("--local", nargs="+", type=Path, help="Local JSONL record files")
    parser.add_argument("--files", nargs="*", help="Record files in --repo (default: every *.jsonl outside sessions/)")
    parser.add_argument("--revision")
    parser.add_argument("--dataset", action="append", help="Label dataset(s) (default: the task's dataset)")
    parser.add_argument("--max-tool-calls", type=int, help="Tool-call limit the episodes ran with")
    parser.add_argument("--mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--output", type=Path, help="Write the per-file summary as JSON")
    args = parser.parse_args(argv)

    task = load_task(args.task)
    limits = Limits(timeout=task.limits.timeout, max_tool_calls=args.max_tool_calls or task.limits.max_tool_calls)
    rows = load_rows(task, args.dataset)
    if args.repo:
        from huggingface_hub import hf_hub_download

        names = args.files or repo_record_files(args.repo, args.revision)
        files = {
            name: Path(hf_hub_download(args.repo, name, repo_type="dataset", revision=args.revision)) for name in names
        }
    else:
        files = {str(path): path for path in args.local}
    report: dict[str, Any] = {}
    for name, path in files.items():
        records = list(read_jsonl(path))
        if not records or not ({"opencode_messages", "tool_events"} & set(records[0])):
            print(f"skip {name}: not episode records")
            continue
        summary = summarize(rescore(records, task, rows, limits=limits, mode=args.mode))
        report[name] = summary
        print(f"\n## {name}")
        for label, entry in summary.items():
            mean = entry["mean_reward"]
            mean_text = f"{mean:.3f}" if mean is not None else "n/a"
            print(f"- {label}: {entry['episodes']} episodes, passed {entry['passed']}, mean reward {mean_text}")
            print(f"  scores {json.dumps({k: round(v, 4) for k, v in entry['scores'].items()})}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    return 0
