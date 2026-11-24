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


def label_repos(task: Task, datasets: Iterable[str] | None = None) -> list[str]:
    """The label datasets to read, in precedence order: `--dataset`, else the task's `label_repos`, else its repo."""
    return list(datasets or task.dataset.label_repos or [task.dataset.repo])


def load_rows(task: Task, datasets: Iterable[str] | None = None, revision: str | None = None) -> dict[str, dict]:
    """Every row of every split of the label datasets, keyed by the task's row id. Earlier datasets win: a later
    one only adds rows the earlier ones lack (as parity tier 1 does)."""
    from datasets import get_dataset_split_names, load_dataset

    rows: dict[str, dict] = {}
    for repo in label_repos(task, datasets):
        for split in get_dataset_split_names(repo, revision=revision):
            for row in load_dataset(repo, split=split, revision=revision):
                rows.setdefault(task.row_id(dict(row)), dict(row))
    return rows


def limits_for(task: Task, name: str, record: dict[str, Any], override: int | None = None) -> Limits:
    """The limits a record ran with: `--max-tool-calls`, else the record's own `limits`, else the task's
    `record_limits(file name)`, else the task's limits."""
    base = (task.record_limits(name) if task.record_limits else None) or task.limits
    stored = record.get("limits") if isinstance(record.get("limits"), dict) else {}
    max_calls = override or stored.get("max_tool_calls") or base.max_tool_calls
    return Limits(timeout=stored.get("timeout") or base.timeout, max_tool_calls=max_calls,
                  max_model_calls=base.max_model_calls)  # fmt: skip


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
    name: str = "",
    max_tool_calls: int | None = None,
) -> list[Rescored]:
    """Score every record. `limits` forces one set of limits; otherwise each record's are found by `limits_for`
    (from the record, or the task's `record_limits(name)` for the file `name`)."""
    out: list[Rescored] = []
    for record in records:
        row_id = str(record["row_id"]) if record.get("version") == 2 else task.row_id(record)
        row = rows.get(row_id)
        if row is None:
            raise KeyError(f"no label row for {row_id!r}")
        episode = record_to_episode(record, task, row_id)
        record_limits = limits or limits_for(task, name, record, max_tool_calls)
        context = episode_context(episode, row, limits=record_limits, task=task)
        result = task.rubrics.score(context, mode=mode)  # type: ignore[arg-type]
        out.append(Rescored(record, episode, result))
    return out


def summarize(items: list[Rescored]) -> dict[str, dict[str, Any]]:
    """Per model label: episodes, passes, unscorable episodes, mean reward, mean of each rubric score, sums of
    rubric metrics, and pooled accuracies.

    `pooled` turns metric pairs `<rubric>/<x>_correct` and `<rubric>/<x>_scored` into one accuracy per rubric over
    every scored item (e.g. per-field URL accuracy, url-extraction's `field_accuracy`), which differs from the
    per-episode mean in `scores` when episodes score different numbers of items.
    """
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
        pooled: dict[str, float] = {}
        for name in {key.split("/", 1)[0] for key in metrics if key.endswith("_scored") and "/" in key}:
            correct = sum(v for k, v in metrics.items() if k.startswith(f"{name}/") and k.endswith("_correct"))
            scored = sum(v for k, v in metrics.items() if k.startswith(f"{name}/") and k.endswith("_scored"))
            if scored:
                pooled[name] = correct / scored
        summary[label] = {
            "episodes": len(group),
            "passed": sum(item.result.passed for item in group),
            "unscorable": sum(item.result.reward is None for item in group),
            "errored": sum(item.result.errored for item in group),
            "pooled": dict(sorted(pooled.items())),
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
    parser = argparse.ArgumentParser(prog="distill rescore", description="Rescore stored episode records")
    parser.add_argument("--task", required=True, help="Task folder or task.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Dataset repo holding episode records")
    source.add_argument("--local", nargs="+", type=Path, help="Local JSONL record files")
    parser.add_argument("--files", nargs="*", help="Record files in --repo (default: every *.jsonl outside sessions/)")
    parser.add_argument("--revision")
    parser.add_argument("--dataset", action="append", help="Label dataset(s) (default: the task's dataset)")
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        help="Tool-call limit the episodes ran with (default: per record, from the record or the task's record_limits)",
    )
    parser.add_argument("--mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--output", type=Path, help="Write the per-file summary as JSON")
    args = parser.parse_args(argv)

    task = load_task(args.task)
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
        rescored = rescore(records, task, rows, mode=args.mode, name=name, max_tool_calls=args.max_tool_calls)
        summary = summarize(rescored)
        report[name] = summary
        print(f"\n## {name}")
        for label, entry in summary.items():
            mean = entry["mean_reward"]
            mean_text = f"{mean:.3f}" if mean is not None else "n/a"
            print(
                f"- {label}: {entry['episodes']} episodes, passed {entry['passed']} "
                f"(unscorable {entry['unscorable']}, counted as not passed), mean reward {mean_text}"
            )
            for rubric, value in entry["pooled"].items():
                print(f"  {rubric} over all scored items: {value:.4f}")
            print(f"  scores {json.dumps({k: round(v, 4) for k, v in entry['scores'].items()})}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    return 0
