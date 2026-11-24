"""Episode records (DESIGN.md §5): the slim JSONL record, the raw harness log, the Pi session, the dataset card.

Layout in a trace / evaluation dataset (`<root>`, published by `distill publish` after a credential scan):

    episodes/<prefix>.jsonl        one record per episode (scores, answer, messages, tool events); no raw logs
    sessions/<prefix>/*.jsonl      one Pi session per episode (the agent-traces viewer)
    summaries/<prefix>.json        the evaluation summary

and in its raw dataset (`<raw root>`, default `<root>-raw`, uploaded to the always-private `<run>-...-raw` repo):

    raw/<prefix>.jsonl             the harness-native messages per episode (for rescoring and debugging)

Raw logs never share a folder or repo with anything that becomes public. Every line is written through
`credentials.record_line`, which redacts credentials.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_distill.credentials import record_line
from agent_distill.episode import Episode, ToolEvent
from agent_distill.export.pi_session import episode_to_pi_session, session_filename, write_session
from agent_distill.rubrics import RubricResult

RECORD_VERSION = 2


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def episode_record(episode: Episode, result: RubricResult, **extra: Any) -> dict[str, Any]:
    return {
        "version": RECORD_VERSION,
        "task": episode.task,
        "row_id": episode.row_id,
        "harness": episode.harness,
        "harness_version": episode.harness_version,
        "model": episode.model,
        **extra,
        "session_id": episode.session_id,
        "answer": _jsonable(episode.answer),
        "answer_mode": episode.answer_mode,
        "answer_error": episode.answer_error,
        "failure": episode.failure,
        "elapsed_seconds": episode.elapsed_seconds,
        "usage": episode.usage,
        "tool_calls": episode.tool_calls,
        "scores": result.scores,
        "reward": result.reward,
        "passed": result.passed,
        "metrics": result.metrics,
        "rubric_errors": result.errors,
        "score_mode": result.mode,
        "metadata": episode.metadata,
        "tool_events": [event.to_dict() for event in episode.tool_events],
        "messages": episode.messages,
    }


def episode_from_record_v2(record: dict[str, Any], answer_model: Any) -> Episode:
    """An agent-distill episode record back to an `Episode` (the stored answer is re-validated)."""
    answer = record.get("answer")
    if answer is not None and hasattr(answer_model, "model_validate"):
        try:
            answer = answer_model.model_validate(answer)
        except ValueError:
            answer = None
    return Episode(
        task=record["task"],
        row_id=str(record["row_id"]),
        harness=record["harness"],
        harness_version=record["harness_version"],
        model=record.get("model", ""),
        messages=record.get("messages") or [],
        tool_events=[ToolEvent.from_dict(event) for event in record.get("tool_events") or []],
        answer=answer,
        answer_mode=record.get("answer_mode", ""),
        answer_error=record.get("answer_error"),
        usage=record.get("usage") or {},
        elapsed_seconds=record.get("elapsed_seconds"),
        failure=record.get("failure"),
        session_id=record.get("session_id"),
        metadata=record.get("metadata") or {},
    )


def raw_root_for(root: Path) -> Path:
    """The default local folder of a dataset's raw logs: a sibling `<root>-raw`, never inside `root`."""
    root = Path(root)
    return root.with_name(root.name + "-raw")


class RecordWriter:
    """Appends records and Pi sessions under `root`, and raw logs under `raw_root`, for one (prefix) file set."""

    def __init__(self, root: Path, prefix: str, raw_root: Path | None = None) -> None:
        self.root = Path(root)
        self.raw_root = Path(raw_root) if raw_root is not None else raw_root_for(self.root)
        if self.raw_root.resolve().is_relative_to(self.root.resolve()):
            raise ValueError(f"raw logs ({self.raw_root}) must not live inside the publishable {self.root}")
        self.prefix = prefix
        self.episodes = self.root / "episodes" / f"{prefix}.jsonl"
        self.raw = self.raw_root / "raw" / f"{prefix}.jsonl"
        self.sessions = self.root / "sessions" / prefix
        for path in (self.episodes.parent, self.raw.parent, self.sessions):
            path.mkdir(parents=True, exist_ok=True)

    def write(
        self, episode: Episode, result: RubricResult, native_trace: list[dict] | None = None, **extra: Any
    ) -> dict:
        record = episode_record(episode, result, **extra)
        with self.episodes.open("a", encoding="utf-8") as handle:
            handle.write(record_line(record))
        with self.raw.open("a", encoding="utf-8") as handle:
            raw = {"session_id": episode.session_id, "row_id": episode.row_id, **extra, "raw": episode.raw}
            handle.write(record_line(raw))
        lines = native_trace or episode_to_pi_session(episode, verification=result.to_dict())
        name = session_filename(episode)
        label = extra.get("label")
        if label:
            name = f"{label}_{name}"
        write_session(_redacted_lines(lines), self.sessions / name)
        return record

    def done(self, key: str = "label") -> set[tuple[str, str]]:
        if not self.episodes.exists():
            return set()
        keys = set()
        for line in self.episodes.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                keys.add((str(record.get(key)), str(record["row_id"])))
        return keys


def _redacted_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [json.loads(record_line(line)) for line in lines]


# --------------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per label: episodes, pass@1, mean reward, mean rubric scores, metric sums, tool calls, failures; plus the
    paired McNemar test of `passed` for every pair of labels evaluated on the same rows."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("label") or record.get("evaluation_model") or "model")].append(record)
    labels: dict[str, Any] = {}
    for label, group in sorted(groups.items()):
        rewards = [r["reward"] for r in group if r.get("reward") is not None]
        scores: dict[str, list[float]] = defaultdict(list)
        metrics: dict[str, float] = defaultdict(float)
        failures: dict[str, int] = defaultdict(int)
        for record in group:
            for name, value in (record.get("scores") or {}).items():
                if value is not None:
                    scores[name].append(value)
            for name, value in (record.get("metrics") or {}).items():
                metrics[name] += value
            if record.get("failure"):
                failures[str(record["failure"])[:120]] += 1
        labels[label] = {
            "episodes": len(group),
            "passed": sum(bool(r.get("passed")) for r in group),
            "pass_at_1": sum(bool(r.get("passed")) for r in group) / len(group),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "scores": {name: sum(v) / len(v) for name, v in sorted(scores.items())},
            "metric_sums": dict(sorted(metrics.items())),
            "mean_tool_calls": sum(r.get("tool_calls") or 0 for r in group) / len(group),
            "failures": dict(sorted(failures.items(), key=lambda item: -item[1])),
        }
    paired = {}
    names = sorted(groups)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a = {r["row_id"]: bool(r.get("passed")) for r in groups[first]}
            b = {r["row_id"]: bool(r.get("passed")) for r in groups[second]}
            shared = sorted(set(a) & set(b))
            only_a = sum(a[k] and not b[k] for k in shared)
            only_b = sum(b[k] and not a[k] for k in shared)
            paired[f"{first}|{second}"] = {"rows": len(shared), f"{first}_only": only_a, f"{second}_only": only_b,
                                            "mcnemar_exact_p": mcnemar_exact(only_a, only_b)}  # fmt: skip
    return {"labels": labels, "paired": paired}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
