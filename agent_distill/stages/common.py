"""The episode loop shared by every stage that runs a model in the harness: eval, traces, the difficulty probe.

One code path for all three is the parity guarantee: a teacher trace is accepted, a probe attempt counts as a success
and an evaluation episode passes by the same harness run and the same rubric set (eval mode).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_distill.harnesses.base import Endpoint
from agent_distill.harnesses.opencode import harness_failure
from agent_distill.records import RecordWriter, read_jsonl
from agent_distill.rubrics import episode_context
from agent_distill.task import Limits, Task


@dataclass
class Plan:
    """What to run for one label: `attempts` per row; with `stop_when_passed`, a row stops at its first pass
    (teacher traces); `budget_usd` stops starting new episodes once the label's recorded spend reaches it."""

    label: str
    attempts: int = 1
    stop_when_passed: bool = False
    budget_usd: float = 0.0
    price: tuple[float, float] | None = None  # USD per 1M input/output tokens, when the harness reports no cost
    extra: dict[str, Any] = field(default_factory=dict)


def progress(writer: RecordWriter, label: str) -> tuple[dict[str, int], set[str], float]:
    """(attempts recorded per row, rows that passed, USD spent) for `label`."""
    attempts: dict[str, int] = defaultdict(int)
    passed: set[str] = set()
    spent = 0.0
    if writer.episodes.exists():
        for record in read_jsonl(writer.episodes):
            if str(record.get("label")) != label:
                continue
            row_id = str(record["row_id"])
            attempts[row_id] += 1
            if record.get("passed"):
                passed.add(row_id)
            spent += float((record.get("usage") or {}).get("cost") or 0)
    return attempts, passed, spent


def pending_work(task: Task, rows: list[dict], writer: RecordWriter, plan: Plan) -> list[tuple[dict, int]]:
    attempts, passed, _ = progress(writer, plan.label)
    work = []
    for row in rows:
        row_id = task.row_id(row)
        if plan.stop_when_passed and row_id in passed:
            continue
        work += [(row, attempt) for attempt in range(attempts[row_id] + 1, plan.attempts + 1)]
    return work


async def run_episodes(
    harness: Any,
    task: Task,
    rows: list[dict],
    endpoint: Endpoint,
    writer: RecordWriter,
    plan: Plan,
    *,
    limits: Limits,
    concurrency: int,
    cache: Path,
    max_restarts: int = 3,
) -> None:
    """Run the plan, resuming from what `writer` already holds and restarting the harness after a harness
    failure (such an episode says nothing about the model and is not recorded)."""
    for restart in range(max_restarts + 1):
        work = pending_work(task, rows, writer, plan)
        if not work:
            return
        if restart:
            print(f"[{plan.label}] restarting the harness after a harness failure ({len(work)} left)", flush=True)
        if not await _run_once(harness, task, work, endpoint, writer, plan, limits, concurrency, cache):
            return
    raise RuntimeError(f"the harness kept failing for {plan.label} after {max_restarts} restarts")


async def _run_once(
    harness: Any,
    task: Task,
    work: list[tuple[dict, int]],
    endpoint: Endpoint,
    writer: RecordWriter,
    plan: Plan,
    limits: Limits,
    concurrency: int,
    cache: Path,
) -> bool:
    """Run `work` on one fresh harness. Returns True when the harness broke (the rest is left for a restart)."""
    broken = asyncio.Event()
    semaphore = asyncio.Semaphore(concurrency)
    _, passed, spent = progress(writer, plan.label)
    state = {"spent": spent, "passed": set(passed), "over_budget": False}
    # One row's attempts run in order (a trace row stops at its first pass); rows run concurrently.
    by_row: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for row, attempt in work:
        by_row[task.row_id(row)].append((row, attempt))

    async with harness.session([endpoint], cache=cache, task=task) as runner:

        async def run_row(index: int, items: list[tuple[dict, int]]) -> None:
            for row, attempt in items:
                row_id = task.row_id(row)
                async with semaphore:
                    if broken.is_set() or (plan.stop_when_passed and row_id in state["passed"]):
                        return
                    if plan.budget_usd and state["spent"] >= plan.budget_usd:
                        state["over_budget"] = True
                        return
                    episode = await harness.run_episode(runner, task, row, endpoint, limits)
                    if failure := harness_failure(episode):
                        print(f"  [{plan.label}] {row_id}: harness failure, not recorded: {failure}", flush=True)
                        broken.set()
                        return
                    episode.metadata["attempt"] = attempt
                    if plan.price and not episode.usage.get("cost"):
                        price_in, price_out = plan.price
                        tokens_in, tokens_out = episode.usage.get("input", 0), episode.usage.get("output", 0)
                        episode.usage["cost"] = (tokens_in * price_in + tokens_out * price_out) / 1e6
                        episode.usage["cost_estimated"] = True
                    context = episode_context(episode, row, limits=limits, task=task)
                    result = await task.rubrics.ascore(context, mode="eval")
                    writer.write(
                        episode, result, harness.native_trace(episode), label=plan.label, attempt=attempt, **plan.extra
                    )
                    state["spent"] += float(episode.usage.get("cost") or 0)
                    if result.passed:
                        state["passed"].add(row_id)
                    print(
                        f"[{plan.label} {index}/{len(by_row)}] {row_id} attempt={attempt} passed={result.passed} "
                        f"reward={result.reward} tools={episode.tool_calls} failure={episode.failure}",
                        flush=True,
                    )

        await asyncio.gather(*(run_row(i, items) for i, items in enumerate(by_row.values(), 1)))
        if not broken.is_set():
            await harness.healthy(runner)
    if state["over_budget"]:
        print(f"[{plan.label}] stopped at the budget: ${state['spent']:.2f} of ${plan.budget_usd:.2f}", flush=True)
    return broken.is_set()


def shard(rows: list[dict], spec: str) -> list[dict]:
    """`i/n`: every n-th row starting at i (disjoint slices for parallel jobs)."""
    index, count = (int(value) for value in spec.split("/"))
    if not 0 <= index < count:
        raise ValueError(f"bad shard {spec!r}")
    return rows[index::count]
