"""The episode loop shared by every stage that runs a model in the harness: eval, traces, the difficulty probe.

One code path for all three is the parity guarantee: a teacher trace is accepted, a probe attempt counts as a success
and an evaluation episode passes by the same harness run and the same rubric set (eval mode).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_distill.harnesses.base import Endpoint, HarnessError
from agent_distill.harnesses.opencode import harness_failure
from agent_distill.reasoning import LeakGuard, leaked
from agent_distill.records import RecordWriter, read_jsonl
from agent_distill.rubrics import ErrorBudget, TooManyErrors, episode_context
from agent_distill.task import Limits, Task


class PublishRefused(RuntimeError):
    """An upload the stage must not make, e.g. unscanned episodes into a public dataset.

    A normal exception, never `SystemExit`: it is raised in a checkpoint's upload thread inside the event loop, where a
    `SystemExit` escapes the loop instead of stopping the stage (pending episodes, harness subprocesses and executor
    threads are then torn down by `asyncio.run`'s shutdown, if at all). A checkpoint that hits it stops the stage.
    """


@dataclass
class Plan:
    """What to run for one label: `attempts` per row; with `stop_when_passed`, a row stops at its first pass
    (teacher traces); `budget_usd` stops starting new episodes once the label's recorded spend reaches it."""

    label: str
    attempts: int = 1
    stop_when_passed: bool = False
    budget_usd: float = 0.0
    price: tuple[float, float] | None = None  # USD per 1M input/output tokens, when the harness reports no cost
    rerun_cut_short: int = 0  # reruns per row of episodes cut short by a timeout or provider error (traces)
    judge: Any = None  # the JudgeRunner judge rubrics use (`rubrics.judge.make_judge_runner`)
    extra: dict[str, Any] = field(default_factory=dict)


# Failures that are the model's own doing, never rerun: the harnesses' tool-call-limit and output-truncation messages.
OWN_FAILURES = ("tool-call limit", "response truncated at the output token limit")


def cut_short(episode: Any) -> str | None:
    """Why an episode stopped before the model finished, or None.

    An episode is cut short when it ended without an answer because of the timeout or a provider/harness error,
    typically mid-tool-call. Such a teacher episode says little about the row, so the traces stage reruns it
    (`Plan.rerun_cut_short`) instead of spending an attempt on it. Hitting the tool-call limit or the output token
    limit (reasoning that never reaches a tool call) is the model's own doing and is not a cut. Evaluation never
    reruns (a student's timeout counts against it).
    """
    failure = str(episode.failure or "")
    if episode.answer is not None or not failure or failure.startswith(OWN_FAILURES):
        return None
    last = episode.messages[-1] if episode.messages else {}
    where = "mid-tool-call" if last.get("role") == "tool" or last.get("tool_calls") else "mid-turn"
    return f"{where}: {str(episode.failure)[:200]}"


def progress(writer: RecordWriter, label: str, rerun_cut_short: int = 0) -> tuple[dict[str, int], set[str], float]:
    """(attempts used per row, rows that passed, USD spent) for `label`.

    Up to `rerun_cut_short` cut-short episodes per row do not use an attempt (they are kept in the records, and
    their spend counts)."""
    attempts: dict[str, int] = defaultdict(int)
    cuts: dict[str, int] = defaultdict(int)
    passed: set[str] = set()
    spent = 0.0
    if writer.episodes.exists():
        for record in read_jsonl(writer.episodes):
            if str(record.get("label")) != label:
                continue
            row_id = str(record["row_id"])
            spent += float((record.get("usage") or {}).get("cost") or 0)
            if record.get("cut_short") and not record.get("passed") and cuts[row_id] < rerun_cut_short:
                cuts[row_id] += 1
                continue
            attempts[row_id] += 1
            if record.get("passed"):
                passed.add(row_id)
    return attempts, passed, spent


def pending_work(task: Task, rows: list[dict], writer: RecordWriter, plan: Plan) -> list[tuple[dict, int]]:
    attempts, passed, _ = progress(writer, plan.label, plan.rerun_cut_short)
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
    checkpoint: Checkpoint | None = None,
    errors: ErrorBudget | None = None,
    leaks: LeakGuard | None = None,
) -> None:
    """Run the plan, resuming from what `writer` already holds and restarting the harness after a harness
    failure (such an episode says nothing about the model and is not recorded). Cut-short episodes are rerun
    up to `plan.rerun_cut_short` times per row. A `checkpoint` uploads the records as they grow. `errors` stops
    the stage (`TooManyErrors`, after recording the episode) once too many episodes were unscorable because a
    rubric errored, e.g. a judge outage; the stages still publish what finished. Every episode is marked
    `metadata["reasoning_leak"]` (agent_distill.reasoning), and `leaks` stops the stage the same way once too many
    episodes leak reasoning into their visible content."""
    if missing := task.missing_commands():
        raise RuntimeError(f"{plan.label}: commands the task needs are not on the PATH: {', '.join(missing)}")
    restarts = 0
    owner = asyncio.current_task()
    if checkpoint is not None and owner is not None:
        checkpoint.bind(owner)  # a refused checkpoint upload cancels the episodes and ends up raised here
    try:
        while (checkpoint is None or checkpoint.error is None) and (work := pending_work(task, rows, writer, plan)):
            outcome = await _run_once(harness, task, work, endpoint, writer, plan, limits, concurrency, cache,
                                      checkpoint, errors, leaks)  # fmt: skip
            if outcome == "budget":
                return
            if outcome == "broken":
                restarts += 1
                if restarts > max_restarts:
                    raise RuntimeError(f"the harness kept failing for {plan.label} after {max_restarts} restarts")
                left = len(pending_work(task, rows, writer, plan))
                print(f"[{plan.label}] restarting the harness after a harness failure ({left} left)", flush=True)
            # "done": every pending row recorded at least one episode; another pass reruns cut-short episodes.
    except asyncio.CancelledError:
        if checkpoint is None or checkpoint.error is None or owner is None:
            raise
        owner.uncancel()  # our own cancellation: the episodes and the harness are stopped; flush() raises the error
    finally:
        if checkpoint is not None:
            await checkpoint.flush()


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
    checkpoint: Checkpoint | None = None,
    errors: ErrorBudget | None = None,
    leaks: LeakGuard | None = None,
) -> str:
    """Run `work` on one fresh harness: "broken" when the harness broke (the rest is left for a restart),
    "budget" when the budget stopped it, else "done".

    A harness failure (an episode's shell could not spawn, the server died, did not start or failed its health
    check: `HarnessError`) is not recorded and stops this harness; episodes that finished are already written."""
    broken = asyncio.Event()
    stopped = asyncio.Event()

    def harness_broke(row_id: str, reason: str) -> None:
        print(f"  [{plan.label}] {row_id}: harness failure, not recorded: {reason}", flush=True)
        broken.set()

    semaphore = asyncio.Semaphore(concurrency)
    _, passed, spent = progress(writer, plan.label, plan.rerun_cut_short)
    state = {"spent": spent, "passed": set(passed), "over_budget": False}
    # One row's attempts run in order (a trace row stops at its first pass); rows run concurrently.
    by_row: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for row, attempt in work:
        by_row[task.row_id(row)].append((row, attempt))

    try:
        await _run_session(harness, task, by_row, endpoint, writer, plan, limits, cache, checkpoint, errors, semaphore,
                           state, broken, stopped, harness_broke, leaks)  # fmt: skip
    except HarnessError as exc:  # the harness did not start (or failed while stopping)
        harness_broke("-", str(exc))
    if state["over_budget"]:
        print(f"[{plan.label}] stopped at the budget: ${state['spent']:.2f} of ${plan.budget_usd:.2f}", flush=True)
    if broken.is_set():
        return "broken"
    return "budget" if state["over_budget"] else "done"


async def _run_session(
    harness: Any,
    task: Task,
    by_row: dict[str, list[tuple[dict, int]]],
    endpoint: Endpoint,
    writer: RecordWriter,
    plan: Plan,
    limits: Limits,
    cache: Path,
    checkpoint: Checkpoint | None,
    errors: ErrorBudget | None,
    semaphore: asyncio.Semaphore,
    state: dict[str, Any],
    broken: asyncio.Event,
    stopped: asyncio.Event,
    harness_broke: Callable[[str, str], None],
    leaks: LeakGuard | None = None,
) -> None:
    async with harness.session([endpoint], cache=cache, task=task) as runner:

        async def run_row(index: int, items: list[tuple[dict, int]]) -> None:
            for row, attempt in items:
                row_id = task.row_id(row)
                async with semaphore:
                    if broken.is_set() or stopped.is_set() or (plan.stop_when_passed and row_id in state["passed"]):
                        return
                    if plan.budget_usd and state["spent"] >= plan.budget_usd:
                        state["over_budget"] = True
                        return
                    try:
                        episode = await harness.run_episode(runner, task, row, endpoint, limits)
                    except HarnessError as exc:
                        harness_broke(row_id, str(exc))
                        return
                    if failure := harness_failure(episode):
                        harness_broke(row_id, failure)
                        return
                    episode.metadata["attempt"] = attempt
                    cut = cut_short(episode)
                    if plan.price and not episode.usage.get("cost"):
                        price_in, price_out = plan.price
                        tokens_in, tokens_out = episode.usage.get("input", 0), episode.usage.get("output", 0)
                        episode.usage["cost"] = (tokens_in * price_in + tokens_out * price_out) / 1e6
                        episode.usage["cost_estimated"] = True
                    leak = leaked(episode)
                    episode.metadata["reasoning_leak"] = leak
                    context = episode_context(episode, row, limits=limits, task=task, judge=plan.judge)
                    result = await task.rubrics.ascore(context, mode="eval")
                    writer.write(episode, result, harness.native_trace(episode), label=plan.label, attempt=attempt,
                                 cut_short=cut, **plan.extra)  # fmt: skip
                    state["spent"] += float(episode.usage.get("cost") or 0)
                    if result.passed:
                        state["passed"].add(row_id)
                    print(
                        f"[{plan.label} {index}/{len(by_row)}] {row_id} attempt={attempt} passed={result.passed} "
                        f"reward={result.reward} tools={episode.tool_calls} failure={episode.failure}"
                        + (f" cut_short={cut}" if cut else ""),
                        flush=True,
                    )
                    if leak:
                        print(f"  [{plan.label}] {row_id}: WARNING reasoning leaked into the visible content",
                              flush=True)  # fmt: skip
                    if checkpoint is not None:
                        checkpoint.maybe()
                    if leaks is not None:
                        try:
                            leaks.record(leak)
                        except TooManyErrors:
                            stopped.set()
                            raise
                    if errors is not None:
                        try:
                            errors.record(result)
                        except TooManyErrors:
                            stopped.set()  # rows still waiting for the semaphore do not start
                            raise
                    if cut and plan.rerun_cut_short and not result.passed:
                        return  # the rest of this row waits for the next pass, which reruns the cut episode

        await asyncio.gather(*(run_row(i, items) for i, items in enumerate(by_row.values(), 1)))
        if not broken.is_set():
            try:
                await harness.healthy(runner)
            except Exception as exc:
                harness_broke("-", f"health check failed: {type(exc).__name__}: {exc}")


class Checkpoint:
    """Uploads a stage's output folder every `every_seconds` while it runs, so a job that dies (or hits its
    timeout) keeps what it recorded and a rerun of the job resumes from it (`resume_from_hub`).

    The folder (and the separate raw-log folder, `raw_dir`) is copied first (writes happen on the event loop, so the
    copy is consistent) and the copies are uploaded in a thread, off the event loop, as `upload(folder, raw_folder)`.
    A failed checkpoint upload is reported and retried later, except a refused one (`PublishRefused`, or a
    `SystemExit` from the upload): that is kept in `error`, cancels the task bound with `bind` (the episode loop, so
    its episodes and harness stop at once) and is raised by `flush` and `maybe`."""

    def __init__(
        self,
        output_dir: Path,
        upload: Callable[[Path, Path | None], None],
        every_seconds: float,
        raw_dir: Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        self.upload = upload
        self.every_seconds = every_seconds
        self.last = time.monotonic()
        self.dirty = False
        self.task: asyncio.Task | None = None
        self.error: PublishRefused | None = None
        self._owner: asyncio.Task | None = None
        self._flushing = False

    def bind(self, owner: asyncio.Task) -> None:
        """Cancel `owner` when an upload is refused."""
        self._owner = owner

    def maybe(self) -> None:
        if self.error is not None:
            raise self.error
        self.dirty = True
        if self.every_seconds <= 0 or time.monotonic() - self.last < self.every_seconds:
            return
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self._upload())

    async def _upload(self) -> None:
        self.last = time.monotonic()
        self.dirty = False
        with tempfile.TemporaryDirectory(prefix="agent-distill-checkpoint-") as tmp:
            copy = Path(tmp) / "out"
            shutil.copytree(self.output_dir, copy)
            raw_copy = None
            if self.raw_dir is not None and self.raw_dir.exists():
                raw_copy = Path(tmp) / "raw-out"
                shutil.copytree(self.raw_dir, raw_copy)
            try:
                await asyncio.to_thread(self.upload, copy, raw_copy)
            except asyncio.CancelledError:
                raise
            except PublishRefused as exc:
                self._refused(exc)
            except Exception as exc:  # the final publish retries; a transient failure must not kill the stage
                self.dirty = True
                print(f"checkpoint upload failed ({type(exc).__name__}: {exc}); continuing", flush=True)
            except BaseException as exc:  # SystemExit from the upload thread: never let it escape the event loop
                self._refused(PublishRefused(str(exc) or type(exc).__name__))

    def _refused(self, error: PublishRefused) -> None:
        self.error = error
        print(f"checkpoint upload refused: {error}; stopping", flush=True)
        if self._owner is not None and not self._owner.done() and not self._flushing:
            self._owner.cancel(f"checkpoint upload refused: {error}")

    async def flush(self) -> None:
        """Wait for the upload in flight; raise the refusal if an upload was refused."""
        self._flushing = True
        try:
            if self.task is not None:
                await self.task
        finally:
            self._flushing = False
        if self.error is not None:
            raise self.error


def resume_from_hub(
    repo: str, output_dir: Path, prefix: str, *, raw_repo: str = "", raw_dir: Path | None = None
) -> int:
    """Download the records a previous run of this job checkpointed (episodes and sessions under `prefix` from
    `repo`, raw logs from the private `raw_repo`) into `output_dir` / `raw_dir`, so the stage resumes instead of
    starting over. Raw logs an older run left in `repo` itself go to `raw_dir` too. Returns the files fetched."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()
    fetched = 0
    for source, kinds in ((repo, ("episodes", "sessions", "raw")), (raw_repo, ("raw",))):
        if not source:
            continue
        try:
            files = api.list_repo_files(source, repo_type="dataset")
        except RepositoryNotFoundError:
            continue
        for name in files:
            kind = name.split("/", 1)[0]
            if kind not in kinds or not (name in (f"{kind}/{prefix}.jsonl",) or name.startswith(f"sessions/{prefix}/")):
                continue
            target = raw_dir if kind == "raw" else output_dir
            if target is None:
                continue
            hf_hub_download(source, name, repo_type="dataset", local_dir=str(target))
            fetched += 1
    if fetched:
        print(f"resumed {fetched} files of {prefix} from {repo}" + (f" and {raw_repo}" if raw_repo else ""), flush=True)
    return fetched


def shard(rows: list[dict], spec: str) -> list[dict]:
    """`i/n`: every n-th row starting at i (disjoint slices for parallel jobs)."""
    index, count = (int(value) for value in spec.split("/"))
    if not 0 <= index < count:
        raise ValueError(f"bad shard {spec!r}")
    return rows[index::count]
