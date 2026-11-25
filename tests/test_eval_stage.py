"""The episode loop's harness-failure handling (eval, probe, traces), the eval stage's publishing, and `distill
publish`'s visibility rules (Hub mocked)."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from agent_distill import pipeline
from agent_distill.config import load_run
from agent_distill.episode import Episode, ToolEvent
from agent_distill.harnesses.base import Endpoint, HarnessError
from agent_distill.records import RecordWriter, read_jsonl
from agent_distill.rubrics import RubricResult
from agent_distill.stages import common
from agent_distill.stages import eval as eval_stage
from agent_distill.task import Limits

ENDPOINT = Endpoint(base_url="http://vllm/v1", model="policy")
NOT_FOUND = httpx.Request("GET", "https://huggingface.co/api/repo")


class Rubrics:
    async def ascore(self, context: Any, mode: str = "eval") -> RubricResult:
        return RubricResult(scores={"r": 1.0}, reward=1.0, passed=True, mode=mode)


TASK = SimpleNamespace(name="t", row_id=lambda row: row["id"], rubrics=Rubrics(), missing_commands=lambda: [])
ROWS = [{"id": str(i)} for i in range(1, 6)]


def episode(row_id: str, *, broken: bool = False) -> Episode:
    output = "posix_spawn failed: ENOENT" if broken else "ok"
    return Episode(task="t", row_id=row_id, harness="opencode", harness_version="2.3.2", model="local/policy",
                   messages=[{"role": "user", "content": "q"}],
                   tool_events=[ToolEvent(kind="shell", name="shell", input={}, success=not broken, output=output)],
                   session_id=f"s{row_id}", raw=[{"id": row_id}])  # fmt: skip


class FakeHarness:
    """Fails `failures` (row id -> list of "dead" / "spawn" outcomes, consumed in order)."""

    def __init__(self, failures: dict[str, list[str]] | None = None, *, start_failures: int = 0) -> None:
        self.failures = {key: list(value) for key, value in (failures or {}).items()}
        self.start_failures = start_failures
        self.sessions = 0
        self.runs: list[str] = []

    @asynccontextmanager
    async def session(self, endpoints, *, cache=None, task=None):
        self.sessions += 1
        if self.start_failures:
            self.start_failures -= 1
            raise HarnessError("OpenCode exited during startup with status 1")
        yield "runner"

    async def healthy(self, runner: Any) -> None:
        pass

    async def run_episode(self, runner, task, row, endpoint, limits):
        self.runs.append(row["id"])
        outcome = (self.failures.get(row["id"]) or ["ok"]).pop(0)
        if outcome == "dead":
            raise HarnessError("OpenCode stopped answering: ConnectError")
        return episode(row["id"], broken=outcome == "spawn")

    def native_trace(self, episode: Episode) -> None:
        return None


@pytest.fixture(autouse=True)
def plain_context(monkeypatch):
    monkeypatch.setattr(common, "episode_context", lambda episode, row, **kwargs: None)


def run_label(harness: FakeHarness, writer: RecordWriter, **kwargs: Any) -> None:
    kwargs = {"concurrency": 1, **kwargs}
    asyncio.run(common.run_episodes(harness, TASK, ROWS, ENDPOINT, writer, common.Plan(label="base"), limits=Limits(),
                                    cache=None, **kwargs))  # fmt: skip


def test_run_label_skips_harness_failures_and_restarts(tmp_path) -> None:
    harness = FakeHarness({"2": ["dead"], "4": ["spawn"]}, start_failures=1)
    writer = RecordWriter(tmp_path / "eval", "base/test")
    run_label(harness, writer)
    records = read_jsonl(writer.episodes)
    assert sorted(r["row_id"] for r in records) == ["1", "2", "3", "4", "5"]  # each recorded once, none broken
    assert all("posix_spawn" not in str(r["tool_events"]) for r in records)
    # session 1 does not start; 2 runs row 1 then breaks on 2; 3 runs 2, 3 and breaks on 4; 4 runs 4 and 5
    assert harness.sessions == 4 and harness.runs == ["1", "2", "2", "3", "4", "4", "5"]
    assert writer.done() == {("base", str(i)) for i in range(1, 6)}
    raws = read_jsonl(writer.raw)
    assert {r["row_id"] for r in raws} == {str(i) for i in range(1, 6)} and raws[0]["raw"] == [{"id": "1"}]


def test_run_label_gives_up_after_max_restarts(tmp_path) -> None:
    harness = FakeHarness({"1": ["dead"] * 10})
    writer = RecordWriter(tmp_path / "eval", "base/test")
    with pytest.raises(RuntimeError, match="kept failing"):
        run_label(harness, writer, max_restarts=2)
    assert harness.sessions == 3 and not writer.episodes.exists()


def test_run_once_reports_a_broken_harness(tmp_path) -> None:
    writer = RecordWriter(tmp_path / "eval", "base/test")

    def run_once(harness: FakeHarness) -> bool:
        work = [(row, 1) for row in ROWS]
        plan = common.Plan(label="base")
        return asyncio.run(common._run_once(harness, TASK, work, ENDPOINT, writer, plan, Limits(), 4, None)) == "broken"

    assert run_once(FakeHarness()) is False
    assert run_once(FakeHarness({"3": ["dead"]})) is True
    assert run_once(FakeHarness(start_failures=1)) is True

    class Unhealthy(FakeHarness):
        async def healthy(self, runner: Any) -> None:
            raise ConnectionError("gone")

    assert run_once(Unhealthy()) is True


# --------------------------------------------------------------------------------------------------
# Hub (mocked)
# --------------------------------------------------------------------------------------------------


class FakeHub:
    def __init__(self, repos: dict[str, dict[str, Any]] | None = None) -> None:
        self.repos = repos or {}  # repo -> {"private": bool, "files": [...]}
        self.calls: list[tuple] = []

    def repo_info(self, repo: str, repo_type: str = "model") -> Any:
        if repo not in self.repos:
            raise RepositoryNotFoundError(f"{repo} not found", response=httpx.Response(404, request=NOT_FOUND))
        return SimpleNamespace(private=self.repos[repo]["private"])

    def create_repo(self, repo: str, repo_type: str = "model", private: bool = False, exist_ok: bool = False) -> None:
        self.calls.append(("create_repo", repo, private))
        self.repos.setdefault(repo, {"private": private, "files": []})

    def list_repo_files(self, repo: str, repo_type: str = "model") -> list[str]:
        self.repo_info(repo)  # not found: RepositoryNotFoundError
        return list(self.repos[repo]["files"])

    def upload_folder(self, *, folder_path: str, repo_id: str, repo_type: str, commit_message: str,
                      ignore_patterns: list[str] | None = None) -> None:  # fmt: skip
        files = sorted(str(p.relative_to(folder_path)) for p in Path(folder_path).rglob("*") if p.is_file())
        files = [f for f in files if not any(f.startswith(i.rstrip("*")) for i in ignore_patterns or [])]
        self.calls.append(("upload_folder", repo_id, files))
        self.repos[repo_id]["files"] += files

    def add_collection_item(self, collection: str, *, item_id: str, item_type: str, note: str, exists_ok: bool):
        self.calls.append(("add_collection_item", item_id, note))

    def update_repo_settings(self, repo: str, *, repo_type: str, private: bool) -> None:
        self.calls.append(("update_repo_settings", repo, private))
        self.repos[repo]["private"] = private


@pytest.fixture
def hub(monkeypatch) -> FakeHub:
    import huggingface_hub

    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    return fake


def written(tmp_path: Path) -> tuple[Path, Path]:
    out = tmp_path / "eval"
    writer = RecordWriter(out, "base/test")
    writer.write(episode("1"), RubricResult(scores={"r": 1.0}, reward=1.0, passed=True), label="base")
    return out, writer.raw_root


def test_eval_publish_creates_private_repos_and_keeps_raw_logs_apart(tmp_path, hub) -> None:
    out, raw = written(tmp_path)
    eval_stage.publish(out, "org/run-eval", "opencode@2.3.2", "t", raw_dir=raw, raw_repo="org/run-eval-raw",
                       collection="org/collection", note="t, opencode@2.3.2: evaluation episodes")  # fmt: skip
    assert hub.repos["org/run-eval"]["private"] and hub.repos["org/run-eval-raw"]["private"]
    assert ("add_collection_item", "org/run-eval", "t, opencode@2.3.2: evaluation episodes") in hub.calls
    assert not any(call[0] == "add_collection_item" and call[1].endswith("-raw") for call in hub.calls)
    eval_files, raw_files = hub.repos["org/run-eval"]["files"], hub.repos["org/run-eval-raw"]["files"]
    assert not any(name.startswith("raw/") for name in eval_files)
    assert any(name.startswith("sessions/") for name in eval_files) and "README.md" in eval_files
    assert raw_files == ["raw/base/test.jsonl"]
    assert "raw/" not in (out / "README.md").read_text().split("never published")[0].split("summaries")[1]


@pytest.mark.parametrize("public", ["org/run-eval", "org/run-eval-raw"])
def test_eval_publish_refuses_a_public_repo(tmp_path, hub, public) -> None:
    out, raw = written(tmp_path)
    hub.repos[public] = {"private": False, "files": []}
    with pytest.raises(common.PublishRefused, match="is public"):
        eval_stage.publish(out, "org/run-eval", "opencode@2.3.2", "t", raw_dir=raw, raw_repo="org/run-eval-raw")
    assert not any(call[0] == "upload_folder" and call[1] == public for call in hub.calls)
    assert hub.repos[public]["private"] is False  # never changed by the eval stage


def test_eval_main_publishes_what_finished_when_a_label_fails(tmp_path, monkeypatch) -> None:
    published: list[tuple] = []

    async def failing_run_label(harness, task, rows, endpoint, writer, plan, **kwargs):
        writer.write(episode("1"), RubricResult(scores={"r": 1.0}, reward=1.0, passed=True), label=plan.label)
        raise RuntimeError("the harness kept failing")

    monkeypatch.setattr(eval_stage, "select_rows", lambda task, split, **kwargs: [{"arxiv_id": "1"}])
    monkeypatch.setattr(eval_stage, "run_episodes", failing_run_label)
    monkeypatch.setattr(eval_stage, "publish", lambda *args, **kwargs: published.append((args, kwargs)))
    monkeypatch.setattr(eval_stage, "refuse_public", lambda *repos: None)
    argv = ["--task", "examples/paper-urls", "--harness", "opencode@2.3.2", "--student", "qwen3.5-9b", "--model",
            "Qwen/Qwen3.5-9B", "--split", "test", "--no-serve", "--output-dir", str(tmp_path / "eval"),
            "--output-repo", "org/run-eval", "--answer-retry", "--cache", str(tmp_path / "cache")]  # fmt: skip
    with pytest.raises(RuntimeError, match="kept failing"):
        eval_stage.main(argv)
    ((args, kwargs),) = published
    assert args[:2] == (tmp_path / "eval", "org/run-eval") and kwargs["raw_repo"] == "org/run-eval-raw"
    assert kwargs["raw_dir"] == tmp_path / "eval-raw"


class BlockingHarness(FakeHarness):
    """Row "1" finishes at once; every other episode runs until it is cancelled (and records that it was stopped,
    like a harness terminating its subprocess)."""

    def __init__(self) -> None:
        super().__init__()
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.closed = 0

    @asynccontextmanager
    async def session(self, endpoints, *, cache=None, task=None):
        self.sessions += 1
        try:
            yield "runner"
        finally:
            self.closed += 1

    async def run_episode(self, runner, task, row, endpoint, limits):
        if row["id"] == "1":
            return episode("1")
        self.started.append(row["id"])
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.append(row["id"])


def finishes(target, timeout: float = 15.0) -> BaseException | None:
    """Run `target` in a daemon thread (a hang fails the test instead of hanging it); return what it raised."""
    outcome: dict[str, BaseException | None] = {}

    def run() -> None:
        try:
            target()
            outcome["raised"] = None
        except BaseException as exc:  # SystemExit included: that is the bug
            outcome["raised"] = exc

    thread = threading.Thread(target=run, daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"still running after {timeout:g}s"
    assert time.monotonic() - started < timeout
    return outcome["raised"]


@pytest.mark.parametrize("refusal", ["public-repo", "system-exit"])
def test_a_refused_checkpoint_upload_stops_the_episode_loop(tmp_path, hub, refusal) -> None:
    out = tmp_path / "eval"
    writer = RecordWriter(out, "base/test")
    hub.repos["org/run-eval"] = {"private": False, "files": []}
    if refusal == "public-repo":  # the real path: publish() -> private_dataset() refuses the public repo
        checkpoint = eval_stage.checkpoint(out, "org/run-eval", 1e-9, "opencode@2.3.2", "t", raw_dir=writer.raw_root,
                                           raw_repo="org/run-eval-raw")  # fmt: skip
    else:  # any SystemExit an upload raises in its thread is turned into a refusal, never escapes the loop

        def upload(folder: Path, raw_folder: Path | None) -> None:
            raise SystemExit("org/run-eval is public: refusing to upload unscanned episodes into it")

        checkpoint = common.Checkpoint(out, upload, 1e-9, raw_dir=writer.raw_root)
    harness = BlockingHarness()
    threads = set(threading.enumerate())

    raised = finishes(lambda: run_label(harness, writer, checkpoint=checkpoint, concurrency=4))

    assert isinstance(raised, common.PublishRefused) and "is public" in str(raised)
    assert harness.started and sorted(harness.stopped) == sorted(harness.started)  # every running episode stopped
    assert harness.closed == harness.sessions == 1  # the harness session was closed, and not restarted
    assert [r["row_id"] for r in read_jsonl(writer.episodes)] == ["1"]
    assert not any(call[0] == "upload_folder" and call[1] == "org/run-eval" for call in hub.calls)
    assert hub.repos["org/run-eval"]["private"] is False
    assert set(threading.enumerate()) <= threads  # no executor or harness thread left behind


def test_eval_main_exits_with_an_error_when_a_checkpoint_upload_is_refused(tmp_path, monkeypatch, hub) -> None:
    hub.repos["org/run-eval"] = {"private": False, "files": []}
    harness = BlockingHarness()
    task = SimpleNamespace(**vars(TASK), dataset=SimpleNamespace(repo="org/data"), judge="")
    monkeypatch.setattr(eval_stage, "load_task", lambda path: task)
    monkeypatch.setattr(eval_stage, "load_harness", lambda profile, **kwargs: harness)
    monkeypatch.setattr(eval_stage, "select_rows", lambda task, split, **kwargs: ROWS)
    monkeypatch.setattr(eval_stage, "refuse_public", lambda *repos: None)  # made public after the startup check
    argv = ["--task", "examples/paper-urls", "--harness", "opencode@2.3.2", "--student", "qwen3.5-9b", "--model",
            "Qwen/Qwen3.5-9B", "--split", "test", "--no-serve", "--output-dir", str(tmp_path / "eval"),
            "--output-repo", "org/run-eval", "--upload-every", "1e-9", "--concurrency", "4",
            "--cache", str(tmp_path / "cache")]  # fmt: skip

    raised = finishes(lambda: eval_stage.main(argv))

    assert isinstance(raised, common.PublishRefused) and "is public" in str(raised)  # a non-zero exit, not a hang
    assert harness.started and sorted(harness.stopped) == sorted(harness.started)
    assert not any(call[0] == "upload_folder" and call[1] == "org/run-eval" for call in hub.calls)


@pytest.mark.parametrize("stage", ["eval", "traces"])
@pytest.mark.parametrize("public", ["org/run-eval", "org/run-eval-raw"])
def test_episode_stages_refuse_a_public_dataset_before_any_episode(tmp_path, monkeypatch, hub, stage, public) -> None:
    from agent_distill.stages import traces as traces_stage

    hub.repos[public] = {"private": False, "files": []}
    hub.repos.setdefault("org/run-eval", {"private": True, "files": []})
    module = eval_stage if stage == "eval" else traces_stage
    harness = BlockingHarness()
    task = SimpleNamespace(**vars(TASK), dataset=SimpleNamespace(repo="org/data"), judge="", limits=Limits())
    monkeypatch.setattr(module, "load_task", lambda path: task)
    monkeypatch.setattr(module, "load_harness", lambda profile, **kwargs: harness)
    rows = []
    monkeypatch.setattr(eval_stage, "select_rows", lambda *args, **kwargs: rows.append(1) or ROWS)
    if stage == "traces":
        monkeypatch.setattr(traces_stage, "trace_rows", lambda *args, **kwargs: rows.append(1) or ROWS)
    common_args = ["--task", "examples/paper-urls", "--harness", "opencode@2.3.2", "--split", "test",
                   "--output-dir", str(tmp_path / "out"), "--output-repo", "org/run-eval",
                   "--cache", str(tmp_path / "cache")]  # fmt: skip
    argv = (["--student", "qwen3.5-9b", "--model", "Qwen/Qwen3.5-9B", "--no-serve"] if stage == "eval"
            else ["--teacher", "kimi-k3.1-pro"]) + common_args  # fmt: skip

    with pytest.raises(common.PublishRefused, match=f"{public} is public"):
        module.main(argv)
    assert not rows and harness.sessions == 0  # no rows loaded, no harness started, nothing spent
    assert not any(call[0] in ("upload_folder", "create_repo") for call in hub.calls)


def test_refuse_public_allows_missing_and_private_repos(hub) -> None:
    hub.repos["org/private"] = {"private": True, "files": []}
    eval_stage.refuse_public("org/private", "org/missing", "")
    assert hub.calls == []  # read-only


RUN_TOML = Path("examples/paper-urls/run.toml")


def make_run(**overrides: Any) -> pipeline.Run:
    return pipeline.make_run(load_run(RUN_TOML, overrides))


@pytest.fixture
def no_scan(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "credential_files", lambda api, repo, files: [])


def test_distill_publish_adds_to_the_collection_before_making_public(hub, no_scan) -> None:
    run = make_run()
    hub.repos[run.repo("eval")] = {"private": True, "files": ["sessions/a/b.jsonl", "episodes/a.jsonl"]}
    hub.repos[run.repo("eval-raw")] = {"private": True, "files": ["raw/a.jsonl"]}
    pipeline.publish(run, dry_run=False)
    names = [call[0] for call in hub.calls]
    assert names == ["add_collection_item", "update_repo_settings"]
    assert not hub.repos[run.repo("eval")]["private"] and hub.repos[run.repo("eval-raw")]["private"]


def test_distill_publish_never_publishes_raw_logs(hub, no_scan) -> None:
    run = make_run()
    hub.repos[run.repo("eval")] = {"private": True, "files": ["sessions/a.jsonl", "raw/base/test.jsonl"]}
    with pytest.raises(SystemExit, match="raw harness logs"):
        pipeline.publish(run, dry_run=False)
    hub.repos[run.repo("eval")]["files"] = ["sessions/a.jsonl"]
    hub.repos[run.repo("eval-raw")] = {"private": False, "files": ["raw/a.jsonl"]}
    with pytest.raises(SystemExit, match="must stay private"):
        pipeline.publish(run, dry_run=False)
    assert hub.calls == []  # nothing changed, not even the public raw repo's visibility
    hub.repos[run.repo("eval-raw")]["private"] = True
    run = make_run(**{"repos": {"eval": "org/sneaky-raw"}})
    with pytest.raises(SystemExit, match="never published"):
        pipeline.publish(run, dry_run=False)


def test_eval_jobs_install_locked_requirements() -> None:
    run = make_run(**{"eval.answer_retry": True})
    job = pipeline.eval_jobs(run, "REV")[0]
    lines = job.command.splitlines()
    locked = lines.index("uv pip install -q --python /work/venv/bin/python --require-hashes -r /work/requirements.txt")
    wheel = lines.index('uv pip install -q --python /work/venv/bin/python --no-deps "$WHEEL"')
    runtime = next(i for i, line in enumerate(lines) if "vllm==0.33.0" in line)
    # The runtime stack is resolved with the wheel, not under the lock's constraints (its pins win, e.g. a
    # huggingface-hub below 2.0 for transformers 5.14).
    assert locked < wheel < runtime and '"$WHEEL"' in lines[runtime] and "constraints" not in lines[runtime]
    assert lines[runtime + 1].startswith("/work/venv/bin/python -m agent_distill.stages.eval ")
    assert "--answer-retry" in job.command and "uv run" not in job.command
    assert f"--raw-repo {run.repo('eval-raw')}" in job.command
    assert "--collection nielsr/agent-distillation-" in job.command


def test_export_requirements_is_hash_pinned(tmp_path) -> None:
    import shutil

    from agent_distill import jobs

    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    requirements, constraints = jobs.export_requirements(tmp_path)
    text = requirements.read_text()
    assert "--hash=sha256:" in text and "datasets==" in text and "huggingface-hub==" in text
    assert "agent-distill==" not in text and "-e " not in text and "pytest" not in text  # no project, no dev
    assert "--hash" not in constraints.read_text() and "datasets==" in constraints.read_text()


def test_dry_run_prints_staging_and_residual_risk(monkeypatch, capsys) -> None:
    from agent_distill import jobs

    monkeypatch.setattr(jobs, "hardware_prices", lambda: {"h200": 0.08})
    assert pipeline.main([str(RUN_TOML), "--stages", "eval", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "requirements.txt" in out and "hash-pinned" in out and "residual risk" in out
    assert "/proc/<pid>/environ" in out and "--require-hashes" in out


def test_answer_retry_setting_is_in_eval_settings() -> None:
    field_names = {field.name for field in dataclasses.fields(load_run(RUN_TOML).eval)}
    assert "answer_retry" in field_names


def test_run_label_stops_on_too_many_rubric_errors(tmp_path, monkeypatch) -> None:
    from agent_distill.rubrics import ErrorBudget, RubricResult, TooManyErrors

    async def failing(context, mode="eval"):
        return RubricResult(scores={"j": None}, reward=None, passed=False, errors={"j": "HTTP 402"})

    monkeypatch.setattr(TASK.rubrics, "ascore", failing)
    writer = RecordWriter(tmp_path / "eval", "base/test")
    with pytest.raises(TooManyErrors, match="HTTP 402"):
        run_label(FakeHarness({}), writer, errors=ErrorBudget(max_rate=0.5, min_episodes=2))
    assert len(read_jsonl(writer.episodes)) == 2  # recorded (as unscorable) before the stop


@pytest.mark.skipif(not os.environ.get("AGENT_DISTILL_RESOLVE_JOB_DEPS"), reason="set AGENT_DISTILL_RESOLVE_JOB_DEPS=1")
def test_every_job_stack_resolves_on_the_locked_environment(tmp_path) -> None:
    """Opt-in (network): every stage's runtime stack, for every built-in student, installs on top of the locked
    job environment the way jobs install it (wheel + stack, no lock constraints)."""
    import subprocess

    from agent_distill import jobs
    from agent_distill.config import load_profile

    requirements, _ = jobs.export_requirements(tmp_path)
    wheel = jobs.build_wheel(tmp_path)
    python = tmp_path / "venv" / "bin" / "python"
    subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(tmp_path / "venv")], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), "--require-hashes", "-r", str(requirements)],
                   check=True)  # fmt: skip
    stacks = {"sft": pipeline.SFT_DEPS, "merge": pipeline.MERGE_DEPS, "grpo": pipeline.GRPO_DEPS}
    for name in ("qwen3.5-4b", "qwen3.5-9b", "inkling-2-mini"):
        student = load_profile("students", name)
        serve = list(student.serve.deps) + (
            [] if student.serve.vllm == "image" else [f"vllm=={student.serve.vllm_version}"]
        )
        stacks[f"{name}:serve"] = serve
        for stage in ("sft", "merge", "grpo"):
            if getattr(student, stage).deps:
                stacks[f"{name}:{stage}"] = list(getattr(student, stage).deps)
    failed = {}
    for label, deps in stacks.items():
        if not deps:
            continue
        result = subprocess.run(["uv", "pip", "install", "--dry-run", "--python", str(python), str(wheel), *deps],
                                capture_output=True, text=True)  # fmt: skip
        if result.returncode:
            failed[label] = result.stderr[-500:]
    assert not failed, failed
