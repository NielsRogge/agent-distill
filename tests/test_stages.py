from __future__ import annotations

from pathlib import Path

import pytest

from agent_distill import pipeline
from agent_distill.config import load_profile, load_run
from agent_distill.episode import Episode, ToolEvent
from agent_distill.harnesses.base import teacher_endpoint
from agent_distill.records import RecordWriter, read_jsonl
from agent_distill.rubrics import RubricResult
from agent_distill.stages.common import Plan, pending_work, progress, shard
from agent_distill.stages.eval import difficulty
from agent_distill.stages.sft import load_accepted, trajectory
from agent_distill.task import load_task

TASK = load_task("examples/paper-urls")
SMOKE = Path("examples/paper-urls/run-smoke.toml")


def write(writer: RecordWriter, row_id: str, passed: bool, label: str = "teacher", attempt: int = 1, cost=0.01,
          harness_version: str = "2.3.2") -> None:  # fmt: skip
    episode = Episode(task="paper-urls", row_id=row_id, harness="opencode", harness_version=harness_version,
                      model="m", usage={"cost": cost}, session_id=f"{row_id}-{attempt}",
                      messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
                                {"role": "assistant", "content": "", "timestamp": 1, "usage": {},
                                 "tool_calls": [{"id": "c1", "type": "function",
                                                 "function": {"name": "StructuredOutput", "arguments": "{}"}}]},
                                {"role": "tool", "tool_call_id": "c1", "content": "ok", "is_error": False},
                                {"role": "assistant", "content": "after the answer"}],
                      tool_events=[ToolEvent(kind="submit", name="StructuredOutput")])  # fmt: skip
    episode.metadata["attempt"] = attempt
    writer.write(episode, RubricResult(scores={}, reward=float(passed), passed=passed), label=label, attempt=attempt)


def rows(*ids: str) -> list[dict]:
    return [{"paper_id": int(i)} for i in ids]


def test_progress_and_pending_work(tmp_path):
    writer = RecordWriter(tmp_path, "teacher/train")
    write(writer, "1", passed=False, attempt=1)
    write(writer, "1", passed=True, attempt=2)
    write(writer, "2", passed=False, attempt=1)
    attempts, passed, spent = progress(writer, "teacher")
    assert attempts == {"1": 2, "2": 1} and passed == {"1"} and spent == pytest.approx(0.03)
    traces = Plan(label="teacher", attempts=3, stop_when_passed=True)
    assert [(r["paper_id"], a) for r, a in pending_work(TASK, rows("1", "2", "3"), writer, traces)] == [
        (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)
    ]  # fmt: skip
    probe = Plan(label="teacher", attempts=2)
    assert [(r["paper_id"], a) for r, a in pending_work(TASK, rows("1", "2"), writer, probe)] == [(2, 2)]
    assert shard(rows("1", "2", "3", "4", "5"), "1/2") == rows("2", "4")
    with pytest.raises(ValueError):
        shard([], "2/2")


def test_difficulty_qualifies_sometimes_solved_rows():
    outcomes = [("a", True), ("a", True), ("b", True), ("b", False), ("c", False), ("c", False), ("d", True)]
    records = [{"row_id": row, "passed": passed} for row, passed in outcomes]
    result = difficulty(records, attempts=2)
    assert result["qualified"] == ["b"] and result["rows"] == 3
    assert result["passes_histogram"] == {"0": 1, "1": 1, "2": 1}


def test_sft_reads_accepted_records_of_one_harness(tmp_path, capsys):
    writer = RecordWriter(tmp_path, "teacher/train")
    write(writer, "1", passed=True)
    write(writer, "1", passed=True, attempt=2)  # a second accepted attempt of the same row: first one wins
    write(writer, "2", passed=False)
    accepted = load_accepted([writer.episodes], harness="opencode@2.3.2")
    assert [r["row_id"] for r in accepted] == ["1"] and accepted[0]["attempt"] == 1
    failed = RecordWriter(tmp_path, "failed")
    write(failed, "2", passed=False)
    with pytest.raises(ValueError, match="no accepted"):
        load_accepted([failed.episodes], harness="opencode@2.3.2")
    # DESIGN.md §6: traces of another harness@version refuse SFT unless --allow-harness-mismatch (still warned)
    write(writer, "3", passed=True, harness_version="1.2.4")
    with pytest.raises(SystemExit, match=r"opencode@1\.2\.4.*--allow-harness-mismatch"):
        load_accepted([writer.episodes], harness="opencode@2.3.2")
    assert len(load_accepted([writer.episodes], harness="opencode@2.3.2", allow_mismatch=True)) == 2
    assert "WARNING" in capsys.readouterr().out


def test_sft_trajectory_stops_at_the_submission_and_strips_metadata(tmp_path):
    writer = RecordWriter(tmp_path, "x")
    write(writer, "1", passed=True)
    (record,) = read_jsonl(writer.episodes)
    messages = trajectory(record, keep_reasoning=False)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert all(set(m) <= {"role", "content", "tool_calls", "tool_call_id"} for m in messages)
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {}


def test_teacher_endpoint_for_opencode():
    teacher = load_profile("teachers", "inkling-2-small")
    endpoint = teacher_endpoint(teacher, load_profile("harnesses", "opencode@2.3.2"), "huggingface")
    assert endpoint.native_id == "huggingface/thinkingmachines/Inkling-2-Small:deepinfra"
    assert endpoint.key_env == "HF_TOKEN" and endpoint.headers == {"X-HF-Bill-To": "huggingface"}
    assert endpoint.native_entry["limit"]["context"] == 131072


def smoke_run(**overrides):
    return pipeline.make_run(load_run(SMOKE, overrides))


def test_smoke_run_jobs():
    run = smoke_run()
    (traces,) = pipeline.trace_jobs(run, "R")
    assert "--teacher glm-5.3" in traces.command and "--bill-to huggingface" in traces.command
    assert "--shard 0/1" in traces.command and "--limit 8" in traces.command and "--budget-usd 2.0000" in traces.command
    assert traces.flavor == "cpu-upgrade"
    (sft,) = pipeline.sft_jobs(run, "R")
    assert "--traces-repo nielsr/qwen35-4b-paper-urls-glm53-opencode-232-smoke-traces" in sft.command
    assert "--max-steps 2" in sft.command and "trl==1.9.2" in sft.command and sft.timeout == "1h"
    (merge,) = pipeline.merge_jobs(run, "R")
    assert "--adapter nielsr/qwen35-4b-paper-urls-glm53-opencode-232-smoke-lora" in merge.command
    assert "--hub-model-id nielsr/qwen35-4b-paper-urls-glm53-opencode-232-smoke-sft-merged" in merge.command
    (probe,) = pipeline.probe_jobs(run, "R")
    assert (
        "--attempts 2" in probe.command and "--temperature 0.8" in probe.command and "--purpose train" in probe.command
    )
    assert "--model nielsr/qwen35-4b-paper-urls-glm53-opencode-232-smoke-sft-merged" in probe.command
    evals = pipeline.eval_jobs(run, "R")
    assert [job.name.rsplit("-", 2)[-2] for job in evals] == ["base", "sft", "grpo", "teacher"]
    assert "--adapter nielsr/qwen35-4b-paper-urls-glm53-opencode-232-smoke-lora --adapter-label sft" in evals[1].command
    assert "--label teacher" in evals[3].command and "--no-stop-when-passed" in evals[3].command
    for job in [traces, sft, merge, probe, *evals]:
        assert "HF_TOKEN" not in job.command


def test_eval_models_variants():
    run = smoke_run(**{"eval.variants": ["grpo"]})
    assert pipeline.eval_models(run) == {
        "grpo": "nielsr/qwen35-4b-paper-urls-glm53-opencode-232-smoke-sft-merged+"
        "nielsr/qwen35-4b-paper-urls-glm53-opencode-232-smoke-grpo-lora"
    }


def test_artifacts_cover_every_stage():
    artifacts = pipeline.artifacts(smoke_run())
    suffixes = [repo.removeprefix("nielsr/qwen35-4b-paper-urls-glm53-opencode-232-smoke-") for repo, _, _ in artifacts]
    assert suffixes == ["traces", "lora", "sft-merged", "grpo-lora", "grpo-merged", "eval"]
    assert all("opencode@2.3.2" in note for _, _, note in artifacts)


class ScriptedHarness:
    """A harness whose episodes follow a script per row: "cut" (timeout mid-tool-call), "fail" or "pass"."""

    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = {row: list(outcomes) for row, outcomes in script.items()}
        self.runs: list[str] = []

    def session(self, endpoints, cache, task):
        import contextlib

        @contextlib.asynccontextmanager
        async def runner():
            yield None

        return runner()

    async def healthy(self, runner) -> None:
        return None

    def native_trace(self, episode):
        return None

    async def run_episode(self, runner, task, row, endpoint, limits):
        row_id = task.row_id(row)
        outcome = self.script[row_id].pop(0)
        self.runs.append(f"{row_id}:{outcome}")
        call = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "c1", "content": "tool running", "is_error": True}]  # fmt: skip
        return Episode(task="paper-urls", row_id=row_id, harness="opencode", harness_version="2.3.2", model="m",
                       messages=messages, usage={"cost": 0.01},
                       failure="episode timed out after 1 seconds" if outcome == "cut" else None,
                       answer=None)  # fmt: skip


def test_cut_short_episodes_are_rerun_without_using_an_attempt(tmp_path, monkeypatch):
    import asyncio

    from agent_distill.stages.common import cut_short, run_episodes
    from agent_distill.task import Limits

    harness = ScriptedHarness({"1": ["cut", "cut", "cut", "fail"], "2": ["cut", "pass"]})

    async def ascore(episode, mode="eval"):  # the outcome of the row's latest scripted run
        last = [run for run in harness.runs if run.startswith(f"{episode.row_id}:")][-1].endswith(":pass")
        return RubricResult(scores={}, reward=float(last), passed=last)

    monkeypatch.setattr(TASK.rubrics, "ascore", ascore)
    monkeypatch.setattr("agent_distill.stages.common.episode_context", lambda episode, row, **kw: episode)
    writer = RecordWriter(tmp_path, "teacher/train")
    plan = Plan(label="teacher", attempts=1, stop_when_passed=True, rerun_cut_short=2)
    asyncio.run(run_episodes(harness, TASK, rows("1", "2"), None, writer, plan, limits=Limits(), concurrency=2,
                             cache=tmp_path))  # fmt: skip
    # row 1: two cuts are rerun, the third cut uses the one attempt; row 2: one cut, then a pass
    assert sorted(harness.runs) == ["1:cut", "1:cut", "1:cut", "2:cut", "2:pass"]
    records = read_jsonl(writer.episodes)
    assert sum(1 for r in records if r["cut_short"]) == 4
    assert records[0]["cut_short"].startswith("mid-tool-call: episode timed out")
    assert len(list(writer.sessions.glob("*.jsonl"))) == len(records)  # reruns get their own session files
    attempts, passed, spent = progress(writer, "teacher", rerun_cut_short=2)
    assert attempts == {"1": 1, "2": 1} and passed == {"2"} and spent == pytest.approx(0.05)
    assert progress(writer, "teacher")[0] == {"1": 3, "2": 2}  # without reruns every cut uses an attempt
    tool_limit = Episode(task="t", row_id="1", harness="opencode", harness_version="2.3.2", model="m",
                         failure="tool-call limit reached (15)")  # fmt: skip
    assert cut_short(tool_limit) is None
    truncated = Episode(task="t", row_id="1", harness="pi", harness_version="1.6.0", model="m",
                        failure="response truncated at the output token limit")  # fmt: skip
    assert cut_short(truncated) is None


def test_checkpoint_uploads_a_consistent_copy(tmp_path):
    import asyncio

    from agent_distill.stages.common import Checkpoint

    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a.jsonl").write_text("1\n")
    uploads: list[list[str]] = []

    def upload(folder, raw_folder=None):
        uploads.append(sorted(p.name for p in folder.iterdir()))

    async def main():
        checkpoint = Checkpoint(tmp_path / "out", upload, every_seconds=0.01)
        checkpoint.maybe()  # too early
        await asyncio.sleep(0.02)
        checkpoint.maybe()
        await checkpoint.flush()
        checkpoint.every_seconds = 0
        checkpoint.maybe()  # 0: only the final publish uploads
        await checkpoint.flush()

    asyncio.run(main())
    assert uploads == [["a.jsonl"]]


def test_episodes_upload_before_raw_logs_and_raw_logs_can_stay_local(tmp_path, monkeypatch):
    from agent_distill.stages import eval as eval_stage

    calls = []

    class FakeApi:
        def list_repo_files(self, repo, repo_type):
            return ["README.md"]

        def upload_folder(self, folder_path, repo_id, repo_type, commit_message, ignore_patterns=None):
            calls.append(repo_id)
            if repo_id.endswith("-raw"):
                raise RuntimeError("Private storage limit reached")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(eval_stage, "private_dataset", lambda api, repo: False)
    raw = tmp_path / "raw"
    (raw / "base").mkdir(parents=True)
    (raw / "base" / "test.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="Private storage"):
        eval_stage.publish(tmp_path / "out", "org/run-eval", "pi@1.6.0", "t", raw_dir=raw, raw_repo="org/run-eval-raw")
    assert calls == ["org/run-eval", "org/run-eval-raw"]  # the episodes were already uploaded
    calls.clear()
    eval_stage.publish(tmp_path / "out", "org/run-eval", "pi@1.6.0", "t", raw_dir=None, raw_repo="")
    assert calls == ["org/run-eval"]


def test_raw_logs_off_passes_no_raw_upload():
    run = pipeline.make_run(load_run(SMOKE, {"hub.raw_logs": False}))
    args = pipeline.output_args(run, "eval")
    assert "--no-raw-upload" in args and "--raw-repo" not in args
    assert "--raw-repo" in pipeline.output_args(pipeline.make_run(load_run(SMOKE)), "traces")
    from agent_distill.stages.eval import parse_args as eval_args
    from agent_distill.stages.traces import parse_args as traces_args

    assert eval_args(["--task", "t", "--harness", "pi@1.6.0", "--student", "s", "--model", "m",
                      "--split", "test", "--no-raw-upload"]).no_raw_upload  # fmt: skip
    assert traces_args(
        ["--task", "t", "--harness", "pi@1.6.0", "--teacher", "glm-5.3", "--no-raw-upload"]
    ).no_raw_upload


def test_probe_jobs_probe_the_whole_sample_not_the_eval_default_limit():
    run = pipeline.make_run(load_run(SMOKE, {"probe.sample": 1162, "probe.shards": 6}))
    built = pipeline.probe_jobs(run, "rev")
    assert len(built) == 6
    for job in built:
        command = job.command
        assert "--sample 1162" in command and "--limit 1162" in command  # select_rows: sample, shard, then limit


def test_an_episode_stage_refuses_to_start_without_the_tasks_commands(tmp_path):
    import asyncio
    import dataclasses

    import pytest

    from agent_distill.stages.common import run_episodes
    from agent_distill.task import Limits

    task = dataclasses.replace(TASK, commands=("sh", "no-such-command-xyz"))
    assert task.missing_commands() == ["no-such-command-xyz"]
    harness = ScriptedHarness({"1": ["pass"]})
    plan = Plan(label="student", attempts=1, stop_when_passed=True)
    with pytest.raises(RuntimeError, match="no-such-command-xyz"):
        asyncio.run(run_episodes(harness, task, rows("1"), None, RecordWriter(tmp_path, "student/test"), plan,
                                 limits=Limits(), concurrency=1, cache=tmp_path))  # fmt: skip
    assert harness.runs == []  # no episode ran
