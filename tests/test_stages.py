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
    write(writer, "3", passed=True, harness_version="1.2.4")
    accepted = load_accepted([writer.episodes], harness="opencode@2.3.2")
    assert [r["row_id"] for r in accepted] == ["1"] and accepted[0]["attempt"] == 1
    assert "opencode@1.2.4" in capsys.readouterr().out
    assert len(load_accepted([writer.episodes], harness="opencode@2.3.2", allow_mismatch=True)) == 2
    with pytest.raises(ValueError, match="no accepted"):
        load_accepted([writer.episodes], harness="pi@1.6.0")


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
