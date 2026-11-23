from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_distill import jobs, pipeline
from agent_distill.config import available_profiles, load_profile, load_run, slugify
from agent_distill.episode import Episode, ToolEvent
from agent_distill.records import RecordWriter, read_jsonl, summarize_records
from agent_distill.rubrics import RubricResult

RUN_TOML = Path("examples/paper-urls/run.toml")


def test_builtin_profiles_load():
    for kind in ("teachers", "students", "harnesses"):
        for name in available_profiles(kind):
            load_profile(kind, name)
    v1, v2 = load_profile("harnesses", "opencode@1.2.4"), load_profile("harnesses", "opencode@2.3.2")
    assert (v1.answer_mode, v2.answer_mode) == ("native", "final_json")
    assert set(v1.download.sha256) == {"x64", "arm64"} == set(v2.download.sha256)
    assert load_profile("teachers", "inkling-2-small").opencode_id == (
        "huggingface/thinkingmachines/Inkling-2-Small:deepinfra"
    )
    with pytest.raises(FileNotFoundError, match="available"):
        load_profile("students", "nope")


def test_project_profiles_shadow_builtins(tmp_path):
    folder = tmp_path / "students"
    folder.mkdir()
    (folder / "qwen3.5-9b.toml").write_text('name = "qwen3.5-9b"\nmodel = "me/my-9b"\n')
    assert load_profile("students", "qwen3.5-9b", [tmp_path]).model == "me/my-9b"
    with pytest.raises(ValueError, match="unknown keys"):
        (folder / "bad.toml").write_text('name = "bad"\nmodel = "x"\ntypo = 1\n')
        load_profile("students", "bad", [tmp_path])


def test_run_config_and_overrides():
    config = load_run(RUN_TOML, {"eval.splits": ["validation"], "harness": "opencode@2.3.2"})
    assert config.eval.splits == ["validation"] and config.harness == "opencode@2.3.2"
    assert config.path(config.task).resolve() == RUN_TOML.parent.resolve()
    assert config.hub.bill_to == "huggingface"


def make_run(**overrides):
    config = load_run(RUN_TOML, overrides)
    return pipeline.make_run(config)


def test_eval_jobs_dry_run():
    run = make_run()
    built = pipeline.eval_jobs(run, "REV")
    assert [job.name for job in built] == [
        "qwen35-9b-paper-urls-parity-opencode-124-eval-base-test",
        "qwen35-9b-paper-urls-parity-opencode-124-eval-grpo-test",
    ]
    command = built[1].command
    assert "--harness opencode@1.2.4" in command and "--model nielsr/qwen35-9b-url-agent-grpo-merged-kimi" in command
    assert "--label grpo" in command and "vllm==0.33.0" in command
    assert "code/REV/" in command and "uv tool install -q 'huggingface-hub>=1.13'" in command  # task job_setup
    assert "--output-repo nielsr/qwen35-9b-paper-urls-parity-opencode-124-eval" in command
    assert "HF_TOKEN" not in command  # the token is a job secret, never in the command


def test_adapter_spec_and_default_name():
    run = make_run(**{"eval.models": {"sft": "Qwen/Qwen3.5-9B+org/lora"}, "name": ""})
    assert run.name == slugify("qwen3.5-9b-paper-urls-kimi-k3.1-pro-opencode@1.2.4")
    (job,) = pipeline.eval_jobs(run, "R")
    assert "--adapter org/lora --adapter-label sft --adapter-only" in job.command


def test_budget_guard(capsys):
    run = make_run(**{"budget.gpu_usd": 1})
    built = pipeline.eval_jobs(run, "R")
    prices = {"h200": 0.083333}
    assert jobs.estimate(built, prices)["worst_case_usd"] == pytest.approx(40.0, rel=1e-3)
    original = jobs.estimate
    jobs.estimate = lambda built: original(built, prices)  # no Hub call in tests
    try:
        with pytest.raises(SystemExit, match="exceeds"):
            pipeline.check_budget(run, built, over_budget=False)
        pipeline.check_budget(run, built, over_budget=True)
    finally:
        jobs.estimate = original


def test_overrides_from_cli():
    argv = ["x.toml", "--stages", "eval", "--splits", "test,validation", "--limit", "5"]
    argv += ["--eval-model", "a=m1", "--eval-model", "b=m2+ad", "--set", "eval.concurrency=4"]
    args = pipeline.parse_args(argv)
    values = pipeline.overrides(args)
    assert values == {"eval.concurrency": 4, "stages": ["eval"], "eval.splits": ["test", "validation"],
                      "eval.limit": 5, "eval.models": {"a": "m1", "b": "m2+ad"}}  # fmt: skip


def test_unimplemented_stages_refuse(tmp_path):
    with pytest.raises(SystemExit, match="M3"):
        pipeline.main([str(RUN_TOML), "--stages", "sft", "--dry-run"])


def test_timeout_minutes():
    assert jobs.timeout_minutes("4h") == 240 and jobs.timeout_minutes("90m") == 90
    assert jobs.timeout_minutes("1d") == 1440


def episode(row_id: str) -> Episode:
    return Episode(task="t", row_id=row_id, harness="opencode", harness_version="1.2.4", model="local/policy",
                   messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                   tool_events=[ToolEvent(kind="shell", name="bash", input={"command": "ls"}, success=True)],
                   session_id=f"s{row_id}", raw=[{"info": {}}])  # fmt: skip


def test_record_writer_layout_and_summary(tmp_path):
    writer = RecordWriter(tmp_path, "base/test")
    for row_id, passed in (("1", True), ("2", False)):
        result = RubricResult(scores={"r": float(passed)}, reward=float(passed), passed=passed, metrics={"m": 1.0})
        writer.write(episode(row_id), result, label="base")
    other = RecordWriter(tmp_path, "grpo/test")
    for row_id in ("1", "2"):
        other.write(episode(row_id), RubricResult(scores={"r": 1.0}, reward=1.0, passed=True), label="grpo")
    assert writer.done() == {("base", "1"), ("base", "2")}
    assert len(list((tmp_path / "sessions" / "base" / "test").glob("*.jsonl"))) == 2
    assert (tmp_path / "raw" / "base" / "test.jsonl").exists()
    records = read_jsonl(writer.episodes) + read_jsonl(other.episodes)
    assert records[0]["version"] == 2 and records[0]["harness_version"] == "1.2.4" and "raw" not in records[0]
    summary = summarize_records(records)
    assert summary["labels"]["base"]["pass_at_1"] == 0.5 and summary["labels"]["base"]["metric_sums"] == {"m": 2.0}
    assert summary["paired"]["base|grpo"]["grpo_only"] == 1 and summary["paired"]["base|grpo"]["rows"] == 2


def test_v2_records_rescore(tmp_path):
    from test_paper_urls import CASES, TASK

    from agent_distill.harnesses.opencode_v1_reader import episode_from_messages
    from agent_distill.rescore import rescore
    from agent_distill.rubrics import episode_context
    from agent_distill.task import Limits

    row, messages, structured = CASES["correct_and_complete"]
    ep = episode_from_messages(messages, answer_model=TASK.answer, task=TASK.name, row_id="1", model="m",
                               structured=structured)  # fmt: skip
    result = TASK.rubrics.score(episode_context(ep, row, limits=Limits(max_tool_calls=15)))
    writer = RecordWriter(tmp_path, "x")
    writer.write(ep, result, label="base")
    (item,) = rescore(read_jsonl(writer.episodes), TASK, {"1": row}, limits=Limits(max_tool_calls=15))
    assert item.result.scores == result.scores and item.result.passed is True
    assert json.loads(json.dumps(item.record))["label"] == "base"
