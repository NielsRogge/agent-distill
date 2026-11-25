from __future__ import annotations

import pytest

from agent_distill import pipeline
from agent_distill.config import load_run
from agent_distill.episode import Episode
from agent_distill.harnesses.tools import harness_tools
from agent_distill.records import RecordWriter
from agent_distill.rubrics import RubricResult
from agent_distill.stages.sft import load_accepted, parse_args, parse_metric_caps
from agent_distill.task import load_task


def test_text_answer_task_omits_submit_tool_in_sft():
    task = load_task("examples/hf-hub-qa")
    tools = harness_tools("pi@1.6.0", task)
    assert [t["function"]["name"] for t in tools] == ["bash", "read"]


def test_sft_filtering_and_multiple_trace_sources(tmp_path):
    w1 = RecordWriter(tmp_path, "part1")
    w2 = RecordWriter(tmp_path, "part2")

    def add(writer: RecordWriter, row_id: str, *, hf_errors: float = 0.0, tool_calls: int = 3) -> None:
        ep = Episode(
            task="hf-hub-qa",
            row_id=row_id,
            harness="pi",
            harness_version="1.6.0",
            model="m",
            messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        )
        # Override tool_calls count via tool_events length
        from agent_distill.episode import ToolEvent

        ep.tool_events = [ToolEvent(kind="shell", name="bash", input={"command": "hf models info x"})] * tool_calls
        writer.write(ep, RubricResult(scores={"hf_errors": hf_errors}, reward=1.0, passed=True), label="teacher")

    add(w1, "r1", hf_errors=0.0, tool_calls=3)
    add(w1, "r2", hf_errors=1.0, tool_calls=4)
    add(w2, "r1", hf_errors=2.0, tool_calls=5)  # duplicate row_id: first wins
    add(w2, "r3", hf_errors=0.0, tool_calls=14)
    add(w2, "r4", hf_errors=0.0, tool_calls=5)

    kept = load_accepted(
        [w1.episodes, w2.episodes],
        harness="pi@1.6.0",
        max_metrics=parse_metric_caps(["hf_errors=0"]),
        max_tool_calls=12,
    )
    assert [r["row_id"] for r in kept] == ["r1", "r4"]

    with pytest.raises(SystemExit, match="bad --max-metric"):
        parse_metric_caps(["hf_errors"])


def test_sft_cli_accepts_multiple_repos_and_optimizer_flags():
    args = parse_args(
        [
            "--task",
            "examples/hf-hub-qa",
            "--harness",
            "pi@1.6.0",
            "--student",
            "qwen3.5-4b",
            "--traces-repo",
            "org/repo-a",
            "--traces-repo",
            "org/repo-b",
            "--max-metric",
            "hf_errors=0",
            "--max-tool-calls",
            "12",
            "--lr-scheduler-type",
            "cosine",
            "--warmup-ratio",
            "0.05",
            "--weight-decay",
            "0.01",
            "--lora-alpha",
            "64",
            "--lora-dropout",
            "0.0",
            "--seed",
            "7",
        ]
    )
    assert args.traces_repo == ["org/repo-a", "org/repo-b"]
    assert args.max_metric == ["hf_errors=0"]
    assert args.max_tool_calls == 12
    assert args.lr_scheduler_type == "cosine"
    assert args.warmup_ratio == pytest.approx(0.05)
    assert args.weight_decay == pytest.approx(0.01)
    assert args.lora_alpha == 64
    assert args.lora_dropout == 0.0
    assert args.seed == 7


def test_experiment_run_configs_build_expected_jobs():
    run = pipeline.make_run(load_run("experiments/hf-cli-distill/run.toml"))
    traces = pipeline.trace_jobs(run, "R")
    assert len(traces) == 4
    assert all("--start 600" in j.command and "--no-raw-upload" in j.command for j in traces)

    (sft,) = pipeline.sft_jobs(run, "R")
    assert "--traces-repo nielsr/cursor-qwen35-4b-think-hf-hub-qa-v2-glm-53-pi-160-traces" in sft.command
    assert "--traces-repo nielsr/qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-traces" in sft.command
    assert "--max-metric hf_errors=0" in sft.command
    assert "flash-linear-attention==0.6.0" in sft.command

    run_grpo = pipeline.make_run(load_run("experiments/hf-cli-distill/run-grpo.toml"))
    probes = pipeline.probe_jobs(run_grpo, "R")
    assert len(probes) == 4
    assert all("--sample 1162" in j.command and "--limit 1162" in j.command for j in probes)
    assert all("--no-raw-upload" in j.command for j in probes)
    (grpo,) = pipeline.grpo_jobs(run_grpo, "R")
    assert "--max-completion-length 4096" in grpo.command
    assert "difficulty/probe/train-0of4-qualified-ids.txt" in grpo.command
