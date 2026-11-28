from __future__ import annotations

import pytest

from agent_distill import pipeline
from agent_distill.config import load_profile, load_run
from agent_distill.episode import Episode, ToolEvent
from agent_distill.records import RecordWriter
from agent_distill.rubrics import RubricResult
from agent_distill.stages.eval import vllm_command
from agent_distill.stages.sft import load_accepted, parse_args, parse_metric_caps


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


def test_sft_refuses_traces_files_with_multiple_repos():
    with pytest.raises(SystemExit, match="--traces-files cannot be combined with multiple --traces-repo"):
        parse_args(
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
                "--traces-files",
                "episodes/train.jsonl",
            ]
        )

    # A single repo (even repeated identically) still allows --traces-files.
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
            "--traces-files",
            "episodes/train.jsonl",
        ]
    )
    assert args.traces_files == ["episodes/train.jsonl"]


def test_fullcot_profile_uses_per_stage_generation_config():
    student = load_profile("students", "qwen3.5-4b-think-fullcot")
    assert student.sft.keep_reasoning is True
    assert student.serve.reasoning_parser == "qwen3"
    assert student.serve.generation_config == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert "--override-generation-config" not in student.serve.vllm_extra_args

    eval_cmd = vllm_command(student, student.model, "")
    probe_cmd = vllm_command(student, student.model, "", temperature=1.0)
    assert eval_cmd[eval_cmd.index("--override-generation-config") + 1] == (
        '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}'
    )
    assert probe_cmd[probe_cmd.index("--override-generation-config") + 1] == (
        '{"temperature": 1.0, "top_p": 1.0, "top_k": -1}'
    )


def test_fullcot_run_configs_build_expected_jobs():
    run = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-fullcot.toml"))
    traces = pipeline.trace_jobs(run, "R")
    assert len(traces) == 4
    assert all("--start 600" in j.command and "--no-raw-upload" in j.command for j in traces)

    (sft,) = pipeline.sft_jobs(run, "R")
    assert "--traces-repo agent-distillation-hf/cursor-qwen35-4b-think-hf-hub-qa-v2-glm-53-pi-160-traces" in sft.command
    assert "--traces-repo agent-distillation-hf/qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-traces" in sft.command
    assert "--max-metric hf_errors=0" in sft.command
    assert "flash-linear-attention==0.6.0" in sft.command

    run_judged = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-fullcot-judged.toml"))
    (judged_eval,) = pipeline.eval_jobs(run_judged, "R")
    assert "--split seed" in judged_eval.command
    assert "--limit 51" in judged_eval.command

    run_grpo = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-fullcot-grpo.toml"))
    probes = pipeline.probe_jobs(run_grpo, "R")
    assert len(probes) == 4
    assert all("--sample 1162" in j.command and "--limit 1162" in j.command for j in probes)
    (grpo,) = pipeline.grpo_jobs(run_grpo, "R")
    assert "--max-concurrent-rollouts 16" in grpo.command
    assert "difficulty/probe/train-0of4-qualified-ids.txt" in grpo.command


def test_open_ended_questions_disjoint_and_well_formed():
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "open_ended_questions", Path("examples/hf-hub-qa/open_ended_questions.py")
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = mod.rows()
    assert len(rows) == 240
    assert len({r["question_id"] for r in rows}) == 240
    refs = [json.loads(r["reference"]) for r in rows if r["reference"]]
    assert len(refs) >= 50
    for ref in refs:
        assert "ids" in ref or ("fact" in ref and "path" in ref)


def test_open_ended_task_and_mixed_run_configs():
    from agent_distill.task import load_task

    oe = load_task("examples/hf-hub-qa/open_ended.py")
    judged = load_task("examples/hf-hub-qa/judged.py")
    fact = load_task("examples/hf-hub-qa")
    assert oe.system_prompt == judged.system_prompt == fact.system_prompt
    assert oe.tools == judged.tools == fact.tools
    assert oe.skills == judged.skills == fact.skills
    assert oe.judge == judged.judge == "gemini-4-hydrogen"
    assert oe.limits == judged.limits

    run_traces = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-open-ended-traces.toml"))
    traces = pipeline.trace_jobs(run_traces, "R")
    assert len(traces) == 4
    assert all("--judge gemini-4-hydrogen" in j.command and "--no-raw-upload" in j.command for j in traces)
    assert all("GEMINI_API_KEY" in j.secret_envs for j in traces)

    run_mixed = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-mixed.toml"))
    (sft,) = pipeline.sft_jobs(run_mixed, "R")
    assert "--traces-repo agent-distillation-hf/qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-traces" in sft.command
    assert (
        "--traces-repo agent-distillation-hf/cursor-qwen35-4b-think-hf-hub-qa-open-ended-glm-53-pi-160-traces"
        in sft.command
    )
    assert "--max-tool-calls 12" in sft.command
    (eval_job,) = pipeline.eval_jobs(run_mixed, "R")
    assert "--limit 377" in eval_job.command

    run_mixed_judged = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-mixed-judged.toml"))
    (judged_eval,) = pipeline.eval_jobs(run_mixed_judged, "R")
    assert "--split seed" in judged_eval.command
    assert "--limit 51" in judged_eval.command
    assert "GEMINI_API_KEY" in judged_eval.secret_envs


def test_followup1_grounded_prompt_task_and_config():
    from agent_distill.task import load_task

    judged = load_task("examples/hf-hub-qa/judged.py")
    grounded = load_task("examples/hf-hub-qa/judged_grounded.py")
    assert grounded.name == "hf-hub-qa-judged-grounded"
    assert grounded.tools == judged.tools
    assert grounded.skills == judged.skills
    assert grounded.judge == judged.judge == "gemini-4-hydrogen"
    assert grounded.limits == judged.limits
    assert grounded.system_prompt_text() != judged.system_prompt_text()
    assert "Strict grounding:" in (grounded.system_prompt_text() or "")
    assert "When you have what you need," in (grounded.system_prompt_text() or "")

    run = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-mixed-judged-grounded.toml"))
    (job,) = pipeline.eval_jobs(run, "R")
    assert "--split seed" in job.command
    assert "--limit 51" in job.command
    assert "GEMINI_API_KEY" in job.secret_envs


def test_followup2_rewrite_helpers_and_grounded_sft_configs():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "rewrite_grounded_traces", Path("examples/hf-hub-qa/rewrite_grounded_traces.py")
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    events = [
        ToolEvent(
            kind="shell",
            name="bash",
            input={"command": "cat /skills/hf-cli/SKILL.md"},
            output="skill text",
        ),
        ToolEvent(
            kind="shell",
            name="bash",
            input={"command": "hf models info repo/a"},
            output='{"license": "apache-2.0"}',
        ),
    ]
    evidence = mod.format_tool_outputs(events)
    assert "SKILL.md" not in evidence
    assert "hf models info repo/a" in evidence
    assert "apache-2.0" in evidence

    messages = [
        {"role": "user", "content": "What license does repo/a use?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "hf models info repo/a"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"license": "apache-2.0"}'},
        {"role": "assistant", "content": "Apache 2.0, released in 2024.", "reasoning_content": "old"},
    ]
    rewritten = mod.replace_final_assistant_answer(messages, "repo/a uses the `apache-2.0` license.")
    assert rewritten[-1]["content"] == "repo/a uses the `apache-2.0` license."
    assert "reasoning_content" not in rewritten[-1]
    assert messages[-1]["content"] == "Apache 2.0, released in 2024."

    run_sft = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-mixed-grounded.toml"))
    (sft,) = pipeline.sft_jobs(run_sft, "R")
    assert "--traces-repo agent-distillation-hf/qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-traces" in sft.command
    assert (
        "--traces-repo agent-distillation-hf/"
        "cursor-qwen35-4b-think-hf-hub-qa-open-ended-grounded-glm-53-pi-160-traces" in sft.command
    )
    assert "--max-tool-calls 12" in sft.command

    run_judged = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-mixed-grounded-judged.toml"))
    (eval_job,) = pipeline.eval_jobs(run_judged, "R")
    assert "--split seed" in eval_job.command
    assert "--limit 51" in eval_job.command


def test_followup3_open_ended_grpo_configs():
    run = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-mixed-grpo.toml"))
    (merge_job,) = pipeline.merge_jobs(run, "R")
    assert (
        "agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-hf-hub-qa-v2-glm-53-pi-160-lora" in merge_job.command
    )

    probes = pipeline.probe_jobs(run, "R")
    assert len(probes) == 2
    assert all("GEMINI_API_KEY" in j.secret_envs for j in probes)
    assert all("--attempts 4" in j.command for j in probes)

    (grpo,) = pipeline.grpo_jobs(run, "R")
    assert "--allow-small-pool" in grpo.command
    assert "--num-generations 8" in grpo.command
    assert "--max-concurrent-rollouts 16" in grpo.command
    assert "GEMINI_API_KEY" in grpo.secret_envs
    assert "difficulty/probe/train-0of2-qualified-ids.txt" in grpo.command
    assert "difficulty/probe/train-1of2-qualified-ids.txt" in grpo.command

    (grpo_merge,) = pipeline.grpo_merge_jobs(run, "R")
    assert (
        "agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-open-ended-grpo-pi-160-grpo-lora"
        in grpo_merge.command
    )

    run_facts = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-mixed-grpo-eval-facts.toml"))
    (facts_eval,) = pipeline.eval_jobs(run_facts, "R")
    assert "--limit 377" in facts_eval.command
    assert (
        "agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-open-ended-grpo-pi-160-grpo-merged"
        in facts_eval.command
    )

    run_judged = pipeline.make_run(load_run("examples/hf-hub-qa/run-think-mixed-grpo-eval-judged.toml"))
    (judged_eval,) = pipeline.eval_jobs(run_judged, "R")
    assert "--split seed" in judged_eval.command
    assert "--limit 51" in judged_eval.command
    assert "GEMINI_API_KEY" in judged_eval.secret_envs


def test_followup4_mcp_judged_tasks_rubrics_and_configs():
    from agent_distill.task import load_task

    mcp_fact = load_task("examples/hf-hub-qa-mcp")
    mcp_judged = load_task("examples/hf-hub-qa-mcp/judged.py")
    mcp_oe = load_task("examples/hf-hub-qa-mcp/open_ended.py")

    assert mcp_judged.tools == mcp_oe.tools == mcp_fact.tools == ()
    assert mcp_judged.mcp_servers == mcp_oe.mcp_servers == mcp_fact.mcp_servers
    assert mcp_judged.system_prompt == mcp_oe.system_prompt == mcp_fact.system_prompt
    assert mcp_judged.judge == mcp_oe.judge == "gemini-4-hydrogen"

    ep_ok = Episode(
        task="hf-hub-qa-mcp-judged",
        row_id="q1",
        harness="pi",
        harness_version="1.6.0",
        model="m",
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "The model uses MIT license."}],
        tool_events=[
            ToolEvent(
                kind="other",
                name="mcp__hf__hub_repo_details",
                input={"repo_ids": ["a/b"]},
                success=True,
                output='{"license": "mit"}',
            )
        ],
        answer="The model uses MIT license.",
    )
    rubric_by_name = {r.name: r for r in mcp_judged.rubrics.rubrics}
    assert rubric_by_name["allowed_tools"](tool_events=ep_ok.tool_events) == 1.0
    assert rubric_by_name["used_hub"](tool_events=ep_ok.tool_events) == 1.0
    assert rubric_by_name["mcp_calls"](tool_events=ep_ok.tool_events) == 1.0
    assert rubric_by_name["mcp_errors"](tool_events=ep_ok.tool_events) == 0.0

    ep_bad_tool = Episode(
        task="hf-hub-qa-mcp-judged",
        row_id="q2",
        harness="pi",
        harness_version="1.6.0",
        model="m",
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"}],
        tool_events=[ToolEvent(kind="shell", name="bash", input={"command": "echo hi"}, success=True)],
        answer="ok",
    )
    assert rubric_by_name["allowed_tools"](tool_events=ep_bad_tool.tool_events) == 0.0
    assert rubric_by_name["used_hub"](tool_events=ep_bad_tool.tool_events) == 0.0

    run_base = pipeline.make_run(load_run("examples/hf-hub-qa-mcp/run-think-judged-baselines.toml"))
    base_jobs = pipeline.eval_jobs(run_base, "R")
    assert len(base_jobs) == 2
    assert all("--split seed" in j.command and "--limit 51" in j.command for j in base_jobs)
    assert all("GEMINI_API_KEY" in j.secret_envs for j in base_jobs)

    run_traces = pipeline.make_run(load_run("examples/hf-hub-qa-mcp/run-think-open-ended-traces.toml"))
    trace_jobs = pipeline.trace_jobs(run_traces, "R")
    assert len(trace_jobs) == 4
    assert all("--judge gemini-4-hydrogen" in j.command and "--no-raw-upload" in j.command for j in trace_jobs)

    run_mixed = pipeline.make_run(load_run("examples/hf-hub-qa-mcp/run-think-mixed.toml"))
    (mcp_sft,) = pipeline.sft_jobs(run_mixed, "R")
    assert "--traces-repo agent-distillation-hf/qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160-traces" in mcp_sft.command
    assert (
        "--traces-repo agent-distillation-hf/"
        "cursor-qwen35-4b-think-hf-hub-qa-mcp-open-ended-glm-53-pi-160-traces" in mcp_sft.command
    )
    assert "--max-tool-calls 12" in mcp_sft.command

    run_mixed_judged = pipeline.make_run(load_run("examples/hf-hub-qa-mcp/run-think-mixed-judged.toml"))
    (mcp_judged_eval,) = pipeline.eval_jobs(run_mixed_judged, "R")
    assert "--split seed" in mcp_judged_eval.command
    assert "--limit 51" in mcp_judged_eval.command
    assert "GEMINI_API_KEY" in mcp_judged_eval.secret_envs
