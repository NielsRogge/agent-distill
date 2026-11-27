"""LFM3-1.7B recipe (a): the SFT renderer gives the student what Pi and vLLM gave it at inference.

The parity tests render the published base-LFM3-1.7B evaluation episodes (Pi 1.6.0 + vLLM 0.34.0, thinking on)
through the SFT stage's own path (`stages.sft.trajectory`, `harnesses.tools.harness_tools`, LFM3's tokenizer and chat
template at the profile's pinned revision) and check them against what vLLM recorded. They need transformers and the
Hub, and skip without them (CI); run them with `uv run --with transformers==5.23.0 --with jinja2 pytest
tests/test_lfm3.py`.
The offline tests check the profile, the serving command and the run configs.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

import pytest

from agent_distill import pipeline
from agent_distill.config import load_profile, load_run
from agent_distill.harnesses.tools import harness_tools
from agent_distill.pipeline import make_run
from agent_distill.stages import sft
from agent_distill.stages.eval import vllm_command
from agent_distill.task import load_task

STUDENT = load_profile("students", "lfm3-1.7b")
TASKS = {
    "hf-hub-qa-mcp": "nielsr/lfm3-17b-hf-hub-qa-mcp-v2-glm-53-pi-160-eval",
    "hf-hub-qa": "nielsr/lfm3-17b-hf-hub-qa-v2-glm-53-pi-160-eval",
}
ROWS = 120  # episodes per task (of 377): enough turns, a few seconds of rendering


@cache
def tokenizer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    try:
        return transformers.AutoTokenizer.from_pretrained(STUDENT.model, revision=STUDENT.revision)
    except OSError as error:  # offline
        pytest.skip(f"LFM3 tokenizer unavailable: {error}")


@cache
def episodes(task: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(TASKS[task], "episodes/base/test.jsonl", repo_type="dataset")
    except Exception as error:  # offline
        pytest.skip(f"LFM3 evaluation episodes unavailable: {error}")
    return [json.loads(line) for line in Path(path).read_text().splitlines()[:ROWS]]


def tools_of(task: str) -> list[dict]:
    return harness_tools("pi@1.6.0", load_task(Path("examples") / task))


def chat_messages(record: dict) -> list[dict]:
    return [m for m in record["messages"] if m.get("role") in {"system", "user", "assistant", "tool"}]


def compacted(message: dict) -> bool:
    return message["role"] == "tool" and len(str(message.get("content") or "")) > sft.MAX_TOOL_CONTENT_CHARS


# --------------------------------------------------------------------------------------------------
# Parity with inference (network)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", sorted(TASKS))
def test_sft_renders_token_for_token_the_prompts_vllm_served(task):
    """Every assistant turn's history, rendered by SFT's path with SFT's tool list and the student's own reasoning kept
    (as Pi sends it back), has exactly the prompt length vLLM reported for that turn (`usage.input`): same system
    prompt, same tool list (names, descriptions, schemas), same history rendering. Turns after a tool output that SFT
    compacts are skipped (a deliberate SFT difference)."""
    tok, tools = tokenizer(), tools_of(task)
    names = [t["function"]["name"] for t in tools]
    compared = 0
    for record in episodes(task):
        assert record["harness"] == "pi" and record["harness_version"] == "1.6.0"
        assert record["metadata"]["tools"] == names  # the tools Pi offered the model
        messages = sft.trajectory(record, keep_reasoning=True)
        raw = chat_messages(record)
        assert len(messages) == len(raw)
        for index, message in enumerate(raw):
            served = (message.get("usage") or {}).get("input")
            if message["role"] != "assistant" or not served or any(compacted(m) for m in raw[:index]):
                continue
            prompt = tok.apply_chat_template(messages[:index], tools=tools, tokenize=False, add_generation_prompt=True)
            assert len(tok(prompt, add_special_tokens=False)["input_ids"]) == served, (record["row_id"], index)
            compared += 1
    assert compared >= 300


CALL = re.compile(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>", re.S)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>\n", re.S)


def parsed(value: str, schema: dict | None):
    """A parameter value as the qwen3_coder parser converts it: by the tool's schema (strings, and every parameter of a
    tool the model invented, stay raw text; other declared types are JSON)."""
    if not schema or schema.get("type", "string") == "string":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@pytest.mark.parametrize("task", sorted(TASKS))
def test_tool_calls_render_as_the_qwen3_coder_xml_vllm_parses(task):
    """The supervised text of a tool-calling turn is the Qwen3-Coder XML the `qwen3_coder` parser reads back into the
    same calls: function names and arguments (strings raw, other values JSON), nothing else after the think block."""
    tok, tools = tokenizer(), tools_of(task)
    schemas = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
    checked = 0
    for record in episodes(task):
        messages = sft.trajectory(record, keep_reasoning=False, strip_visible_text=True)
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or not message.get("tool_calls"):
                continue
            prompt = tok.apply_chat_template(messages[:index], tools=tools, tokenize=False, add_generation_prompt=True)
            through = tok.apply_chat_template(messages[: index + 1], tools=tools, tokenize=False)
            target = through[sft.supervised_start(prompt, through, "empty") :]
            calls = CALL.findall(target)
            assert re.fullmatch(r"(?:<tool_call>\n.*?</tool_call>)+<\|im_end\|>\n", target, re.S), target[:200]
            expected = [(c["function"]["name"], c["function"]["arguments"]) for c in message["tool_calls"]]
            got = [(name, {k: parsed(v, schemas.get(name, {}).get(k)) for k, v in PARAMETER.findall(body)})
                   for name, body in calls]  # fmt: skip
            assert got == expected, (record["row_id"], index)
            checked += 1
    assert checked >= 200


@pytest.mark.parametrize("task", sorted(TASKS))
def test_empty_think_target_supervises_from_after_the_empty_think_block(task):
    """Recipe (a): every assistant turn renders `<think></think>` (history turns exactly as the template renders a
    turn without kept reasoning at inference), the block is context, and supervision starts right after it; the
    generation prompt itself is the inference one (`<|im_start|>assistant\\n<think>`)."""
    tok, tools = tokenizer(), tools_of(task)
    trajectories = [sft.trajectory(r, keep_reasoning=False, strip_visible_text=True) for r in episodes(task)[:40]]
    examples = sft.render_trajectory_examples(trajectories, tok, tools, "empty")
    assert len(examples) == len(trajectories)  # prefix-stable: one example per trajectory, no per-turn fallback
    for messages, example in zip(trajectories, examples, strict=True):
        text, spans = example["prompt"], json.loads(example["spans"])
        assert len(spans) == sum(m["role"] == "assistant" for m in messages)
        for start, end in spans:
            assert text[:start].endswith("<|im_start|>assistant\n<think></think>")
            assert text[start:end].endswith("<|im_end|>\n") and "<think>" not in text[start:end]
    first = tok.apply_chat_template(trajectories[0][:2], tools=tools, tokenize=False, add_generation_prompt=True)
    assert first.endswith("<|im_start|>assistant\n<think>")


# --------------------------------------------------------------------------------------------------
# Profile, serving and run configs (offline)
# --------------------------------------------------------------------------------------------------


def test_profile_pins_the_revision_and_the_training_stack():
    assert re.fullmatch(r"[0-9a-f]{40}", STUDENT.revision)
    assert "--revision" not in STUDENT.serve.vllm_extra_args  # one pin, from the profile
    assert "transformers==5.23.0" in STUDENT.sft.deps and "peft==0.20.0" in STUDENT.sft.deps
    assert STUDENT.sft.flavor == "l4x1" and STUDENT.sft.replay
    assert {"--think-target", "empty", "--strip-visible-text"} <= set(STUDENT.sft.args)
    targets = set(sft.lora_target_modules(STUDENT.lora.sft_target_modules))
    assert targets == {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"}
    assert STUDENT.lora.rank == 16 == sft.PAPER_DEFAULTS["lora_rank"]


def test_revision_applies_to_the_base_model_only():
    base = vllm_command(STUDENT, STUDENT.model, "/adapter")
    assert base[base.index("--revision") + 1] == STUDENT.revision
    assert base[base.index("--tokenizer-revision") + 1] == STUDENT.revision
    assert base[base.index("--max-lora-rank") + 1] == "16" and "--trust-remote-code" in base
    assert "--revision" not in vllm_command(STUDENT, "nielsr/some-merged-model", "")
    assert STUDENT.revision_of(STUDENT.model) == STUDENT.revision and STUDENT.revision_of("other/model") is None
    qwen = load_profile("students", "qwen3.5-4b-think")
    assert qwen.revision_of(qwen.model) is None and "--revision" not in vllm_command(qwen, qwen.model, "")


@pytest.mark.parametrize("task", sorted(TASKS))
def test_recipe_a_run_configs_build_sft_on_l4_and_adapter_eval(task):
    run = make_run(load_run(Path("examples") / task / "run-lfm3-think.toml"))
    assert run.config.student == "lfm3-1.7b" and run.config.eval.limit == 377
    assert run.repo("replay") == "nielsr/lfm3-17b-replay-openr1-oasst"
    assert run.repo("traces") == f"nielsr/qwen35-4b-{task}-v2-glm-53-pi-160-traces"
    built = {stage: job for stage in ("replay", "sft", "eval") for (job,) in pipeline.stage_waves(stage, run, "REV")}
    assert built["sft"].flavor == "l4x1" and "transformers==5.23.0" in built["sft"].command
    assert "--replay-repo nielsr/lfm3-17b-replay-openr1-oasst" in built["sft"].command
    assert "--output-repo nielsr/lfm3-17b-replay-openr1-oasst" in built["replay"].command
    command = built["eval"].command  # the adapter served on the pinned base (no merge stage)
    assert "--model LiquidAI/LFM3-1.7B " in command and f"--adapter {run.repo('lora')} " in command


def test_mcp_drift_override_reaches_only_the_runs_that_set_it():
    from agent_distill.mcp import ALLOW_DRIFT_ENV

    mcp = make_run(load_run(Path("examples/hf-hub-qa-mcp/run-lfm3-think.toml")))
    (job,) = pipeline.eval_jobs(mcp, "REV")
    assert job.env[ALLOW_DRIFT_ENV] == "1"
    cli = make_run(load_run(Path("examples/hf-hub-qa/run-lfm3-think.toml")))
    assert all(ALLOW_DRIFT_ENV not in job.env for job in pipeline.eval_jobs(cli, "REV"))
    base = make_run(load_run(Path("examples/hf-hub-qa-mcp/run-lfm3.toml")))
    assert all(ALLOW_DRIFT_ENV not in job.env for job in pipeline.eval_jobs(base, "REV"))
