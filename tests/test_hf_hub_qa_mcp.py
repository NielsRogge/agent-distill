"""The hf-hub-qa-mcp example and MCP support in the Pi adapter (no network: the Hub and MCP server are faked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_distill import mcp
from agent_distill.config import load_profile
from agent_distill.episode import Episode, ToolEvent
from agent_distill.harnesses.base import Endpoint
from agent_distill.harnesses.pi import mcp_json, pi_command, tool_names
from agent_distill.rubrics import episode_context
from agent_distill.task import TextAnswer, load_task, task_module

TASK = load_task("examples/hf-hub-qa-mcp")
EXAMPLE = Path("examples/hf-hub-qa-mcp")
TOOLS = ["mcp__hf__hub_repo_search", "mcp__hf__hub_repo_details", "mcp__hf__hf_fs"]


def module(name: str):
    return task_module(EXAMPLE, name)


rubrics = module("rubrics")


# --------------------------------------------------------------------------------------------------
# Task and Pi wiring
# --------------------------------------------------------------------------------------------------


def test_task_shape():
    assert isinstance(TASK.answer, TextAnswer) and TASK.tools == ()
    (server,) = TASK.mcp_servers
    assert server.url == "https://huggingface.co/mcp" and server.tools == (
        "hub_repo_search",
        "hub_repo_details",
        "hf_fs",
    )
    summary = server.summary()
    assert summary["server_version"] == "0.4.39" and len(summary["tools_sha256"]) == 64


def test_snapshot_digest_matches_its_tools():
    data = json.loads((EXAMPLE / "mcp-tools.json").read_text())
    assert mcp.tools_digest(data["tools"]) == data["tools_sha256"]
    assert {t["name"] for t in data["tools"]} >= set(data["allowed_tools"])
    assert all(t["annotations"]["readOnlyHint"] for t in data["tools"])


def test_pi_gets_only_the_allowed_mcp_tools_and_no_credentials():
    assert tool_names(TASK) == TOOLS  # no bash, read or submit
    config = mcp_json(TASK)
    server = config["mcpServers"]["hf"]
    assert config["autoEnableCodemode"] is False
    assert server == {
        "url": "https://huggingface.co/mcp",
        "exposure": "hidden",
        "timeout": 120,
        "toolExposure": {"hub_repo_search": "direct", "hub_repo_details": "direct", "hf_fs": "direct"},
    }
    assert "headers" not in server and "oauth" not in server and "hf_" not in json.dumps(config).replace("hf_fs", "")
    endpoint = Endpoint(base_url="http://127.0.0.1:1/v1", model="m")
    command = pi_command(Path("node"), Path("cli.js"), TASK, endpoint, Path("s"), "q", "sys")
    assert "--no-mcp" not in command and "--no-extensions" in command
    assert command[command.index("builtin:mcp") - 1] == "-e"  # --no-extensions turns built-in MCP off
    assert command[command.index("--tools") + 1] == ",".join(TOOLS)


def test_tasks_without_mcp_keep_mcp_off():
    paper = load_task("examples/paper-urls")
    command = pi_command(Path("node"), Path("cli.js"), paper, Endpoint(base_url="http://x/v1", model="m"), Path("s"),
                         "q", None)  # fmt: skip
    assert "--no-mcp" in command and "builtin:mcp" not in command


def test_drift_check_refuses_a_changed_tool_list(monkeypatch):
    (server,) = TASK.mcp_servers
    tools = json.loads(server.snapshot.read_text())["tools"]
    monkeypatch.setattr(mcp, "fetch_tools", lambda url, timeout=30: (tools, {"version": "0.4.39"}))
    assert mcp.check(server)["drift"] is False
    changed = [*tools, {"name": "hub_repo_create", "description": "write", "inputSchema": {}}]
    monkeypatch.setattr(mcp, "fetch_tools", lambda url, timeout=30: (changed, {"version": "0.5.0"}))
    with pytest.raises(mcp.McpDrift, match="changed since its snapshot"):
        mcp.check(server)
    monkeypatch.setenv(mcp.ALLOW_DRIFT_ENV, "1")
    result = mcp.check(server)
    assert result["drift"] is True and result["live_server_version"] == "0.5.0"


# --------------------------------------------------------------------------------------------------
# Answer checks
# --------------------------------------------------------------------------------------------------

LICENSES = ["apache-2.0", "mit", "cc-by-4.0", "llama3.1", "llama3", "openrail++", "other"]


def test_builder_skips_placeholder_licenses_and_keeps_dual_ones():
    questions = module("questions")
    assert questions.known("mit") and questions.known(["cc-by-sa-3.0", "gfdl"])
    assert not questions.known("other") and not questions.known(["mit", "unknown"]) and not questions.known([])


def test_built_rows_pass_oracle_and_reject_no_op_and_canary():
    questions = module("questions")
    cands = [
        {"template": "model_license", "entity": "model:a/x", "subject": "a/x", "gold": "apache-2.0",
         "live": {"type": "model", "id": "a/x", "field": "license"}, "extra": {}},
        {"template": "model_license", "entity": "model:b/y", "subject": "b/y", "gold": "mit",
         "live": {"type": "model", "id": "b/y", "field": "license"}, "extra": {}},
        {"template": "model_params", "entity": "model:a/x", "subject": "a/x", "gold": 1000,
         "live": {"type": "model", "id": "a/x", "field": "params"}, "extra": {}},
        {"template": "model_params", "entity": "model:b/y", "subject": "b/y", "gold": 5000,
         "live": {"type": "model", "id": "b/y", "field": "params"}, "extra": {}},
    ]  # fmt: skip
    rows = questions.finalize(cands, "2026-11-25")
    kept, dropped = questions.validate(rows)
    assert len(kept) == 4 and not dropped
    assert {json.loads(r["canary"]) for r in kept if r["template"] == "model_license"} == {"apache-2.0", "mit"}
    assert all(r["split"] in ("train", "validation", "test") for r in kept)
    assert questions.split_of("model:a/x") == next(r["split"] for r in kept if r["entity"] == "model:a/x")


# --------------------------------------------------------------------------------------------------
# Rubrics
# --------------------------------------------------------------------------------------------------

ROW = {
    "question_id": "q1",
    "kind": "category",
    "gold": '"apache-2.0"',
    "question": "What is the license of a/x?",
    "extra": json.dumps({"hedge": LICENSES}),
    "live": json.dumps({"type": "model", "id": "a/x", "field": "license"}),
}


def call(name="mcp__hf__hub_repo_details", output="license: apache-2.0, id acme/Model-1", success=True) -> ToolEvent:
    return ToolEvent(kind="fetch", name=name, input={"repo_ids": ["a/x"]}, success=success, output=output)


def score(answer, events, live="apache-2.0", monkeypatch=None):
    if monkeypatch is not None:
        rubrics.live.cache_clear()
        monkeypatch.setattr(rubrics, "live_value", lambda spec: live)
    episode = Episode(task="hf-hub-qa-mcp", row_id="q1", harness="pi", harness_version="1.6.0", model="m",
                      answer=answer, answer_mode="text", tool_events=list(events))  # fmt: skip
    return TASK.rubrics.score(episode_context(episode, ROW, task=TASK), mode="eval")


def test_correct_grounded_answer_passes_with_efficiency_bonus(monkeypatch):
    result = score("The license of `acme/Model-1` is Apache 2.0.", [call()], monkeypatch=monkeypatch)
    assert result.passed and result.scores["fact_correct"] == 1.0 and result.scores["grounded"] == 1.0
    assert result.reward == pytest.approx(1.0 + 0.1 * (1 - 1 / 20))


def test_answer_from_memory_or_other_tools_fails(monkeypatch):
    memory = score("Apache 2.0.", [], monkeypatch=monkeypatch)
    assert not memory.passed and memory.reward == 0.0 and memory.scores["used_hub"] == 0.0
    shell = ToolEvent(kind="shell", name="bash", input={"command": "hf models info a/x"}, success=True, output="")
    other = score("Apache 2.0.", [call(), shell], monkeypatch=monkeypatch)
    assert not other.passed and other.reward == 0.0 and other.scores["allowed_tools"] == 0.0


def test_wrong_answer_scores_zero_but_a_drifted_row_is_unscorable(monkeypatch):
    wrong = score("It is MIT licensed.", [call()], monkeypatch=monkeypatch)
    assert wrong.scores["fact_correct"] == 0.0 and not wrong.passed
    drifted = score("It is MIT licensed.", [call()], live="mit", monkeypatch=monkeypatch)
    assert drifted.scores["fact_correct"] is None and not drifted.passed


def test_ungrounded_ids_fail(monkeypatch):
    result = score("Apache 2.0, see `acme/Model-1` and `b/Other-Model`.", [call()], monkeypatch=monkeypatch)
    assert result.scores["grounded"] == 0.5 and not result.passed


def test_profiles_exist():
    assert load_profile("teachers", "glm-5.3").model.startswith("zai-org/GLM-5.3")
    assert load_profile("students", "qwen3.5-4b").model == "Qwen/Qwen3.5-4B"


def test_v2_templates_and_triples_follow_their_first_model():
    v2 = module("questions_v2")
    assert set(v2.QUESTIONS) == set(v2.KINDS) and set(v2.HEDGED) <= set(v2.KINDS)
    assert (
        v2.differs(10, 9) and not v2.differs(100, 101) and v2.differs("mit", "apache-2.0") and not v2.differs("", "x")
    )
    questions = module("questions")
    cands = [
        {"template": "params_sum", "entity": "sum:a/A|b/B|c/C", "subject": "a/A, b/B and c/C", "gold": 600,
         "live": {"type": "sum", "ids": ["a/A", "b/B", "c/C"], "field": "params"}, "split_entity": "model:a/A",
         "extra": {}},
        {"template": "params_sum", "entity": "sum:d/D|e/E|f/F", "subject": "d/D, e/E and f/F", "gold": 900,
         "live": {"type": "sum", "ids": ["d/D", "e/E", "f/F"], "field": "params"}, "split_entity": "model:d/D",
         "extra": {}},
    ]  # fmt: skip
    rows, dropped = questions.validate(questions.finalize(cands, "2026-11-25", questions=v2.QUESTIONS, kinds=v2.KINDS,
                                                          hedged=v2.HEDGED))  # fmt: skip
    assert len(rows) == 2 and not dropped
    assert rows[0]["split"] == questions.split_of("model:a/A")
    assert "a/A, b/B and c/C combined" in rows[0]["question"]
