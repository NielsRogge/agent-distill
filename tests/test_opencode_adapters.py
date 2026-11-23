"""The OpenCode 1.2.4 and 2.3.2 adapters score every parity episode like url-extraction.

Each paper-urls parity case (tests/test_paper_urls.py) is replayed through the adapter's real `run_episode` against
a fake OpenCode server (httpx MockTransport): as 1.2.4 messages with native structured output, and converted to
2.3.2 messages (flat messages, `shell` instead of `bash`, the answer as a JSON-only final message). The scores and
the GRPO reward must match url-extraction's expectations in both versions.

`AGENT_DISTILL_REAL_OPENCODE=1` additionally downloads the pinned binaries and runs them against a fake model.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_paper_urls import CASES, EXPECTED, MAX_TOOL_CALLS, TASK, url_extraction_reward

from agent_distill.config import load_profile
from agent_distill.harnesses import opencode
from agent_distill.harnesses.base import Endpoint, load_harness
from agent_distill.harnesses.opencode_v1 import OpenCodeV1
from agent_distill.harnesses.opencode_v1 import Runner as RunnerV1
from agent_distill.harnesses.opencode_v2 import AGENT_ID, OpenCodeV2, normalize_messages, permissions, server_env
from agent_distill.harnesses.opencode_v2 import Runner as RunnerV2
from agent_distill.rubrics import episode_context
from agent_distill.task import Limits

V1 = load_profile("harnesses", "opencode@1.2.4")
V2 = load_profile("harnesses", "opencode@2.3.2")
ENDPOINT = Endpoint(base_url="http://vllm/v1", model="policy")
ROW_OF = {case: CASES[case][0] for case in CASES}


class FakeV1:
    def __init__(self, messages: list[dict[str, Any]], structured: Any) -> None:
        self.messages, self.structured = messages, structured
        self.requests: list[tuple[str, str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append((request.method, path, body))
        if path == "/session":
            return httpx.Response(200, json={"id": "ses_test"})
        if path.endswith("/message") and request.method == "POST":
            info = {"structured": self.structured} if self.structured is not None else {}
            return httpx.Response(200, json={"info": info})
        if path.endswith("/message"):
            return httpx.Response(200, json=self.messages)
        return httpx.Response(200, json=True)


def to_v2(messages: list[dict[str, Any]], structured: Any) -> list[dict[str, Any]]:
    """A 1.x parity episode as the messages an OpenCode 2 server returns."""
    result: list[dict[str, Any]] = []
    for number, message in enumerate(messages):
        if message["info"]["role"] == "user":
            text = "".join(part.get("text", "") for part in message["parts"])
            result.append({"id": f"u{number}", "type": "user", "time": {"created": 1}, "text": text})
            continue
        content: list[dict[str, Any]] = []
        for part in message["parts"]:
            if part["type"] != "tool":
                content.append({"type": part["type"], "text": part.get("text", "")})
                continue
            if part["tool"] == "StructuredOutput":
                continue
            state = part["state"]
            v2_state: dict[str, Any] = {"status": state["status"], "input": state["input"]}
            if state["status"] == "completed":
                v2_state["content"] = [{"type": "text", "text": state.get("output", "")}]
            else:
                v2_state["error"] = {"type": "tool.execution", "message": state.get("error", "")}
            name = "shell" if part["tool"] == "bash" else part["tool"]
            content.append({"type": "tool", "id": part["callID"], "name": name, "state": v2_state, "time": {}})
        if structured is not None:  # 2.x has no structured output: the answer is a JSON-only final message
            content.append({"type": "text", "text": json.dumps(structured)})
        result.append(
            {"id": f"a{number}", "type": "assistant", "time": {"created": 2, "completed": 3}, "agent": AGENT_ID,
             "model": {"id": "policy", "providerID": "local"}, "content": content, "finish": "stop", "cost": 0.001,
             "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}}}
        )  # fmt: skip
    result.append({"id": "idle", "type": "idle", "time": {"created": 4}, "outcome": "succeeded"})
    return result


class FakeV2:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.requests: list[tuple[str, str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Basic ")
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append((request.method, path, body))
        if path == "/api/session":
            return httpx.Response(200, json={"data": {"id": "ses_v2"}})
        if path.endswith("/prompt"):
            return httpx.Response(200, json={"data": {"id": "u0", "type": "user"}})
        if path.endswith("/wait"):
            return httpx.Response(204)
        if path.endswith("/message"):
            return httpx.Response(200, json={"data": self.messages, "cursor": {"previous": None, "next": None}})
        if path == "/api/info":
            return httpx.Response(200, json={"version": "2.3.2"})
        if path.endswith("/interrupt"):
            return httpx.Response(200, json={"interrupted": True})
        return httpx.Response(404, json={"message": path})


def server(version: str, password: str | None = None) -> opencode.Server:
    return opencode.Server("http://opencode", version, Path("/tmp/ws"), None, password)


def run_v1(case: str):
    row, messages, structured = CASES[case]
    fake = FakeV1(messages, structured)

    async def go():
        async with httpx.AsyncClient(base_url="http://opencode", transport=httpx.MockTransport(fake)) as client:
            runner = RunnerV1(server("1.2.4"), client)
            return await OpenCodeV1(V1).run_episode(runner, TASK, row, ENDPOINT, Limits(timeout=30, max_tool_calls=15))

    return asyncio.run(go()), fake


def run_v2(case: str):
    row, messages, structured = CASES[case]
    fake = FakeV2(to_v2(messages, structured))

    async def go():
        async with httpx.AsyncClient(
            base_url="http://opencode", auth=("opencode", "pw"), transport=httpx.MockTransport(fake)
        ) as client:
            runner = RunnerV2(server("2.3.2", "pw"), client, TASK)
            return await OpenCodeV2(V2).run_episode(runner, TASK, row, ENDPOINT, Limits(timeout=30, max_tool_calls=15))

    return asyncio.run(go()), fake


@pytest.mark.parametrize("version", ["1.2.4", "2.3.2"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_adapters_score_like_url_extraction(case: str, version: str) -> None:
    episode, _ = run_v1(case) if version == "1.2.4" else run_v2(case)
    assert episode.harness_id == f"opencode@{version}"
    row = ROW_OF[case]
    context = episode_context(episode, row, limits=Limits(max_tool_calls=MAX_TOOL_CALLS))
    evaluation = TASK.rubrics.score(context, mode="eval")
    train = TASK.rubrics.score(context, mode="train")
    fields, complete = EXPECTED[case]
    assert (evaluation.scores["protocol"] == 1.0) is complete
    # 2.x has no StructuredOutput tool call, so a correct complete episode uses one tool call fewer.
    assert train.reward == pytest.approx(url_extraction_reward(fields, complete, episode.tool_calls))
    kinds = {event.kind for event in episode.tool_events}
    assert "other" not in kinds


def test_v1_request_body() -> None:
    _, fake = run_v1("correct_and_complete")
    posts = [body for method, path, body in fake.requests if method == "POST" and path.endswith("/message")]
    body = posts[0]
    assert body["model"] == {"providerID": "local", "modelID": "policy"}
    assert body["system"] == TASK.system_prompt_text()
    assert {name for name, on in body["tools"].items() if on} == {"bash", "read", "grep", "glob", "webfetch"}
    assert body["format"]["type"] == "json_schema" and body["format"]["schema"] == TASK.answer.model_json_schema()
    assert body["parts"] == [{"type": "text", "text": TASK.render_prompt(ROW_OF["correct_and_complete"])}]


def test_v1_structured_output_fallback_message() -> None:
    from agent_distill.harnesses.opencode_v1 import message_body

    body = message_body(TASK, ROW_OF["correct_and_complete"], ("local", "policy"), structured=False)
    assert body["parts"][0]["text"].endswith("with reasoning, github_url, and project_page_url.")
    assert "format" not in body


def test_v2_requests() -> None:
    _, fake = run_v2("correct_and_complete")
    session = next(body for method, path, body in fake.requests if path == "/api/session")
    assert session["agent"] == AGENT_ID and session["model"] == {"providerID": "local", "id": "policy"}
    assert session["permissions"] == permissions(TASK.tools)
    prompt = next(body for method, path, body in fake.requests if path.endswith("/prompt"))
    assert prompt == {"text": TASK.render_prompt(ROW_OF["correct_and_complete"])}


def test_v2_permissions_hide_everything_but_the_task_tools() -> None:
    def hidden(action: str, rules: list[dict[str, str]]) -> bool:  # OpenCode's `whollyDisabled`
        rule = next((r for r in reversed(rules) if fnmatch.fnmatchcase(action, r["action"])), None)
        return bool(rule) and rule["resource"] == "*" and rule["effect"] == "deny"

    rules = permissions(TASK.tools)
    agent = [
        {"action": "*", "resource": "*", "effect": "allow"},
        *rules,
        {"action": "skill", "resource": "browser", "effect": "deny"},
    ]
    effective = agent + rules  # OpenCode appends its own rules after the agent's; the session's come last
    for tool in ("shell", "read", "grep", "glob", "webfetch"):
        assert not hidden(tool, effective)
    for tool in ("skill", "execute", "question", "edit", "write", "patch", "subagent", "websearch", "todowrite"):
        assert hidden(tool, effective)
    assert not hidden("skill", agent)  # why the session repeats the rules


def test_v2_system_prompt_is_url_extractions() -> None:
    prompt = TASK.system_prompt_for("opencode", "2.3.2", "final_json")
    assert "You may use Shell, Read, Grep, Glob, and WebFetch" in prompt
    schema = {k: v for k, v in TASK.answer.model_json_schema().items() if k != "description"}
    assert json.dumps(schema, separators=(",", ":")) in prompt and "{schema}" not in prompt


def test_v2_normalize_interrupted_and_failed() -> None:
    raw = [
        {"id": "u", "type": "user", "time": {"created": 1}, "payload": {"text": "hi"}},
        {"id": "a", "type": "assistant", "time": {"created": 2}, "model": {"id": "m", "providerID": "p"}, "content": [
            {"type": "tool", "id": "t1", "name": "shell", "state": {"status": "running", "input": '{"command": "ls"'}},
            {"type": "tool", "id": "t2", "name": "read", "state": {"status": "error", "input": {"path": "x"},
                                                                   "error": {"type": "e", "message": "nope"}}},
        ]},
        {"id": "i", "type": "idle", "outcome": "failed", "error": {"type": "provider", "message": "rate limited"}},
    ]  # fmt: skip
    normalized = normalize_messages(raw)
    assert normalized[0]["parts"][0]["text"] == "hi"
    first, second = normalized[1]["parts"]
    assert first["state"]["status"] == "error" and "running" in first["state"]["error"]
    assert first["state"]["input"] == {"raw": '{"command": "ls"'}
    assert second["state"]["error"] == "nope"
    assert normalized[1]["info"]["error"]["data"]["message"] == "rate limited"


def test_server_envs_keep_secrets_out(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value_that_must_not_leak")
    monkeypatch.setenv("OPENCODE_PASSWORD", "pw")
    student = Endpoint(base_url="http://vllm/v1", model="policy")
    teacher = Endpoint(
        base_url="",
        model="zai-org/GLM-5.2",
        key_env="HF_TOKEN",
        native_id="huggingface/zai-org/GLM-5.2",
        headers={"X-HF-Bill-To": "huggingface"},
    )
    env = server_env(TASK, [student], "2.3.2", tmp_path)
    assert "HF_TOKEN" not in env and "OPENCODE_PASSWORD" not in env
    assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1" and env["XDG_DATA_HOME"].startswith(str(tmp_path))
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["agents"][AGENT_ID]["system"] == TASK.system_prompt_for("opencode", "2.3.2", "final_json")
    assert config["provider"]["local"]["options"]["baseURL"] == "http://vllm/v1"
    teacher_env = opencode.server_env([teacher])
    assert teacher_env["HF_TOKEN"] == "hf_secret_value_that_must_not_leak"  # the teacher's own key only
    teacher_config = opencode.provider_config([teacher])
    assert teacher_config["provider"]["huggingface"]["options"]["headers"] == {"X-HF-Bill-To": "huggingface"}
    assert opencode.model_ref(teacher) == ("huggingface", "zai-org/GLM-5.2")


def test_version_parsing_and_profiles() -> None:
    assert opencode.parse_version("1.2.4\n") == "1.2.4"
    assert opencode.parse_version("opencode v2.3.2") == "2.3.2"
    assert opencode.parse_version("") is None
    assert isinstance(load_harness(V1), OpenCodeV1) and isinstance(load_harness(V2), OpenCodeV2)
    with pytest.raises(ValueError):
        OpenCodeV1(V2)


# --------------------------------------------------------------------------------------------------
# Opt-in: the real pinned binaries against a fake model
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("AGENT_DISTILL_REAL_OPENCODE"), reason="set AGENT_DISTILL_REAL_OPENCODE=1")
@pytest.mark.parametrize("profile", [V1, V2], ids=["1.2.4", "2.3.2"])
def test_real_opencode_against_a_fake_model(profile, tmp_path, monkeypatch) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests: list[dict[str, Any]] = []
    answer = {"reasoning": "x", "github_url": None, "project_page_url": None}

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(request)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            base = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "policy"}
            tools = [tool["function"]["name"] for tool in request.get("tools") or []]
            if "StructuredOutput" in tools:
                call = {"index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "StructuredOutput", "arguments": json.dumps(answer)}}  # fmt: skip
                delta, finish = {"role": "assistant", "tool_calls": [call]}, "tool_calls"
            else:
                delta, finish = {"role": "assistant", "content": json.dumps(answer)}, "stop"
            for chunk in (
                {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ):  # fmt: skip
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    model_server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=model_server.serve_forever, daemon=True).start()
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak_into_the_agent_env")
    endpoint = Endpoint(base_url=f"http://127.0.0.1:{model_server.server_address[1]}/v1", model="policy")
    harness = load_harness(profile)
    cache = Path(os.environ.get("AGENT_DISTILL_CACHE", tmp_path))
    row = ROW_OF["correct_and_complete"]

    async def go():
        async with harness.session([endpoint], cache=cache, task=TASK) as runner:
            await harness.healthy(runner)
            return await harness.run_episode(runner, TASK, row, endpoint, Limits(timeout=60, max_tool_calls=15))

    episode = asyncio.run(go())
    model_server.shutdown()
    assert episode.answer is not None and episode.answer.model_dump() == answer, episode.failure
    assert episode.harness_id == profile.id
    agent_requests = [r for r in requests if any(m.get("role") == "user" for m in r["messages"])]
    tools = sorted(tool["function"]["name"] for tool in agent_requests[-1].get("tools") or [])
    if profile.version == "2.3.2":
        assert len(requests) == 1  # no title-generation call
        assert tools == ["glob", "grep", "read", "shell", "webfetch"]
        assert agent_requests[-1]["messages"][0]["content"].startswith(
            TASK.system_prompt_for("opencode", "2.3.2", "final_json")
        )
    else:
        assert tools == ["StructuredOutput", "bash", "glob", "grep", "read", "webfetch"]
        assert episode.answer_mode == "native"
    assert "hf_should_not_leak" not in json.dumps(requests)
