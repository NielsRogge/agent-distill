from __future__ import annotations

from pathlib import Path

import pytest

from agent_distill.config import load_profile
from agent_distill.harnesses.base import Endpoint, load_harness, teacher_endpoint
from agent_distill.harnesses.opencode import key_proxies
from agent_distill.harnesses.pi import (
    EXTENSION,
    HERE,
    episode_from_session,
    extension_config,
    models_json,
    pi_command,
    tool_names,
)
from agent_distill.rubrics import RubricSet
from agent_distill.task import DatasetSpec, FileAnswer, Limits, Task, TextAnswer, load_task

PAPERS = load_task("examples/paper-urls")
PROFILE = load_profile("harnesses", "pi@1.6.0")


def html_task(**kw) -> Task:
    return Task(name="html", dataset=DatasetSpec(repo="x"), prompt="p", answer=FileAnswer("index.html"),
                rubrics=RubricSet([]), row_id=lambda row: str(row["id"]), tools=("shell", "read", "write"),
                **kw)  # fmt: skip


def session(*messages: dict) -> list[dict]:
    header = {"type": "session", "version": 3, "id": "s1", "timestamp": "2026-11-23T00:00:00Z", "cwd": "/ws"}
    return [header] + [{"type": "message", "id": str(i), "message": m} for i, m in enumerate(messages)]


SYSTEM = {"role": "system", "content": "", "sections": {"preamble": "Be good.", "cwd": "<cwd>/ws</cwd>"},
          "toolsAdded": [{"name": "bash"}, {"name": "submit"}], "timestamp": 1}  # fmt: skip
USER = {"role": "user", "content": [{"type": "text", "text": "Make a page"}], "timestamp": 2}
USAGE = {"input": 10, "output": 5, "cacheRead": 0, "cost": {"total": 0.001}}


def assistant(*calls: tuple[str, str, dict], text: str = "", stop: str = "toolUse", **extra) -> dict:
    blocks = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": text}]
    blocks += [{"type": "toolCall", "id": cid, "name": name, "arguments": args} for cid, name, args in calls]
    return {"role": "assistant", "content": blocks, "provider": "agent-distill", "model": "m", "usage": USAGE,
            "stopReason": stop, "timestamp": 3, **extra}  # fmt: skip


def result(cid: str, name: str, text: str, error: bool = False) -> dict:
    return {"role": "toolResult", "toolCallId": cid, "toolName": name, "content": [{"type": "text", "text": text}],
            "isError": error, "timestamp": 4}  # fmt: skip


def test_profile_and_wheel_data():
    harness = load_harness(PROFILE)
    assert harness.name == "pi" and harness.version == "1.6.0" and PROFILE.answer_mode == "submit_tool"
    assert (HERE / "pi@1.6.0" / "package-lock.json").exists() and EXTENSION.exists()
    lock = (HERE / "pi@1.6.0" / "package-lock.json").read_text()
    assert '"node_modules/@earendil-works/pi-coding-agent": {\n      "version": "1.6.0"' in lock


def test_models_json_names_the_key_never_its_value(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecretsecretsecret")
    teacher = teacher_endpoint(load_profile("teachers", "glm-5.3"), PROFILE, "org")
    with pytest.raises(ValueError, match="key_proxies"):
        models_json(teacher)  # a real key never reaches Pi's config
    with key_proxies([teacher]) as (routed,):
        provider = models_json(routed)["providers"]["agent-distill"]
        assert provider["apiKey"] == routed.session_key and provider["baseUrl"].startswith("http://127.0.0.1:")
        assert "headers" not in provider and "hf_secret" not in str(models_json(routed))  # the proxy adds them
    assert provider["models"][0]["maxTokens"] == 65536 and provider["models"][0]["input"] == ["text"]
    student = Endpoint("http://127.0.0.1:8000/v1", "student", max_output_tokens=2048)
    assert models_json(student)["providers"]["agent-distill"]["apiKey"] == "EMPTY"
    capture = Endpoint("http://127.0.0.1:9000/v1", "policy", api_key="capture-123")
    assert models_json(capture)["providers"]["agent-distill"]["apiKey"] == "capture-123"
    assert extension_config(html_task(), Limits(max_tool_calls=7), teacher) == {
        "submit": {"mode": "file", "path": "index.html"}, "max_tool_calls": 7, "strip_env": []
    }  # fmt: skip
    schema = extension_config(PAPERS, Limits(), student)["submit"]
    assert (
        schema["mode"] == "schema"
        and "properties" in schema["schema"]
        and extension_config(PAPERS, Limits(), student)["strip_env"] == []
    )


def test_command_isolates_pi():
    command = pi_command(Path("/n"), Path("/cli.js"), html_task(), Endpoint("u", "m"), Path("/s"), "go", "sys")
    for flag in ("--no-context-files", "--no-extensions", "--no-skills", "--no-mcp", "--no-approve"):
        assert flag in command
    assert command[command.index("-e") + 1] == str(EXTENSION)
    assert command[command.index("--tools") + 1] == "bash,read,write,submit" == ",".join(tool_names(html_task()))
    assert command[-2:] == ["--", "go"] and command[command.index("--system-prompt") + 1] == "sys"


def text_task(**kw) -> Task:
    return Task(name="qa", dataset=DatasetSpec(repo="x"), prompt="p", answer=TextAnswer(), rubrics=RubricSet([]),
                row_id=lambda row: str(row["id"]), tools=("shell", "read"), **kw)  # fmt: skip


def test_text_answer_task_has_no_submit_tool_and_loads_its_skills():
    task = text_task(skills=(Path("/skills/hf-cli"),))
    assert tool_names(task) == ["bash", "read"] and task.answer_schema() is None
    assert extension_config(task, Limits(max_tool_calls=20), Endpoint("u", "m"))["submit"] == {"mode": "none"}
    command = pi_command(Path("/n"), Path("/cli.js"), task, Endpoint("u", "m"), Path("/s"), "go", None)
    assert command[command.index("--tools") + 1] == "bash,read"
    assert "--no-skills" in command and command[command.index("--skill") + 1] == "/skills/hf-cli"
    assert "--skill" not in pi_command(Path("/n"), Path("/cli.js"), html_task(), Endpoint("u", "m"), Path("/s"),
                                       "go", None)  # fmt: skip


def test_text_answer_episode():
    answer = "The top trending model is [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)."
    lines = session(SYSTEM, USER, assistant(("c1", "bash", {"command": "hf models list"})),
                    result("c1", "bash", "Qwen/Qwen3.5-9B"), assistant(text=answer, stop="stop"))  # fmt: skip
    episode = episode_from_session(lines, task=text_task(), row_id="1", version="1.6.0")
    assert episode.answer == answer and episode.answer_mode == "text" and episode.answer_error is None
    assert episode.failure is None and episode.metadata["submitted"] is False
    # Cut off at the tool-call limit: the last assistant turn still calls a tool, so there is no answer.
    limited = session(SYSTEM, USER, assistant(("c1", "bash", {"command": "hf models list"}), text="Let me look."),
                      result("c1", "bash", "tool-call limit reached (1)", error=True))  # fmt: skip
    episode = episode_from_session(limited, task=text_task(), row_id="1", version="1.6.0")
    assert episode.answer is None and "no final answer" in episode.answer_error
    assert episode.failure == "tool-call limit reached (1)"


def test_file_answer_episode(tmp_path):
    (tmp_path / "index.html").write_text("<html>hi</html>")
    lines = session(SYSTEM, USER, assistant(("c1", "write", {"path": "index.html"}), ("c2", "bash", {"command": "x"})),
                    result("c1", "write", "ok"), result("c2", "bash", "boom", error=True),
                    assistant(("c3", "submit", {})), result("c3", "submit", "index.html submitted."))  # fmt: skip
    episode = episode_from_session(lines, task=html_task(), row_id="1", version="1.6.0", workspace=tmp_path)
    assert episode.answer == "<html>hi</html>" and episode.answer_mode == "file" and episode.failure is None
    assert [(e.kind, e.success) for e in episode.tool_events] == [("write", True), ("shell", False), ("submit", True)]
    assert episode.tool_events[1].error == "boom" and episode.metadata["submitted"] is True
    assert episode.usage == {"input": 20, "output": 10, "cost": 0.002}
    assert episode.harness_id == "pi@1.6.0" and episode.session_id == "s1" and episode.model == "agent-distill/m"
    roles = [m["role"] for m in episode.messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "tool"]
    assert episode.messages[0]["content"] == "Be good.\n\n<cwd>/ws</cwd>"
    assert episode.messages[2]["reasoning_content"] == "hmm" and len(episode.messages[2]["tool_calls"]) == 2
    missing = episode_from_session(lines, task=html_task(), row_id="1", version="1.6.0", workspace=tmp_path / "no")
    assert missing.answer is None and missing.answer_error == "index.html was not written"


def test_schema_answer_failures_and_fallbacks():
    answer = {"reasoning": "found it", "github_url": "https://github.com/a/b", "project_page_url": None}
    good = PAPERS.answer.model_validate(answer)
    lines = session(SYSTEM, USER, assistant(("c1", "submit", good.model_dump(mode="json"))),
                    result("c1", "submit", "Answer submitted."))  # fmt: skip
    episode = episode_from_session(lines, task=PAPERS, row_id="1", version="1.6.0")
    assert episode.answer == good and episode.answer_mode == "submit_tool"
    # No submit: the JSON final message is the fallback.
    text = good.model_dump_json()
    fallback = episode_from_session(session(SYSTEM, USER, assistant(text=text, stop="stop")), task=PAPERS, row_id="1",
                                    version="1.6.0")  # fmt: skip
    assert fallback.answer == good and fallback.answer_mode == "final_json"
    # The extension's blocked call past the limit, a truncated response, a provider error.
    limited = session(SYSTEM, USER, assistant(("c1", "bash", {})),
                      result("c1", "bash", "tool-call limit reached (1)", error=True))  # fmt: skip
    assert episode_from_session(limited, task=PAPERS, row_id="1", version="1.6.0").failure.startswith(
        "tool-call limit reached"
    )
    truncated = session(SYSTEM, USER, assistant(text="<html", stop="length"))
    assert "output token limit" in episode_from_session(truncated, task=PAPERS, row_id="1", version="1.6.0").failure
    errored = session(SYSTEM, USER, assistant(stop="error", errorMessage="502 bad gateway"))
    assert "502" in episode_from_session(errored, task=PAPERS, row_id="1", version="1.6.0").failure
    # A call without a result (the run was killed mid-tool-call) is unfinished.
    unfinished = episode_from_session(session(SYSTEM, USER, assistant(("c1", "bash", {}))), task=PAPERS, row_id="1",
                                      version="1.6.0", failure="episode timed out after 1 seconds")  # fmt: skip
    assert unfinished.tool_events[0].error == "unfinished"


def test_captured_tool_definitions():
    from agent_distill.harnesses.tools import harness_tools

    names = [tool["function"]["name"] for tool in harness_tools("pi@1.6.0", html_task())]
    assert names == ["bash", "read", "write", "submit"]
    submit = next(tool for tool in harness_tools("pi@1.6.0", PAPERS) if tool["function"]["name"] == "submit")
    assert "github_url" in submit["function"]["parameters"]["properties"]


def test_terminate_kills_the_agent_when_cancelled_while_waiting_for_it() -> None:
    """A second cancellation during the SIGTERM grace period (a stage shutting down) still kills the process group."""
    import asyncio
    import time

    from agent_distill.harnesses.pi import _terminate

    async def main() -> int:
        process = await asyncio.create_subprocess_exec(
            "sh", "-c", "trap '' TERM; while :; do sleep 0.05; done", start_new_session=True
        )
        await asyncio.sleep(0.2)  # let the shell install its trap
        terminating = asyncio.create_task(_terminate(process))
        await asyncio.sleep(0.2)
        terminating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await terminating
        return await asyncio.wait_for(process.wait(), 5)

    started = time.monotonic()
    assert asyncio.run(main()) != 0  # killed, long before the 10-second grace period
    assert time.monotonic() - started < 5


def test_event_stream_splits_on_lf_only_and_survives_bad_records_and_observers() -> None:
    """`on_event` gets every JSON record of Pi's stream: split on LF only (U+2028 is valid inside a string), across
    read boundaries; a record that does not parse, or an observer that raises, never stops the stream."""
    import asyncio
    import json

    from agent_distill.harnesses.pi import _events

    records = [{"type": "session", "id": "s"}, {"type": "message_update", "delta": "a" + chr(0x2028) + "b"},
               {"type": "tool_execution_start", "toolName": "bash"}]  # fmt: skip
    data = "\n".join(json.dumps(r, ensure_ascii=False) for r in records[:2]) + "\nnot json\n\n"
    data += json.dumps(records[2]) + "\n"
    seen: list[dict] = []

    def observe(event: dict) -> None:
        seen.append(event)
        if event["type"] == "session":
            raise RuntimeError("a broken observer")

    async def main() -> None:
        reader = asyncio.StreamReader()
        raw = data.encode()
        for start in range(0, len(raw), 7):  # records split across reads
            reader.feed_data(raw[start : start + 7])
        reader.feed_eof()
        await _events(reader, observe)

    asyncio.run(main())
    assert seen == records
