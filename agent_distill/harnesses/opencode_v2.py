"""The `opencode@2.3.2` harness adapter (DESIGN.md §6).

Ported from url-extraction `harness/opencode_v2.py` (branch `claude/opencode-v2-glm53-4b`). What differs from 1.x:

- Routes live under `/api/`, behind HTTP Basic auth (`opencode:<password>`). The server generates the password and
  prints it once; it is read from stdout, so it never enters the agent's environment.
- A prompt request carries only text. The system prompt and the tool set are a config agent (`agent-distill`, in
  OPENCODE_CONFIG_CONTENT): its `system` replaces OpenCode's base prompt, and permission rules deny every tool except
  the ones the task's tool kinds map to. The rules are repeated on the session, because OpenCode appends its own
  rules (e.g. `skill:browser` deny) after the agent's, and a tool is hidden only when the last rule matching it
  denies every resource.
- There is no structured output: the agent ends with a JSON-only message (the contract and schema are in the system
  prompt, `Task.system_prompt_for(..., "final_json")`), validated by the task's answer model.
- Prompts are asynchronous: post the prompt, then wait for the session to go idle.
- Messages are flat (`{id, type, content: [...]}`, one assistant message per model step). `normalize_messages`
  converts them to the 1.x `{info, parts}` shape, so the Episode conversion, scoring and Pi export are shared with
  1.x. Tool names stay as OpenCode 2 reports them (`shell`, not `bash`); tool kinds normalize them.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agent_distill.config import HarnessProfile
from agent_distill.episode import Episode
from agent_distill.harnesses import opencode
from agent_distill.harnesses.base import Endpoint
from agent_distill.harnesses.opencode_v1_reader import episode_from_messages
from agent_distill.task import FileAnswer, Limits, Task

AGENT_ID = "agent-distill"
PAGE_LIMIT = 200
KIND_TOOLS = {
    "shell": ("shell",),
    "fetch": ("webfetch",),
    "read": ("read",),
    "search": ("grep", "glob"),
    "edit": ("edit", "patch"),
    "write": ("write",),
}


def permissions(kinds: tuple[str, ...] | list[str]) -> list[dict[str, str]]:
    allowed = [tool for kind in kinds for tool in KIND_TOOLS.get(kind, ())]
    return [
        {"action": "*", "resource": "*", "effect": "deny"},
        *({"action": tool, "resource": "*", "effect": "allow"} for tool in allowed),
        # As 1.x's `external_directory: deny`: tools stay inside the episode's workspace (shell can still cd).
        {"action": "external_directory", "resource": "*", "effect": "deny"},
    ]


def server_config(task: Task, endpoints: list[Endpoint], version: str) -> dict[str, Any]:
    config = opencode.provider_config(endpoints)
    config["agents"] = {
        AGENT_ID: {
            "system": task.system_prompt_for("opencode", version, "final_json") or "",
            "mode": "primary",
            "description": f"agent-distill task {task.name}",
            "permissions": permissions(task.tools),
        }
    }
    return config


def xdg_dirs(workspace: Path) -> dict[str, str]:
    names = (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"), ("XDG_STATE_HOME", "state"))
    return {key: str(workspace / f".opencode-{name}") for key, name in names}


def server_env(task: Task, endpoints: list[Endpoint], version: str, workspace: Path) -> dict[str, str]:
    """No server password in the environment (it is generated), no ~/.claude prompts or skills, the bundled model
    catalog instead of a live models.dev fetch, per-workspace data/config/state directories."""
    env = opencode.server_env(
        endpoints,
        {
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_CONFIG_CONTENT": json.dumps(server_config(task, endpoints, version)),
            **xdg_dirs(workspace),
        },
    )
    for key in ("OPENCODE_PASSWORD", "OPENCODE_SERVER_PASSWORD"):
        env.pop(key, None)
    return env


# --------------------------------------------------------------------------------------------------
# Messages: OpenCode 2 -> the 1.x shape
# --------------------------------------------------------------------------------------------------


def _tool_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):  # a call still streaming its arguments
        with suppress(ValueError):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        return {"raw": value} if value else {}
    return {}


def _tool_text(state: dict[str, Any]) -> str:
    items = state.get("content") or []
    texts = [str(item.get("text", "")) for item in items if isinstance(item, dict) and item.get("type") == "text"]
    files = [str(item.get("uri", "")) for item in items if isinstance(item, dict) and item.get("type") == "file"]
    return "\n".join(texts + [f"[file {uri}]" for uri in files])


def _error(error: Any) -> dict[str, Any] | None:
    if not error:
        return None
    if isinstance(error, dict):
        return {"name": str(error.get("type") or "error"), "data": {"message": str(error.get("message") or error)}}
    return {"name": "error", "data": {"message": str(error)}}


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenCode 2 session messages -> 1.x-style `{info, parts}` messages (user and assistant only).

    Tool state: `completed` keeps its text content as `output`; `error` keeps `error.message`; a call still
    `streaming`/`running` when the episode ended (interrupted) becomes an `error` with that status. The final
    `idle` message's failure, if any, is attached to the last assistant message as its error.
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        kind = message.get("type")
        created = (message.get("time") or {}).get("created")
        if kind == "user":
            text = message.get("text")
            if text is None:
                text = (message.get("payload") or {}).get("text", "")
            info = {"id": message.get("id"), "role": "user", "time": {"created": created}}
            result.append({"info": info, "parts": [{"type": "text", "text": str(text)}]})
        elif kind == "assistant":
            model = message.get("model") or {}
            times = message.get("time") or {}
            info = {
                "id": message.get("id"),
                "role": "assistant",
                "providerID": model.get("providerID"),
                "modelID": model.get("id"),
                "agent": message.get("agent"),
                "finish": message.get("finish"),
                "cost": message.get("cost") or 0,
                "tokens": message.get("tokens") or {},
                "time": {"created": created, "completed": times.get("completed")},
            }
            if error := _error(message.get("error")):
                info["error"] = error
            parts: list[dict[str, Any]] = []
            for item in message.get("content") or []:
                item_type = item.get("type")
                if item_type in {"text", "reasoning"}:
                    parts.append({"type": item_type, "text": str(item.get("text") or "")})
                elif item_type == "tool":
                    state = item.get("state") or {}
                    status = state.get("status")
                    normalized: dict[str, Any] = {"status": status, "input": _tool_input(state.get("input"))}
                    if status == "completed":
                        normalized["output"] = _tool_text(state)
                    else:
                        error = state.get("error")
                        message_text = error.get("message") if isinstance(error, dict) else error
                        normalized["status"] = "error"
                        normalized["error"] = str(message_text or f"tool call {status or 'did not complete'}")
                    item_time = item.get("time") or {}
                    if isinstance(item_time, dict) and item_time.get("completed"):
                        normalized["time"] = {"end": item_time.get("completed")}
                    call_id = str(item.get("id"))
                    parts.append({"type": "tool", "id": call_id, "callID": call_id, "tool": str(item.get("name", "")),
                                  "state": normalized})  # fmt: skip
            result.append({"info": info, "parts": parts})
        elif kind == "idle" and message.get("outcome") == "failed" and message.get("error"):
            for entry in reversed(result):
                if entry["info"].get("role") == "assistant":
                    entry["info"].setdefault("error", _error(message.get("error")))
                    break
    return result


def tool_call_count(messages: list[dict[str, Any]]) -> int:
    return sum(
        item.get("type") == "tool"
        for message in messages
        if message.get("type") == "assistant"
        for item in message.get("content") or []
    )


# --------------------------------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------------------------------


async def api(
    client: httpx.AsyncClient, method: str, path: str, body: dict[str, Any] | None = None, **kwargs: Any
) -> Any:
    response = await client.request(method.upper(), path, json=body, **kwargs)
    response.raise_for_status()
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


async def list_messages(client: httpx.AsyncClient, session_id: str) -> list[dict[str, Any]]:
    """Every message of a session, oldest first (the endpoint pages at most 200 at a time)."""
    messages: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": PAGE_LIMIT, "order": "asc"}
        if cursor:
            params["cursor"] = cursor
        page = await api(client, "get", f"/api/session/{session_id}/message", params=params)
        data = page.get("data") or []
        messages.extend(data)
        cursor = (page.get("cursor") or {}).get("next")
        if not data or not cursor or len(data) < PAGE_LIMIT:
            return messages


async def wait_idle(client: httpx.AsyncClient, session_id: str) -> None:
    """Block until the session's loop has finished the prompt (an `idle` message follows the last user message)."""
    while True:
        with suppress(Exception):
            await api(client, "post", f"/api/experimental/session/{session_id}/wait")
        messages = await list_messages(client, session_id)
        if messages and messages[-1].get("type") == "idle":
            return
        await asyncio.sleep(0.5)


async def interrupt(client: httpx.AsyncClient, session_id: str) -> None:
    with suppress(Exception):
        await api(client, "post", f"/api/session/{session_id}/interrupt")


@dataclass
class Runner:
    server: opencode.Server
    client: httpx.AsyncClient
    task: Task


class OpenCodeV2:
    def __init__(self, profile: HarnessProfile) -> None:
        if not profile.version.startswith("2."):
            raise ValueError(f"OpenCodeV2 cannot run OpenCode {profile.version}")
        self.profile = profile
        self.name = profile.name
        self.version = profile.version

    def install(self, cache: Path) -> Path:
        return opencode.install(self.profile, cache)

    @asynccontextmanager
    async def session(
        self, endpoints: list[Endpoint], *, cache: Path, task: Task, request_timeout: float = 600
    ) -> AsyncIterator[Runner]:
        """`opencode serve` (2.x) in a fresh workspace with the task's config agent; one server per task."""
        executable = await asyncio.to_thread(self.install, cache)
        workspace = tempfile.TemporaryDirectory(prefix="agent-distill-opencode2-")
        root = Path(workspace.name)
        env = server_env(task, endpoints, self.version, root)
        server = await asyncio.to_thread(opencode.start_server, executable, self.version, root, env, startup_timeout=60)
        try:
            async with httpx.AsyncClient(
                base_url=server.url, auth=server.auth, timeout=httpx.Timeout(request_timeout, connect=10)
            ) as client:
                yield Runner(server, client, task)
        finally:
            await asyncio.to_thread(opencode.stop_server, server)
            workspace.cleanup()

    async def healthy(self, runner: Runner) -> None:
        (await runner.client.get("/api/info")).raise_for_status()

    async def run_episode(
        self, runner: Runner, task: Task, row: dict[str, Any], endpoint: Endpoint, limits: Limits, *, title: str = ""
    ) -> Episode:
        if isinstance(task.answer, FileAnswer):
            raise NotImplementedError("file answers in OpenCode 2 are not supported yet; use Pi")
        if task is not runner.task:
            raise ValueError("an OpenCode 2 server is configured for one task (its config agent)")
        client = runner.client
        model = opencode.model_ref(endpoint)
        row_id = task.row_id(row)
        started = time.monotonic()
        body = {
            "title": title or f"{task.name} {row_id}",
            "agent": AGENT_ID,
            "model": {"providerID": model[0], "id": model[1]},
            "permissions": permissions(task.tools),
            "location": {"directory": str(runner.server.workspace)},
        }
        session_id = str((await api(client, "post", "/api/session", body))["data"]["id"])
        failure: str | None = None
        timed_out = False
        watchdog_reason: str | None = None

        async def watchdog() -> None:
            nonlocal watchdog_reason
            if not limits.max_tool_calls:
                return
            while True:
                await asyncio.sleep(2)
                with suppress(Exception):
                    if tool_call_count(await list_messages(client, session_id)) >= limits.max_tool_calls:
                        watchdog_reason = f"tool-call limit reached ({limits.max_tool_calls})"
                        await interrupt(client, session_id)
                        return

        watchdog_task = asyncio.create_task(watchdog())
        try:
            async with asyncio.timeout(limits.timeout):
                await api(client, "post", f"/api/session/{session_id}/prompt", {"text": task.render_prompt(row)})
                await wait_idle(client, session_id)
        except TimeoutError:
            timed_out = True
            failure = f"episode timed out after {limits.timeout:g} seconds"
            await interrupt(client, session_id)
        except Exception as exc:  # keep failed provider attempts for diagnosis and resumption
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            watchdog_task.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog_task
        if watchdog_reason or timed_out:
            with suppress(Exception):  # let the interrupt settle so the last tool call is recorded as stopped
                async with asyncio.timeout(15):
                    await wait_idle(client, session_id)
        raw = await list_messages(client, session_id)
        episode = episode_from_messages(
            normalize_messages(raw),
            answer_model=task.answer,
            task=task.name,
            row_id=row_id,
            model=f"{model[0]}/{model[1]}",
            structured=None,
            system_prompt=task.system_prompt_for("opencode", self.version, "final_json"),
            failure=failure or watchdog_reason,
            session_id=session_id,
            elapsed_seconds=round(time.monotonic() - started, 3),
            metadata={"cwd": str(runner.server.workspace)},
        )
        episode.harness_version = self.version
        episode.raw = raw
        return episode

    def native_trace(self, episode: Episode) -> None:
        return None
