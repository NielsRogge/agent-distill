"""The `opencode@1.2.4` harness adapter (DESIGN.md §6).

Ported from url-extraction `harness/opencode.py` (`run_one`, `message_body`, `opencode_server`): one
`opencode serve` per session, an episode is a session whose prompt request carries the system prompt, the tool
toggles and the structured-output schema (`format: json_schema`). `POST /session/{id}/message` blocks until the
turn ends; a watchdog aborts the session at the tool-call limit, and when the provider rejects native structured
output the prompt is retried asking for a JSON-only message. Messages become an `Episode` through
`opencode_v1_reader`, the reader that rescored every published url-extraction record.
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
from agent_distill.harnesses.base import Endpoint, HarnessError
from agent_distill.harnesses.opencode_v1_reader import episode_from_messages
from agent_distill.task import FileAnswer, Limits, Task

# Every tool OpenCode 1.2.4 knows; the request enables only those the task's tool kinds map to.
OPENCODE_TOOLS = (
    "invalid", "question", "bash", "read", "glob", "grep", "edit", "write", "task", "webfetch", "todowrite",
    "websearch", "codesearch", "skill", "apply_patch",
)  # fmt: skip
KIND_TOOLS = {
    "shell": ("bash",),
    "fetch": ("webfetch",),
    "read": ("read",),
    "search": ("grep", "glob"),
    "edit": ("edit", "apply_patch"),
    "write": ("write",),
}
SESSION_PERMISSIONS = [
    {"permission": "*", "pattern": "*", "action": "allow"},
    {"permission": "external_directory", "pattern": "*", "action": "deny"},
    {"permission": "doom_loop", "pattern": "*", "action": "deny"},
]
STRUCTURED_OUTPUT_RETRY = (
    "The provider rejected native structured output. Complete the task above and return only one JSON object "
    "with {fields}."
)


def allowed_tools(kinds: tuple[str, ...] | list[str]) -> set[str]:
    return {tool for kind in kinds for tool in KIND_TOOLS.get(kind, ())}


def message_body(task: Task, row: dict[str, Any], model: tuple[str, str], *, structured: bool = True) -> dict[str, Any]:
    """The prompt request: system prompt, tool toggles, structured-output schema, user prompt."""
    allowed = allowed_tools(task.tools)
    body: dict[str, Any] = {
        "model": {"providerID": model[0], "modelID": model[1]},
        "system": task.system_prompt_for("opencode", "1.2.4", "native"),
        "tools": {name: name in allowed for name in OPENCODE_TOOLS},
    }
    schema = task.answer_schema()
    if structured and schema is not None:
        body["format"] = {"type": "json_schema", "schema": schema, "retryCount": 2}
        body["parts"] = [{"type": "text", "text": task.render_prompt(row)}]
    elif structured:
        body["parts"] = [{"type": "text", "text": task.render_prompt(row)}]
    else:
        fields = list((schema or {}).get("properties", {}))
        listed = ", ".join(fields[:-1]) + f", and {fields[-1]}" if len(fields) > 1 else "".join(fields)
        body["parts"] = [{"type": "text", "text": STRUCTURED_OUTPUT_RETRY.format(fields=listed)}]
    if body["system"] is None:
        del body["system"]
    return body


async def raw_json(client: httpx.AsyncClient, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    response = await client.request(method.upper(), path, json=body)
    response.raise_for_status()
    return response.json()


@dataclass
class Runner:
    """A running OpenCode 1.2.4 server and an HTTP client for it."""

    server: opencode.Server
    client: httpx.AsyncClient


class OpenCodeV1:
    def __init__(self, profile: HarnessProfile, *, answer_retry: bool = False) -> None:
        if not profile.version.startswith("1."):
            raise ValueError(f"OpenCodeV1 cannot run OpenCode {profile.version}")
        if answer_retry:
            raise ValueError("answer_retry is for OpenCode 2.x (1.x forces a tool call on every turn)")
        self.profile = profile
        self.name = profile.name
        self.version = profile.version

    def install(self, cache: Path | None = None) -> Path:
        return opencode.install(self.profile, cache)

    @asynccontextmanager
    async def session(
        self,
        endpoints: list[Endpoint],
        *,
        cache: Path | None = None,
        task: Task | None = None,
        request_timeout: float = 600,
    ) -> AsyncIterator[Runner]:
        """`opencode serve` in a fresh temporary workspace, configured for `endpoints`, without credentials: keyed
        endpoints are reached through a key proxy with a session key (one per session: 1.x shares the server)."""
        executable = await asyncio.to_thread(self.install, cache)
        workspace = tempfile.TemporaryDirectory(prefix="agent-distill-opencode-")
        try:
            with opencode.key_proxies(endpoints) as routed:
                env = opencode.server_env(
                    routed, {"OPENCODE_CONFIG_CONTENT": json.dumps(opencode.provider_config(routed))}
                )
                server = await asyncio.to_thread(
                    opencode.start_server, executable, self.version, Path(workspace.name), env, startup_timeout=30
                )
                try:
                    async with httpx.AsyncClient(
                        base_url=server.url, timeout=httpx.Timeout(request_timeout, connect=10)
                    ) as client:
                        yield Runner(server, client)
                finally:
                    await asyncio.to_thread(opencode.stop_server, server)
        finally:
            workspace.cleanup()

    async def healthy(self, runner: Runner) -> None:
        await raw_json(runner.client, "get", "/global/health")

    async def run_episode(
        self, runner: Runner, task: Task, row: dict[str, Any], endpoint: Endpoint, limits: Limits, *, title: str = ""
    ) -> Episode:
        if isinstance(task.answer, FileAnswer):
            raise NotImplementedError("opencode@1.2.4 shares one workspace between sessions; file answers need Pi")
        client = runner.client
        model = opencode.model_ref(endpoint)
        row_id = task.row_id(row)
        started = time.monotonic()
        try:
            body = {"title": title or f"{task.name} {row_id}", "permission": SESSION_PERMISSIONS}
            session = await raw_json(client, "post", "/session", body)
        except Exception as exc:  # the server is gone: nothing about the model to record
            raise HarnessError(f"OpenCode did not create a session: {type(exc).__name__}: {exc}") from exc
        session_id = session["id"]
        failure: str | None = None
        response: dict[str, Any] = {}
        watchdog_reason: str | None = None

        async def watchdog() -> None:
            nonlocal watchdog_reason
            if not limits.max_tool_calls:
                return
            while True:
                await asyncio.sleep(2)
                current = await raw_json(client, "get", f"/session/{session_id}/message")
                calls = sum(part.get("type") == "tool" for message in current for part in message.get("parts", []))
                if calls >= limits.max_tool_calls:
                    watchdog_reason = f"tool-call limit reached ({limits.max_tool_calls})"
                    await raw_json(client, "post", f"/session/{session_id}/abort")
                    return

        watchdog_task = asyncio.create_task(watchdog())
        try:
            async with asyncio.timeout(limits.timeout):
                path = f"/session/{session_id}/message"
                response = await raw_json(client, "post", path, message_body(task, row, model))
                if (response.get("info") or {}).get("error"):
                    response = await raw_json(client, "post", path, message_body(task, row, model, structured=False))
        except TimeoutError:
            failure = f"episode timed out after {limits.timeout:g} seconds"
            with suppress(Exception):
                await raw_json(client, "post", f"/session/{session_id}/abort")
        except Exception as exc:  # keep failed provider attempts for diagnosis and resumption
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            watchdog_task.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog_task
        try:
            messages = await raw_json(client, "get", f"/session/{session_id}/message")
        except Exception as exc:  # the server died mid-episode: a harness failure, not a model failure
            raise HarnessError(f"OpenCode stopped answering: {type(exc).__name__}: {exc}") from exc
        episode = episode_from_messages(
            messages,
            answer_model=task.answer,
            task=task.name,
            row_id=row_id,
            model=f"{model[0]}/{model[1]}",
            structured=(response.get("info") or {}).get("structured"),
            system_prompt=task.system_prompt_for("opencode", self.version, "native"),
            failure=failure or watchdog_reason,
            session_id=session_id,
            elapsed_seconds=round(time.monotonic() - started, 3),
            metadata={"cwd": str(runner.server.workspace)},
        )
        episode.raw = messages  # the harness-native messages, written to the private raw dataset
        return episode

    def native_trace(self, episode: Episode) -> None:
        return None
