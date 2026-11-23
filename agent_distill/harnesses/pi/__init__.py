"""The `pi@1.6.0` harness adapter (DESIGN.md §6).

Pi (`@earendil-works/pi-coding-agent`) runs one process per episode:

    node cli.js --mode json --no-context-files --no-extensions -e extension.ts --no-skills --no-prompt-templates
        --no-themes --no-mcp --no-approve --session-dir <dir> --provider agent-distill --model <id>
        --tools <tools>,submit --system-prompt <task prompt> <row prompt>

- **Pinned install.** Node (an official tarball, sha256-checked) and Pi with its whole dependency tree from the
  lockfile shipped next to this module (`npm ci`, which checks every package's integrity). Never a `pi` or `node`
  found on PATH.
- **Endpoint.** `PI_CODING_AGENT_DIR` is a per-episode directory whose `models.json` declares one OpenAI-compatible
  provider (`agent-distill`) with the endpoint's model. A key is referenced by its environment variable name
  (`"$HF_TOKEN"`), never written to disk; a non-secret literal key (the GRPO capture session id) is written as is.
- **Credentials.** The Pi process gets a credential-free environment plus the one key its endpoint names. The
  extension's `bash` drops that key again, so the agent's commands never see it.
- **Extension** (`extension.ts`): the `submit` tool (the answer schema, or no arguments for a `FileAnswer`), the
  exact tool-call limit, the credential-free `bash`.
- **Records.** Pi's own session file is the episode's native trace (the agent-traces viewer reads it as is); the
  episode's chat messages and tool events are converted from it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_distill.config import HarnessProfile
from agent_distill.credentials import strip_credentials
from agent_distill.episode import Episode, ToolEvent
from agent_distill.harnesses.base import Endpoint, architecture, download_verified
from agent_distill.harnesses.opencode_v1_reader import answer_from_text, json_text
from agent_distill.task import FileAnswer, Limits, Task

HERE = Path(__file__).parent
EXTENSION = HERE / "extension.ts"
PROVIDER = "agent-distill"
CLI = "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
# Task tool kinds -> Pi built-in tools, and back.
KIND_TOOLS = {
    "shell": ("bash",),
    "read": ("read",),
    "search": ("grep", "find", "ls"),
    "edit": ("edit",),
    "write": ("write",),
}
TOOL_KINDS = {tool: kind for kind, tools in KIND_TOOLS.items() for tool in tools} | {"submit": "submit"}
TOOL_LIMIT_MARKER = "tool-call limit reached"
_INSTALL_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------------------------


def install_node(profile: HarnessProfile, cache: Path) -> Path:
    """The pinned Node binary (the profile's `[download]`), extracted once into `cache`."""
    arch = architecture()
    sha = profile.download.sha256.get(arch)
    if not sha:
        raise RuntimeError(f"{profile.id}: no pinned Node sha256 for {arch}")
    url = profile.download.url.format(arch=arch)
    root = Path(cache) / Path(url).name.removesuffix(".tar.gz")
    node = root / "bin" / "node"
    if not node.exists():
        archive = download_verified(url, sha)
        staging = Path(tempfile.mkdtemp(prefix="node-", dir=cache))
        with tarfile.open(fileobj=__import__("io").BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(staging, filter="tar")
        (extracted,) = staging.iterdir()
        if root.exists():
            shutil.rmtree(root)
        extracted.rename(root)
        shutil.rmtree(staging, ignore_errors=True)
    return node


def install(profile: HarnessProfile, cache: Path) -> tuple[Path, Path]:
    """(node, Pi's cli.js): Pi and its dependencies from the shipped lockfile (`npm ci` checks every package's
    integrity), with the pinned Node. The installed Pi must report the profile's version."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    lock = HERE / f"pi@{profile.version}"
    if not (lock / "package-lock.json").exists():
        raise RuntimeError(f"{profile.id}: no lockfile shipped in {lock}")
    with _INSTALL_LOCK:
        node = install_node(profile, cache)
        root = cache / f"pi-{profile.version}"
        cli = root / CLI
        if not cli.exists():
            root.mkdir(parents=True, exist_ok=True)
            for name in ("package.json", "package-lock.json"):
                shutil.copy(lock / name, root / name)
            npm = node.parent.parent / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
            env = {**strip_credentials(dict(os.environ)), "PATH": f"{node.parent}:{os.environ.get('PATH', '')}"}
            subprocess.run([str(node), str(npm), "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=root,
                           env=env, check=True, capture_output=True, timeout=600)  # fmt: skip
        reported = subprocess.run([str(node), str(cli), "--version"], capture_output=True, text=True, timeout=60,
                                  env=pi_base_env()).stdout.strip()  # fmt: skip
    if reported != profile.version:
        raise RuntimeError(f"{cli} reports Pi {reported!r}, but the profile pins {profile.version}")
    return node, cli


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def pi_base_env() -> dict[str, str]:
    """A credential-free environment with Pi's network side effects off (no telemetry, version check or catalog
    refresh)."""
    env = strip_credentials(dict(os.environ))
    env.update(PI_OFFLINE="1", PI_TELEMETRY="0", PI_SKIP_VERSION_CHECK="1")
    return env


def models_json(endpoint: Endpoint) -> dict[str, Any]:
    if endpoint.key_env:
        api_key = f"${endpoint.key_env}"  # interpolated by Pi from its environment; never written to disk
    else:
        api_key = endpoint.api_key or "EMPTY"
    model = {
        "id": endpoint.model,
        "name": endpoint.name or endpoint.model,
        "contextWindow": endpoint.context,
        "maxTokens": endpoint.max_output_tokens,
        "input": ["text"],  # text-only episodes (DESIGN.md §10): never send images to a text-only student
        "reasoning": False,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        **dict(endpoint.native_entry),
    }
    provider: dict[str, Any] = {
        "baseUrl": endpoint.base_url.rstrip("/"),
        "api": "openai-completions",
        "apiKey": api_key,
        "models": [model],
    }
    if endpoint.headers:
        provider["headers"] = dict(endpoint.headers)
    return {"providers": {PROVIDER: provider}}


def extension_config(task: Task, limits: Limits, endpoint: Endpoint) -> dict[str, Any]:
    if isinstance(task.answer, FileAnswer):
        submit: dict[str, Any] = {"mode": "file", "path": task.answer.path}
    else:
        schema = task.answer_schema() or {"type": "object"}
        submit = {"mode": "schema", "schema": {k: v for k, v in schema.items() if k not in ("title", "description")}}
    return {
        "submit": submit,
        "max_tool_calls": limits.max_tool_calls or 0,
        "strip_env": [endpoint.key_env] if endpoint.key_env else [],
    }


def tool_names(task: Task) -> list[str]:
    names = [tool for kind in task.tools for tool in KIND_TOOLS.get(kind, ())]
    if "shell" not in task.tools and "bash" in names:  # pragma: no cover - KIND_TOOLS keeps bash under shell
        names.remove("bash")
    return [*dict.fromkeys(names), "submit"]


def pi_command(node: Path, cli: Path, task: Task, endpoint: Endpoint, session_dir: Path, prompt: str,
               system_prompt: str | None) -> list[str]:  # fmt: skip
    command = [
        str(node), str(cli), "--mode", "json",
        "--no-context-files", "--no-extensions", "-e", str(EXTENSION),
        "--no-skills", "--no-prompt-templates", "--no-themes", "--no-mcp", "--no-approve",
        "--session-dir", str(session_dir),
        "--provider", PROVIDER, "--model", endpoint.model,
        "--thinking", "off",
        "--tools", ",".join(tool_names(task)),
    ]  # fmt: skip
    if system_prompt:
        command += ["--system-prompt", system_prompt]
    return [*command, "--", prompt]


# --------------------------------------------------------------------------------------------------
# Pi session -> Episode
# --------------------------------------------------------------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(str(block.get("text", "")) for block in content or [] if block.get("type") == "text")


def system_text(message: dict[str, Any]) -> str:
    """The system prompt Pi composed: its content followed by its named sections (preamble, cwd, ...)."""
    parts = [_text(message.get("content"))] + [str(v) for v in (message.get("sections") or {}).values() if v]
    return "\n\n".join(part for part in parts if part)


def session_messages(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [line["message"] for line in lines if line.get("type") == "message" and line.get("message")]


def chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pi messages -> OpenAI-style chat messages (reasoning as `reasoning_content`; Pi's own `timestamp`, `model`,
    `usage` and `error` kept for the exporters)."""
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            result.append({"role": "system", "content": system_text(message), "timestamp": message.get("timestamp")})
        elif role == "user":
            result.append({"role": "user", "content": _text(message.get("content")),
                           "timestamp": message.get("timestamp")})  # fmt: skip
        elif role == "assistant":
            blocks = message.get("content") or []
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text"),
                "timestamp": message.get("timestamp"),
                "model": message.get("model"),
                "provider": message.get("provider"),
                "usage": _usage(message.get("usage")),
            }
            thinking = [str(b.get("thinking", "")).strip() for b in blocks if b.get("type") == "thinking"]
            if any(thinking):
                entry["reasoning_content"] = "\n\n".join(t for t in thinking if t)
            calls = [
                {"id": str(b.get("id")), "type": "function",
                 "function": {"name": str(b.get("name", "")), "arguments": json_text(b.get("arguments") or {})}}
                for b in blocks if b.get("type") == "toolCall"
            ]  # fmt: skip
            if calls:
                entry["tool_calls"] = calls
            if message.get("stopReason") in ("error", "aborted"):
                entry["error"] = {"stopReason": message.get("stopReason"), "message": message.get("errorMessage")}
            result.append(entry)
        elif role == "toolResult":
            result.append({
                "role": "tool",
                "tool_call_id": str(message.get("toolCallId")),
                "name": str(message.get("toolName", "")),
                "content": _text(message.get("content")),
                "is_error": bool(message.get("isError")),
                "timestamp": message.get("timestamp"),
            })  # fmt: skip
    return result


def _usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    return {
        "input": int(usage.get("input") or 0),
        "output": int(usage.get("output") or 0),
        "cache_read": int(usage.get("cacheRead") or 0),
        "cost": float((usage.get("cost") or {}).get("total") or 0),
    }


def tool_events(messages: list[dict[str, Any]]) -> list[ToolEvent]:
    """Every tool call in order, paired with its result (a call without a result is unfinished)."""
    results = {str(m.get("toolCallId")): m for m in messages if m.get("role") == "toolResult"}
    events = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for block in message.get("content") or []:
            if block.get("type") != "toolCall":
                continue
            name = str(block.get("name", ""))
            result = results.get(str(block.get("id")))
            text = _text(result.get("content")) if result else ""
            failed = result is None or bool(result.get("isError"))
            events.append(ToolEvent(kind=TOOL_KINDS.get(name.lower(), "other"), name=name,
                                    input=block.get("arguments") or {}, success=not failed,
                                    output=None if failed else text,
                                    error=(text or "unfinished") if failed else None))  # fmt: skip
    return events


def submitted_arguments(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The arguments of the last successful `submit` call."""
    ok = {str(m.get("toolCallId")) for m in messages if m.get("role") == "toolResult" and not m.get("isError")}
    found = None
    for message in messages:
        for block in message.get("content") or [] if message.get("role") == "assistant" else []:
            if block.get("type") == "toolCall" and block.get("name") == "submit" and str(block.get("id")) in ok:
                found = block.get("arguments") or {}
    return found


def episode_from_session(
    lines: list[dict[str, Any]],
    *,
    task: Task,
    row_id: str,
    version: str,
    workspace: Path | None = None,
    failure: str | None = None,
    elapsed_seconds: float | None = None,
) -> Episode:
    """A Pi session (its JSONL lines) -> `Episode`. A `FileAnswer` is read from `workspace`."""
    messages = session_messages(lines)
    chat = chat_messages(messages)
    events = tool_events(messages)
    header = next((line for line in lines if line.get("type") == "session"), {})
    model = next((f"{m.get('provider')}/{m.get('model')}" for m in messages if m.get("role") == "assistant"), "")
    usage = {"input": 0, "output": 0, "cost": 0.0}
    for entry in chat:
        if entry["role"] == "assistant":
            for key in usage:
                usage[key] += entry["usage"].get(key, 0)
    submitted = submitted_arguments(messages)
    answer: Any = None
    answer_error: str | None = None
    if isinstance(task.answer, FileAnswer):
        mode = "file"
        path = Path(workspace or ".") / task.answer.path
        if path.is_file():
            answer = path.read_text(encoding="utf-8", errors="replace")
        else:
            answer_error = f"{task.answer.path} was not written"
    else:
        mode = "submit_tool" if submitted is not None else "final_json"
        if submitted is not None:
            try:
                answer = task.answer.model_validate(submitted) if hasattr(task.answer, "model_validate") else submitted
            except ValueError as exc:
                answer_error = str(exc)[:500]
        else:
            last = next((e["content"] for e in reversed(chat) if e["role"] == "assistant" and e["content"]), "")
            answer = answer_from_text(task.answer, last)
            if answer is None:
                answer_error = "no submit call and no JSON answer in the final message"
    if failure is None:
        limit = next((e for e in events if e.error and TOOL_LIMIT_MARKER in e.error), None)
        errors = [e.get("error") for e in chat if e["role"] == "assistant" and e.get("error")]
        last = next((m for m in reversed(messages) if m.get("role") == "assistant"), {})
        if limit is not None:
            failure = limit.error
        elif last.get("stopReason") == "length" and submitted is None:
            failure = "response truncated at the output token limit"
        elif errors and submitted is None:
            failure = f"provider error: {(errors[-1] or {}).get('message') or errors[-1]}"[:500]
    return Episode(
        task=task.name,
        row_id=row_id,
        harness="pi",
        harness_version=version,
        model=model,
        messages=chat,
        tool_events=events,
        answer=answer,
        answer_mode=mode,
        answer_error=answer_error,
        usage=usage,
        elapsed_seconds=elapsed_seconds,
        failure=failure,
        session_id=str(header.get("id") or "") or None,
        metadata={"submitted": submitted is not None, "tools": _tools(messages)},
    )


def _tools(messages: list[dict[str, Any]]) -> list[str]:
    system = next((m for m in messages if m.get("role") == "system"), {})
    return [str(tool.get("name")) for tool in system.get("toolsAdded") or []]


def read_session(session_dir: Path) -> list[dict[str, Any]]:
    files = sorted(Path(session_dir).glob("*.jsonl"))
    if not files:
        return []
    lines = []
    for raw in files[-1].read_text(encoding="utf-8").split("\n"):
        if raw.strip():
            with suppress(ValueError):
                lines.append(json.loads(raw))
    return lines


# --------------------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------------------


@dataclass
class Runner:
    node: Path
    cli: Path
    task: Task
    root: Path  # per-session scratch: one workspace + config directory per episode


class Pi:
    def __init__(self, profile: HarnessProfile) -> None:
        if profile.name != "pi":
            raise ValueError(f"the Pi adapter cannot run {profile.id}")
        self.profile = profile
        self.name = profile.name
        self.version = profile.version

    def install(self, cache: Path) -> Path:
        return install(self.profile, cache)[1]

    @asynccontextmanager
    async def session(
        self, endpoints: list[Endpoint], *, cache: Path, task: Task, request_timeout: float = 600
    ) -> AsyncIterator[Runner]:
        """Pi has no server: the session installs it; every episode is its own process and workspace."""
        node, cli = await asyncio.to_thread(install, self.profile, cache)
        with tempfile.TemporaryDirectory(prefix="agent-distill-pi-") as root:
            yield Runner(node, cli, task, Path(root))

    async def healthy(self, runner: Runner) -> None:
        if not runner.cli.exists():
            raise RuntimeError(f"{runner.cli} vanished")

    async def run_episode(
        self, runner: Runner, task: Task, row: dict[str, Any], endpoint: Endpoint, limits: Limits, *, title: str = ""
    ) -> Episode:
        row_id = task.row_id(row)
        episode_dir = Path(tempfile.mkdtemp(prefix="episode-", dir=runner.root))
        workspace, agent_dir, sessions = episode_dir / "workspace", episode_dir / "agent", episode_dir / "sessions"
        for path in (workspace, agent_dir, sessions):
            path.mkdir()
        if task.setup is not None:
            await asyncio.to_thread(task.setup, row, workspace)
        (agent_dir / "models.json").write_text(json.dumps(models_json(endpoint), indent=2))
        config = agent_dir / "agent-distill.json"
        config.write_text(json.dumps(extension_config(task, limits, endpoint)))
        env = pi_base_env()
        if endpoint.key_env and endpoint.key_env in os.environ:
            env[endpoint.key_env] = os.environ[endpoint.key_env]
        env.update(PI_CODING_AGENT_DIR=str(agent_dir), AGENT_DISTILL_PI_CONFIG=str(config))
        system_prompt = task.system_prompt_for("pi", self.version, "submit_tool")
        command = pi_command(runner.node, runner.cli, task, endpoint, sessions, task.render_prompt(row), system_prompt)
        started = time.monotonic()
        failure: str | None = None
        process = await asyncio.create_subprocess_exec(
            *command, cwd=workspace, env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True, limit=64 * 1024 * 1024,
        )  # fmt: skip
        stderr = bytearray()

        async def drain() -> None:  # stdout must be read continuously or Pi stalls; the session file is the record
            assert process.stdout is not None and process.stderr is not None
            await asyncio.gather(_discard(process.stdout), _collect(process.stderr, stderr))

        try:
            async with asyncio.timeout(limits.timeout):
                await drain()
                await process.wait()
        except TimeoutError:
            failure = f"episode timed out after {limits.timeout:g} seconds"
        finally:
            await _terminate(process)
        if failure is None and process.returncode:
            tail = stderr.decode(errors="replace").strip().splitlines()[-3:]
            failure = f"pi exited with {process.returncode}: {' | '.join(tail)}"[:500]
        lines = read_session(sessions)
        elapsed = round(time.monotonic() - started, 3)
        episode = episode_from_session(lines, task=task, row_id=row_id, version=self.version, workspace=workspace,
                                       failure=failure, elapsed_seconds=elapsed)  # fmt: skip
        episode.metadata["cwd"] = str(workspace)
        episode.raw = {"session": lines, "stderr": stderr.decode(errors="replace")[-4000:]}
        return episode

    def native_trace(self, episode: Episode) -> list[dict[str, Any]] | None:
        raw = episode.raw if isinstance(episode.raw, dict) else {}
        return raw.get("session") or None


async def _discard(stream: asyncio.StreamReader) -> None:
    while await stream.read(1 << 16):
        pass


async def _collect(stream: asyncio.StreamReader, sink: bytearray) -> None:
    while chunk := await stream.read(1 << 16):
        sink.extend(chunk)
        del sink[: max(0, len(sink) - 65536)]


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), 10)
    except TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
