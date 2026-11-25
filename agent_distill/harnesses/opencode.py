"""Shared OpenCode plumbing for the 1.2.4 and 2.3.2 adapters: install, provider config, server lifecycle.

Ported from url-extraction `harness/opencode.py` (main, and the OpenCode v2 branch `claude/opencode-v2-glm53-4b`).
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from agent_distill.config import HarnessProfile
from agent_distill.credentials import strip_credentials
from agent_distill.harnesses.base import (
    Endpoint,
    HarnessError,
    architecture,
    default_cache,
    download_verified,
    extract_member,
    member_bytes,
)
from agent_distill.harnesses.key_proxy import KeyProxy

LOCAL_PROVIDER = "local"
HF_ROUTER = "https://router.huggingface.co/v1"

# Tool failures that mean the OpenCode server itself is broken (e.g. its workspace vanished, so every shell spawn
# fails), not that the model did something wrong. Episodes hitting them say nothing about the model.
HARNESS_FAILURE_MARKERS = ("posix_spawn",)


def parse_version(output: str) -> str | None:
    """`opencode --version` output -> "1.2.4". 1.x prints `1.2.4`, 2.x prints `opencode v2.3.2`."""
    tokens = output.strip().split()
    if not tokens:
        return None
    version = tokens[-1].lstrip("v")
    return version if re.fullmatch(r"\d+\.\d+\.\d+(?:[-+.\w]*)?", version) else None


def binary_version(executable: str | Path) -> str | None:
    try:
        result = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_version(result.stdout or result.stderr)


def install(profile: HarnessProfile, cache: Path | None = None) -> Path:
    """The pinned OpenCode binary in `cache` (default: the per-user cache); never one found on PATH.

    The release archive is kept in the cache and re-hashed against the profile's sha256 on every call, and the
    binary is re-extracted whenever it differs from the archive's, so a cached file that was replaced or modified is
    never run.
    """
    arch = architecture()
    sha = profile.download.sha256.get(arch)
    if not sha:
        raise RuntimeError(f"{profile.id}: no pinned sha256 for {arch}")
    folder = Path(cache or default_cache()) / f"opencode-{profile.version}-{arch}"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    archive_path, destination = folder / "opencode.tar.gz", folder / "opencode"
    archive = archive_path.read_bytes() if archive_path.exists() else b""
    if hashlib.sha256(archive).hexdigest() != sha:
        archive = download_verified(profile.download.url.format(version=profile.version, arch=arch), sha)
        archive_path.write_bytes(archive)
    binary = member_bytes(archive, "opencode")
    current = destination.read_bytes() if destination.exists() else b""
    if current != binary:
        extract_member(archive, "opencode", destination)
    reported = binary_version(destination)
    if reported != profile.version:
        raise RuntimeError(f"{destination} reports OpenCode {reported}, but the profile pins {profile.version}")
    return destination


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def provider_config(endpoints: list[Endpoint]) -> dict[str, Any]:
    """OpenCode 1.x `provider` config for the endpoints (2.x migrates it on read).

    An endpoint with a native id `huggingface/...` uses OpenCode's built-in Hugging Face provider, with its headers
    (e.g. `X-HF-Bill-To`) and an inline model entry for ids missing from OpenCode's catalog. Any other endpoint
    becomes an OpenAI-compatible provider named `local` (vLLM, the GRPO capture proxy). An endpoint that needs a key
    must have been routed through a `KeyProxy` first (`key_proxies`): the harness then gets the proxy's URL and
    session key, never the real key.
    """
    providers: dict[str, Any] = {}
    for endpoint in endpoints:
        if endpoint.key_env:
            raise ValueError(f"{endpoint.model}: an endpoint with a key must go through key_proxies()")
        provider, model = model_ref(endpoint)
        if provider == LOCAL_PROVIDER:
            entry = providers.setdefault(
                LOCAL_PROVIDER,
                {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Local OpenAI-compatible server",
                    "options": {"baseURL": endpoint.base_url.rstrip("/"), "apiKey": "EMPTY"},
                    "models": {},
                },
            )
            if endpoint.session_key:  # through the key proxy (which adds the real key and the headers)
                entry["options"]["apiKey"] = endpoint.session_key
            elif endpoint.api_key:  # a non-secret key, e.g. the OpenEnv capture session id
                entry["options"]["apiKey"] = endpoint.api_key
            if endpoint.headers:
                entry["options"].setdefault("headers", {}).update(endpoint.headers)
            entry["models"][model] = {
                "name": endpoint.name or model,
                "limit": {"context": endpoint.context, "output": endpoint.max_output_tokens},
                **endpoint.native_entry,
            }
        else:
            entry = providers.setdefault(provider, {})
            if endpoint.session_key:  # through the key proxy (which adds the real key and the headers)
                entry.setdefault("options", {}).update(baseURL=endpoint.base_url.rstrip("/"),
                                                       apiKey=endpoint.session_key)  # fmt: skip
            if endpoint.headers:
                entry.setdefault("options", {}).setdefault("headers", {}).update(endpoint.headers)
            if endpoint.native_entry:
                entry.setdefault("models", {})[model] = dict(endpoint.native_entry)
    config: dict[str, Any] = {"provider": providers}
    small = [endpoint for endpoint in endpoints if endpoint.small_model]
    if small:
        config["small_model"] = "/".join(model_ref(small[0]))
    return config


WORKDIR_GUARD = Path(__file__).parent / "opencode_plugins" / "workdir-guard.js"
DEFAULT_WORKSPACE_PREFIX = "agent-distill-opencode-"


def with_workdir_guard(config: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """`config` plus the workdir-guard plugin: a bash call naming a missing `workdir` fails with an error the model
    sees, instead of hanging until the episode times out (OpenCode 1.2.4 never spawns such a command)."""
    if enabled:
        config = {**config, "plugin": [*config.get("plugin", []), WORKDIR_GUARD.resolve().as_uri()]}
    return config


def workspace_prefix(task: Any, default: str = DEFAULT_WORKSPACE_PREFIX) -> str:
    """The temporary workspace's name prefix: the task's `workspace_prefix`, else `default`. The agent sees the path
    (OpenCode puts the working directory in its prompt), so parity runs use the reference harness's."""
    return getattr(task, "workspace_prefix", "") or default


def model_ref(endpoint: Endpoint) -> tuple[str, str]:
    """(providerID, modelID) OpenCode uses for an endpoint."""
    if endpoint.native_id:
        provider, _, model = endpoint.native_id.partition("/")
        return provider, model
    return LOCAL_PROVIDER, endpoint.model


def server_env(endpoints: list[Endpoint], extra: dict[str, str] | None = None) -> dict[str, str]:
    """The `opencode serve` environment (inherited by every shell the agent runs).

    Credentials are stripped and none is passed back in: teachers are reached through a `KeyProxy`, whose session
    key is in the provider config, and students on local vLLM need none.
    """
    env = strip_credentials(dict(os.environ))
    env.update(extra or {})
    return env


@contextmanager
def key_proxies(endpoints: list[Endpoint]) -> Iterator[list[Endpoint]]:
    """`endpoints` with every keyed endpoint routed through its own `KeyProxy` (stopped on exit).

    The proxy forwards to the endpoint's URL (the Hugging Face router for a `huggingface/...` id without one), adding
    the real key and the endpoint's headers; the returned endpoint has the proxy's URL, a session key, no `key_env`
    and no headers. `rekey` mints a fresh session key per episode.
    """
    proxies: list[KeyProxy] = []
    routed: list[Endpoint] = []
    try:
        for endpoint in endpoints:
            if not endpoint.key_env:
                routed.append(endpoint)
                continue
            upstream = endpoint.base_url or (HF_ROUTER if endpoint.native_id.startswith("huggingface/") else "")
            if not upstream:
                raise ValueError(f"{endpoint.model}: no base URL to proxy to")
            proxy = KeyProxy(upstream, endpoint.key_env, endpoint.headers).start()
            proxies.append(proxy)
            PROXIES[proxy.url] = proxy
            routed.append(replace(endpoint, base_url=proxy.url, key_env=None, headers={},
                                  session_key=proxy.new_key()))  # fmt: skip
        yield routed
    finally:
        for proxy in proxies:
            PROXIES.pop(proxy.url, None)
            proxy.stop()


PROXIES: dict[str, KeyProxy] = {}  # running key proxies by URL (for per-episode session keys)


@contextmanager
def rekey(endpoints: list[Endpoint]) -> Iterator[list[Endpoint]]:
    """`endpoints` with a fresh session key for every proxied endpoint, revoked on exit (one per episode)."""
    minted: list[tuple[KeyProxy, str]] = []
    result = []
    for endpoint in endpoints:
        proxy = PROXIES.get(endpoint.base_url) if endpoint.session_key else None
        if proxy is None:
            result.append(endpoint)
            continue
        key = proxy.new_key()
        minted.append((proxy, key))
        result.append(replace(endpoint, session_key=key))
    try:
        yield result
    finally:
        for proxy, key in minted:
            proxy.revoke(key)


def harness_failure(episode: Any) -> str | None:
    for event in episode.tool_events:
        text = str(event.output or "") + str(event.error or "")
        for marker in HARNESS_FAILURE_MARKERS:
            if marker in text:
                return text.strip().splitlines()[-1][:200] if text.strip() else marker
    return None


@dataclass
class Server:
    url: str
    version: str
    workspace: Path
    process: subprocess.Popen | None = None
    password: str | None = None

    def __str__(self) -> str:  # never print the password
        return self.url

    @property
    def auth(self) -> tuple[str, str] | None:
        return ("opencode", self.password) if self.password else None


def _read_password(process: subprocess.Popen, deadline: float) -> str | None:
    """The `server password ...` line of a 2.x server's stdout, or None once `deadline` passes or it exits.

    A reader thread feeds the lines into a queue (and keeps draining stdout afterwards), so a server that prints
    nothing cannot block startup past the deadline.
    """
    lines: queue.Queue[str | None] = queue.Queue()
    done = threading.Event()

    def reader() -> None:
        with suppress(Exception):
            for line in process.stdout:  # type: ignore[union-attr]
                if not done.is_set():
                    lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                line = lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                return None
            if line.startswith("server password "):
                return line.split()[-1].strip()
        return None
    finally:
        done.set()  # the reader keeps draining stdout, discarding it


# Many OpenCode servers cold-starting at once crash or stall (shared provider-package cache): start one at a time.
_STARTUP_LOCK = threading.Lock()


def xdg_dirs(workspace: Path) -> dict[str, str]:
    """Per-workspace data, config and state directories (OpenCode keeps its session database there)."""
    names = (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"), ("XDG_STATE_HOME", "state"))
    return {key: str(workspace / f".opencode-{name}") for key, name in names}


def start_server(
    executable: Path, version: str, workspace: Path, env: dict[str, str], *, startup_timeout: float = 120
) -> Server:
    """Start `opencode serve` on 127.0.0.1 and wait until it answers. 2.x: read the generated password."""
    with _STARTUP_LOCK:
        return _start_server(executable, version, workspace, env, startup_timeout=startup_timeout)


def _start_server(
    executable: Path, version: str, workspace: Path, env: dict[str, str], *, startup_timeout: float
) -> Server:
    v2 = int(version.split(".")[0]) >= 2
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [str(executable), "serve", "--hostname", "127.0.0.1", "--port", str(port)],
        cwd=str(workspace),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if v2 else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if v2 else subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    server = Server(url, version, workspace, process)
    deadline = time.monotonic() + startup_timeout
    try:
        if v2:
            server.password = _read_password(process, deadline)
            if server.password is None:
                status = process.poll()
                reason = f"exited with status {status}" if status is not None else "printed no password in time"
                raise HarnessError(f"OpenCode {reason} during startup")
        health = "/api/info" if v2 else "/global/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise HarnessError(f"OpenCode exited during startup with status {process.returncode}")
            with suppress(httpx.HTTPError):
                if httpx.get(url + health, auth=server.auth, timeout=1).is_success:
                    return server
            time.sleep(0.1)
        raise HarnessError(f"OpenCode did not become healthy within {startup_timeout:g} seconds")
    except BaseException:
        stop_server(server)
        raise


def stop_server(server: Server) -> None:
    process = server.process
    if process is None or process.poll() is not None:
        return
    import signal

    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def config_json(config: dict[str, Any]) -> str:
    return json.dumps(config)
