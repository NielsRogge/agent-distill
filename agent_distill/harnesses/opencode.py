"""Shared OpenCode plumbing for the 1.2.4 and 2.3.2 adapters: install, provider config, server lifecycle.

Ported from url-extraction `harness/opencode.py` (main, and the OpenCode v2 branch `claude/opencode-v2-glm53-4b`).
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agent_distill.config import HarnessProfile
from agent_distill.credentials import strip_credentials
from agent_distill.harnesses.base import Endpoint, architecture, download_verified, extract_member

LOCAL_PROVIDER = "local"

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


def install(profile: HarnessProfile, cache: Path) -> Path:
    """The pinned OpenCode binary in `cache` (downloaded and sha256-checked once); never one found on PATH."""
    arch = architecture()
    destination = Path(cache) / f"opencode-{profile.version}-{arch}" / "opencode"
    if not destination.exists():
        sha = profile.download.sha256.get(arch)
        if not sha:
            raise RuntimeError(f"{profile.id}: no pinned sha256 for {arch}")
        url = profile.download.url.format(version=profile.version, arch=arch)
        extract_member(download_verified(url, sha), "opencode", destination)
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

    An endpoint with a native id `huggingface/...` uses OpenCode's built-in Hugging Face provider (the router), with
    its headers (e.g. `X-HF-Bill-To`) and an inline model entry for ids missing from OpenCode's catalog. Any other
    endpoint becomes an OpenAI-compatible provider named `local` (vLLM, the GRPO capture proxy).
    """
    providers: dict[str, Any] = {}
    for endpoint in endpoints:
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
            if endpoint.key_env:
                entry["options"]["apiKey"] = f"{{env:{endpoint.key_env}}}"
            elif endpoint.api_key:
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
            if endpoint.headers:
                entry.setdefault("options", {}).setdefault("headers", {}).update(endpoint.headers)
            if endpoint.native_entry:
                entry.setdefault("models", {})[model] = dict(endpoint.native_entry)
    config: dict[str, Any] = {"provider": providers}
    small = [endpoint for endpoint in endpoints if endpoint.small_model]
    if small:
        config["small_model"] = "/".join(model_ref(small[0]))
    return config


def model_ref(endpoint: Endpoint) -> tuple[str, str]:
    """(providerID, modelID) OpenCode uses for an endpoint."""
    if endpoint.native_id:
        provider, _, model = endpoint.native_id.partition("/")
        return provider, model
    return LOCAL_PROVIDER, endpoint.model


def server_env(endpoints: list[Endpoint], extra: dict[str, str] | None = None) -> dict[str, str]:
    """The `opencode serve` environment (inherited by every shell the agent runs).

    Credentials are stripped; only the keys the endpoints name are passed back in (a teacher's HF token). A student
    on local vLLM therefore runs without any token in its environment.
    """
    env = strip_credentials(dict(os.environ))
    for endpoint in endpoints:
        if endpoint.key_env and endpoint.key_env in os.environ:
            env[endpoint.key_env] = os.environ[endpoint.key_env]
    env.update(extra or {})
    return env


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


def _drain(stream: Any) -> None:
    with suppress(Exception):
        for _ in stream:
            pass


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
            for line in process.stdout:  # type: ignore[union-attr]
                if line.startswith("server password "):
                    server.password = line.split()[-1].strip()
                    break
                if time.monotonic() > deadline:
                    break
            threading.Thread(target=_drain, args=(process.stdout,), daemon=True).start()
            if server.password is None:
                raise RuntimeError(f"OpenCode exited during startup with status {process.poll()}")
        health = "/api/info" if v2 else "/global/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"OpenCode exited during startup with status {process.returncode}")
            with suppress(httpx.HTTPError):
                if httpx.get(url + health, auth=server.auth, timeout=1).is_success:
                    return server
            time.sleep(0.1)
        raise TimeoutError(f"OpenCode did not become healthy within {startup_timeout:g} seconds")
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
