"""The harness adapter interface (DESIGN.md §6)."""

from __future__ import annotations

import hashlib
import io
import platform
import tarfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from agent_distill.hooks import resolve

if TYPE_CHECKING:
    from agent_distill.config import HarnessProfile
    from agent_distill.episode import Episode
    from agent_distill.task import Limits, Task


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible model endpoint.

    `key_env` names the environment variable holding the API key (never the key itself); `None` means the
    endpoint needs no key (local vLLM). `context` / `max_output_tokens` are what the harness is told.
    """

    base_url: str
    model: str
    key_env: str | None = None
    name: str = ""  # display name
    context: int = 65536
    max_output_tokens: int = 1024
    headers: Mapping[str, str] = field(default_factory=dict)  # e.g. {"X-HF-Bill-To": "huggingface"}
    native_id: str = ""  # harness-native model id, e.g. OpenCode's "huggingface/zai-org/GLM-5.2"
    native_entry: Mapping[str, Any] = field(default_factory=dict)  # harness-native model declaration
    # A non-secret API key sent as-is, e.g. the OpenEnv capture session id the capture proxy reads as its key.
    api_key: str = ""
    # Also route the harness's auxiliary model calls (e.g. OpenCode 1.x session titles) to this endpoint.
    small_model: bool = False


class Harness(Protocol):
    name: str
    version: str

    def install(self, cache: Path) -> Path:
        """The pinned binary (downloaded and verified into `cache`); never a binary found on PATH."""
        ...

    def session(self, endpoints: list[Endpoint], *, cache: Path, task: Task, request_timeout: float = 600) -> Any:
        """An async context manager yielding a runner: a running harness configured for `endpoints` and `task`.

        Agent processes get no credentials except the keys the endpoints name (a teacher's HF token).
        """
        ...

    async def healthy(self, runner: Any) -> None:
        """Raise unless the running harness still answers."""
        ...

    async def run_episode(
        self, runner: Any, task: Task, row: dict[str, Any], endpoint: Endpoint, limits: Limits, *, title: str = ""
    ) -> Episode:
        """Run one episode of `task` on `row` with the model at `endpoint`, inside a running `session`."""
        ...

    def native_trace(self, episode: Episode) -> list[dict[str, Any]] | None:
        """The harness's own session file for the agent-traces viewer, if it has one (Pi); else None."""
        ...


HF_ROUTER = "https://router.huggingface.co/v1"


def teacher_endpoint(teacher: Any, harness: HarnessProfile, bill_to: str = "") -> Endpoint:
    """A teacher served by HF Inference Providers, billed to `bill_to` (`X-HF-Bill-To`), with the `HF_TOKEN` key.

    OpenCode reaches it through its built-in Hugging Face provider (`huggingface/<model>[:provider]`, with an inline
    model entry for ids missing from its catalog); other harnesses use the router's OpenAI-compatible API.
    """
    headers = {"X-HF-Bill-To": bill_to} if bill_to else {}
    native_id = teacher.opencode_id if harness.name == "opencode" else ""
    native_entry = teacher.opencode_model_entry if harness.name == "opencode" else {}
    return Endpoint(base_url=HF_ROUTER, model=teacher.routed_model, key_env="HF_TOKEN", name=teacher.name,
                    headers=headers, native_id=native_id, native_entry=native_entry)  # fmt: skip


def load_harness(profile: HarnessProfile) -> Harness:
    cls = resolve(profile.adapter)
    return cls(profile)


def architecture() -> str:
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if arch is None:
        raise RuntimeError(f"Unsupported architecture: {platform.machine()}")
    return arch


def download_verified(url: str, sha256: str, timeout: float = 180) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{url}: sha256 {digest} does not match the pinned {sha256}")
    return data


def extract_member(archive: bytes, member: str, destination: Path) -> Path:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"archive has no {member}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(handle.read())
    destination.chmod(0o755)
    return destination
