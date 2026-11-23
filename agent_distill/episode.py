"""The harness-neutral episode: what every adapter returns and every rubric scores (DESIGN.md §5)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

ToolKind = Literal["shell", "fetch", "read", "search", "edit", "write", "submit", "other"]
TOOL_KINDS: tuple[str, ...] = ("shell", "fetch", "read", "search", "edit", "write", "submit", "other")

_URL_RE = re.compile(r"https?://[^\s\"']+")


@dataclass
class ToolEvent:
    """One tool call. `kind` is normalized across harnesses; `name` is the harness's own tool name."""

    kind: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    output: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in TOOL_KINDS:
            raise ValueError(f"unknown tool kind {self.kind!r}; expected one of {TOOL_KINDS}")

    @property
    def command(self) -> str:
        """The shell command (`""` for non-shell tools)."""
        return str(self.input.get("command", "")) if self.kind == "shell" else ""

    @property
    def url(self) -> str:
        """The fetched URL: the `url` argument of a fetch tool, or the first URL in a shell command."""
        if self.kind == "fetch":
            return str(self.input.get("url", ""))
        if self.kind == "shell":
            match = _URL_RE.search(self.command)
            return match.group(0) if match else ""
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "input": self.input,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolEvent:
        return cls(**{key: data.get(key) for key in ("kind", "name", "input", "success", "output", "error")})


@dataclass
class ModelCall:
    """One captured model call of a GRPO rollout (the fields a Tinker TrajectoryGroup / verifiers Trace needs)."""

    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]


@dataclass
class Episode:
    """One finished episode of a task in a harness.

    `messages` are OpenAI-style chat messages (reasoning as `reasoning_content`; adapters may add `timestamp`,
    `usage`, `model` keys, which the Pi exporter uses). `answer` is the parsed answer, or None (no answer).
    """

    task: str
    row_id: str
    harness: str
    harness_version: str
    model: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    answer: Any = None
    answer_mode: str = ""
    answer_error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    failure: str | None = None
    session_id: str | None = None
    model_calls: list[ModelCall] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    @property
    def harness_id(self) -> str:
        return f"{self.harness}@{self.harness_version}"

    @property
    def tool_calls(self) -> int:
        return len(self.tool_events)
