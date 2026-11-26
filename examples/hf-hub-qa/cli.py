"""The CLI side of the hf-hub-qa tasks: what counts as using the Hub, writing to it, and the agent's tool text.

Shared by both task files of this folder (task.py: Hub facts; judged.py: the judged questions), so both gate
episodes the same way. The MCP twin (examples/hf-hub-qa-mcp) has its own tool gates; everything else, the fact
checker (agent_distill.rubrics.facts) and grounding (agent_distill.rubrics.grounding), is shared.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import Any

from agent_distill.rubrics import metric, process, programmatic

WRITE_COMMANDS = re.compile(
    r"\bhf\s+(?:upload|sync|cp|auth\s+(?:login|logout|switch)"
    r"|repos?\s+(?:create|delete|settings|move|(?:branch|tag)\s+(?:create|delete)|duplicate)"
    r"|(?:jobs|endpoints|webhooks|buckets|collections|discussions|sandbox)\s+"
    r"(?:run|uv|create|deploy|delete|pause|resume|update|scale-to-zero|add-item|update-item|delete-item|comment|close"
    r"|reopen|merge|rename|cancel|scheduled)"
    r"|spaces\s+(?:secrets|variables)\s+(?:add|delete|set)|spaces\s+(?:duplicate|restart|pause|create|delete))\b"
)
HUB_COMMAND = re.compile(r"(?:^|[\s;&|(`])hf\s+[a-z]|huggingface\.co/api/|huggingface_hub")
HELP = re.compile(r"(?:^|\s)(?:--help|-h)(?:\s|$)")


def shell_events(tool_events: list[Any]) -> list[Any]:
    return [e for e in tool_events or [] if getattr(e, "kind", "") == "shell"]


def command_of(event: Any) -> str:
    data = getattr(event, "input", None) or {}
    return str(data.get("command", "")) if isinstance(data, dict) else str(data)


def tool_text(tool_events: list[Any]) -> str:
    return "\n".join(f"{command_of(e)}\n{getattr(e, 'output', '') or ''}" for e in tool_events or [])


@programmatic(weight=0.0)
def answered(answer: Any) -> float:
    return float(bool(str(answer or "").strip()))


@process(weight=0.0)
def read_only(tool_events: list[Any]) -> float:
    writes = [c for c in map(command_of, shell_events(tool_events)) if WRITE_COMMANDS.search(c)]
    return 0.0 if any(not HELP.search(c) for c in writes) else 1.0


@process(weight=0.0)
def used_hub(tool_events: list[Any]) -> float:
    return float(any(HUB_COMMAND.search(command_of(e)) and getattr(e, "success", False)
                     for e in shell_events(tool_events)))  # fmt: skip


@lru_cache(maxsize=256)
def run_hf(argv: tuple[str, ...]) -> Any:
    """The pinned CLI, logged out like the agent (no token), JSON out. Cached for a stage's lifetime."""
    binary = os.environ.get("AGENT_DISTILL_HF_CLI") or shutil.which("hf")
    if not binary:
        raise RuntimeError("the hf CLI is not installed (task job_setup installs it)")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))}
    env.update(HF_HUB_DISABLE_UPDATE_CHECK="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    with tempfile.TemporaryDirectory() as home:
        env["HOME"] = home  # no cached token either
        env.pop("HF_HOME", None)
        done = subprocess.run([binary, *argv, "--format", "json"], capture_output=True, text=True, timeout=120,
                              env=env, check=True)  # fmt: skip
    return json.loads(done.stdout)


@metric
def hf_calls(tool_events: list[Any]) -> float:
    return float(sum(1 for e in shell_events(tool_events) if HUB_COMMAND.search(command_of(e))))


@metric
def answer_chars(answer: Any) -> float | None:
    return float(len(str(answer))) if answer else None


HF_CALL = re.compile(r"(?:^|[\s;&|(`])hf\s+[a-z]")
CLI_ERROR = re.compile(
    r"Error: (?:No such (?:command|option)|Invalid value|Missing (?:argument|option)|Got unexpected)"
)


def rejected(event: Any) -> bool:
    """An `hf` call the CLI rejected: an invented command or option, or an invalid value."""
    text = f"{getattr(event, 'output', '') or ''}{getattr(event, 'error', '') or ''}"
    return bool(HF_CALL.search(command_of(event)) and CLI_ERROR.search(text))


@metric
def hf_errors(tool_events: list[Any]) -> float:
    """How many `hf` calls the CLI rejected: how well the agent knows the CLI, which the pass rate hides (v1
    validation: base Qwen3.5-4B 33% of its calls, GLM-5.3 5%)."""
    return float(sum(1 for e in shell_events(tool_events) if rejected(e)))


@metric
def read_skill(tool_events: list[Any]) -> float:
    """1 if the agent read the hf-cli skill (SKILL.md) at least once."""
    return float(any("SKILL.md" in json.dumps(getattr(e, "input", None) or {}) for e in tool_events or []))
