"""`Episode` -> Pi session (format version 3) for the Hub's agent-traces viewer (DESIGN.md §5).

Generalizes url-extraction's `trace_export.record_to_pi_session`: the header carries `harness` (custom-harness
convention of https://huggingface.co/docs/hub/en/session-traces-format) plus the episode's metadata, then a
model change, the system prompt, the conversation (reasoning as `thinking` blocks), and a final `verification`
custom message with the rubric scores. Credentials are redacted. Pi episodes are published as Pi's own session
instead (`Harness.native_trace`); this exporter is for every other harness.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent_distill.credentials import jsonl_line, redact_credentials
from agent_distill.episode import Episode

MAX_TOOL_OUTPUT_CHARS = 60_000
SESSIONS_DIR = "sessions"


def _iso(ms: float | None) -> str:
    moment = dt.datetime.fromtimestamp(ms / 1000, dt.UTC) if ms else dt.datetime.now(dt.UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:80] or "x"


def _redact_value(value: Any) -> Any:
    return json.loads(redact_credentials(json.dumps(value, ensure_ascii=False)))


def session_filename(episode: Episode) -> str:
    model = _slug(str(episode.metadata.get("evaluation_model") or episode.model or "model").split("/")[-1])
    attempt = episode.metadata.get("attempt")
    suffix = f"_a{attempt}" if attempt is not None else ""
    return f"{_slug(episode.row_id)}_{model}{suffix}.jsonl"


def _pi_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    return {
        "input": usage.get("input", 0),
        "output": usage.get("output", 0),
        "cacheRead": usage.get("cache_read", 0),
        "cacheWrite": usage.get("cache_write", 0),
        "totalTokens": usage.get("total", 0),
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": usage.get("cost", 0)},
    }


def episode_to_pi_session(
    episode: Episode, *, verification: dict[str, Any] | None = None, title: str | None = None
) -> list[dict[str, Any]]:
    """Session lines, header first. `verification` (e.g. `RubricResult.to_dict()`) becomes the last message."""
    timestamps = [m.get("timestamp") for m in episode.messages if m.get("timestamp")]
    first_ms = timestamps[0] if timestamps else None
    session_id = (
        episode.session_id
        or hashlib.sha1(json.dumps(episode.messages, default=str, sort_keys=True).encode()).hexdigest()[:16]
    )
    passed = (verification or {}).get("passed")
    header = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "harness": episode.harness,
        "harness_version": episode.harness_version,
        "name": title
        or " · ".join(
            part
            for part in (episode.row_id, episode.model.split("/")[-1], {True: "passed", False: "failed"}.get(passed))
            if part
        ),
        "timestamp": _iso(first_ms),
        "cwd": str(episode.metadata.get("cwd", "")),
        # Extra fields are ignored by the viewer and keep the episode self-describing.
        "task": episode.task,
        "row_id": episode.row_id,
        "model": episode.model,
        **{key: value for key, value in episode.metadata.items() if isinstance(value, str | int | float | bool)},
    }
    # Metadata strings (e.g. `failure_reason` with a provider's error text) can carry credentials.
    header = _redact_value(header)
    lines: list[dict[str, Any]] = [header]
    parent: str | None = None
    counter = 0
    last_ms: float | None = first_ms

    def add(entry: dict[str, Any], ms: float | None) -> None:
        nonlocal parent, counter, last_ms
        counter += 1
        entry_id = f"{counter:08x}"
        ms = ms or last_ms
        last_ms = ms
        lines.append({"id": entry_id, "parentId": parent, "timestamp": _iso(ms), **entry})
        parent = entry_id

    provider, _, model_id = episode.model.partition("/") if "/" in episode.model else ("", "", episode.model)
    first_assistant = next((m for m in episode.messages if m.get("role") == "assistant" and m.get("model")), None)
    if first_assistant:
        provider = first_assistant.get("provider") or provider
        model_id = first_assistant.get("model") or model_id
    add({"type": "model_change", "provider": provider, "modelId": model_id}, first_ms)

    pending_calls: list[dict[str, Any]] = []
    pending_meta: dict[str, Any] | None = None
    pending_content: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}

    def flush_assistant() -> None:
        nonlocal pending_meta, pending_content, pending_calls
        if pending_meta is None:
            return
        meta = pending_meta
        message = {
            "role": "assistant",
            "content": pending_content,
            "api": "openai-completions",
            "provider": meta.get("provider") or provider,
            "model": meta.get("model") or model_id,
            "usage": _pi_usage(meta.get("usage")),
            "stopReason": "toolUse" if pending_calls else ("error" if meta.get("error") else "stop"),
            "timestamp": meta.get("timestamp"),
        }
        if meta.get("error"):
            message["errorMessage"] = redact_credentials(json.dumps(meta["error"], ensure_ascii=False))[:2000]
        add({"type": "message", "message": message}, meta.get("timestamp"))
        pending_meta, pending_content, pending_calls = None, [], []

    for message in episode.messages:
        role = message.get("role")
        ms = message.get("timestamp")
        if role == "system":
            add(
                {
                    "type": "message",
                    "message": {
                        "role": "custom",
                        "customType": "system_prompt",
                        "content": redact_credentials(str(message.get("content") or "")),
                        "display": True,
                        "timestamp": ms or first_ms,
                    },
                },
                ms or first_ms,
            )
        elif role == "user":
            flush_assistant()
            text = redact_credentials(str(message.get("content") or ""))
            add({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}, ms)
        elif role == "assistant":
            # Consecutive assistant entries of one model turn (text, then tool calls) form one Pi message.
            same_turn = pending_meta is not None and pending_meta.get("timestamp") == ms and not pending_calls
            if not same_turn:
                flush_assistant()
                pending_meta = message
            if message.get("reasoning_content"):
                pending_content.append(
                    {
                        "type": "thinking",
                        "thinking": redact_credentials(str(message["reasoning_content"])),
                        "thinkingSignature": "reasoning",
                    }
                )
            if message.get("content"):
                pending_content.append({"type": "text", "text": redact_credentials(str(message["content"]))})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {"raw": function.get("arguments")}
                tool_names[str(call.get("id"))] = str(function.get("name", ""))
                pending_content.append(
                    {
                        "type": "toolCall",
                        "id": call.get("id"),
                        "name": function.get("name"),
                        "arguments": _redact_value(arguments),
                    }
                )
                pending_calls.append(call)
        elif role == "tool":
            flush_assistant()
            # Redact before truncating, so a token cut at the limit cannot leak partially.
            text = redact_credentials(str(message.get("content") or ""))
            if len(text) > MAX_TOOL_OUTPUT_CHARS:
                text = (
                    text[:MAX_TOOL_OUTPUT_CHARS]
                    + f"\n[truncated {len(text) - MAX_TOOL_OUTPUT_CHARS} chars for the viewer]"
                )
            call_id = message.get("tool_call_id")
            add(
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": call_id,
                        "toolName": message.get("name") or tool_names.get(str(call_id), ""),
                        "content": [{"type": "text", "text": redact_credentials(text)}],
                        "isError": bool(message.get("is_error")),
                        "timestamp": ms,
                    },
                },
                ms,
            )
    flush_assistant()

    if verification is not None:
        add(verification_entry(episode, verification), None)
    return lines


def verification_entry(episode: Episode, verification: dict[str, Any]) -> dict[str, Any]:
    """The final custom message with the rubric scores (shown by the agent-traces viewer)."""
    scores = verification.get("scores") or {}
    summary = [f"{name}: {value}" for name, value in scores.items()]
    summary.append(f"reward: {verification.get('reward')}  passed: {verification.get('passed')}")
    if episode.failure:
        summary.append(f"failure: {episode.failure}")
    return {
        "type": "message",
        "message": {
            "role": "custom",
            "customType": "verification",
            "content": redact_credentials("\n".join(summary)),
            "display": True,
            "details": _redact_value({"answer": _jsonable(episode.answer), **verification}),
        },
    }


def annotate_native_session(
    lines: list[dict[str, Any]], episode: Episode, verification: dict[str, Any]
) -> list[dict[str, Any]]:
    """A harness's own Pi session (Pi episodes) with what every exported session carries: `harness` and
    `harness_version` in the header and the verification as the last message, chained to the last entry."""
    lines = [dict(line) for line in lines]
    if lines and lines[0].get("type") == "session":
        lines[0].update({"harness": episode.harness, "harness_version": episode.harness_version,
                         "task": episode.task, "row_id": episode.row_id, "model": episode.model})  # fmt: skip
    entries = [line for line in lines[1:] if line.get("id")]
    last = entries[-1] if entries else None
    stamp = (last or {}).get("timestamp") or (lines[0].get("timestamp") if lines else None)
    lines.append({"id": "verification", "parentId": last["id"] if last else None, "timestamp": stamp,
                  **verification_entry(episode, verification)})  # fmt: skip
    return lines


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def write_session(lines: Iterable[dict[str, Any]], path: Path) -> None:
    """Write session lines as JSONL that `str.splitlines()` splits correctly, with credentials redacted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(jsonl_line(line) + "\n")
