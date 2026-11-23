"""OpenCode 1.2.4 messages -> `Episode` (read-only; DESIGN.md §5, §6).

Every published url-extraction record stores the raw OpenCode 1.2.4 session messages (`opencode_messages`). This
module turns them, or a whole stored record, into harness-neutral episodes so the ported rubrics can rescore
them (parity tier 1). The `opencode@1.2.4` adapter (M2) reuses the same conversion for live episodes.
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from typing import Any

from agent_distill.episode import Episode, ToolEvent

HARNESS = "opencode"
VERSION = "1.2.4"

# OpenCode 1.2.4 tool name (lowercased) -> normalized kind. Anything else is "other".
TOOL_KINDS: dict[str, str] = {
    "bash": "shell",
    "shell": "shell",
    "webfetch": "fetch",
    "web_fetch": "fetch",
    "read": "read",
    "grep": "search",
    "glob": "search",
    "list": "search",
    "codesearch": "search",
    "websearch": "search",
    "edit": "edit",
    "apply_patch": "edit",
    "write": "write",
    "structuredoutput": "submit",
    # OpenCode 2.x names
    "patch": "edit",
}


def tool_kind(name: str) -> str:
    return TOOL_KINDS.get(name.lower(), "other")


def tool_events(messages: list[dict[str, Any]]) -> list[ToolEvent]:
    """Every tool part, in order (including the StructuredOutput submission, as OpenCode counts it)."""
    events: list[ToolEvent] = []
    for message in messages:
        for part in message.get("parts", []):
            if part.get("type") != "tool":
                continue
            state = part.get("state") or {}
            name = str(part.get("tool", ""))
            events.append(
                ToolEvent(
                    kind=tool_kind(name),
                    name=name,
                    input=state.get("input") or {},
                    success=state.get("status") == "completed",
                    error=state.get("error"),
                    output=state.get("output"),
                )
            )
    return events


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def answer_from_text(model: Any, text: str) -> Any | None:
    """A JSON answer in `text`: the whole text, a fenced ```json block, or the outermost {...}."""
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE))
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        with suppress(ValueError, TypeError):
            return model.model_validate_json(candidate)
    return None


def answer_from_messages(model: Any, messages: list[dict[str, Any]]) -> Any | None:
    """The last assistant text that parses as an answer (when native structured output was unavailable)."""
    texts = [
        str(part.get("text", ""))
        for message in messages
        if (message.get("info") or {}).get("role") == "assistant"
        for part in message.get("parts", [])
        if part.get("type") == "text" and part.get("text")
    ]
    for text in reversed(texts):
        answer = answer_from_text(model, text)
        if answer is not None:
            return answer
    return None


def provider_errors(messages: list[dict[str, Any]]) -> list[str]:
    errors = [((message.get("info") or {}).get("error") or {}).get("data", {}).get("message") for message in messages]
    return [str(error) for error in errors if error]


def _usage(info: dict[str, Any]) -> dict[str, Any]:
    tokens = info.get("tokens") or {}
    cache = tokens.get("cache") or {}
    return {
        "input": tokens.get("input", 0),
        "output": tokens.get("output", 0),
        "cache_read": cache.get("read", 0),
        "cache_write": cache.get("write", 0),
        "total": tokens.get("total", 0),
        "cost": float(info.get("cost") or 0),
    }


def chat_messages(messages: list[dict[str, Any]], system_prompt: str | None = None) -> list[dict[str, Any]]:
    """OpenCode parts -> ordered OpenAI-style chat messages, with nothing truncated.

    Reasoning parts become `reasoning_content` on the assistant message that follows them. Each message also
    carries `timestamp` (ms), and assistant messages `model`/`provider`/`usage`/`error`, for the Pi exporter.
    Tool results carry `is_error`.
    """
    result: list[dict[str, Any]] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    for message in messages:
        info = message.get("info") or {}
        role = info.get("role")
        created = (info.get("time") or {}).get("created")
        if role == "user":
            text = "\n".join(part.get("text", "") for part in message.get("parts", []) if part.get("type") == "text")
            if text:
                result.append({"role": "user", "content": text, "timestamp": created})
            continue
        if role != "assistant":
            continue
        meta = {
            "timestamp": created,
            "model": info.get("modelID"),
            "provider": info.get("providerID"),
            "usage": _usage(info),
        }
        if info.get("error"):
            meta["error"] = info["error"]
        text_parts: list[str] = []
        reasoning_parts: list[str] = []

        def assistant(
            content: str, meta: dict[str, Any] = meta, reasoning_parts: list[str] = reasoning_parts, **extra: Any
        ) -> dict[str, Any]:
            entry: dict[str, Any] = {"role": "assistant", "content": content, **extra, **meta}
            if reasoning_parts:
                entry["reasoning_content"] = "\n\n".join(reasoning_parts)
                reasoning_parts.clear()
            return entry

        for part in message.get("parts", []):
            kind = part.get("type")
            if kind == "reasoning" and str(part.get("text") or "").strip():
                reasoning_parts.append(str(part["text"]).strip())
            elif kind == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            elif kind == "tool":
                if text_parts:
                    result.append(assistant("\n".join(text_parts)))
                    text_parts.clear()
                state = part.get("state") or {}
                call_id = str(part.get("callID") or part.get("id"))
                call = {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": str(part.get("tool", "")), "arguments": json_text(state.get("input") or {})},
                }
                result.append(assistant("", tool_calls=[call]))
                completed = state.get("status") == "completed"
                output = state.get("output") if completed else (state.get("error") or f"tool {state.get('status')}")
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": str(part.get("tool", "")),
                        "content": output if isinstance(output, str) else json_text(output),
                        "is_error": not completed,
                        "timestamp": (state.get("time") or {}).get("end") or created,
                    }
                )
        if text_parts or reasoning_parts:
            result.append(assistant("\n".join(text_parts)))
    return result


def episode_from_messages(
    messages: list[dict[str, Any]],
    *,
    answer_model: Any,
    task: str,
    row_id: str,
    model: str,
    structured: Any = None,
    system_prompt: str | None = None,
    failure: str | None = None,
    session_id: str | None = None,
    elapsed_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Episode:
    """One OpenCode 1.2.4 session as an `Episode`.

    `structured` is OpenCode's validated structured output (`info.structured` of the prompt response, or the
    `findings` a record stored); without a valid one, the answer is recovered from a JSON final message.
    """
    answer = None
    answer_error = None
    answer_mode = "native"
    if structured is not None:
        try:
            answer = answer_model.model_validate(structured)
        except ValueError as exc:
            answer_error = f"invalid structured output: {exc}"
    if answer is None:
        answer = answer_from_messages(answer_model, messages)
        if answer is not None:
            answer_mode = "final_json"
    errors = provider_errors(messages)
    if errors:
        failure = failure or f"provider error: {errors[-1]}"
    usage: dict[str, Any] = {"input": 0, "output": 0, "cost": 0.0}
    for message in messages:
        info = message.get("info") or {}
        if info.get("role") == "assistant":
            item = _usage(info)
            usage["input"] += item["input"]
            usage["output"] += item["output"]
            usage["cost"] += item["cost"]
    return Episode(
        task=task,
        row_id=row_id,
        harness=HARNESS,
        harness_version=VERSION,
        model=model,
        messages=chat_messages(messages, system_prompt),
        tool_events=tool_events(messages),
        answer=answer,
        answer_mode=answer_mode if answer is not None else "",
        answer_error=answer_error,
        usage=usage,
        elapsed_seconds=elapsed_seconds,
        failure=failure,
        session_id=session_id,
        metadata=metadata or {},
        raw=messages,
    )


RECORD_METADATA = (
    "paper_id",
    "arxiv_id",
    "evaluation_model",
    "attempt",
    "difficulty_tier",
    "accepted",
    "attempt_complete",
    "failure_reason",
    "verification",
    "findings",
    "label_quality",
)


def episode_from_record(record: dict[str, Any], *, answer_model: Any, task: str, row_id: str) -> Episode:
    """A stored url-extraction episode record (version 1) as an `Episode`.

    The stored `findings` is re-validated against the answer schema; when it fails or is missing, the answer is
    recovered from the final message, as url-extraction's rescoring does. Stored verification fields go into
    `metadata` so a rescoring report can compare old and new.
    """
    messages = record.get("opencode_messages") or []
    system_prompt = next(
        (str(m.get("content") or "") for m in record.get("trajectory") or [] if m.get("role") == "system"), None
    )
    episode = episode_from_messages(
        messages,
        answer_model=answer_model,
        task=task,
        row_id=row_id,
        model=str(record.get("model") or ""),
        structured=record.get("findings"),
        system_prompt=system_prompt,
        failure=None,
        session_id=record.get("session_id"),
        elapsed_seconds=record.get("elapsed_seconds"),
        metadata={key: record.get(key) for key in RECORD_METADATA if key in record},
    )
    if not messages and record.get("tool_events"):  # slim records without raw messages
        episode.tool_events = [
            ToolEvent(
                kind=tool_kind(str(event.get("tool_name", ""))),
                name=str(event.get("tool_name", "")),
                input=event.get("tool_input") or {},
                success=bool(event.get("success")),
                error=event.get("error"),
                output=event.get("response"),
            )
            for event in record["tool_events"]
        ]
    return episode
