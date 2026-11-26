"""Race core: one question, two agents (the teacher and its distilled student) in the same pinned harness.

Everything comes from the library and the run configs named in race.toml: the task (prompt, tools, skills, MCP
servers, rubrics), the teacher and student profiles, the harness adapter (Pi, with per-episode UID isolation where the
process may switch users) and the run's evaluation limits. Each race runs both episodes concurrently through the
adapter's `run_episode`; its JSON event stream (`on_event`) drives the live view, and the finished episodes are graded
with the task's rubric set in eval mode, exactly as the `eval` stage grades them.

Credentials: the agents run shell commands and read untrusted content. They never see a credential: the Pi adapter
starts them with a credential-free environment, and both endpoints (the router for the teacher, the Inference Endpoint
for the student) are reached through the library's per-episode key proxy, which holds the token in this process only.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import config as race_config
import httpx

from agent_distill.episode import Episode
from agent_distill.harnesses import isolation
from agent_distill.harnesses.base import Endpoint, default_cache, load_harness
from agent_distill.rubrics.core import episode_context
from agent_distill.stages.eval import select_rows

CONFIG = race_config.load()
ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "").rstrip("/")
WAKE_TIMEOUT = float(os.environ.get("WAKE_TIMEOUT", "900"))

# --------------------------------------------------------------------------------------------------
# Questions: each race's evaluation rows (the split and limit the published numbers were measured on)
# --------------------------------------------------------------------------------------------------

_rows: dict[str, list[dict[str, Any]]] = {}
_rows_ready = threading.Event()


def _load_rows() -> None:
    for race in CONFIG.races:
        try:
            task = race.task()
            rows = select_rows(task, race.split, limit=race.limit)
            _rows[race.id] = [{**row, "_id": task.row_id(row)} for row in rows]
        except Exception as exc:
            print(f"[race] could not load {race.id} questions: {exc}", flush=True)
            _rows[race.id] = []
    _rows_ready.set()


threading.Thread(target=_load_rows, daemon=True).start()


def questions(race_id: str) -> list[dict[str, str]]:
    _rows_ready.wait(120)
    return [{"id": row["_id"], "question": str(CONFIG.race(race_id).task().render_prompt(row))}
            for row in _rows.get(race_id, [])]  # fmt: skip


def find_row(race_id: str, row_id: str) -> dict[str, Any] | None:
    _rows_ready.wait(120)
    return next((row for row in _rows.get(race_id, []) if row["_id"] == row_id), None)


# --------------------------------------------------------------------------------------------------
# Thinking tokens: the student's reasoning counted with its own tokenizer (cheap, local)
# --------------------------------------------------------------------------------------------------

_tokenizer: Any = None


def _load_tokenizer() -> None:
    global _tokenizer
    with suppress(Exception):
        from tokenizers import Tokenizer

        _tokenizer = Tokenizer.from_pretrained(CONFIG.student().model)


threading.Thread(target=_load_tokenizer, daemon=True).start()


def count_tokens(text: str) -> int | None:
    if not text:
        return 0
    if _tokenizer is None:
        return None
    return len(_tokenizer.encode(text, add_special_tokens=False).ids)


# --------------------------------------------------------------------------------------------------
# One agent's live state, built from the harness's event stream
# --------------------------------------------------------------------------------------------------


def short_tool(name: str) -> str:
    """`mcp__hf__hub_repo_search` -> `hub_repo_search`; built-in tools keep their name."""
    return name.split("__", 2)[2] if name.startswith("mcp__") and name.count("__") >= 2 else name


def tool_arg(name: str, args: Any) -> str:
    if not isinstance(args, dict):
        return str(args or "")
    for key in ("command", "path", "file_path", "query", "pattern"):
        if isinstance(args.get(key), str) and len(args) <= 2:
            return args[key]
    return json.dumps(args, ensure_ascii=False, separators=(", ", ": "))


def result_text(result: Any) -> str:
    if isinstance(result, dict):
        return "\n".join(str(b.get("text", "")) for b in result.get("content") or [] if isinstance(b, dict))
    return str(result or "")


@dataclass
class AgentRun:
    side: str  # "teacher" | "student"
    label: str
    model: str
    count_thinking: bool = False
    status: str = "waiting"  # waiting | waking | running | done
    started: float | None = None
    finished: float | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    episode: Episode | None = None
    grade: dict[str, Any] | None = None
    error: str | None = None
    usage_in: int = 0
    usage_out: int = 0
    _message: int = -1
    _blocks: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    _tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    _live_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.monotonic()) - self.started

    # Pi's JSON event stream (pi docs/json.md): deltas for the live view, `message_end` is authoritative.
    def on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start" and (event.get("message") or {}).get("role") == "assistant":
            self._message += 1
            self._live_usage = {}
        elif kind == "message_update":
            self._live_usage = event.get("usage") or {}
            self._update(event.get("assistantMessageEvent") or {})
        elif kind == "message_end" and (event.get("message") or {}).get("role") == "assistant":
            message = event["message"]
            for index, block in enumerate(message.get("content") or []):
                self._final_block(index, block)
            usage = message.get("usage") or {}
            self.usage_in += int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
            self.usage_out += int(usage.get("output") or 0)
            self._live_usage = {}
            if message.get("stopReason") in ("error", "aborted") and message.get("errorMessage"):
                self.error = str(message.get("errorMessage"))[:300]
        elif kind == "tool_execution_start":
            row = self._tool_row(str(event.get("toolCallId")), str(event.get("toolName") or ""))
            row.update(status="running", arg=tool_arg(row["tool"], event.get("args")))
        elif kind == "tool_execution_end":
            row = self._tool_row(str(event.get("toolCallId")), str(event.get("toolName") or ""))
            text = result_text(event.get("result"))
            row.update(status="error" if event.get("isError") else "ok", out=text[:600], out_len=len(text))

    def _block(self, index: int, kind: str) -> dict[str, Any]:
        key = (self._message, index)
        if key not in self._blocks:
            row = {"kind": kind, "text": "", "status": "live", "turn": self._message}
            self._blocks[key] = row
            self.rows.append(row)
        return self._blocks[key]

    def _tool_row(self, call_id: str, name: str) -> dict[str, Any]:
        if call_id not in self._tools:
            row = {"kind": "tool", "tool": short_tool(name), "arg": "", "status": "writing", "turn": self._message}
            self._tools[call_id] = row
            self.rows.append(row)
        return self._tools[call_id]

    def _update(self, update: dict[str, Any]) -> None:
        kind, index = update.get("type"), int(update.get("contentIndex") or 0)
        if kind in ("thinking_start", "thinking_delta"):
            self._block(index, "think")["text"] += str(update.get("delta") or "")
        elif kind in ("text_start", "text_delta"):
            self._block(index, "text")["text"] += str(update.get("delta") or "")
        elif kind == "toolcall_start":
            row = self._tool_row(str(update.get("id")), str(update.get("toolName") or ""))
            self._blocks[(self._message, index)] = row
        elif kind == "toolcall_end":
            call = update.get("toolCall") or {}
            row = self._tool_row(str(call.get("id")), str(call.get("name") or ""))
            row.update(arg=tool_arg(row["tool"], call.get("arguments")), status="calling")

    def _final_block(self, index: int, block: dict[str, Any]) -> None:
        kind = block.get("type")
        if kind == "thinking":
            row = self._block(index, "think")
            row.update(text=str(block.get("thinking") or ""), status="done")
        elif kind == "text":
            row = self._block(index, "text")
            row.update(text=str(block.get("text") or ""), status="done")
        elif kind == "toolCall":
            row = self._tool_row(str(block.get("id")), str(block.get("name") or ""))
            row["arg"] = row["arg"] or tool_arg(row["tool"], block.get("arguments"))

    # What the page draws.
    def snapshot(self, price_in: float, price_out: float, gpu_per_hour: float) -> dict[str, Any]:
        rows = []
        thinking_tokens = 0
        for row in self.rows:
            if row["kind"] == "think":
                if not row["text"].strip():
                    continue
                tokens = count_tokens(row["text"]) if self.count_thinking else None
                if tokens is not None:
                    row["tokens"] = tokens
                    thinking_tokens += tokens
                rows.append({k: v for k, v in row.items() if k != "turn"})
            elif row["kind"] == "text":
                if row["text"].strip():
                    rows.append({**{k: v for k, v in row.items() if k != "turn"}, "text": row["text"][-2000:]})
            else:
                rows.append({k: v for k, v in row.items() if k != "turn"})
        answer = None
        if self.episode is not None and self.episode.answer:
            answer = str(self.episode.answer)
            if rows and rows[-1]["kind"] == "text" and rows[-1]["text"].strip() == answer.strip():
                rows.pop()  # shown as the answer below the panel
        tokens_out = self.usage_out + int(self._live_usage.get("output") or 0)
        tool_calls = sum(1 for r in self.rows if r["kind"] == "tool")
        cost = (self.usage_in * price_in + tokens_out * price_out) / 1e6 if self.side == "teacher" else (
            self.elapsed / 3600 * gpu_per_hour)  # fmt: skip
        return {
            "side": self.side,
            "label": self.label,
            "model": self.model,
            "status": self.status,
            "elapsed": round(self.elapsed, 1),
            "tool_calls": tool_calls,
            "rows": rows,
            "thinking_tokens": thinking_tokens if self.count_thinking and _tokenizer is not None else None,
            "output_tokens": tokens_out,
            "cost": round(cost, 5),
            "answer": answer,
            "grade": self.grade,
            "error": self.error if self.status == "done" else None,
        }


# --------------------------------------------------------------------------------------------------
# The student endpoint: wait out a scale-to-zero cold start before the race starts (fair start)
# --------------------------------------------------------------------------------------------------


class EndpointRefused(RuntimeError):
    """The endpoint answered, but refused this Space's token (a missing or under-scoped `HF_TOKEN` secret)."""


async def endpoint_ready(wait: float = 5) -> bool:
    """Whether the student endpoint serves now. Raises `EndpointRefused` on 401/403: waiting would not help."""
    if not ENDPOINT_URL:
        return False
    try:
        async with httpx.AsyncClient(timeout=wait) as client:
            response = await client.get(f"{ENDPOINT_URL}/v1/models", headers=_auth())
    except httpx.HTTPError:
        return False
    if response.status_code in (401, 403):
        raise EndpointRefused(
            f"the student endpoint refused this Space's token (HTTP {response.status_code}); its HF_TOKEN secret "
            "needs inference.endpoints.infer.write on the endpoint's namespace"
        )
    return response.status_code == 200


def _auth() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def wake_endpoint() -> bool:
    """Wait until the endpoint answers; a request to a scaled-to-zero endpoint is what starts its cold start."""
    deadline = time.monotonic() + WAKE_TIMEOUT
    while time.monotonic() < deadline:
        if await endpoint_ready(wait=30):
            return True
        await asyncio.sleep(5)
    return False


# Agent isolation: checked once at startup (as the stage does before any spend), shown in /api/status.
ISOLATION: dict[str, Any] = {"mode": isolation.mode(), "ok": None, "error": None}


def _preflight() -> None:
    try:
        isolation.preflight()
        ISOLATION["ok"] = True
    except Exception as exc:  # reported; every race's session runs it again and refuses
        ISOLATION.update(ok=False, error=str(exc)[:300])


threading.Thread(target=_preflight, daemon=True).start()


# --------------------------------------------------------------------------------------------------
# Grading: the task's rubric set in eval mode (as the `eval` stage scores)
# --------------------------------------------------------------------------------------------------


async def grade(race: race_config.Race, episode: Episode, row: dict[str, Any]) -> dict[str, Any]:
    task = race.task()
    clean = {k: v for k, v in row.items() if k != "_id"}
    try:
        result = await task.rubrics.ascore(episode_context(episode, clean, task=task, limits=race.limits()),
                                           mode="eval")  # fmt: skip
    except Exception as exc:
        return {"passed": None, "error": f"{type(exc).__name__}: {exc}"[:300], "scores": {}}
    shown = {r.name: result.scores.get(r.name) for r in task.rubrics if r.kind != "metric"}
    return {
        "passed": None if result.errored else result.passed,
        "scores": shown,
        "error": "; ".join(f"{k}: {v}" for k, v in result.errors.items())[:300] or None,
    }


# --------------------------------------------------------------------------------------------------
# A race
# --------------------------------------------------------------------------------------------------


@dataclass
class Race:
    race: race_config.Race
    question: str
    row: dict[str, Any] | None  # a dataset row (graded) or None (a typed question: not graded)
    teacher: AgentRun = field(init=False)
    student: AgentRun = field(init=False)
    started: float = field(default_factory=time.monotonic)
    graded: bool = False

    def __post_init__(self) -> None:
        teacher = self.race.teacher()
        self.teacher = AgentRun("teacher", teacher.model.split("/")[-1], teacher.model)
        self.student = AgentRun("student", self.race.student_label, self.race.adapter, count_thinking=True)
        self.price_in, self.price_out = teacher.price.input, teacher.price.output

    def frame(self) -> dict[str, Any]:
        gpu = CONFIG.endpoint.price_per_hour
        limits = self.race.limits()
        snaps = {s: getattr(self, s).snapshot(self.price_in, self.price_out, gpu) for s in ("teacher", "student")}
        finished = [s for s in snaps.values() if s["status"] == "done" and s["answer"]]
        first = min(finished, key=lambda s: s["elapsed"])["side"] if finished else None
        return {
            "t": round(time.monotonic() - self.started, 2),
            "race": {
                "id": self.race.id,
                "name": self.race.name,
                "harness": self.race.config.harness,
                "question": self.question,
                "row_id": (self.row or {}).get("_id"),
                "graded": self.row is not None,
                "limits": {"timeout": limits.timeout, "max_tool_calls": limits.max_tool_calls},
            },
            **snaps,
            "first": first,
            "isolation": isolation.mode(),
        }

    async def run(self) -> None:
        task = self.race.task()
        limits = self.race.limits()
        row = self.row or {self.race.question_field: self.question}
        row = {k: v for k, v in row.items() if k != "_id"}
        if self.row is None:  # a typed question has no dataset id
            task = dataclasses.replace(task, row_id=lambda _row: "typed")
        self.student.status = "waking"
        try:
            if not await endpoint_ready() and not await wake_endpoint():
                self.student.status, self.student.error = "done", "the student's endpoint did not wake up"
        except EndpointRefused as exc:
            self.student.status, self.student.error = "done", str(exc)
        harness = load_harness(self.race.harness())
        endpoints: list[tuple[AgentRun, Endpoint]] = [(self.teacher, self.race.teacher_endpoint())]
        if self.student.status != "done":
            endpoints.append((self.student, self.race.student_endpoint(ENDPOINT_URL)))
        self.started = time.monotonic()
        async with harness.session([e for _, e in endpoints], cache=default_cache(), task=task) as runner:

            async def one(run: AgentRun, endpoint: Endpoint) -> None:
                run.status, run.started = "running", time.monotonic()
                try:
                    run.episode = await harness.run_episode(runner, task, row, endpoint, limits,
                                                            on_event=run.on_event)  # fmt: skip
                    if run.episode.failure:
                        run.error = run.episode.failure
                    elif run.episode.answer is None and run.episode.answer_error:
                        run.error = run.episode.answer_error
                except Exception as exc:
                    run.error = f"{type(exc).__name__}: {exc}"[:300]
                finally:
                    run.finished = time.monotonic()
                    run.status = "done"

            await asyncio.gather(*(one(run, endpoint) for run, endpoint in endpoints))
        if self.row is not None:
            for run in (self.teacher, self.student):
                if run.episode is not None:
                    run.grade = await grade(self.race, run.episode, self.row)
        self.graded = True
