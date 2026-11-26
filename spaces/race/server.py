"""Web server for the race: a static page, the races and their questions, and a server-sent-events stream of race
frames (everything the page draws, as plain JSON)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import core
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

STATIC = Path(__file__).parent / "static"
MAX_RACES = int(os.environ.get("MAX_CONCURRENT_RACES", "2"))
MAX_QUESTION_CHARS = 600
_slots = asyncio.Semaphore(MAX_RACES)


async def index(_: Request) -> Response:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


async def config(_: Request) -> Response:
    cfg = core.CONFIG
    races = []
    for race in cfg.races:
        races.append({
            "id": race.id, "name": race.name, "harness": race.config.harness, "tools": race.tools_note,
            "teacher": race.teacher().model, "student": race.student_label, "adapter": race.adapter,
            "split": race.split, "dataset": race.task().dataset.repo,
            "limits": {"timeout": race.limits().timeout, "max_tool_calls": race.limits().max_tool_calls},
            "replay": (STATIC / "replays" / f"{race.id}.json").exists(),
        })  # fmt: skip
    e = cfg.endpoint
    endpoint = {"name": f"{e.namespace}/{e.name}", "gpu": e.instance_type, "price_per_hour": e.price_per_hour,
                "idle_minutes": e.scale_to_zero_minutes}  # fmt: skip
    return JSONResponse({"title": cfg.title, "lede": cfg.lede, "races": races, "endpoint": endpoint})


async def questions(request: Request) -> Response:
    race_id = request.query_params.get("race", "")
    if race_id not in {r.id for r in core.CONFIG.races}:
        return JSONResponse({"error": "unknown race"}, status_code=404)
    return JSONResponse(await asyncio.to_thread(core.questions, race_id))


async def status(_: Request) -> Response:
    missing = sorted({c for race in core.CONFIG.races for c in race.task().missing_commands()})
    return JSONResponse({
        "student_ready": await _ready(),
        "isolation": core.ISOLATION,
        "missing_commands": missing,
        "busy": _slots.locked(),
    })  # fmt: skip


async def _ready() -> bool | str:
    try:
        return await core.endpoint_ready()
    except core.EndpointRefused as exc:
        return str(exc)


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def race_events(race: core.Race) -> AsyncIterator[str]:
    if _slots.locked():
        yield sse("queued", {"message": f"All {MAX_RACES} race slots are busy; waiting for one to free up…"})
    async with _slots:
        runner = asyncio.create_task(race.run())
        try:
            while not runner.done():
                yield sse("frame", race.frame())
                await asyncio.wait({runner}, timeout=0.5)
            error = runner.exception()
            yield sse("frame", race.frame())
            if error is not None:
                print(f"[race] failed: {error!r}", flush=True)
                yield sse("failed", {"message": f"The race failed: {type(error).__name__}: {str(error)[:200]}"})
            for run in (race.teacher, race.student):
                grade = (run.grade or {}).get("passed")
                print(f"[race] {race.race.id} {run.side}: {run.elapsed:.1f}s, "
                      f"{sum(r['kind'] == 'tool' for r in run.rows)} tool calls, passed={grade}"
                      + (f", error={run.error!r}" if run.error else ""), flush=True)  # fmt: skip
            yield sse("done", {})
        finally:
            if not runner.done():  # the visitor left: stop the agents (the adapter kills their processes)
                runner.cancel()


async def race(request: Request) -> Response:
    params = request.query_params
    race_id, row_id, typed = params.get("race", ""), params.get("row", ""), params.get("q", "").strip()
    try:
        race_cfg = core.CONFIG.race(race_id)
    except StopIteration:
        return JSONResponse({"error": "unknown race"}, status_code=404)
    row = None
    if row_id:
        row = await asyncio.to_thread(core.find_row, race_id, row_id)
        if row is None:
            return JSONResponse({"error": "unknown question"}, status_code=404)
        question = race_cfg.task().render_prompt({k: v for k, v in row.items() if k != "_id"})
    elif typed:
        if len(typed) > MAX_QUESTION_CHARS:
            return JSONResponse({"error": f"Questions are limited to {MAX_QUESTION_CHARS} characters."},
                                status_code=400)  # fmt: skip
        question = typed
    else:
        return JSONResponse({"error": "Pick a question or type one."}, status_code=400)
    return StreamingResponse(race_events(core.Race(race_cfg, question, row)), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})  # fmt: skip


app = Starlette(routes=[
    Route("/", index),
    Route("/api/config", config),
    Route("/api/questions", questions),
    Route("/api/status", status),
    Route("/api/race", race),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
])  # fmt: skip
