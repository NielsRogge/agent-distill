"""Record a race for the page's replay button: run it against a running Space (or local server) and save its frames.

    python record.py --url http://127.0.0.1:7860 --race mcp --row f545bfcd9ae7   # -> static/replays/mcp.json

Frames that change nothing are dropped, long thoughts are cut to their last characters, and the rest is thinned to at
most `--max-frames` (keeping every frame where a row or a status appears or changes), so the file stays small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
THOUGHT_CHARS = 1500


def frames_from(url: str, race: str, row: str, timeout: float) -> list[dict[str, Any]]:
    frames = []
    with httpx.stream("GET", f"{url.rstrip('/')}/api/race", params={"race": race, "row": row}, timeout=timeout) as r:
        r.raise_for_status()
        event = ""
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "frame":
                frames.append(json.loads(line[6:]))
            elif line.startswith("data: ") and event == "failed":
                raise SystemExit(f"the race failed: {line[6:]}")
    return frames


def slim(frame: dict[str, Any]) -> dict[str, Any]:
    for side in ("teacher", "student"):
        for row in frame[side]["rows"]:
            if row.get("kind") == "think" and len(row.get("text", "")) > THOUGHT_CHARS:
                row["text"] = "…" + row["text"][-THOUGHT_CHARS:]
            if row.get("out"):
                row["out"] = row["out"][:300]
    return frame


def structure(frame: dict[str, Any]) -> str:
    """What must never be skipped in a replay: rows appearing, statuses changing, answers and grades."""
    sides = []
    for side in ("teacher", "student"):
        run = frame[side]
        rows = [(r.get("kind"), r.get("status"), r.get("arg")) for r in run["rows"]]
        sides.append([rows, run["status"], run["answer"], run["grade"]])
    return json.dumps(sides)


def thin(frames: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    kept, last = [], None
    for frame in frames:
        key = json.dumps({s: frame[s] for s in ("teacher", "student")}, sort_keys=True)
        if key != last:
            kept.append(frame)
            last = key
    if len(kept) <= limit:
        return kept
    essential = {0, len(kept) - 1}
    essential |= {i for i in range(1, len(kept)) if structure(kept[i]) != structure(kept[i - 1])}
    rest = [i for i in range(len(kept)) if i not in essential]
    room = max(0, limit - len(essential))
    step = len(rest) / room if room else float("inf")
    sampled = {rest[int(k * step)] for k in range(room)} if room else set()
    return [kept[i] for i in sorted(essential | sampled)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--race", required=True)
    parser.add_argument("--row", required=True, help="a question id of the race's split")
    parser.add_argument("--from-sse", type=Path, help="read a saved event stream instead of running a race")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.from_sse:
        lines = args.from_sse.read_text().splitlines()
        frames = [json.loads(line[6:]) for line in lines if line.startswith('data: {"t"')]
    else:
        frames = frames_from(args.url, args.race, args.row, args.timeout)
    frames = thin([slim(f) for f in frames], args.max_frames)
    out = HERE / "static" / "replays" / f"{args.race}.json"
    out.write_text(json.dumps(frames, ensure_ascii=False, separators=(",", ":")))
    print(f"{out}: {len(frames)} frames, {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
