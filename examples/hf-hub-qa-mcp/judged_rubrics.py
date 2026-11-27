"""Rubrics of the judged hf-hub-qa-mcp questions (`judged.py` / `open_ended.py`).

MCP twin of `hf-hub-qa/judged_rubrics.py`: shares the exact same grounding scorer, live `hf` CLI reference check
(executed by the stage process, never exposed to the MCP agent), Gemini 4 Hydrogen judge prompt/thresholds (`>= 7/10`),
and pass rule (`answered`, `allowed_tools`, `used_hub`, `grounded == 1`, `reference >= 2/3`, `answer_quality >= 7/10`)
so CLI and MCP open-ended numbers compare apples-to-apples.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from functools import lru_cache
from typing import Any

from agent_distill.rubrics import RubricSet, gate, programmatic, weighted
from agent_distill.rubrics.grounding import grounded_score
from agent_distill.rubrics.judge import judge

from .rubrics import allowed_tools, answer_chars, answered, mcp_calls, mcp_errors, tool_text, used_hub

PASS_SCORE = 7  # of 10
SCALE = (1.0, 10.0)
CRITERIA = ("correctness", "relevance", "completeness", "clarity")
REFERENCE_PASS = 2 / 3
TOOL_LOG_CHARS = 14000
OUTPUT_CHARS = 1800
HF_VERSION = "2.4.0"

# Stage-only setup for the live `reference` scorer (`run_hf`); the MCP agent itself has `tools=()` and no shell access.
HF_HUB_SPEC = f"huggingface_hub=={HF_VERSION}"  # pragma: allowlist secret
SCORER_JOB_SETUP = [
    "export UV_TOOL_DIR=/opt/hf-cli/tools UV_TOOL_BIN_DIR=/opt/hf-cli/bin",
    f"uv tool install -q --python 3.12 --python-preference only-managed {HF_HUB_SPEC}",
    'export PATH="/opt/hf-cli/bin:$PATH" HF_HUB_DISABLE_UPDATE_CHECK=1',
]


@lru_cache(maxsize=256)
def run_hf(args: tuple[str, ...]) -> Any:
    """Run one anonymous read-only `hf ... --format json` command for live reference scoring (cached per stage)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_TOKEN") and k != "HUGGING_FACE_HUB_TOKEN"}
    env["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"
    proc = subprocess.run(["hf", *args, "--format", "json"], capture_output=True, text=True, timeout=60, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"hf {' '.join(args)} failed: {proc.stderr[:200]}")
    return json.loads(proc.stdout)


@programmatic(weight=1.0)
def grounded(answer: Any, tool_events: list[Any]) -> tuple[float, dict[str, float]] | None:
    return grounded_score(str(answer or ""), tool_text(tool_events))


def dig(data: Any, path: str) -> Any:
    for part in path.split("."):
        data = data.get(part) if isinstance(data, dict) else None
    return data


def normalize(text: str) -> str:
    return re.sub(r"[\s_-]+", "", text.lower())


@programmatic(weight=1.0)
def reference(answer: Any, row: dict[str, Any]) -> tuple[float, dict[str, float]] | None:
    spec = json.loads(row.get("reference") or "null")
    if not spec or not answer:
        return None
    text = str(answer)
    if "fact" in spec:
        value = dig(run_hf(tuple(spec["fact"])), spec["path"])
        if value in (None, ""):
            return None
        return float(normalize(str(value)) in normalize(text))
    listing = run_hf(tuple(spec["ids"]))
    k = int(spec.get("k", 5))
    window = [str(item.get(spec.get("key", "id"))) for item in listing][: 2 * k]
    lowered = text.lower()
    hits = sum(1 for i in window if i.lower() in lowered)
    return min(1.0, hits / k), {"hits": float(hits), "k": float(k)}


JUDGE_PROMPT = """You are grading an assistant that answers questions about the Hugging Face Hub. It looked things
up with the Hugging Face MCP server tools; the Hub changes daily, so judge the answer against what the tools returned
(shown below), not against your own memory of the Hub.

Question: {question}

Tool calls and their output (truncated):
{tool_log}

The assistant's final answer:
{answer}

Score each criterion from 1 (poor) to 10 (excellent):
- correctness: every claim is supported by the tool output or is uncontroversial background knowledge; no invented
  repos, papers, numbers or features. Penalise heavily anything that contradicts the tool output.
- relevance: it answers the question that was asked (the right task, filters, time frame, organisation).
- completeness: it gives enough specific items or detail to be useful (ids, a reason for each), and says what the
  Hub could not tell it, if anything.
- clarity: concise, well organised, easy to act on.

Answer with one line per criterion, then the overall score, exactly in this form:
correctness: <1-10>
relevance: <1-10>
completeness: <1-10>
clarity: <1-10>
SCORE: <1-10>"""


def tool_log(tool_events: list[Any]) -> str:
    parts = []
    for event in tool_events or []:
        out = str(getattr(event, "output", "") or getattr(event, "error", "") or "")
        if len(out) > OUTPUT_CHARS:
            out = out[:OUTPUT_CHARS] + f"... [{len(out) - OUTPUT_CHARS} more characters]"
        call = json.dumps(getattr(event, "input", None) or {})[:300]
        parts.append(f"$ [{getattr(event, 'name', '')}] {call}\n{out}")
    text = "\n\n".join(parts)
    return text if len(text) <= TOOL_LOG_CHARS else text[:TOOL_LOG_CHARS] + "\n... [truncated]"


@judge(JUDGE_PROMPT, scale=SCALE, criteria=CRITERIA)
def answer_quality(answer: Any, row: dict[str, Any], tool_events: list[Any]) -> dict[str, Any] | None:
    if not str(answer or "").strip():
        return None
    return {"question": row.get("question", ""), "tool_log": tool_log(tool_events) or "(no tool calls)",
            "answer": str(answer)}  # fmt: skip


def passed(scores: dict[str, float | None]) -> bool:
    quality = scores.get("answer_quality")
    threshold = (PASS_SCORE - SCALE[0]) / (SCALE[1] - SCALE[0])
    ref = scores.get("reference")
    return (
        scores.get("answered") == 1.0
        and scores.get("allowed_tools") == 1.0
        and scores.get("used_hub") == 1.0
        and scores.get("grounded") in (None, 1.0)
        and (ref is None or ref >= REFERENCE_PASS - 1e-9)
        and quality is not None
        and quality >= threshold - 1e-9
    )


RUBRICS = RubricSet(
    [answered, allowed_tools, used_hub, grounded, reference, answer_quality, mcp_calls, mcp_errors, answer_chars],
    reward=gate(
        "answered",
        gate("allowed_tools", gate("used_hub", weighted({"answer_quality": 2.0, "grounded": 1.0, "reference": 1.0}))),
    ),
    passed=passed,
)
