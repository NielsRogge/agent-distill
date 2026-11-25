"""Rubrics of the judged hf-hub-qa questions (judged.py): open-ended questions about the Hub, scored by a judge.

- `answered` (gate): the episode ended with a final message (not cut off while still calling tools).
- `read_only` (gate): no command that writes to the Hub or logs in (`hf upload`, `hf repos create|delete`, `hf jobs
  run`, ...; `--help` on one is fine). The agent has no token, so such commands fail anyway; trying them still
  fails the episode.
- `used_hub` (gate): at least one successful `hf` command (or Hub API call). An answer from memory fails, however
  good: the task is to look things up.
- `grounded`: of the Hub ids the answer names, the share that appears in the episode's own tool output
  (agent_distill.rubrics.grounding, shared with the MCP twin). Ids the agent never saw are guesses. Not scored
  (None) when the answer names no ids.
- `reference`: for questions with a canonical answer (questions.py), the stage runs the reference `hf` command when
  it scores and checks the answer against it: recall of the top-k ids (a match anywhere in the listing's top 2k
  counts) or the stated fact. Not scored (None) without a reference.
- `answer_quality` (judge): correctness against what the tools returned, relevance, completeness and clarity, 1-10.
- Metrics, recorded only: `hf_calls`, `hf_errors` (calls the CLI rejected), `read_skill`, answer length.

reward = gate(answered, gate(read_only, gate(used_hub, weighted(answer_quality 2, grounded 1, reference 1))));
an episode passes when it answered, stayed read-only, used the Hub, every named id was seen (grounded = 1), the
reference (if any) is at least 2/3 met and the judge scores at least 7/10.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent_distill.rubrics import RubricSet, gate, programmatic, weighted
from agent_distill.rubrics.grounding import grounded_score
from agent_distill.rubrics.judge import judge

from .cli import (
    answer_chars,
    answered,
    command_of,
    hf_calls,
    hf_errors,
    read_only,
    read_skill,
    run_hf,
    tool_text,
    used_hub,
)

PASS_SCORE = 7  # of 10
SCALE = (1.0, 10.0)
CRITERIA = ("correctness", "relevance", "completeness", "clarity")
REFERENCE_PASS = 2 / 3
TOOL_LOG_CHARS = 14000  # what the judge sees of the agent's tool calls and their output
OUTPUT_CHARS = 1800  # per tool call


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
up with the `hf` command-line tool; the Hub changes daily, so judge the answer against what the tools returned
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
        call = command_of(event) or json.dumps(getattr(event, "input", None) or {})[:300]
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
        and scores.get("read_only") == 1.0
        and scores.get("used_hub") == 1.0
        and scores.get("grounded") in (None, 1.0)
        and (ref is None or ref >= REFERENCE_PASS - 1e-9)
        and quality is not None
        and quality >= threshold - 1e-9
    )


RUBRICS = RubricSet(
    [answered, read_only, used_hub, grounded, reference, answer_quality, hf_calls, hf_errors, read_skill, answer_chars],
    reward=gate(
        "answered",
        gate("read_only", gate("used_hub", weighted({"answer_quality": 2.0, "grounded": 1.0, "reference": 1.0}))),
    ),
    passed=passed,
)
