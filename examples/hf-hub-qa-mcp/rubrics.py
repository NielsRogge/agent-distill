"""Rubrics of the hf-hub-qa-mcp task (no judge: every score is deterministic; the fact and grounding checks are the
library's, shared with hf-hub-qa).

- `answered` (gate): the episode ended with a final message (not cut off while still calling tools).
- `allowed_tools` (gate): every tool call was one of the task's MCP tools. Pi only declares those, so this is a
  backstop; anything else fails the episode.
- `used_hub` (gate): at least one successful MCP call. An answer from memory fails, however right.
- `fact_correct`: the answer states the row's frozen gold value (facts.py). When it does not, the value is re-read
  from the Hub (anonymously); if the Hub changed since the snapshot, the row has drifted and the episode is
  unscorable (None), never a 0.
- `grounded`: of the Hub ids the answer names (agent_distill.rubrics.grounding, shared with hf-hub-qa), the share that
  appears in the episode's own MCP calls and results. Not scored (None) when the answer names no ids.
- Metrics: `mcp_calls`, `answer_chars`.

reward = gate(answered, gate(allowed_tools, gate(used_hub, weighted(fact_correct 1, grounded 0.25))))
         + 0.1 * (1 - mcp_calls / 20) when the episode passed    (the tool-efficiency bonus, for GRPO)
passed = answered, allowed tools only, used the Hub, fact_correct == 1 and grounded in (None, 1).
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from agent_distill.hub_facts import live_value
from agent_distill.rubrics import RubricSet, bonus_if, facts, gate, metric, process, programmatic, weighted
from agent_distill.rubrics.grounding import grounded_score

SERVER = "hf"
TOOLS = ("hub_repo_search", "hub_repo_details", "hf_fs")
ALLOWED = {f"mcp__{SERVER}__{tool}" for tool in TOOLS}
EFFICIENCY_CALLS = 20


def mcp_events(tool_events: list[Any]) -> list[Any]:
    return [e for e in tool_events or [] if str(getattr(e, "name", "")).startswith("mcp__")]


def tool_text(tool_events: list[Any]) -> str:
    return "\n".join(f"{json.dumps(getattr(e, 'input', None) or {})}\n{getattr(e, 'output', '') or ''}"
                     for e in tool_events or [])  # fmt: skip


@programmatic(weight=0.0)
def answered(answer: Any) -> float:
    return float(bool(str(answer or "").strip()))


@process(weight=0.0)
def allowed_tools(tool_events: list[Any]) -> float:
    return float(all(getattr(e, "name", "") in ALLOWED for e in tool_events or []))


@process(weight=0.0)
def used_hub(tool_events: list[Any]) -> float:
    return float(any(getattr(e, "success", False) for e in mcp_events(tool_events)))


@lru_cache(maxsize=4096)
def live(spec: str) -> Any:
    """The row's value on the Hub now (anonymous), cached for a stage's lifetime."""
    return live_value(json.loads(spec))


def drifted(row: dict[str, Any]) -> bool:
    kind, gold, extra = row["kind"], json.loads(row["gold"]), json.loads(row.get("extra") or "{}")
    now = live(row["live"])
    return now is None or not facts.check(kind, gold, facts.render(kind, now), **extra)


@programmatic(weight=1.0)
def fact_correct(answer: Any, row: dict[str, Any]) -> float | None:
    if not answer:
        return 0.0
    kind, gold, extra = row["kind"], json.loads(row["gold"]), json.loads(row.get("extra") or "{}")
    if facts.check(kind, gold, str(answer), **extra):
        return 1.0
    return None if drifted(row) else 0.0


@programmatic(weight=0.25)
def grounded(answer: Any, tool_events: list[Any]) -> tuple[float, dict[str, float]] | None:
    return grounded_score(str(answer or ""), tool_text(tool_events))


@metric
def mcp_calls(tool_events: list[Any]) -> float:
    return float(len(mcp_events(tool_events)))


@metric
def answer_chars(answer: Any) -> float | None:
    return float(len(str(answer))) if answer else None


def efficiency(scores: dict[str, float | None]) -> float:
    calls = scores.get("mcp_calls") or 0.0
    return 0.1 * max(0.0, 1.0 - calls / EFFICIENCY_CALLS)


def passed(scores: dict[str, float | None]) -> bool:
    return (
        scores.get("answered") == 1.0
        and scores.get("allowed_tools") == 1.0
        and scores.get("used_hub") == 1.0
        and scores.get("fact_correct") == 1.0
        and scores.get("grounded") in (None, 1.0)
    )


RUBRICS = RubricSet(
    [answered, allowed_tools, used_hub, fact_correct, grounded, mcp_calls, answer_chars],
    reward=bonus_if(
        efficiency,
        gate("answered", gate("allowed_tools", gate("used_hub", weighted({"fact_correct": 1.0, "grounded": 0.25})))),
        when=lambda scores: float(passed(scores)),  # only on top of a full, gated pass
    ),
    passed=passed,
)
