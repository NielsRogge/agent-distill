"""Rubrics of the hf-hub-qa task (task.py): Hub facts, scored deterministically (no judge).

The same rubrics as the MCP twin (examples/hf-hub-qa-mcp), with the CLI's tool gates:

- `answered` (gate): the episode ended with a final message (not cut off while still calling tools).
- `read_only` (gate): no command that writes to the Hub or logs in (cli.py). The CLI counterpart of the MCP twin's
  `allowed_tools`: the agent has a shell, so the gate is on what it runs rather than which tools exist.
- `used_hub` (gate): at least one successful `hf` command (or Hub API call). An answer from memory fails, however
  right.
- `fact_correct`: the answer states the row's frozen gold value (agent_distill.rubrics.facts). When it does not, the
  value is re-read from the Hub, anonymously (agent_distill.hub_facts); if the Hub changed since the snapshot, the
  row has drifted and the episode is unscorable (None), never a 0.
- `grounded`: of the Hub ids the answer names, the share that appears in the episode's own tool output
  (agent_distill.rubrics.grounding). Not scored (None) when the answer names no ids.
- Metrics: `hf_calls`, `answer_chars`.

reward = gate(answered, gate(read_only, gate(used_hub, weighted(fact_correct 1, grounded 0.25))))
         + 0.1 * (1 - hf_calls / 20) when the episode passed    (the tool-efficiency bonus, for GRPO)
passed = answered, read-only, used the Hub, fact_correct == 1 and grounded in (None, 1).
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from agent_distill.hub_facts import live_value
from agent_distill.rubrics import RubricSet, bonus_if, facts, gate, programmatic, weighted
from agent_distill.rubrics.grounding import grounded_score

from .cli import answer_chars, answered, hf_calls, read_only, tool_text, used_hub

EFFICIENCY_CALLS = 20


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


def efficiency(scores: dict[str, float | None]) -> float:
    calls = scores.get("hf_calls") or 0.0
    return 0.1 * max(0.0, 1.0 - calls / EFFICIENCY_CALLS)


def passed(scores: dict[str, float | None]) -> bool:
    return (
        scores.get("answered") == 1.0
        and scores.get("read_only") == 1.0
        and scores.get("used_hub") == 1.0
        and scores.get("fact_correct") == 1.0
        and scores.get("grounded") in (None, 1.0)
    )


RUBRICS = RubricSet(
    [answered, read_only, used_hub, fact_correct, grounded, hf_calls, answer_chars],
    reward=bonus_if(
        efficiency,
        gate("answered", gate("read_only", gate("used_hub", weighted({"fact_correct": 1.0, "grounded": 0.25})))),
        when=lambda scores: float(passed(scores)),  # only on top of a full, gated pass
    ),
    passed=passed,
)
