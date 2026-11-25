"""Reasoning leakage: a model's thinking showing up in its visible answers.

A thinking model served without a reasoning parser (Qwen3.5 in its default thinking mode, for example) writes its
reasoning into the message content, ending with `</think>`. The answer the rubrics score then contains the
reasoning: checkers read its numbers and names, and the reasoning uses up the output-token budget. SFT on teacher
turns without reasoning (`keep_reasoning = false`) also expects a student that does not think. Serve such students
with thinking off (e.g. the `qwen3.5-4b-nothink` profile) or with a reasoning parser.

The episode stages (`traces`, `eval`) mark every episode (`metadata["reasoning_leak"]`), warn, count leaks in their
summaries (`reasoning_leaks`) and stop, through `LeakGuard`, once too many episodes leak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_distill.rubrics import TooManyErrors

MARKERS = re.compile(r"</?think(?:ing)?>", re.IGNORECASE)


def leaked(episode: Any) -> bool:
    """Does any assistant turn's visible content (or the answer, if it is text) carry a reasoning marker?"""
    texts = [m.get("content") for m in getattr(episode, "messages", None) or [] if m.get("role") == "assistant"]
    answer = getattr(episode, "answer", None)
    if isinstance(answer, str):
        texts.append(answer)
    return any(isinstance(text, str) and MARKERS.search(text) for text in texts)


class ReasoningLeak(TooManyErrors):
    """Raised by `LeakGuard` when too many episodes leak reasoning into their visible content."""


@dataclass
class LeakGuard:
    """Stops a stage once more than `max_rate` of at least `min_episodes` episodes leaked reasoning (a serving
    misconfiguration, not the model's skill), so a whole evaluation or trace run is not spent on it. `max_rate`
    None turns the stop off (episodes are still marked and counted)."""

    max_rate: float | None = 0.2  # None or negative: never stop
    min_episodes: int = 20
    seen: int = 0
    leaks: int = 0

    def record(self, leak: bool) -> None:
        self.seen += 1
        self.leaks += int(leak)
        if (
            self.max_rate is not None
            and self.max_rate >= 0
            and self.seen >= self.min_episodes
            and self.leaks > self.max_rate * self.seen
        ):
            raise ReasoningLeak(
                f"{self.leaks} of {self.seen} episodes leaked reasoning (</think> or <think> in the visible content; "
                f"limit {self.max_rate:.0%}). Serve the model with thinking off (e.g. a *-nothink student profile) "
                "or with a reasoning parser; --max-reasoning-leak-rate raises the limit."
            )
