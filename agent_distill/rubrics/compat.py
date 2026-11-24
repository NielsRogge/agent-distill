"""Reward functions written for other libraries, as rubrics (DESIGN.md §4, §12).

- `from_callable`: Tinker-cookbook's `RewardFn`, `async (messages) -> float | (float, metrics)` (sync works too).
- `from_verifiers`: classic `verifiers` reward functions, `fn(prompt, completion, answer, state, info, **kw)`,
  receiving only the arguments they name.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from agent_distill.rubrics.core import Rubric


def from_callable(fn: Callable[..., Any], *, name: str | None = None, weight: float = 1.0, **options: Any) -> Rubric:
    """Wrap `fn(messages)` (Tinker `RewardFn` shape) as a programmatic rubric."""

    async def rubric(messages: list[dict[str, Any]]) -> Any:
        value = fn(messages)
        return await value if inspect.isawaitable(value) else value

    return Rubric(fn=rubric, name=name or getattr(fn, "__name__", "reward_fn"), weight=weight, **options)


def split_prompt_completion(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """verifiers' view: the prompt is everything before the first assistant message, the completion the rest."""
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            return messages[:index], messages[index:]
    return list(messages), []


def from_verifiers(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    weight: float = 1.0,
    answer_key: str = "answer",
    parser: Any = None,
    **options: Any,
) -> Rubric:
    """Wrap a classic verifiers reward function as a programmatic rubric."""
    signature = inspect.signature(fn)
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())

    async def rubric(episode: Any, row: Mapping[str, Any], messages: list[dict[str, Any]]) -> Any:
        prompt, completion = split_prompt_completion(messages)
        available = {
            "prompt": prompt,
            "completion": completion,
            "answer": row.get(answer_key, ""),
            "state": {"episode": episode, "answer": episode.answer},
            "info": dict(row),
            "task": getattr(episode, "task", None),
            "parser": parser,
            **dict(row),
        }
        arguments = available if takes_kwargs else {k: v for k, v in available.items() if k in signature.parameters}
        value = fn(**arguments)
        return await value if inspect.isawaitable(value) else value

    return Rubric(fn=rubric, name=name or fn.__name__, weight=weight, **options)
