"""Rubrics and rubric sets (DESIGN.md §4).

A rubric is a function that names the inputs it wants (`row`, `answer`, `tool_events`, ...) and returns a score
in [0, 1] (bool and int are cast), `None` for "unscorable", or `(score, metrics)`. A `RubricSet` runs its rubrics
on one episode and turns the scores into a `reward` (GRPO) and a `passed` verdict (trace acceptance, the
difficulty probe, pass@1). The same set is used everywhere; rubrics tagged `train_only` / `eval_only` are the only
allowed differences between training and evaluation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

Mode = Literal["train", "eval"]
Scores = Mapping[str, float | None]
Expr = Callable[[Scores], float | None]

# Arguments a rubric may name. `judge` is the judge runner (judge rubrics); `limits` the run's Limits.
INJECTABLE = frozenset(
    {"episode", "row", "answer", "messages", "tool_events", "workspace", "artifacts", "judge", "limits", "task"}
)


@dataclass
class Rubric:
    fn: Callable[..., Any]
    name: str
    kind: Literal["programmatic", "process", "judge", "metric"] = "programmatic"
    weight: float = 1.0
    train_only: bool = False
    eval_only: bool = False
    neutral: float | None = 1.0  # the value `reward` sees when the rubric is inactive in a mode
    judge_spec: Any = None  # JudgeSpec for judge rubrics
    _params: tuple[str, ...] = field(default=(), repr=False)
    _takes_kwargs: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.train_only and self.eval_only:
            raise ValueError(f"rubric {self.name!r} cannot be both train_only and eval_only")
        signature = inspect.signature(self.fn)
        params = []
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                self._takes_kwargs = True
            elif parameter.kind in (inspect.Parameter.VAR_POSITIONAL,):
                continue
            else:
                if parameter.name not in INJECTABLE and parameter.default is inspect.Parameter.empty:
                    allowed = sorted(INJECTABLE)
                    raise TypeError(
                        f"rubric {self.name!r} asks for {parameter.name!r}; injectable arguments are {allowed}"
                    )
                params.append(parameter.name)
        self._params = tuple(params)

    def active(self, mode: Mode) -> bool:
        return not ((mode == "train" and self.eval_only) or (mode == "eval" and self.train_only))

    def arguments(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if self._takes_kwargs:
            return dict(context)
        return {name: context[name] for name in self._params if name in context}

    async def ascore(self, context: Mapping[str, Any]) -> tuple[float | None, dict[str, float]]:
        if self.kind == "judge":
            from agent_distill.rubrics.judge import run_judge_rubric

            return await run_judge_rubric(self, context)
        value = self.fn(**self.arguments(context))
        if inspect.isawaitable(value):
            value = await value
        return normalize(value, self.name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def normalize(value: Any, name: str) -> tuple[float | None, dict[str, float]]:
    metrics: dict[str, float] = {}
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        value, extra = value
        metrics = {f"{name}/{key}": float(item) for key, item in extra.items()}
    if value is None:
        return None, metrics
    return float(value), metrics


def _decorator(kind: str, default_weight: float):
    def decorate(fn: Callable[..., Any] | None = None, /, **options: Any):
        def wrap(function: Callable[..., Any]) -> Rubric:
            options.setdefault("weight", default_weight)
            return Rubric(fn=function, name=options.pop("name", function.__name__), kind=kind, **options)  # type: ignore[arg-type]

        return wrap(fn) if fn is not None else wrap

    decorate.__name__ = kind
    decorate.__doc__ = f"Declare a {kind} rubric: `@{kind}` or `@{kind}(weight=..., train_only=..., name=...)`."
    return decorate


programmatic = _decorator("programmatic", 1.0)
process = _decorator("process", 1.0)
metric = _decorator("metric", 0.0)


# --------------------------------------------------------------------------------------------------
# Reward expressions
# --------------------------------------------------------------------------------------------------


def _expr(value: str | float | Expr) -> Expr:
    if isinstance(value, str):
        return lambda scores: scores.get(value)
    if callable(value):
        return value
    constant = float(value)
    return lambda scores: constant


def score(name: str) -> Expr:
    """The score of one rubric."""
    return _expr(name)


def weighted(weights: Mapping[str, float]) -> Expr:
    """Weighted mean of the named scores, over the ones that are not None (None if all are None)."""

    def evaluate(scores: Scores) -> float | None:
        pairs = [(scores.get(name), weight) for name, weight in weights.items()]
        pairs = [(value, weight) for value, weight in pairs if value is not None]
        total = sum(weight for _, weight in pairs)
        if not pairs or total == 0:
            return None
        return sum(value * weight for value, weight in pairs) / total

    return evaluate


def gate(condition: str | Expr, expr: str | Expr) -> Expr:
    """0 when `condition` scores 0; otherwise `expr` (an unscorable condition passes)."""
    cond, inner = _expr(condition), _expr(expr)

    def evaluate(scores: Scores) -> float | None:
        value = cond(scores)
        if value is not None and value <= 0:
            return 0.0
        return inner(scores)

    return evaluate


def penalize(condition: str | Expr, factor: float, expr: str | Expr) -> Expr:
    """`expr * (1 - factor)` when `condition` scores below 1; otherwise `expr`."""
    cond, inner = _expr(condition), _expr(expr)

    def evaluate(scores: Scores) -> float | None:
        value = inner(scores)
        if value is None:
            return None
        flag = cond(scores)
        if factor and flag is not None and flag < 1:
            return value * (1.0 - factor)
        return value

    return evaluate


def bonus_if(
    bonus: str | float | Expr,
    expr: str | Expr,
    *,
    when: str | Expr | None = None,
    when_value_at_least: float | None = None,
) -> Expr:
    """`expr + bonus` when `when` scores 1 (or the value of `expr` is at least `when_value_at_least`)."""
    extra, inner = _expr(bonus), _expr(expr)
    cond = _expr(when) if when is not None else None
    if cond is None and when_value_at_least is None:
        raise ValueError("bonus_if needs `when` or `when_value_at_least`")

    def evaluate(scores: Scores) -> float | None:
        value = inner(scores)
        if value is None:
            return None
        if when_value_at_least is not None and value < when_value_at_least:
            return value
        if cond is not None:
            flag = cond(scores)
            if flag is None or flag < 1:
                return value
        add = extra(scores)
        return value + (add or 0.0)

    return evaluate


# --------------------------------------------------------------------------------------------------
# Rubric sets
# --------------------------------------------------------------------------------------------------


@dataclass
class RubricResult:
    scores: dict[str, float | None]
    reward: float | None
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    mode: Mode = "eval"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "reward": self.reward,
            "passed": self.passed,
            "metrics": self.metrics,
            "errors": self.errors,
            "mode": self.mode,
        }


class RubricSet:
    """The task's rubrics plus how their scores become a reward and a pass/fail verdict.

    `reward` defaults to the weighted mean of the non-metric rubrics; `passed` to `reward >= threshold`.
    """

    def __init__(
        self,
        rubrics: Iterable[Rubric],
        *,
        reward: Expr | None = None,
        passed: Callable[[Scores], bool] | None = None,
        threshold: float = 1.0,
    ) -> None:
        self.rubrics = list(rubrics)
        names = [rubric.name for rubric in self.rubrics]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate rubric names: {sorted(duplicates)}")
        self.threshold = threshold
        self._reward = reward or weighted({r.name: r.weight for r in self.rubrics if r.kind != "metric"})
        self._passed = passed

    def __iter__(self):
        return iter(self.rubrics)

    def __getitem__(self, name: str) -> Rubric:
        for rubric in self.rubrics:
            if rubric.name == name:
                return rubric
        raise KeyError(name)

    def reward_scores(self, scores: Scores, mode: Mode) -> dict[str, float | None]:
        """The scores `reward` sees in `mode`: inactive rubrics are replaced by their neutral value."""
        view = dict(scores)
        for rubric in self.rubrics:
            if not rubric.active(mode):
                view[rubric.name] = rubric.neutral
        return view

    def reward(self, scores: Scores, mode: Mode = "train") -> float | None:
        return self._reward(self.reward_scores(scores, mode))

    def passed(self, scores: Scores, mode: Mode = "eval") -> bool:
        view = self.reward_scores(scores, mode)
        if self._passed is not None:
            return bool(self._passed(view))
        value = self._reward(view)
        return value is not None and value >= self.threshold

    async def ascore(self, context: Mapping[str, Any], *, mode: Mode = "eval") -> RubricResult:
        """Run every rubric active in `mode` (inactive ones are skipped, e.g. eval-only judges during GRPO)."""
        active = [rubric for rubric in self.rubrics if rubric.active(mode) or rubric.kind != "judge"]
        outcomes = await asyncio.gather(*(_safe(rubric, context) for rubric in active))
        scores: dict[str, float | None] = {}
        metrics: dict[str, float] = {}
        errors: dict[str, str] = {}
        for rubric, (value, extra, error) in zip(active, outcomes, strict=True):
            scores[rubric.name] = value
            metrics.update(extra)
            if error:
                errors[rubric.name] = error
        return RubricResult(
            scores=scores,
            reward=self.reward(scores, mode),
            passed=self.passed(scores, mode),
            metrics=metrics,
            errors=errors,
            mode=mode,
        )

    def score(self, context: Mapping[str, Any], *, mode: Mode = "eval") -> RubricResult:
        """Synchronous `ascore` (for scripts and tests; inside an event loop use `ascore`)."""
        return asyncio.run(self.ascore(context, mode=mode))


async def _safe(rubric: Rubric, context: Mapping[str, Any]) -> tuple[float | None, dict[str, float], str | None]:
    try:
        value, metrics = await rubric.ascore(context)
        return value, metrics, None
    except Exception as exc:  # a broken rubric scores 0, loudly, instead of killing a run
        logger.warning("rubric %s failed: %s: %s", rubric.name, type(exc).__name__, exc)
        return 0.0, {}, f"{type(exc).__name__}: {exc}"


def episode_context(episode: Any, row: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """The injectable arguments for one episode."""
    return {
        "episode": episode,
        "row": row,
        "answer": episode.answer,
        "messages": episode.messages,
        "tool_events": episode.tool_events,
        **extra,
    }
