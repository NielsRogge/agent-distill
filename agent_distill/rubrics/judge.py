"""LLM / vision-judge rubrics (DESIGN.md §4).

A judge rubric's function returns the template variables for its prompt (and optional `images`, as PNG bytes);
the judge runner (passed to the rubric set as `judge` in the context) sends the rendered prompt to the judge
model and the score is parsed from the reply and scaled to [0, 1]. Verdicts are cached by (judge model, rendered
prompt, images); the rendered prompt contains the template, so editing a prompt invalidates old verdicts, and the
scale and pattern are applied to the cached reply each time, so editing them never returns a stale score. Only
replies that parse are cached. Trace filtering, GRPO, evaluation and rescoring never pay twice for a judgement.

Concrete clients (HF router, Gemini) are in `rubrics.clients`, selected by judge profiles; any object with
`async complete(prompt: str, images: list[bytes]) -> str` and a `model` attribute works.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent_distill.rubrics.core import Rubric


class JudgeClient(Protocol):
    model: str

    async def complete(self, prompt: str, images: list[bytes]) -> str: ...


@dataclass(frozen=True)
class JudgeSpec:
    prompt: str  # str.format template
    scale: tuple[float, float] = (0.0, 1.0)
    # First group is the score. The *last* match in the reply counts (a reply may discuss sub-scores first, e.g.
    # "Overall score: 6 ... SCORE: 9"), and Markdown emphasis around the label is allowed ("**SCORE:** 8").
    pattern: str = r"(?i)\bscore\s*\**\s*[:=]\s*\**\s*(-?\d+(?:\.\d+)?)"
    # Named sub-scores in the same reply ("creativity: 7"), recorded as metrics `<rubric>/<criterion>`, scaled
    # like the score.
    criteria: tuple[str, ...] = ()


class JudgeCache:
    """Append-only JSONL cache of verdicts, safe across threads of one process."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._data[entry["key"]] = entry["reply"]

    @staticmethod
    def key(model: str, prompt: str, images: list[bytes]) -> str:
        digest = hashlib.sha256()
        for part in (model.encode(), prompt.encode(), *(hashlib.sha256(image).digest() for image in images)):
            digest.update(len(part).to_bytes(8, "big") + part)
        return digest.hexdigest()

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, reply: str) -> None:
        with self._lock:
            self._data[key] = reply
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "reply": reply}) + "\n")


class JudgeRunner:
    """What judge rubrics receive as `judge`: a client plus the verdict cache."""

    def __init__(self, client: JudgeClient, cache: JudgeCache | None = None) -> None:
        self.client = client
        self.cache = cache or JudgeCache()
        self.calls = 0  # uncached calls (for the budget)

    async def __call__(
        self, prompt: str, images: list[bytes] | None = None, *, check: Callable[[str], Any] | None = None
    ) -> str:
        """The judge's reply. With `check` (e.g. the score parser), only replies it accepts are cached; a cached
        reply it rejects (written before a parser change) is asked again. Errors are never cached."""
        images = images or []
        key = JudgeCache.key(self.client.model, prompt, images)
        cached = self.cache.get(key)
        if cached is not None and _accepted(cached, check):
            return cached
        reply = await self.client.complete(prompt, images)
        self.calls += 1
        if _accepted(reply, check):
            self.cache.put(key, reply)
        return reply


def _accepted(reply: str, check: Callable[[str], Any] | None) -> bool:
    if not reply.strip():
        return False
    if check is None:
        return True
    try:
        check(reply)
    except ValueError:
        return False
    return True


def judge(
    prompt: str,
    *,
    scale: tuple[float, float] = (0.0, 1.0),
    pattern: str | None = None,
    criteria: tuple[str, ...] = (),
    weight: float = 1.0,
    name: str | None = None,
    train_only: bool = False,
    eval_only: bool = False,
) -> Callable[[Callable[..., Any]], Rubric]:
    """Declare a judge rubric. The function returns a mapping of template variables (plus optional `images`)."""
    spec = JudgeSpec(prompt=prompt, scale=scale, pattern=pattern or JudgeSpec.pattern, criteria=tuple(criteria))

    def wrap(function: Callable[..., Any]) -> Rubric:
        return Rubric(
            fn=function,
            name=name or function.__name__,
            kind="judge",
            weight=weight,
            train_only=train_only,
            eval_only=eval_only,
            judge_spec=spec,
        )

    return wrap


def parse_score(reply: str, spec: JudgeSpec) -> float:
    matches = list(re.finditer(spec.pattern, reply))
    match = matches[-1] if matches else None
    if not match:
        raise ValueError(f"no score in judge reply: {reply[:200]!r}")
    low, high = spec.scale
    value = float(match.group(1))
    if not low <= value <= high:
        raise ValueError(f"judge score {value} outside {spec.scale}")
    return (value - low) / (high - low) if high != low else 0.0


async def run_judge_rubric(rubric: Rubric, context: Mapping[str, Any]) -> tuple[float | None, dict[str, float]]:
    runner = context.get("judge")
    if runner is None:
        raise RuntimeError(f"judge rubric {rubric.name!r} needs a judge runner in the context")
    variables = rubric.fn(**rubric.arguments(context))
    if inspect.isawaitable(variables):
        variables = await variables
    if variables is None:  # the rubric declines (e.g. nothing to judge): unscorable
        return None, {}
    variables = dict(variables)
    images = list(variables.pop("images", []) or [])
    prompt = rubric.judge_spec.prompt.format(**variables)
    spec = rubric.judge_spec
    if isinstance(runner, JudgeRunner):
        reply = await runner(prompt, images, check=lambda text: parse_score(text, spec))
    else:  # any `async (prompt, images) -> str`
        reply = await runner(prompt, images)
    metrics = {f"{rubric.name}/{name}": value for name, value in parse_criteria(reply, spec).items()}
    return parse_score(reply, spec), metrics


def parse_criteria(reply: str, spec: JudgeSpec) -> dict[str, float]:
    low, high = spec.scale
    found: dict[str, float] = {}
    for name in spec.criteria:
        # The last mention wins (a reply may discuss a criterion before giving its final value), and markdown around
        # the name or the number ("**visual_quality:** 8") is allowed.
        matches = re.findall(rf"(?i)\b{re.escape(name)}\**\s*[:=]\s*\**\s*(-?\d+(?:\.\d+)?)", reply)
        if matches and low <= float(matches[-1]) <= high and high != low:
            found[name] = (float(matches[-1]) - low) / (high - low)
    return found


def make_judge_runner(
    name: str, *, cache_dir: str | Path | None = None, bill_to: str = "", profiles: list[Path] | None = None
) -> JudgeRunner | None:
    """A judge runner for a judge profile (`profiles/judges/<name>.toml`), caching verdicts in
    `<cache_dir>/judge-cache/<name>.jsonl`; None for an empty name."""
    if not name:
        return None
    from agent_distill.config import load_profile
    from agent_distill.rubrics.clients import judge_client

    client = judge_client(load_profile("judges", name, profiles), bill_to)
    cache = JudgeCache(Path(cache_dir) / "judge-cache" / f"{name}.jsonl" if cache_dir else None)
    return JudgeRunner(client, cache)
