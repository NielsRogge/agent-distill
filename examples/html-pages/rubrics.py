"""Rubrics of the HTML-pages task (DESIGN.md §10).

- `page_ok` (gate): the "zero broken files" checks from the same `preview` inspection the agent runs: the page
  loads, has no console errors or uncaught exceptions at load or during the scripted interaction, is not blank,
  makes no network requests, and does not overflow horizontally at 390 px or 1440 px.
- `page_quality` (judge): a vision judge sees the page in motion: an 8-frame filmstrip of a scripted ~10 s session
  (wait, click, drag, controls, keys, scroll), the phone view, and an activity log of what the page did (sound,
  animation, input listeners, controls). It scores visual quality, creativity, fit to the concept and interactivity,
  1-10, plus an overall score. A broken page is not judged (the gate already scores it 0). Against human pair
  labels this view disagreed least with people (README); `page_quality_stills` is the earlier three-screenshot
  judge, kept for comparison.
- Metrics, recorded only: page size, basic accessibility, and whether the agent previewed before submitting.

reward = gate(page_ok, page_quality); an episode passes (trace acceptance, evaluation pass rate) when the page is
not broken and the judge's overall score is at least 7/10.
"""

from __future__ import annotations

import re
from typing import Any

from agent_distill.browser.observe import observe_html
from agent_distill.browser.preview import inspect_html
from agent_distill.rubrics import RubricSet, gate, metric, process, programmatic
from agent_distill.rubrics.judge import judge

PASS_SCORE = 7  # of 10
SCALE = (1.0, 10.0)
CRITERIA = ("visual_quality", "creativity", "concept_fit", "interactivity")

STILLS_PROMPT = """You are an exacting art director judging a single-file web page made for this concept:

Title: {title}
Idea: {idea}
Category: {category}

Image 1: the page on a desktop (1440x900) right after loading.
Image 2: the same page after a few seconds of clicks, key presses and mouse movement.
Image 3: the page on a phone (390 px wide).
Measured: animates on its own: {animated}; changes when interacted with: {interactive}.

Score each criterion from 1 (poor) to 10 (exceptional), calibrated so that 5 is a competent but forgettable page
and 9-10 is something people would share:
- visual_quality: composition, typography, colour, polish, nothing broken or clipped; the phone view counts too.
- creativity: a bold, specific, non-generic idea, not a template.
- concept_fit: how well the page realises this concept.
- interactivity: does interaction or motion add something meaningful (judge from the measurements and images)?

Answer with one line per criterion, then the overall score, exactly in this form:
visual_quality: <1-10>
creativity: <1-10>
concept_fit: <1-10>
interactivity: <1-10>
SCORE: <1-10>"""


@programmatic(weight=0.0)
async def page_ok(answer: Any) -> tuple[float, dict[str, float]] | float:
    if not answer:
        return 0.0
    report = await inspect_html(str(answer))
    gates = report.gates()
    return float(all(gates.values())), {name: float(ok) for name, ok in gates.items()}


JUDGE_PROMPT = """You are an exacting art director judging a single-file web page made for this concept:

Title: {title}
Idea: {idea}
Category: {category}

The page was played in a browser for about 10 seconds by a scripted visitor: it waited, clicked the middle, dragged
across the page, used up to four controls, pressed keys and scrolled. Image 1 is a filmstrip of 8 desktop frames
(1440x900) in time order, each labelled with what had just happened. Image 2 is the page on a phone (390 px wide).
The activity log says what the page did, including sound, which images cannot show:
{log}

Score each criterion from 1 (poor) to 10 (exceptional), calibrated so that 5 is a competent but forgettable page
and 9-10 is something people would share:
- visual_quality: composition, typography, colour, polish, nothing broken or clipped; the phone view counts too.
- creativity: a bold, specific, non-generic idea, not a template.
- concept_fit: how well the page realises this concept.
- interactivity: how much motion, the response to the visitor and sound add. A page that looks the same in every
  frame and ignores the visitor is static, whatever its first frame looks like.

Answer with one line per criterion, then the overall score, exactly in this form:
visual_quality: <1-10>
creativity: <1-10>
concept_fit: <1-10>
interactivity: <1-10>
SCORE: <1-10>"""


@judge(JUDGE_PROMPT, scale=SCALE, criteria=CRITERIA)
async def page_quality(answer: Any, row: dict[str, Any]) -> dict[str, Any] | None:
    if not answer:
        return None
    report = await inspect_html(str(answer))
    if not all(report.gates().values()):
        return None  # broken: the gate scores it, no judge spend
    seen = await observe_html(str(answer))
    return {
        "title": row.get("title", ""),
        "idea": row.get("idea", ""),
        "category": row.get("category", ""),
        "log": seen.log,
        "images": [seen.filmstrip, seen.phone],
    }


@judge(STILLS_PROMPT, scale=SCALE, criteria=CRITERIA)
async def page_quality_stills(answer: Any, row: dict[str, Any]) -> dict[str, Any] | None:
    """The earlier judge view (three screenshots), kept for comparison; not part of RUBRICS."""
    if not answer:
        return None
    report = await inspect_html(str(answer))
    if not all(report.gates().values()):
        return None  # broken: the gate scores it, no judge spend
    shots = report.screenshots
    return {
        "title": row.get("title", ""),
        "idea": row.get("idea", ""),
        "category": row.get("category", ""),
        "animated": "yes" if report.animated else "no",
        "interactive": "yes" if report.changed_by_interaction else "no",
        "images": [shots["desktop"], shots["desktop_after"], shots["phone"]],
    }


@metric
def size_kb(answer: Any) -> float | None:
    return round(len(str(answer).encode()) / 1024, 1) if answer else None


@metric
def accessibility_basics(answer: Any) -> float | None:
    """The share of three basics: a `lang` attribute, a non-empty `<title>`, `alt` on every `<img>`."""
    if not answer:
        return None
    html = str(answer)
    images = re.findall(r"<img\b[^>]*>", html, re.I)
    checks = [
        bool(re.search(r"<html\b[^>]*\blang=", html, re.I)),
        bool(re.search(r"<title>\s*\S", html, re.I)),
        all(re.search(r"\balt=", tag, re.I) for tag in images),
    ]
    return sum(checks) / len(checks)


@process(weight=0.0)
def previewed(tool_events: list[Any]) -> float:
    """Whether the agent ran `./preview` before submitting (recorded, not rewarded)."""
    for event in tool_events:
        if event.kind == "submit":
            return 0.0
        if event.kind == "shell" and "preview" in str((event.input or {}).get("command", "")):
            return 1.0
    return 0.0


def passed(scores: dict[str, float | None]) -> bool:
    quality = scores.get("page_quality")
    threshold = (PASS_SCORE - SCALE[0]) / (SCALE[1] - SCALE[0])
    return scores.get("page_ok") == 1.0 and quality is not None and quality >= threshold - 1e-9


RUBRICS = RubricSet(
    [page_ok, page_quality, size_kb, accessibility_basics, previewed],
    reward=gate("page_ok", "page_quality"),
    passed=passed,
)
