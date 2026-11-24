"""Rubrics of the paper-URL task, ported from url-extraction's `harness/scoring.py` (commit 9fb9a39).

The reward reproduces url-extraction's GRPO reward exactly:

    reward = bonus_if(tool_efficiency, penalize(protocol, 0.5, gate(no_shortcut, url_correctness)), when reward >= 1)

and `passed` is url-extraction's success criterion (every verified field correct and the episode complete), used
for trace acceptance and the difficulty probe. `no_shortcut` is `train_only`: evaluation scores the answer, RL
refuses to reward looking the labels up. The research-protocol check matches normalized tool kinds, so the same
rubric scores OpenCode 1.2.4, OpenCode 2.x and Pi episodes.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from agent_distill.episode import ToolEvent
from agent_distill.rubrics import RubricSet, bonus_if, gate, metric, penalize, process, programmatic

from .schema import PaperCandidate, candidate_from_row

FIELDS = ("github", "project_page")
# Label qualities whose reference (a URL or a verified null) is trusted enough to score against.
SCOREABLE_QUALITIES = frozenset({"gold", "silver", "verified_null"})
PROTOCOL_WEIGHT = 0.5
W_TOOL_EFFICIENCY = 0.1
TOOL_BUDGET = 15.0


# --------------------------------------------------------------------------------------------------
# Rows and labels
# --------------------------------------------------------------------------------------------------


def scoreable_fields(row: dict[str, Any]) -> list[str]:
    """Requested fields (`missing_*`) whose label is verified."""
    return [
        name
        for name in FIELDS
        if bool(row[f"missing_{name}"]) and row.get(f"{name}_label_quality") in SCOREABLE_QUALITIES
    ]


def fully_scoreable(row: dict[str, Any]) -> bool:
    """Every requested field has a verified label (the evaluation population)."""
    requested = [name for name in FIELDS if bool(row[f"missing_{name}"])]
    return bool(requested) and len(scoreable_fields(row)) == len(requested)


def partly_scoreable(row: dict[str, Any]) -> bool:
    """At least one requested field has a verified label; unverified fields are never scored."""
    return bool(scoreable_fields(row))


# --------------------------------------------------------------------------------------------------
# URL verification
# --------------------------------------------------------------------------------------------------


def canonical_url(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = urlparse(text if "://" in text else f"https://{text}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    host = parsed.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if host == "github.com":
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            path = "/" + "/".join(parts[:2])
    return urlunparse(("https", host, path.lower(), "", "", ""))


def verify_fields(row: dict[str, Any], answer: Any | None) -> dict[str, bool | None]:
    """Per requested field: correct, wrong, or None (not scored). With no answer every verified field is wrong."""
    fields: dict[str, bool | None] = {name: None for name in FIELDS}
    for name in scoreable_fields(row):
        if answer is None:
            fields[name] = False
            continue
        predicted = canonical_url(getattr(answer, f"{name}_url"))
        accepted = {canonical_url(row.get(f"reference_{name}_url"))}
        accepted |= {canonical_url(alias) for alias in row.get(f"reference_{name}_aliases") or []}
        fields[name] = predicted in accepted
    return fields


# --------------------------------------------------------------------------------------------------
# Research protocol (judged from normalized tool events)
# --------------------------------------------------------------------------------------------------

RESEARCH_HOSTS = {"api.github.com", "github.com", "huggingface.co", "arxiv.org", "google.com", "paperswithcode.com"}


def _successful_shell(events: list[ToolEvent], fragment: str) -> bool:
    fragment = fragment.lower()
    return any(event.success and event.kind == "shell" and fragment in event.command.lower() for event in events)


def _masks_hf_exit_status(event: ToolEvent) -> bool:
    if event.kind != "shell":
        return False
    command = event.command.strip().lower()
    return command.startswith(("hf papers info ", "hf papers read ")) and any(op in command for op in ("|", ";", "&&"))


def _is_external_page_fetch(event: ToolEvent) -> bool:
    if event.kind == "fetch":
        url = event.url
    elif event.kind == "shell":
        if "curl" not in event.command.lower():
            return False
        url = event.url
    else:
        return False
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return bool(host) and host not in RESEARCH_HOSTS


def _successful_arxiv_fetch(events: list[ToolEvent], arxiv_id: str, paths: tuple[str, ...]) -> bool:
    for event in events:
        url = event.url.lower()
        if (
            event.success
            and event.kind == "fetch"
            and "arxiv.org" in url
            and arxiv_id.lower() in url
            and any(path in url for path in paths)
        ):
            return True
    return False


def _successful_page_fetch(events: list[ToolEvent], url: str) -> bool:
    wanted = url.rstrip("/").lower()
    for event in events:
        if not event.success:
            continue
        if event.kind == "fetch" and event.url.rstrip("/").lower() == wanted:
            return True
        if event.kind == "shell" and "curl" in event.command.lower() and wanted in event.command.lower():
            return True
    return False


def protocol_violation(candidate: PaperCandidate, answer: Any, events: list[ToolEvent]) -> str | None:
    """Why the episode did not follow the research protocol of the system prompt (None if it did)."""
    github_discovered = False
    for event in events:
        if _masks_hf_exit_status(event):
            return "an hf papers command masked its exit status"
        if github_discovered and _is_external_page_fetch(event):
            return "project-page content was fetched after GitHub discovery"
        if event.success and "github.com/" in str(event.output or "").lower():
            github_discovered = True

    metadata_checked = _successful_shell(events, f"hf papers info {candidate.arxiv_id}") or _successful_arxiv_fetch(
        events, candidate.arxiv_id, ("/abs/",)
    )
    if not metadata_checked:
        return "paper metadata was not successfully inspected"

    unresolved = (candidate.missing_github and answer.github_url is None) or (
        candidate.missing_project_page and answer.project_page_url is None
    )
    paper_checked = _successful_shell(events, f"hf papers read {candidate.arxiv_id}") or _successful_arxiv_fetch(
        events, candidate.arxiv_id, ("/html/", "/pdf/")
    )
    if unresolved and not paper_checked:
        return "paper content was not successfully inspected"

    if candidate.missing_github and answer.github_url is None:
        project_url = answer.project_page_url or candidate.known_project_page_url
        if project_url and not _successful_page_fetch(events, str(project_url)):
            return "project page was not inspected for a GitHub link"
    return None


def incomplete_reason(
    row: dict[str, Any], answer: Any, events: list[ToolEvent], max_tool_calls: int | None
) -> str | None:
    """url-extraction's `incomplete_reason`: no answer, a protocol violation, or the tool-call limit."""
    reason = (
        "no valid structured output" if answer is None else protocol_violation(candidate_from_row(row), answer, events)
    )
    if max_tool_calls is not None and len(events) >= max_tool_calls:
        reason = f"tool-call limit reached ({max_tool_calls})"
    return reason


# Rule-based shortcut guard: the labels live in a public Hub dataset, so a policy rewarded for correct URLs could
# learn to look them up instead of researching. Matched against what the agent writes, never against tool outputs.
SHORTCUT_PATTERNS = (
    re.compile(r"paper-url-extraction|pwc-url-extraction|\bnielsr/|community-science-llm-traces", re.IGNORECASE),
    re.compile(r"\bhf\s+download\b|load_dataset|snapshot_download|hf_hub_download"),
    re.compile(r"\bHF_TOKEN\b|\bhf\s+auth\b|huggingface-cli\s+login|/\.cache/huggingface/token"),
)


def shortcut_violation(events: list[ToolEvent]) -> str | None:
    for event in events:
        authored = json.dumps(event.input or {})
        for pattern in SHORTCUT_PATTERNS:
            if pattern.search(authored):
                return f"{event.name}: {pattern.pattern[:40]}"
    return None


# --------------------------------------------------------------------------------------------------
# Rubrics
# --------------------------------------------------------------------------------------------------


@programmatic
def url_correctness(row, answer):
    """Fraction of verified requested fields that are correct (None: nothing verifiable in this row)."""
    fields = verify_fields(row, answer)
    scored = {name: value for name, value in fields.items() if value is not None}
    if not scored:
        return None
    metrics = {f"{name}_correct": float(value) for name, value in scored.items()}
    metrics |= {f"{name}_scored": 1.0 for name in scored}
    return sum(scored.values()) / len(scored), metrics


@metric
def all_fields_correct(row, answer):
    """url-extraction's `all_scoreable_correct`."""
    scored = [value for value in verify_fields(row, answer).values() if value is not None]
    return answer is not None and bool(scored) and all(scored)


@process
def protocol(row, answer, tool_events, limits=None):
    """1 when the episode answered, followed the research protocol and stayed under the tool-call limit."""
    max_tool_calls = getattr(limits, "max_tool_calls", None)
    return incomplete_reason(row, answer, tool_events, max_tool_calls) is None


@programmatic(train_only=True, weight=0.0)
def no_shortcut(tool_events):
    """0 when the agent tried to look the labels up on the Hub (RL only)."""
    return shortcut_violation(tool_events) is None


@metric
def tool_efficiency(tool_events):
    """`0.1 * (1 - tool_calls / 15)`: the bonus a fully correct, complete episode earns."""
    return W_TOOL_EFFICIENCY * max(0.0, min(1.0, 1.0 - len(tool_events) / TOOL_BUDGET))


REWARD = bonus_if(
    "tool_efficiency",
    penalize("protocol", PROTOCOL_WEIGHT, gate("no_shortcut", "url_correctness")),
    when_value_at_least=1.0,
)

RUBRICS = RubricSet(
    [url_correctness, all_fields_correct, protocol, no_shortcut, tool_efficiency],
    reward=REWARD,
    passed=lambda scores: scores.get("all_fields_correct") == 1.0 and scores.get("protocol") == 1.0,
)
