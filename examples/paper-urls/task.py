"""Paper URL extraction: find an arXiv paper's official GitHub repository and project page.

Ported from url-extraction's `harness/task.py` (commit 9fb9a39): the same system prompt, user prompt and strict
answer schema, so episodes are comparable with every published url-extraction result. The rubrics are in
`rubrics.py`, the answer schema in `schema.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_distill.task import DatasetSpec, Limits, Task

from .rubrics import FIELDS, RUBRICS, fully_scoreable
from .schema import PaperCandidate, URLFindings, candidate_from_row

HERE = Path(__file__).parent
SYSTEM_PROMPT = (HERE / "system_prompt.txt").read_text().strip()


def user_prompt(row: dict[str, Any]) -> str:
    candidate = candidate_from_row(row)
    requested: list[str] = []
    if candidate.missing_github:
        requested.append("GitHub repository")
    if candidate.missing_project_page:
        requested.append("project page")
    return "\n".join(
        [
            f"Paper ID: {candidate.arxiv_id}",
            f"Title: {candidate.title}",
            f"Abstract: {candidate.abstract}",
            f"Find only: {', '.join(requested)}",
            f"Known GitHub URL: {candidate.known_github_url or 'none'}",
            f"Known project page URL: {candidate.known_project_page_url or 'none'}",
        ]
    )


def record_limits(name: str) -> Limits:
    """url-extraction's teacher trace generation (`train*`, `distill*` files) ran with 20 tool calls, its
    evaluations and baselines with 15."""
    stem = name.rsplit("/", 1)[-1]
    return Limits(timeout=180, max_tool_calls=20 if stem.startswith(("train", "distill")) else 15)


task = Task(
    name="paper-urls",
    dataset=DatasetSpec(
        repo="nielsr/paper-url-extraction-v1",
        splits={"train": "train", "eval": ["test", "validation"]},
        filter=fully_scoreable,
        # As parity/tier1.py: v1 labels first, later datasets only for papers v1 lacks (test_hard in v2).
        label_repos=(
            "nielsr/paper-url-extraction-v1",
            "nielsr/paper-url-extraction-v1-validated",
            "nielsr/paper-url-extraction-v2",
        ),
    ),
    system_prompt=SYSTEM_PROMPT,
    prompt=user_prompt,
    answer=URLFindings,
    rubrics=RUBRICS,
    row_id=lambda row: str(row["paper_id"]),
    tools=("shell", "fetch", "read", "search"),
    network=True,
    limits=Limits(timeout=180, max_tool_calls=15),
    # OpenCode 2.x has no structured output: its prompt carries the JSON-only answer contract (url-extraction's
    # harness/system_prompt_v2.txt from the OpenCode v2 experiment, "Shell" instead of "Bash").
    system_prompts={"opencode@2": HERE / "system_prompt_v2.txt"},
    # The agent runs `hf papers info/read` (anonymous: agents get no HF token).
    job_setup=["command -v hf >/dev/null || uv tool install -q 'huggingface-hub>=1.13'"],
    record_limits=record_limits,
)

__all__ = ["FIELDS", "SYSTEM_PROMPT", "PaperCandidate", "URLFindings", "candidate_from_row", "task", "user_prompt"]
