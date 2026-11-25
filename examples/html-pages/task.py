"""Stunning HTML pages: one self-contained, creative, interactive page per concept (DESIGN.md §10).

Episodes run in Pi with text-only tools: the agent never sees the page; `./preview` (headless Chromium) reports
what is broken as text. Only the judge sees screenshots. The answer is the workspace's `index.html`, submitted
with Pi's `submit` tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_distill.browser.preview import write_tool
from agent_distill.task import DatasetSpec, FileAnswer, Limits, Task

from .rubrics import RUBRICS

HERE = Path(__file__).parent


def user_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Make a stunning single-file HTML page for this concept.",
            f"Title: {row['title']}",
            f"Idea: {row['idea']}",
            f"Category: {row['category']}",
        ]
    )


def setup(row: dict[str, Any], workspace: Path) -> None:
    write_tool(workspace)


task = Task(
    name="html-pages",
    dataset=DatasetSpec(repo="nielsr/html-concepts", splits={"train": "train", "eval": ["test", "validation"]}),
    system_prompt=HERE / "system_prompt.txt",
    prompt=user_prompt,
    answer=FileAnswer("index.html"),
    rubrics=RUBRICS,
    row_id=lambda row: str(row["concept_id"]),
    tools=("shell", "read", "edit", "write"),
    setup=setup,
    network=False,
    limits=Limits(timeout=900, max_tool_calls=30),
    judge="gemini-4-hydrogen",  # the user's choice (2026-11-23); Qwen3.8-27B is the second judge
    extras=("browser",),
    # Chromium for `preview` and the gates, at the version the pinned Playwright drives, outside root's home: the
    # agents that run `preview` are other users (agent_distill.harnesses.isolation).
    job_setup=[
        "export PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright",
        "uvx --from playwright==1.65.2 playwright install --with-deps chromium >/dev/null",
    ],
)

__all__ = ["task", "user_prompt"]
