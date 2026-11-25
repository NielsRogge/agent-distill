"""Hugging Face Hub questions: an agent answers questions about the Hub with the `hf` CLI (DESIGN.md §10).

Episodes run in Pi with `bash` and `read`, the pinned `hf` CLI (huggingface_hub 2.4.0, no token: read-only) and the
`hf-cli` Agent Skill generated from that same version (`skills/hf-cli/SKILL.md`). There is no submit tool: the
agent's final message, in natural language, is the answer (`TextAnswer`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_distill.task import DatasetSpec, Limits, Task, TextAnswer

from .rubrics import RUBRICS

HERE = Path(__file__).parent
HF_VERSION = "2.4.0"  # the CLI the agent runs and the skill was generated from (`hf skills preview`)


def user_prompt(row: dict[str, Any]) -> str:
    return str(row["question"])


task = Task(
    name="hf-hub-qa",
    dataset=DatasetSpec(repo="nielsr/hf-hub-questions", splits={"train": "seed", "eval": ["seed"]}),
    system_prompt=HERE / "system_prompt.txt",
    prompt=user_prompt,
    answer=TextAnswer(),
    rubrics=RUBRICS,
    row_id=lambda row: str(row["question_id"]),
    tools=("shell", "read"),
    skills=(HERE / "skills" / "hf-cli",),
    network=True,  # the Hub API, anonymously
    limits=Limits(timeout=600, max_tool_calls=20),
    judge="gemini-4-hydrogen",
    # The pinned CLI, outside root's home so isolated agents can run it, on the PATH agents inherit.
    job_setup=[
        "export UV_TOOL_DIR=/opt/hf-cli/tools UV_TOOL_BIN_DIR=/opt/hf-cli/bin",
        f"uv tool install -q --python 3.12 --python-preference only-managed huggingface_hub=={HF_VERSION}",
        'export PATH="/opt/hf-cli/bin:$PATH" HF_HUB_DISABLE_UPDATE_CHECK=1',
    ],
)

__all__ = ["task", "user_prompt"]
