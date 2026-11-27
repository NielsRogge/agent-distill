"""Open-ended training task for `hf-hub-qa`.

240 questions, disjoint from `judged.py` / `agent-distillation-hf/hf-hub-questions`.

Uses the exact same agent setup (`system_prompt.txt`, `bash` + `read`, pinned `hf` CLI 2.4.0, `hf-cli` skill) and
rubrics (`judged_rubrics.py`) as `judged.py` so accepted traces mix directly with `task.py` factoid traces in SFT.
"""

from __future__ import annotations

from agent_distill.task import DatasetSpec, Limits, Task, TextAnswer

from .judged_rubrics import RUBRICS
from .task import COMMANDS, HERE, JOB_SETUP, user_prompt

task = Task(
    name="hf-hub-qa-open-ended",
    dataset=DatasetSpec(repo="agent-distillation-hf/hf-hub-questions-train", splits={"train": "train", "eval": []}),
    system_prompt=HERE / "system_prompt.txt",
    prompt=user_prompt,
    answer=TextAnswer(),
    rubrics=RUBRICS,
    row_id=lambda row: str(row["question_id"]),
    tools=("shell", "read"),
    skills=(HERE / "skills" / "hf-cli",),
    network=True,
    limits=Limits(timeout=600, max_tool_calls=20),
    judge="gemini-4-hydrogen",
    job_setup=JOB_SETUP,
    commands=COMMANDS,
)

__all__ = ["task"]
