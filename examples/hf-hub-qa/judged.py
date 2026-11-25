"""The judged hf-hub-qa questions: open-ended questions about the Hub (questions.py), answered with the `hf` CLI.

The secondary evaluation of both twins: the same agent setup as task.py, on `nielsr/hf-hub-questions`, scored by
grounding, a live reference where one exists, and a judge (judged_rubrics.py). Run it with `task = "judged.py"`.
"""

from __future__ import annotations

from agent_distill.task import DatasetSpec, Limits, Task, TextAnswer

from .judged_rubrics import RUBRICS
from .task import COMMANDS, HERE, JOB_SETUP, user_prompt

task = Task(
    name="hf-hub-qa-judged",
    dataset=DatasetSpec(repo="nielsr/hf-hub-questions", splits={"train": "seed", "eval": ["seed"]}),
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
