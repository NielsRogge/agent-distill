"""Follow-up 1 ablation: `judged.py` with an explicit strict-grounding instruction (`system_prompt_grounded.txt`).

Everything else (dataset `nielsr/hf-hub-questions`, tools, `hf-cli` skill, pinned `hf` CLI 2.4.0, limits, and
`judged_rubrics.py`) is identical to `judged.py` so this isolates how many 4B confabulation/grounding failures are
fixable purely at inference time vs. baked into SFT weights.
"""

from __future__ import annotations

from agent_distill.task import DatasetSpec, Limits, Task, TextAnswer

from .judged_rubrics import RUBRICS
from .task import COMMANDS, HERE, JOB_SETUP, user_prompt

_BASE_PROMPT = (HERE / "system_prompt.txt").read_text(encoding="utf-8")
_GROUNDING_ADDENDUM = (HERE / "system_prompt_grounded.txt").read_text(encoding="utf-8").strip()
_GROUNDED_PROMPT = _BASE_PROMPT.replace(
    "\nWhen you have what you need,",
    f"\n{_GROUNDING_ADDENDUM}\n\nWhen you have what you need,",
    1,
)

task = Task(
    name="hf-hub-qa-judged-grounded",
    dataset=DatasetSpec(repo="nielsr/hf-hub-questions", splits={"train": "seed", "eval": ["seed"]}),
    system_prompt=_GROUNDED_PROMPT,
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
