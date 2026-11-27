"""Judged open-ended evaluation for the MCP twin (`nielsr/hf-hub-questions`, 51 seed questions).

Uses the exact same anonymous 3-tool HF MCP setup as `examples/hf-hub-qa-mcp/task.py` (`hub_repo_search`,
`hub_repo_details`, `hf_fs`; no shell or CLI access) and scores with `judged_rubrics.py` (parity with CLI `judged.py`).
"""

from __future__ import annotations

from agent_distill.task import DatasetSpec, Limits, Task, TextAnswer

from .judged_rubrics import RUBRICS, SCORER_JOB_SETUP
from .task import HERE, HF_MCP, user_prompt

task = Task(
    name="hf-hub-qa-mcp-judged",
    dataset=DatasetSpec(repo="nielsr/hf-hub-questions", splits={"train": "seed", "eval": ["seed"]}),
    system_prompt=HERE / "system_prompt.txt",
    prompt=user_prompt,
    answer=TextAnswer(),
    rubrics=RUBRICS,
    row_id=lambda row: str(row["question_id"]),
    tools=(),
    mcp_servers=(HF_MCP,),
    network=True,
    limits=Limits(timeout=600, max_tool_calls=20),
    judge="gemini-4-hydrogen",
    job_setup=SCORER_JOB_SETUP,
)

__all__ = ["task"]
