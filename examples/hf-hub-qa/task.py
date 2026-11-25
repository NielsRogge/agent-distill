"""Hugging Face Hub facts with the `hf` CLI: the CLI twin of hf-hub-qa-mcp (README.md).

Episodes run in Pi with `bash` and `read`, the pinned `hf` CLI (huggingface_hub 2.4.0, no token: read-only) and the
`hf-cli` Agent Skill generated from that same version (`skills/hf-cli/SKILL.md`). There is no submit tool: the
agent's final message, in natural language, is the answer (`TextAnswer`), checked against a frozen gold value.

Both twins train and evaluate on `nielsr/hf-hub-facts-v2` with the same deterministic checker, so only the tool surface
differs. The open-ended judged questions are a second task in this folder (judged.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_distill.task import DatasetSpec, Limits, Task, TextAnswer

from .rubrics import RUBRICS

HERE = Path(__file__).parent
HF_VERSION = "2.4.0"  # the CLI the agent runs and the skill was generated from (`hf skills preview`)
# Shell tools the agents reach for (the teacher pipes `curl` into `jq`). Job images differ: the teacher's uv image has
# curl but no jq, the student's huggingface/trl image has none of them, so install all three everywhere. Otherwise SFT
# imitates teacher commands that fail in the student's image, and every student number is understated (README).
SHELL_TOOLS = ("curl", "jq", "wget")
# The pinned CLI, outside root's home so isolated agents can run it, on the PATH agents inherit.
JOB_SETUP = [
    " && ".join(f"command -v {tool} >/dev/null" for tool in SHELL_TOOLS)
    + " || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
    + " ".join(SHELL_TOOLS)
    + " >/dev/null)",
    "export UV_TOOL_DIR=/opt/hf-cli/tools UV_TOOL_BIN_DIR=/opt/hf-cli/bin",
    f"uv tool install -q --python 3.12 --python-preference only-managed huggingface_hub=={HF_VERSION}",
    'export PATH="/opt/hf-cli/bin:$PATH" HF_HUB_DISABLE_UPDATE_CHECK=1',
]


def user_prompt(row: dict[str, Any]) -> str:
    return str(row["question"])


task = Task(
    name="hf-hub-qa",
    # v2: multi-hop and count questions, shared with the MCP twin. v1 (nielsr/hf-hub-facts, single lookups) was too
    # easy: base Qwen3.5-4B passed 96% of its validation rows through MCP (README).
    dataset=DatasetSpec(repo="nielsr/hf-hub-facts-v2", splits={"train": "train", "eval": ["test", "validation"]}),
    version="2",
    system_prompt=HERE / "system_prompt.txt",
    prompt=user_prompt,
    answer=TextAnswer(),
    rubrics=RUBRICS,
    row_id=lambda row: str(row["question_id"]),
    tools=("shell", "read"),
    skills=(HERE / "skills" / "hf-cli",),
    network=True,  # the Hub API, anonymously
    limits=Limits(timeout=300, max_tool_calls=15),  # the MCP twin's limits
    job_setup=JOB_SETUP,
)

__all__ = ["JOB_SETUP", "SHELL_TOOLS", "task", "user_prompt"]
