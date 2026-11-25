"""Hugging Face Hub facts through the Hugging Face MCP server: the MCP twin of hf-hub-qa (README.md).

Episodes run in Pi 1.6.0 with no built-in tools: only three tools of the anonymous HF MCP server
(`hub_repo_search`, `hub_repo_details`, `hf_fs`), pinned by the snapshot in mcp-tools.json. There is no submit tool:
the agent's final message, in natural language, is the answer (`TextAnswer`), checked against a frozen gold value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_distill.mcp import McpServer
from agent_distill.task import DatasetSpec, Limits, Task, TextAnswer

from .rubrics import RUBRICS, SERVER, TOOLS

HERE = Path(__file__).parent
HF_MCP = McpServer(name=SERVER, url="https://huggingface.co/mcp", tools=TOOLS, snapshot=HERE / "mcp-tools.json")


def user_prompt(row: dict[str, Any]) -> str:
    return str(row["question"])


task = Task(
    name="hf-hub-qa-mcp",
    # v2: multi-hop and count questions (questions_v2.py). v1 (nielsr/hf-hub-facts, single lookups) was too easy:
    # base Qwen3.5-4B passed 96% of its validation rows (README).
    dataset=DatasetSpec(repo="nielsr/hf-hub-facts-v2", splits={"train": "train", "eval": ["test", "validation"]}),
    system_prompt=HERE / "system_prompt.txt",
    prompt=user_prompt,
    answer=TextAnswer(),
    rubrics=RUBRICS,
    row_id=lambda row: str(row["question_id"]),
    tools=(),  # no built-in tools: no shell, file or CLI fallback
    mcp_servers=(HF_MCP,),
    network=True,  # the HF MCP server, anonymously
    limits=Limits(timeout=300, max_tool_calls=15),
    version="2",
)

__all__ = ["task", "user_prompt"]
