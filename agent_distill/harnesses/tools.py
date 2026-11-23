"""The tool definitions a harness sends to the model, for rendering SFT examples exactly as the student will see
them at inference time (DESIGN.md §6).

`tools/<harness>-<version>.json` holds the definitions captured from the harness (url-extraction captured both
OpenCode versions from a live server). Only the tools the task's tool kinds enable are kept; a structured-output
submit tool gets the task's answer schema.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent_distill.task import Task

TOOLS_DIR = Path(__file__).resolve().parent / "tools"
SUBMIT_TOOLS = {"structuredoutput", "submit"}


def tool_schemas(harness: str, version: str, task: Task, allowed: set[str]) -> list[dict[str, Any]]:
    """The captured definitions of `allowed` tools (native names), plus the submit tool with the answer schema."""
    path = TOOLS_DIR / f"{harness}-{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"no captured tool definitions for {harness}@{version} ({path})")
    tools = json.loads(path.read_text())
    result = []
    schema = task.answer_schema()
    for tool in tools:
        name = tool["function"]["name"]
        if name.lower() in SUBMIT_TOOLS:
            if schema is None:
                continue
            tool = copy.deepcopy(tool)
            tool["function"]["parameters"] = {k: v for k, v in schema.items() if k not in {"title", "description"}}
            result.append(tool)
        elif name in allowed:
            result.append(tool)
    return result


def harness_tools(harness_id: str, task: Task) -> list[dict[str, Any]]:
    """The tools a task's episodes in `harness_id` ("opencode@1.2.4") offer the model."""
    name, _, version = harness_id.partition("@")
    if name == "opencode":
        if version.startswith("1."):
            from agent_distill.harnesses.opencode_v1 import allowed_tools
        else:
            from agent_distill.harnesses.opencode_v2 import KIND_TOOLS

            def allowed_tools(kinds):  # type: ignore[no-redef]
                return {tool for kind in kinds for tool in KIND_TOOLS.get(kind, ())}

        return tool_schemas(name, version, task, allowed_tools(task.tools))
    raise NotImplementedError(f"no tool definitions for {harness_id}")
