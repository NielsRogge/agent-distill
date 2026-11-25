"""The tool definitions a harness sends to the model, for rendering SFT examples exactly as the student will see
them at inference time (DESIGN.md §6).

`tools/<harness>-<version>.json` holds the definitions captured from the harness: url-extraction captured both
OpenCode versions from a live server, and Pi's come from a Pi session's system message. Only the tools the task's
tool kinds enable are kept; a structured-output submit tool gets the task's answer schema. A `TextAnswer` task has no
submit tool (its final message is the answer, and Pi offers none). A task's MCP tools (`Task.mcp_servers`, Pi only)
come from each server's pinned snapshot, declared the way Pi's MCP extension declares a `direct` tool: the name
`mcp__<server>__<tool>`, the tool's description (else its title) and its input schema as an object schema.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from agent_distill.task import Task, TextAnswer

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
            if isinstance(task.answer, TextAnswer):  # the final message is the answer: no submit tool
                continue
            if schema is None:  # a file answer: Pi's argument-free submit as captured
                if not tool["function"]["parameters"].get("properties"):
                    result.append(tool)
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
    if name == "pi":
        from agent_distill.harnesses.pi import KIND_TOOLS as PI_TOOLS

        builtin = tool_schemas(name, version, task, {tool for kind in task.tools for tool in PI_TOOLS.get(kind, ())})
        return builtin + pi_mcp_tools(task)
    raise NotImplementedError(f"no tool definitions for {harness_id}")


def pi_mcp_name(server: str, tool: str) -> str:
    """Pi's name for an MCP tool: every character other than a letter, digit or `_` becomes `_`."""
    return "__".join(re.sub(r"[^A-Za-z0-9_]", "_", part) for part in ("mcp", server, tool))


def pi_mcp_tools(task: Task) -> list[dict[str, Any]]:
    """The allowed tools of the task's MCP servers, from their snapshots, as Pi 1.6.0 declares them to the model
    (`extensions/mcp/tools.js` `createMcpToolDefinition`, `pi-mcp` `normalizeMcpInputSchema`)."""
    result = []
    for server in getattr(task, "mcp_servers", ()) or ():
        snapshot = {tool["name"]: tool for tool in server.snapshot_data().get("tools", [])}
        missing = [tool for tool in server.tools if tool not in snapshot]
        if missing:
            raise ValueError(f"MCP server {server.name}: no snapshot of the allowed tools {missing}")
        for name in server.tools:
            tool = snapshot[name]
            schema = dict(tool.get("inputSchema") or {})
            schema["type"] = schema.get("type") or "object"
            schema.setdefault("properties", {})
            description = str(tool.get("description") or "").strip() or str(tool.get("title") or "")
            description = description or f"MCP tool {name} from server {server.name}"
            function = {"name": pi_mcp_name(server.name, name), "description": description, "parameters": schema}
            result.append({"type": "function", "function": function})
    return result
