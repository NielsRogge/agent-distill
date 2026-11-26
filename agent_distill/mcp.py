"""MCP servers a task's agent may use, pinned by a snapshot of their tool list.

A task names each server it needs (`Task.mcp_servers`) with the tools the agent may call. The server's `tools/list`
is frozen in a JSON snapshot next to the task (`McpServer.snapshot`), because remote servers change their tools over
time. Before any episode runs, the stage fetches the live list and compares its digest with the snapshot's
(`check`): on a mismatch it refuses to start, unless AGENT_DISTILL_ALLOW_MCP_DRIFT=1, which records both digests.

Only anonymous servers are supported: the configuration the agent's harness gets holds a URL and nothing else, so
no credential can reach the agent through it.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL = "2025-06-18"
ALLOW_DRIFT_ENV = "AGENT_DISTILL_ALLOW_MCP_DRIFT"


@dataclass(frozen=True)
class McpServer:
    name: str  # the harness-side server name; tools become e.g. `mcp__<name>__<tool>` in Pi
    url: str  # streamable HTTP endpoint, reached anonymously
    tools: tuple[str, ...]  # the allow-list: every other tool of the server stays hidden from the agent
    snapshot: Path | None = None  # frozen `tools/list` ({"tools": [...], "tools_sha256": ...})

    def snapshot_data(self) -> dict[str, Any]:
        return json.loads(self.snapshot.read_text()) if self.snapshot else {}

    def summary(self) -> dict[str, Any]:
        """What manifests, records and cards say about the server."""
        data = self.snapshot_data()
        return {"name": self.name, "url": self.url, "tools": list(self.tools),
                "server_version": data.get("server_version"), "tools_sha256": data.get("tools_sha256"),
                "snapshot_date": data.get("snapshot_date")}  # fmt: skip


class McpDrift(RuntimeError):
    pass


def tools_digest(tools: list[dict[str, Any]]) -> str:
    """sha256 of the tool list, sorted by name, as canonical JSON."""
    canonical = json.dumps(sorted(tools, key=lambda t: t["name"]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _post(url: str, body: dict[str, Any], session: str | None, timeout: float) -> tuple[dict[str, Any], str | None]:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session:
        headers["Mcp-Session-Id"] = session
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode()
        session = response.headers.get("Mcp-Session-Id") or session
    for line in text.splitlines():  # plain JSON, or server-sent events whose data lines hold the JSON
        line = line.removeprefix("data:").strip()
        if line.startswith("{"):
            message = json.loads(line)
            if message.get("id") == body.get("id"):
                if "error" in message:
                    raise RuntimeError(f"{url}: {message['error']}")
                return message.get("result") or {}, session
    raise RuntimeError(f"{url}: no reply to {body.get('method')}")


def fetch_tools(url: str, timeout: float = 30) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(tools, serverInfo) of a streamable-HTTP MCP server, anonymously."""
    init, session = _post(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "agent-distill", "version": "0"}}},
        None, timeout)  # fmt: skip
    listed, _ = _post(url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session, timeout)
    return list(listed.get("tools") or []), dict(init.get("serverInfo") or {})


def check(server: McpServer, timeout: float = 30) -> dict[str, Any]:
    """Compare the live tool list with the snapshot; the server's summary plus the live digest and version.

    Raises `McpDrift` on a mismatch (or when an allowed tool is missing) unless AGENT_DISTILL_ALLOW_MCP_DRIFT=1."""
    summary = server.summary()
    tools, info = fetch_tools(server.url, timeout)
    live = tools_digest(tools)
    summary.update(live_tools_sha256=live, live_server_version=info.get("version"), drift=False)
    missing = sorted(set(server.tools) - {t.get("name") for t in tools})
    if summary["tools_sha256"] and (live != summary["tools_sha256"] or missing):
        summary["drift"] = True
        if os.environ.get(ALLOW_DRIFT_ENV) != "1":
            raise McpDrift(
                f"MCP server {server.name} ({server.url}) changed since its snapshot {server.snapshot}: tools sha256 "
                f"{live} (server {info.get('version')}) vs {summary['tools_sha256']} "
                f"(server {summary['server_version']}){f', missing {missing}' if missing else ''}. "
                f"Refresh the snapshot, or set {ALLOW_DRIFT_ENV}=1 to run anyway (recorded)."
            )
    return summary
