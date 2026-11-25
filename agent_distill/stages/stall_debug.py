"""Diagnostics for agent commands that hang inside the harness (`--debug-stalls` on the eval stage).

Debug-only: on HF Jobs, the GRPO model's first `hf papers info ... --format json` sometimes never returns (the
OpenCode bash tool stays "running" past its own `timeout`), which never happens locally. This module prints, to the
job log:

- once at start: which `hf` the agent's shell finds, its version and interpreter, DNS for huggingface.co, and the
  latency of one anonymous Hub API call;
- every `interval` seconds: every process in the harness's process tree (descendants of an `opencode` process) that
  has run longer than `min_age` seconds, with its ancestors, state, kernel wait channel, open file descriptors and the
  TCP connections behind its sockets, and whether its children are alive.

It reads /proc only and never touches the agents' processes.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

TCP_STATES = {"01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1", "05": "FIN_WAIT2",
              "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT", "09": "LAST_ACK", "0A": "LISTEN",
              "0B": "CLOSING"}  # fmt: skip


def log(message: str) -> None:
    print(f"[stall-debug] {message}", flush=True)


def _run(command: list[str], timeout: float = 20) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return (result.stdout + result.stderr).strip()[:600]
    except Exception as exc:  # diagnostics must never fail the stage
        return f"{type(exc).__name__}: {exc}"


def environment_report() -> None:
    log(f"PATH={os.environ.get('PATH', '')}")
    log(f"which -a hf: {_run(['bash', '-lc', 'which -a hf'])!r}")
    hf = shutil.which("hf")
    if hf:
        log(f"hf version: {_run([hf, 'version'])!r}")
        try:
            log(f"hf shebang: {Path(hf).read_text(errors='replace').splitlines()[0][:200]!r}")
        except Exception as exc:
            log(f"hf shebang unreadable: {exc}")
    try:
        log(f"DNS huggingface.co: {sorted({a[4][0] for a in socket.getaddrinfo('huggingface.co', 443)})}")
    except Exception as exc:
        log(f"DNS huggingface.co failed: {exc}")
    for proxy in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "NO_PROXY"):
        if os.environ.get(proxy):
            log(f"{proxy} is set")
    if hf:
        started = time.monotonic()
        out = _run(["bash", "-c", f"env -u HF_TOKEN {hf} papers info 2508.09968 --format json | head -c 80"], 120)
        log(f"anonymous `hf papers info` from the stage: {time.monotonic() - started:.1f}s {out!r}")


def _stat(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    head, _, tail = raw.rpartition(")")
    return [head.split(" (", 1)[0], head.split(" (", 1)[-1], *tail.split()]


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:300]
    except OSError:
        return "?"


def _sockets() -> dict[str, str]:
    """socket inode -> "local -> remote STATE" for every TCP connection in this network namespace."""
    found: dict[str, str] = {}
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) > 9:
                found[parts[9]] = f"{_addr(parts[1])} -> {_addr(parts[2])} {TCP_STATES.get(parts[3], parts[3])}"
    return found


def _addr(value: str) -> str:
    host, port = value.split(":")
    if len(host) == 8:
        octets = [str(int(host[i : i + 2], 16)) for i in (6, 4, 2, 0)]
        return f"{'.'.join(octets)}:{int(port, 16)}"
    return f"[v6 {host[-8:]}]:{int(port, 16)}"


def describe(pid: int, age: float, parents: dict[int, int], sockets: dict[str, str]) -> None:
    stat = _stat(pid) or []
    state = stat[2] if len(stat) > 2 else "?"
    try:
        wchan = Path(f"/proc/{pid}/wchan").read_text().strip()
    except OSError:
        wchan = "?"
    chain, cursor = [], parents.get(pid)
    while cursor and len(chain) < 6:
        chain.append(f"{cursor}:{_cmdline(cursor)[:80]}")
        cursor = parents.get(cursor)
    fds = []
    try:
        for fd in sorted(os.listdir(f"/proc/{pid}/fd"), key=int):
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if target.startswith("socket:["):
                target += " " + sockets.get(target[8:-1], "(not TCP)")
            fds.append(f"{fd}={target}")
    except OSError as exc:
        fds.append(f"unreadable: {exc}")
    children = [child for child, parent in parents.items() if parent == pid]
    log(f"pid {pid} age {age:.0f}s state {state} wchan {wchan!r} cmd {_cmdline(pid)!r}")
    log(f"  ancestors: {chain}")
    log(f"  children: {[f'{c}:{_cmdline(c)[:60]}' for c in children]}")
    log(f"  fds: {fds[:30]}")


def scan(min_age: float, seen: set[tuple[int, int]]) -> None:
    ticks = os.sysconf("SC_CLK_TCK")
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    parents: dict[int, int] = {}
    started: dict[int, float] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            stat = _stat(int(entry))
            if stat and len(stat) > 21:
                parents[int(entry)] = int(stat[3])
                started[int(entry)] = uptime - int(stat[21]) / ticks
    roots = {
        pid for pid in parents if any(Path(token).name.startswith("opencode") for token in _cmdline(pid).split(" ")[:2])
    }
    sockets = _sockets()
    for pid, age in started.items():
        if age < min_age or pid in roots:
            continue
        cursor, inside = parents.get(pid), False
        while cursor:
            if cursor in roots:
                inside = True
                break
            cursor = parents.get(cursor)
        bucket = int(age // 60)
        if inside and (pid, bucket) not in seen:
            seen.add((pid, bucket))
            describe(pid, age, parents, sockets)


def start(interval: float = 20, min_age: float = 90) -> threading.Thread:
    """Report the environment, then watch the harness's process tree in a daemon thread."""
    environment_report()
    seen: set[tuple[int, int]] = set()

    def loop() -> None:
        while True:
            try:
                scan(min_age, seen)
            except Exception as exc:
                log(f"scan failed: {type(exc).__name__}: {exc}")
            time.sleep(interval)

    thread = threading.Thread(target=loop, name="stall-debug", daemon=True)
    thread.start()
    return thread
