"""Agent isolation: every episode's agent runs as its own unprivileged OS user (DESIGN.md §15).

Stages run as root in HF Jobs containers. An agent started as root could read the job's `HF_TOKEN` from
`/proc/<stage pid>/environ`, read other episodes' answers, or delete the stage's records (an SFT student did: the
M5 evaluation lost `/eval/episodes` mid-run). When the stage runs as root, the adapter instead starts the agent as a
per-episode UID from `UID_BASE` upwards (no `/etc/passwd` entry needed) that:

- owns only its episode directory (0700), and is the only user in it;
- cannot read root's processes (`/proc/<pid>/environ` is 0400) or 0700 directories (the stage's outputs, `/root`);
- can read and run only what it is granted (`grant_read`): the harness install, the stage's Python and Chromium;
- is killed with every process it left behind (`kill_user`) before its UID goes back to the pool.

The stage reads files the agent controlled (its answer, its session) only after every process of the agent's UID is
dead (`kill_user`), relative to a directory fd pinned before the agent started (`open_root`), opening every path
component with O_NOFOLLOW (`read_agent_file`, `list_agent_dir`). An agent cannot point the stage, running as root,
at a secret through a symlink, not even from a process that escaped its process group.

`preflight()` checks, before any episode or spend, that this process can hand directories to an agent UID and start
processes as it. `AGENT_DISTILL_AGENT_USERS=off` keeps agents as the stage's user (local runs as root that need it);
a stage that is not root cannot switch users and runs agents as itself, with a warning (`mode()` is recorded in each
episode record and stage summary).
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

UID_BASE = 61000
POOL_SIZE = 256
SWITCH_ENV = "AGENT_DISTILL_AGENT_USERS"
_lock = threading.Lock()
_in_use: set[int] = set()


# Directories that are never opened up recursively, whatever asks (a system Python's prefix is /usr).
SYSTEM_DIRS = {Path(p) for p in ("/", "/usr", "/usr/local", "/opt", "/etc", "/var", "/home", "/root", "/srv", "/tmp")}
_warned = False


def switched_off() -> bool:
    return os.environ.get(SWITCH_ENV, "auto").lower() in {"off", "0", "false", "no"}


def enabled() -> bool:
    return os.geteuid() == 0 and not switched_off()


def mode() -> str:
    """How agents run in this process: "own-uid", "stage-user (switched off)" or "stage-user (not root)"."""
    if enabled():
        return "own-uid"
    return "stage-user (switched off)" if switched_off() else "stage-user (not root)"


def warn_if_shared() -> None:
    """Say once, loudly, when agents share the stage's user (and so can read its token and records)."""
    global _warned
    if enabled() or _warned:
        return
    _warned = True
    print(f"WARNING: agent isolation is off ({mode()}): agents run as the stage's user and can read its environment "
          "(HF_TOKEN, judge keys) and its records. Run stages as root (HF Jobs do) for per-agent users.",
          file=sys.stderr, flush=True)  # fmt: skip


def preflight() -> None:
    """Fail before any episode (and any spend) if agents cannot be given their own users here."""
    if not enabled():
        warn_if_shared()
        return
    probe = Path(tempfile.mkdtemp(prefix="agent-distill-isolation-check-"))
    try:
        with agent_user() as uid:
            assert uid is not None
            give(probe, uid)
            command = ["/bin/sh", "-c", 'touch "$1/ok"', "sh", str(probe)]
            done = subprocess.run(command, user=uid, group=uid, extra_groups=[], env={"PATH": "/usr/bin:/bin"},
                                  capture_output=True, timeout=30)  # fmt: skip
            if done.returncode or not (probe / "ok").exists():
                raise RuntimeError(done.stderr.decode(errors="replace")[-300:] or f"exit {done.returncode}")
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        raise RuntimeError(
            f"agent isolation is on (running as root) but this process cannot run an agent as its own user ({exc}). "
            "It needs CAP_SETUID, CAP_SETGID and CAP_CHOWN. Fix the container, or set "
            f"{SWITCH_ENV}=off to run agents as the stage's user (they can then read its token and records)."
        ) from exc
    finally:
        shutil.rmtree(probe, ignore_errors=True)


@contextmanager
def agent_user() -> Iterator[int | None]:
    """A UID nobody else is using for the episode's agent, or None when agents run as the stage's user."""
    if not enabled():
        yield None
        return
    with _lock:
        uid = next((UID_BASE + i for i in range(POOL_SIZE) if UID_BASE + i not in _in_use), None)
        if uid is None:
            raise RuntimeError(f"more than {POOL_SIZE} concurrent agents")
        _in_use.add(uid)
    kill_user(uid)  # anything a crashed earlier stage left behind under this UID
    try:
        yield uid
    finally:
        kill_user(uid)
        with _lock:
            _in_use.discard(uid)


def give(path: Path, uid: int) -> None:
    """Hand a directory tree the stage just created to the agent's UID, private to it (0700 at the top)."""
    path = Path(path)
    for root, dirs, files in os.walk(path):
        for name in [*dirs, *files]:
            os.lchown(os.path.join(root, name), uid, uid)
    os.chown(path, uid, uid)
    path.chmod(0o700)


def grant_read(paths: Iterable[Path]) -> None:
    """Let other users read and run `paths` (recursively) and traverse, but not list, their ancestors."""
    homes = {Path("/root").resolve(), Path.home().resolve()}
    for path in paths:
        path = Path(path).resolve()
        if not path.exists():
            continue
        if path in SYSTEM_DIRS:
            raise RuntimeError(f"refusing to open up {path} recursively; grant the specific directories agents need")
        if any(path == home or path.is_relative_to(home) for home in homes):
            raise RuntimeError(
                f"agents would need {path}, inside the stage user's home, which holds its credentials; install it "
                "elsewhere (XDG_CACHE_HOME, PLAYWRIGHT_BROWSERS_PATH) or set AGENT_DISTILL_AGENT_USERS=off"
            )
        for parent in path.parents:
            mode = parent.stat().st_mode
            if not mode & stat.S_IXOTH:
                parent.chmod(stat.S_IMODE(mode) | stat.S_IXOTH)
        items = [path] if path.is_file() else [path, *path.rglob("*")]
        for item in items:
            if item.is_symlink():
                continue
            mode = item.stat().st_mode
            extra = stat.S_IROTH | (stat.S_IXOTH if stat.S_ISDIR(mode) or mode & stat.S_IXUSR else 0)
            if stat.S_IMODE(mode) & extra != extra:
                item.chmod(stat.S_IMODE(mode) | extra)


def private(path: Path) -> None:
    """Make a stage output directory unreadable to agents (0700, owned by the stage)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def kill_user(uid: int) -> None:
    """SIGKILL every process owned by `uid`."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        with suppress(OSError):
            if entry.stat().st_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)


def open_root(path: Path) -> int:
    """A directory fd to read an episode through later, pinned before the agent can touch the directory."""
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)


def _relative_parts(path: Path, root: Path) -> list[str] | None:
    try:
        parts = list(Path(path).relative_to(root).parts)
    except ValueError:
        return None
    return None if not parts or any(p in ("..", ".", "") for p in parts) else parts


def _open_dirs(root: Path, root_fd: int | None, parts: list[str]) -> int:
    """An fd of `root/parts...` opened one component at a time, none of them a symlink."""
    fd = os.dup(root_fd) if root_fd is not None else open_root(root)
    try:
        for part in parts:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = nxt
    except OSError:
        os.close(fd)
        raise
    return fd


def read_agent_file(path: Path, root: Path, *, root_fd: int | None = None, limit: int = 64 * 1024 * 1024) -> str | None:
    """Read a file the agent controlled, below `root` (through `root_fd` when pinned); None if it is missing, not a
    regular file, or reached through a symlink anywhere below `root`."""
    parts = _relative_parts(Path(path), Path(root))
    if parts is None:
        return None
    try:
        parent = _open_dirs(Path(root), root_fd, parts[:-1])
    except OSError:
        return None
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
    except OSError:
        return None
    finally:
        os.close(parent)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return None
        return handle.read(limit).decode("utf-8", errors="replace")


def list_agent_dir(path: Path, root: Path, *, root_fd: int | None = None) -> list[str]:
    """Entry names of a directory the agent controlled, reached without following symlinks ([] if unreachable)."""
    parts = [] if Path(path) == Path(root) else _relative_parts(Path(path), Path(root))
    if parts is None:
        return []
    try:
        fd = _open_dirs(Path(root), root_fd, parts)
    except OSError:
        return []
    try:
        return os.listdir(fd)
    finally:
        os.close(fd)
