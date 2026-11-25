"""Agent isolation: every episode's agent runs as its own unprivileged OS user (DESIGN.md §15).

Stages run as root in HF Jobs containers. An agent started as root could read the job's `HF_TOKEN` from
`/proc/<stage pid>/environ`, read other episodes' answers, or delete the stage's records (an SFT student did: the
M5 evaluation lost `/eval/episodes` mid-run). When the stage runs as root, the adapter instead starts the agent as a
per-episode UID from `UID_BASE` upwards (no `/etc/passwd` entry needed) that:

- owns only its episode directory (0700), and is the only user in it;
- cannot read root's processes (`/proc/<pid>/environ` is 0400) or 0700 directories (the stage's outputs, `/root`);
- can read and run only what it is granted (`grant_read`): the harness install, the stage's Python and Chromium;
- is killed with every process it left behind (`kill_user`) before its UID goes back to the pool.

The stage reads files the agent controlled (its answer, its session) with `read_agent_file`, which refuses symlinks
and anything outside the episode directory, so an agent cannot point the stage, running as root, at a secret.

`AGENT_DISTILL_AGENT_USERS=off` keeps agents as the stage's user (local runs as root that need it); a stage that is
not root cannot switch users and runs agents as itself, as before.
"""

from __future__ import annotations

import os
import signal
import stat
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

UID_BASE = 61000
POOL_SIZE = 256
SWITCH_ENV = "AGENT_DISTILL_AGENT_USERS"
_lock = threading.Lock()
_in_use: set[int] = set()


def enabled() -> bool:
    return os.geteuid() == 0 and os.environ.get(SWITCH_ENV, "auto").lower() not in {"off", "0", "false", "no"}


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


def read_agent_file(path: Path, root: Path, *, limit: int = 64 * 1024 * 1024) -> str | None:
    """Read a file the agent controlled; None if it is missing, not a regular file, a symlink or outside `root`."""
    path, root = Path(path), Path(root).resolve()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_relative_to(root) or resolved != path.absolute():
        return None  # a symlink somewhere on the way (or outside the episode)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return None
        return handle.read(limit).decode("utf-8", errors="replace")
