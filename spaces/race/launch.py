"""Container entrypoint: take the HF token over from start.sh's one-time file, then serve the race page.

The token is read and the file deleted before anything else runs. It then lives only in this process's memory (the
library's key proxy reads it from `os.environ`, which is not what `/proc/<pid>/environ` shows: that is the
environment this process was started with, and start.sh removed the token from it). This process is also made
non-dumpable, so processes of the same user cannot read its memory or its `/proc` files either (agents normally run
as their own UIDs; this covers a container where they cannot). Agents get a credential-free environment from the Pi
adapter and reach the models only through the per-episode key proxy.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

PR_SET_DUMPABLE = 4


def take_token() -> None:
    path = Path(os.environ.pop("RACE_TOKEN_FILE", "") or "/nonexistent")
    token = ""
    if path.is_file():
        token = path.read_text().strip()
        path.unlink(missing_ok=True)
    token = token or os.environ.pop("HF_TOKEN", "")  # a local run without start.sh
    if token:
        os.environ["HF_TOKEN"] = token


def non_dumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE, 0) failed")


if __name__ == "__main__":
    non_dumpable()  # before the token is read
    take_token()
    import server
    import uvicorn

    uvicorn.run(server.app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), log_level="warning")
