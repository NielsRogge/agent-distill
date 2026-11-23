"""Credential patterns: never written to records, scanned for before a dataset is made public (DESIGN.md §8)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Credentials an agent can print from its environment (e.g. `env | grep HF`).
CREDENTIAL_RE = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AIza[0-9A-Za-z_-]{35})"
)
CREDENTIAL_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


def redact_credentials(text: str) -> str:
    return CREDENTIAL_RE.sub(lambda match: match.group(0)[:3] + "***REDACTED***", text)


def strip_credentials(env: dict[str, str]) -> dict[str, str]:
    """A copy of `env` without credential variables (for agent processes)."""
    return {key: value for key, value in env.items() if key not in CREDENTIAL_ENV_VARS}


def record_line(record: dict[str, Any]) -> str:
    """One JSONL line for an episode record, UTF-8, with credential-shaped strings redacted.

    Tool output can contain lone UTF-16 surrogates (e.g. math symbols cut out of paper text), which UTF-8
    cannot encode; those records fall back to ASCII-escaped JSON, which is still valid and round-trips.
    """
    line = json.dumps(record, ensure_ascii=False)
    try:
        line.encode("utf-8")
    except UnicodeEncodeError:
        line = json.dumps(record)
    return redact_credentials(line) + "\n"


def scan_text(text: str) -> list[str]:
    """Credential-shaped strings in `text`, masked (never return the full value)."""
    return [match.group(0)[:6] + "..." for match in CREDENTIAL_RE.finditer(text)]


def scan_files(paths: Iterable[Path]) -> dict[str, int]:
    """Number of credential-shaped strings per file that has any (text files only)."""
    hits: dict[str, int] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (IsADirectoryError, PermissionError):
            continue
        count = len(CREDENTIAL_RE.findall(text))
        if count:
            hits[str(path)] = count
    return hits
