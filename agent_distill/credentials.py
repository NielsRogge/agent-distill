"""Credential patterns: never written to records, scanned for before a dataset is made public (DESIGN.md §8)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Credentials an agent can print from its environment (e.g. `env | grep HF`).
# Records are redacted after JSON serialisation, where a token printed at the start of a line becomes `\nhf_...`
# and a coloured one `\u001b[0mhf_...`, so the character before a token is often a letter: prefixed tokens
# (`hf_`, `ghp_`, `github_pat_`, Google's `AIza`) are matched anywhere. `sk-` keys must not follow a letter or digit
# (unless that ends a JSON escape such as `\n` or a colour code), so words such as "task-..." are not matched
# (url-extraction #22).
_SK_START = (
    r"(?:(?<![A-Za-z0-9])|(?<=\\[nrtbf])|(?<=\\u[0-9a-fA-F]{4})"
    # ...or the end of an ANSI colour code (`\u001b[0m`, `\u001b[1;31m`).
    r"|(?<=\[[0-9]m)|(?<=\[[0-9][0-9]m)|(?<=;[0-9]m)|(?<=;[0-9][0-9]m))"
)
CREDENTIAL_RE = re.compile(
    r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{35}|"
    + _SK_START
    + r"sk-[A-Za-z0-9_-]{20,}"
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


CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def is_credential_var(name: str) -> bool:
    return name in CREDENTIAL_ENV_VARS or name.upper().endswith(CREDENTIAL_SUFFIXES)


def strip_credentials(env: dict[str, str]) -> dict[str, str]:
    """A copy of `env` without credential variables (for agent processes): the known names, and any name ending in
    `_API_KEY`, `_TOKEN`, `_SECRET` or `_PASSWORD` (e.g. a judge profile's own `key_env`)."""
    return {key: value for key, value in env.items() if not is_credential_var(key)}


# Characters `str.splitlines()` (and so the Hub's dataset viewer) treats as line breaks that `json.dumps` leaves
# raw with `ensure_ascii=False`. (`json.dumps` already escapes the C0 ones such as \x0b, \x0c, \x1c-\x1e and \r;
# they are listed too so the table matches url-extraction's and stays correct if the serialiser changes.)
LINE_BREAK_ESCAPES = {
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
    "\x85": "\\u0085",
    "\x0b": "\\u000b",
    "\x0c": "\\u000c",
    "\x1c": "\\u001c",
    "\x1d": "\\u001d",
    "\x1e": "\\u001e",
    "\r": "\\r",
}


def jsonl_line(value: Any) -> str:
    """One JSON line (no trailing newline) that every JSONL reader splits correctly, credentials redacted.

    UTF-8, with Unicode line breaks escaped. Tool output can contain lone UTF-16 surrogates (e.g. math symbols cut
    out of paper text), which UTF-8 cannot encode; those values fall back to ASCII-escaped JSON, which is still
    valid and round-trips.
    """
    line = json.dumps(value, ensure_ascii=False)
    try:
        line.encode("utf-8")
    except UnicodeEncodeError:
        line = json.dumps(value)
    for char, escaped in LINE_BREAK_ESCAPES.items():
        line = line.replace(char, escaped)
    return redact_credentials(line)


def record_line(record: dict[str, Any]) -> str:
    """One JSONL line for an episode record (`jsonl_line` plus the newline)."""
    return jsonl_line(record) + "\n"


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
