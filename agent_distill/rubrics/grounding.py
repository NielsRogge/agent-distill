"""Grounding: are the Hub ids an answer names ones the agent actually saw in its tool output?

Shared by the hf-hub-qa (CLI) and hf-hub-qa-mcp (MCP) examples; each passes its own tool text (shell commands and
their output, or MCP calls and their results), so the check is the same whatever the tools.

- Repo ids written as ids (in `code` or a huggingface.co link) always count, so an invented id is caught.
- A bare `word/word` in prose ("PEFT/LoRA", "F16/BF16") counts only if it exists on the Hub (anonymous, cached).
- Organisation pages (huggingface.co/Qwen/models), file names (`*.gguf`) and size or precision shorthand
  (`Kronos-mini/small`, `-bf16/f16`) are not repo ids.
- Paper ids (2205.02340) count wherever they appear.
"""

from __future__ import annotations

import re
from functools import lru_cache

REPO_ID = re.compile(r"(?<![\w./-])([A-Za-z0-9][\w.-]{0,95})/([A-Za-z0-9][\w.-]{0,95})(?![\w/-])")
PAPER_ID = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5})(?![\w]|\.\d)")
HUB_LINK = re.compile(r"huggingface\.co/(?:(?:datasets|spaces)/)?([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)")
FILE_OR_VARIANT = re.compile(
    r"\.(?:gguf|safetensors|bin|json|md|py|txt|onnx|pt|pth|yaml|parquet)$"
    r"|^(?:small|base|large|mini|tiny|medium|xl|\d+(?:\.\d+)?[kmb]|[bf]p?\d{1,2}|q\d\w*|int\d|\d+bit)$",
    re.IGNORECASE,
)
ORG_PAGES = {"models", "datasets", "spaces", "collections", "papers", "activity", "posts", "followers"}
NOT_IDS = {"and/or", "input/output", "read/write", "train/test", "true/false", "yes/no", "on/off"}
CODE_SPAN = re.compile(r"`([^`\n]{3,200})`")
LINK_TARGET = re.compile(r"\]\((https?://[^)\s]+)\)")


def _repo_candidates(text: str) -> set[str]:
    found = set()
    for owner, name in REPO_ID.findall(text):
        candidate = f"{owner}/{name}".rstrip(".")
        if candidate.lower() in NOT_IDS or owner.lower() in {"http", "https", "www"}:
            continue
        if owner.lower() in {"huggingface.co", "hf.co", "datasets", "spaces", "models", "papers"}:
            continue
        if name.lower() in ORG_PAGES:  # huggingface.co/Qwen/models: an organisation's listing, not a repo
            continue
        if FILE_OR_VARIANT.search(name):  # `x-bf16/f16.gguf` (a file), `Kronos-mini/small` (shorthand for two repos)
            continue
        found.add(candidate)
    return found


@lru_cache(maxsize=4096)
def exists_on_hub(repo_id: str) -> bool:
    """Is this a public model, dataset or Space (anonymous, like the agent)?"""
    from huggingface_hub import HfApi

    api = HfApi(token=False)
    for kind in ("model", "dataset", "space"):
        try:
            if api.repo_exists(repo_id, repo_type=kind):
                return True
        except Exception:  # an invalid id, a rate limit: not evidence that it exists
            continue
    return False


def cited_ids(answer: str, *, check_hub: bool = True) -> set[str]:
    """Hub ids an answer names (see the module docstring)."""
    marked = set()
    for span in CODE_SPAN.findall(answer):
        marked |= _repo_candidates(span)
    marked.update(
        link.rstrip(".")
        for link in HUB_LINK.findall(answer)
        if not link.startswith("papers/") and link.rstrip(".").split("/")[-1].lower() not in ORG_PAGES
    )
    prose = _repo_candidates(LINK_TARGET.sub("", CODE_SPAN.sub(" ", answer))) - marked
    found = marked | {c for c in prose if check_hub and exists_on_hub(c)}
    found.update(PAPER_ID.findall(answer))
    return found


def grounded_score(answer: str, tool_text: str, *, check_hub: bool = True) -> tuple[float, dict[str, float]] | None:
    """The share of the answer's Hub ids that appear in the tool text, with counts; None when it names no ids."""
    ids = cited_ids(answer, check_hub=check_hub)
    if not ids:
        return None
    seen = tool_text.lower()
    hits = sum(1 for i in ids if i.lower() in seen)
    return hits / len(ids), {"ids": float(len(ids)), "unseen": float(len(ids) - hits)}
