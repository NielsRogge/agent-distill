"""Rubrics of the hf-hub-qa task.

- `answered` (gate): the episode ended with a final message (not cut off while still calling tools).
- `read_only` (gate): no command that writes to the Hub or logs in (`hf upload`, `hf repos create|delete`, `hf jobs
  run`, ...; `--help` on one is fine). The agent has no token, so such commands fail anyway; trying them still
  fails the episode.
- `used_hub` (gate): at least one successful `hf` command (or Hub API call). An answer from memory fails, however
  good: the task is to look things up.
- `grounded`: of the Hub ids the answer names (repo ids written as ids, in `code` or a huggingface.co link; bare
  `word/word` pairs only if they exist on the Hub; paper ids), the share that appears in the episode's own tool
  output. Ids the agent never saw are guesses. Not scored (None) when
  the answer names no ids.
- `reference`: for questions with a canonical answer (questions.py), the stage runs the reference `hf` command when
  it scores and checks the answer against it: recall of the top-k ids (a match anywhere in the listing's top 2k
  counts) or the stated fact. Not scored (None) without a reference.
- `answer_quality` (judge): correctness against what the tools returned, relevance, completeness and clarity, 1-10.
- Metrics, recorded only: number of `hf` calls, answer length.

reward = gate(answered, gate(read_only, gate(used_hub, weighted(answer_quality 2, grounded 1, reference 1))));
an episode passes when it answered, stayed read-only, used the Hub, every named id was seen (grounded = 1), the
reference (if any) is at least 2/3 met and the judge scores at least 7/10.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import Any

from agent_distill.rubrics import RubricSet, gate, metric, process, programmatic, weighted
from agent_distill.rubrics.judge import judge

PASS_SCORE = 7  # of 10
SCALE = (1.0, 10.0)
CRITERIA = ("correctness", "relevance", "completeness", "clarity")
REFERENCE_PASS = 2 / 3
TOOL_LOG_CHARS = 14000  # what the judge sees of the agent's tool calls and their output
OUTPUT_CHARS = 1800  # per tool call

WRITE_COMMANDS = re.compile(
    r"\bhf\s+(?:upload|sync|cp|auth\s+(?:login|logout|switch)"
    r"|repos?\s+(?:create|delete|settings|move|branch|tag\s+(?:create|delete)|duplicate)"
    r"|(?:jobs|endpoints|webhooks|buckets|collections|discussions|sandbox)\s+"
    r"(?:run|uv|create|deploy|delete|pause|resume|update|scale-to-zero|add-item|update-item|delete-item|comment|close"
    r"|reopen|merge|rename|cancel|scheduled)"
    r"|spaces\s+(?:secrets|variables)\s+(?:add|delete|set)|spaces\s+(?:duplicate|restart|pause|create|delete))\b"
)
HUB_COMMAND = re.compile(r"(?:^|[\s;&|(`])hf\s+[a-z]|huggingface\.co/api/|huggingface_hub")
REPO_ID = re.compile(r"(?<![\w./-])([A-Za-z0-9][\w.-]{0,95})/([A-Za-z0-9][\w.-]{0,95})(?![\w/-])")
PAPER_ID = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5})(?![\w]|\.\d)")
HUB_LINK = re.compile(r"huggingface\.co/(?:(?:datasets|spaces)/)?([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)")
ORG_PAGES = {"models", "datasets", "spaces", "collections", "papers", "activity", "posts", "followers"}
HELP = re.compile(r"(?:^|\s)(?:--help|-h)(?:\s|$)")
NOT_IDS = {"and/or", "input/output", "read/write", "train/test", "true/false", "yes/no", "on/off"}


def shell_events(tool_events: list[Any]) -> list[Any]:
    return [e for e in tool_events or [] if getattr(e, "kind", "") == "shell"]


def command_of(event: Any) -> str:
    data = getattr(event, "input", None) or {}
    return str(data.get("command", "")) if isinstance(data, dict) else str(data)


def tool_text(tool_events: list[Any]) -> str:
    return "\n".join(f"{command_of(e)}\n{getattr(e, 'output', '') or ''}" for e in tool_events or [])


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
    """Hub ids an answer names. Repo ids written as ids (in `code` or a huggingface.co link) always count, so an
    invented id is caught; a bare `word/word` in prose ("PEFT/LoRA", "F16/BF16") counts only if it exists on the Hub.
    Paper ids (2205.02340) count wherever they appear."""
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


@programmatic(weight=0.0)
def answered(answer: Any) -> float:
    return float(bool(str(answer or "").strip()))


@process(weight=0.0)
def read_only(tool_events: list[Any]) -> float:
    writes = [c for c in map(command_of, shell_events(tool_events)) if WRITE_COMMANDS.search(c)]
    return 0.0 if any(not HELP.search(c) for c in writes) else 1.0


@process(weight=0.0)
def used_hub(tool_events: list[Any]) -> float:
    return float(any(HUB_COMMAND.search(command_of(e)) and getattr(e, "success", False)
                     for e in shell_events(tool_events)))  # fmt: skip


@programmatic(weight=1.0)
def grounded(answer: Any, tool_events: list[Any]) -> tuple[float, dict[str, float]] | None:
    ids = cited_ids(str(answer or ""))
    if not ids:
        return None
    seen = tool_text(tool_events).lower()
    hits = sum(1 for i in ids if i.lower() in seen)
    return hits / len(ids), {"ids": float(len(ids)), "unseen": float(len(ids) - hits)}


@lru_cache(maxsize=256)
def run_hf(argv: tuple[str, ...]) -> Any:
    """The pinned CLI, logged out like the agent (no token), JSON out. Cached for a stage's lifetime."""
    binary = os.environ.get("AGENT_DISTILL_HF_CLI") or shutil.which("hf")
    if not binary:
        raise RuntimeError("the hf CLI is not installed (task job_setup installs it)")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))}
    env.update(HF_HUB_DISABLE_UPDATE_CHECK="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    with tempfile.TemporaryDirectory() as home:
        env["HOME"] = home  # no cached token either
        env.pop("HF_HOME", None)
        done = subprocess.run([binary, *argv, "--format", "json"], capture_output=True, text=True, timeout=120,
                              env=env, check=True)  # fmt: skip
    return json.loads(done.stdout)


def dig(data: Any, path: str) -> Any:
    for part in path.split("."):
        data = data.get(part) if isinstance(data, dict) else None
    return data


def normalize(text: str) -> str:
    return re.sub(r"[\s_-]+", "", text.lower())


@programmatic(weight=1.0)
def reference(answer: Any, row: dict[str, Any]) -> tuple[float, dict[str, float]] | None:
    spec = json.loads(row.get("reference") or "null")
    if not spec or not answer:
        return None
    text = str(answer)
    if "fact" in spec:
        value = dig(run_hf(tuple(spec["fact"])), spec["path"])
        if value in (None, ""):
            return None
        return float(normalize(str(value)) in normalize(text))
    listing = run_hf(tuple(spec["ids"]))
    k = int(spec.get("k", 5))
    window = [str(item.get(spec.get("key", "id"))) for item in listing][: 2 * k]
    lowered = text.lower()
    hits = sum(1 for i in window if i.lower() in lowered)
    return min(1.0, hits / k), {"hits": float(hits), "k": float(k)}


JUDGE_PROMPT = """You are grading an assistant that answers questions about the Hugging Face Hub. It looked things
up with the `hf` command-line tool; the Hub changes daily, so judge the answer against what the tools returned
(shown below), not against your own memory of the Hub.

Question: {question}

Tool calls and their output (truncated):
{tool_log}

The assistant's final answer:
{answer}

Score each criterion from 1 (poor) to 10 (excellent):
- correctness: every claim is supported by the tool output or is uncontroversial background knowledge; no invented
  repos, papers, numbers or features. Penalise heavily anything that contradicts the tool output.
- relevance: it answers the question that was asked (the right task, filters, time frame, organisation).
- completeness: it gives enough specific items or detail to be useful (ids, a reason for each), and says what the
  Hub could not tell it, if anything.
- clarity: concise, well organised, easy to act on.

Answer with one line per criterion, then the overall score, exactly in this form:
correctness: <1-10>
relevance: <1-10>
completeness: <1-10>
clarity: <1-10>
SCORE: <1-10>"""


def tool_log(tool_events: list[Any]) -> str:
    parts = []
    for event in tool_events or []:
        out = str(getattr(event, "output", "") or getattr(event, "error", "") or "")
        if len(out) > OUTPUT_CHARS:
            out = out[:OUTPUT_CHARS] + f"... [{len(out) - OUTPUT_CHARS} more characters]"
        call = command_of(event) or json.dumps(getattr(event, "input", None) or {})[:300]
        parts.append(f"$ [{getattr(event, 'name', '')}] {call}\n{out}")
    text = "\n\n".join(parts)
    return text if len(text) <= TOOL_LOG_CHARS else text[:TOOL_LOG_CHARS] + "\n... [truncated]"


@judge(JUDGE_PROMPT, scale=SCALE, criteria=CRITERIA)
def answer_quality(answer: Any, row: dict[str, Any], tool_events: list[Any]) -> dict[str, Any] | None:
    if not str(answer or "").strip():
        return None
    return {"question": row.get("question", ""), "tool_log": tool_log(tool_events) or "(no tool calls)",
            "answer": str(answer)}  # fmt: skip


@metric
def hf_calls(tool_events: list[Any]) -> float:
    return float(sum(1 for e in shell_events(tool_events) if HUB_COMMAND.search(command_of(e))))


@metric
def answer_chars(answer: Any) -> float | None:
    return float(len(str(answer))) if answer else None


def passed(scores: dict[str, float | None]) -> bool:
    quality = scores.get("answer_quality")
    threshold = (PASS_SCORE - SCALE[0]) / (SCALE[1] - SCALE[0])
    ref = scores.get("reference")
    return (
        scores.get("answered") == 1.0
        and scores.get("read_only") == 1.0
        and scores.get("used_hub") == 1.0
        and scores.get("grounded") in (None, 1.0)
        and (ref is None or ref >= REFERENCE_PASS - 1e-9)
        and quality is not None
        and quality >= threshold - 1e-9
    )


RUBRICS = RubricSet(
    [answered, read_only, used_hub, grounded, reference, answer_quality, hf_calls, answer_chars],
    reward=gate(
        "answered",
        gate("read_only", gate("used_hub", weighted({"answer_quality": 2.0, "grounded": 1.0, "reference": 1.0}))),
    ),
    passed=passed,
)
