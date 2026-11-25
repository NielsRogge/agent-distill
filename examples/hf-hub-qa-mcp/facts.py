"""Deterministic answer checks for Hub fact questions (no judge, no network).

Each question has one gold value of a kind:

- `category` (license, pipeline tag, library, Space SDK): the answer names the value as whole tokens ("Apache 2.0",
  "apache-2.0" and "Apache License 2.0" all match `apache-2.0`; `text-generation` does not match
  `text2text-generation`). With `hedge` set, naming another value of the same field as well (`hedge`: the field's
  vocabulary) fails: "Apache 2.0 or MIT" is not an answer.
- `repo`: the answer names the repo id (case-insensitive, whole id).
- `number`: some number in the answer ("8,030,261,248", "8.03B", "8.03 billion", "494M") is within 1% of the value.
- `title`: the paper's title (or its part before a colon, if that has at least three words) appears in the answer.
- `first_author`: the author is named, before any other author of the paper.
- `arxiv`: the answer names one of the gold arXiv ids and no other arXiv id.

`check(kind, gold, answer, **extra) -> bool`. This module is shared by the CLI and MCP twins of the task, so both are
scored by the same verifier.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

FILLER = {"license", "licence", "licensed", "version", "the", "under", "a", "an", "library", "sdk", "pipeline", "tag"}
TOLERANCE = 0.01
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9, "t": 1e12, "trillion": 1e12}
NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion|[kmbt])?(?![a-z])", re.I
)
ARXIV = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\w]|\.\d)")


def fold(text: str) -> str:
    """Lowercase ASCII: accents dropped."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def tokens(text: str) -> list[str]:
    """Words of [a-z0-9.] (a sentence's final dot dropped), without filler words like "license"."""
    words = [w.strip(".") for w in re.split(r"[^a-z0-9.]+", fold(text))]
    return [w for w in words if w and w not in FILLER]


def compact(text: str) -> str:
    return "".join(tokens(text))


def spans(text: str, longest: int = 6) -> set[str]:
    """Every run of up to `longest` consecutive tokens, joined."""
    words = tokens(text)
    return {"".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, min(len(words), i + longest) + 1)}


def names_value(value: str, answer: str) -> bool:
    target = compact(value)
    return bool(target) and target in spans(answer)


def check_category(gold: str, answer: str, hedge: list[str] | None = None) -> bool:
    if not names_value(gold, answer):
        return False
    gold_c = compact(gold)
    others = [v for v in hedge or [] if compact(v) and compact(v) != gold_c and compact(v) not in gold_c]
    return not any(names_value(other, answer) for other in others)


def check_repo(gold: str, answer: str) -> bool:
    pattern = rf"(?<![\w./-]){re.escape(gold.lower())}(?![\w/-]|\.\w)"
    return re.search(pattern, answer.lower()) is not None


def numbers(answer: str) -> list[float]:
    found = []
    for digits, unit in NUMBER.findall(answer):
        value = float(digits.replace(",", ""))
        found.append(value * SCALE.get(unit.lower(), 1.0) if unit else value)
    return found


def check_number(gold: float, answer: str) -> bool:
    return any(abs(value - gold) <= TOLERANCE * abs(gold) for value in numbers(answer))


def check_title(gold: str, answer: str) -> bool:
    text = compact(answer)
    main = gold.split(":")[0]
    candidates = [gold] + ([main] if len(tokens(main)) >= 3 else [])
    return any(compact(c) and compact(c) in text for c in candidates)


def _position(name: str, answer: str) -> int:
    text = fold(answer)
    words = [w for w in re.split(r"[^a-z0-9]+", fold(name)) if w]
    if not words:
        return -1
    match = re.search(r"[^a-z0-9]+".join(map(re.escape, words)), text)
    return match.start() if match else -1


def check_first_author(gold: str, answer: str, authors: list[str] | None = None) -> bool:
    first = _position(gold, answer)
    if first < 0:
        return False
    others = [_position(a, answer) for a in authors or [] if fold(a) != fold(gold)]
    return not any(0 <= p < first for p in others)


def check_arxiv(gold: list[str], answer: str) -> bool:
    named = set(ARXIV.findall(answer))
    return bool(named & set(gold)) and not (named - set(gold))


def check(kind: str, gold: Any, answer: str | None, **extra: Any) -> bool:
    if not answer:
        return False
    if kind == "category":
        return check_category(str(gold), answer, extra.get("hedge"))
    if kind == "repo":
        return check_repo(str(gold), answer)
    if kind == "number":
        return check_number(float(gold), answer)
    if kind == "title":
        return check_title(str(gold), answer)
    if kind == "first_author":
        return check_first_author(str(gold), answer, extra.get("authors"))
    if kind == "arxiv":
        return check_arxiv(list(gold), answer)
    raise ValueError(f"unknown answer kind {kind!r}")


def render(kind: str, gold: Any) -> str:
    """The gold value as a plain sentence (the oracle answer for task validation)."""
    if kind == "number":
        return f"It has {int(gold):,} parameters."
    if kind == "arxiv":
        return f"The linked paper is arXiv {next(iter(gold))}."
    return f"The answer is {gold}."
