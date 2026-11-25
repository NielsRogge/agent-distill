"""Deterministic answer checks for fact questions (no judge, no network): shared by the hf-hub-qa (CLI) and
hf-hub-qa-mcp (MCP) examples, so both are scored by the same verifier.

Each question has one gold value of a kind:

- `category` (license, pipeline tag, library, Space SDK): the answer names the value as whole tokens ("Apache 2.0",
  "apache-2.0" and "Apache License 2.0" all match `apache-2.0`; `text-generation` does not match
  `text2text-generation`). A list gold (a dual license) needs every value. With `hedge` (the field's vocabulary), a
  hedged answer fails: another value named before the gold, or offered as an alternative ("Apache 2.0 or MIT",
  "Apache-2.0/MIT"); other values named afterwards in context are fine ("Docker, unlike Gradio/Streamlit").
- `repo`: the answer names the repo id (case-insensitive, whole id).
- `number`: some number in the answer ("8,030,261,248", "8.03B", "8.03 billion", "494M") is within 1% of the value.
- `title`: the paper's title (or its part before a colon, if that has at least three words) appears in the answer.
- `first_author`: the author is named, before any other author of the paper.
- `arxiv`: the answer names one of the gold arXiv ids and no other arXiv id.
- `count`: the first standalone integer in the answer is the value ("The repo has **12** GGUF files"). Digits inside
  words, ids, file names and decimals do not count (`Q4_K_M`, `model-00001-of-00004`, `Qwen3.5-9B`, `4.5`), so an
  answer that lists files after stating the count is fine, and one that only lists them fails.

`check(kind, gold, answer, **extra) -> bool`; `render(kind, gold)` is the gold value as a sentence (the oracle answer
used to validate questions).
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


def words(text: str) -> list[str]:
    """Words of [a-z0-9.] (a sentence's final dot dropped) and "/" separators, without filler words like "license"."""
    found = [w.strip(".") if w != "/" else w for w in re.findall(r"[a-z0-9.]+|/", fold(text))]
    return [w for w in found if w and w not in FILLER]


def tokens(text: str) -> list[str]:
    """`words` without the separators."""
    return [w for w in words(text) if w != "/"]


def compact(text: str) -> str:
    return "".join(tokens(text))


def mentions(value: str, answer: str, longest: int = 6) -> list[tuple[int, int]]:
    """(start, end) word positions where a run of up to `longest` consecutive words (not across "/") spells `value`."""
    target = compact(value)
    if not target:
        return []
    ws = words(answer)
    found = []
    for i in range(len(ws)):
        joined = ""
        for j in range(i, min(len(ws), i + longest)):
            if ws[j] == "/":
                break
            joined += ws[j]
            if joined == target:
                found.append((i, j + 1))
            if len(joined) >= len(target):
                break
    return found


def spans(text: str, longest: int = 6) -> set[str]:
    """Every run of up to `longest` consecutive tokens, joined."""
    ws = tokens(text)
    return {"".join(ws[i:j]) for i in range(len(ws)) for j in range(i + 1, min(len(ws), i + longest) + 1)}


def names_value(value: str, answer: str) -> bool:
    return bool(mentions(value, answer))


JOIN_WINDOW = 3  # words between two values that "or" or "/" makes alternatives ("Apache 2.0 or MIT")


def check_category(gold: str | list[str], answer: str, hedge: list[str] | None = None) -> bool:
    """Every gold value is named. With `hedge` (the field's vocabulary), the answer must not hedge: no other value
    named before the first gold value ("Either MIT or Apache 2.0", "MIT. (Some say Apache 2.0.)"), and none offered
    as an alternative to a gold value, joined by "or" or "/" within a few words ("Apache 2.0 or MIT", "Apache-2.0/MIT").
    Other values named after the answer, in context, are fine ("Docker, unlike Gradio/Streamlit")."""
    golds = [str(g) for g in (gold if isinstance(gold, list) else [gold])]
    found = [mentions(g, answer) for g in golds]
    if not golds or not all(found):
        return False
    gold_spans = [span for spans_ in found for span in spans_]
    first = min(start for start, _ in gold_spans)
    gold_c = [compact(g) for g in golds]
    others = [v for v in hedge or [] if compact(v) and not any(compact(v) == g or compact(v) in g for g in gold_c)]
    ws = words(answer)
    other_spans = sorted(span for other in others for span in mentions(other, answer))
    if any(start < first for start, _ in other_spans):
        return False
    for g_start, g_end in gold_spans:  # the nearest other value on each side, if offered as an alternative
        after = next(((s, e) for s, e in other_spans if s >= g_end), None)
        before = next(((s, e) for s, e in reversed(other_spans) if e <= g_start), None)
        for between in ([ws[g_end : after[0]]] if after else []) + ([ws[before[1] : g_start]] if before else []):
            if len(between) <= JOIN_WINDOW and ("or" in between or "/" in between):
                return False
    return True


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


STANDALONE_INT = re.compile(r"(?<![\w.,/:-])(\d{1,3}(?:,\d{3})+|\d+)(?![\w/-]|[.,]\d)")


def check_count(gold: int, answer: str) -> bool:
    found = STANDALONE_INT.search(answer.replace("*", " ").replace("`", " "))
    return found is not None and int(found.group(1).replace(",", "")) == int(gold)


def check(kind: str, gold: Any, answer: str | None, **extra: Any) -> bool:
    if not answer:
        return False
    if kind == "category":
        return check_category(gold if isinstance(gold, list) else str(gold), answer, extra.get("hedge"))
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
    if kind == "count":
        return check_count(int(gold), answer)
    raise ValueError(f"unknown answer kind {kind!r}")


def render(kind: str, gold: Any) -> str:
    """The gold value as a plain sentence (the oracle answer for task validation)."""
    if kind == "number":
        return f"It has {int(gold):,} parameters."
    if kind == "arxiv":
        return f"The linked paper is arXiv {next(iter(gold))}."
    if kind == "count":
        return f"There are {int(gold)} of them."
    if isinstance(gold, list):
        return f"The answer is {' and '.join(map(str, gold))}."
    return f"The answer is {gold}."
