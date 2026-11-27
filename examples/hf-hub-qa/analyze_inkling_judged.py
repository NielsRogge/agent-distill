"""Analyse base Inkling-2-Mini on the 51 open-ended judged questions (`judged.py`) against the published records.

First record per row everywhere. Paired comparisons are exact McNemar tests on the 51 shared rows. Also checks
whether Inkling fails the questions the Qwen students fail (per-question difficulty = baselines that fail it).
Prints a JSON report.

    uv run --extra hub python examples/hf-hub-qa/analyze_inkling_judged.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

from agent_distill.records import mcnemar_exact

NS = "nielsr"
JUDGED = "hf-hub-qa-judged-glm-53-pi-160-eval"
BASE, SFT = "episodes/base/seed.jsonl", "episodes/sft/seed.jsonl"
BASELINES = {
    "qwen-base-think": (f"cursor-qwen35-4b-think-cfx-a-curated-{JUDGED}", BASE),
    "qwen-recipe-a": (f"qwen35-4b-think-cfx-a-{JUDGED}", SFT),
    "qwen-recipe-a-curated": (f"qwen35-4b-think-cfx-a-curated-{JUDGED}", SFT),
    "qwen-recipe-a-mixed": (f"cursor-qwen35-4b-think-cfx-a-mixed-{JUDGED}", SFT),
    "qwen-fullcot": (f"cursor-qwen35-4b-think-{JUDGED}", SFT),
    "teacher": ("hf-hub-qa-pilot-glm-53-pi-160-eval", "episodes/teacher/seed.jsonl"),
}
INKLING = (f"inkling-2-mini-8k-{JUDGED}", BASE)
QUALITY_PASS = (7 - 1) / (10 - 1)  # judged_rubrics: PASS_SCORE 7 on the 1-10 scale
REFERENCE_PASS = 2 / 3


def load(source: tuple[str, str] | str) -> list[dict[str, Any]]:
    if isinstance(source, str):
        path = Path(source)
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(f"{NS}/{source[0]}", source[1], repo_type="dataset"))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    first: dict[str, dict[str, Any]] = {}
    for record in records:
        row = record["row_id"]
        if row not in first or (record.get("attempt") or 1) < (first[row].get("attempt") or 1):
            first[row] = record
    return first


def reasons(record: dict[str, Any]) -> list[str]:
    """Every reason a failed episode failed (an episode can miss several bars at once)."""
    failure = str(record.get("failure") or "")
    scores = record.get("scores") or {}
    metrics = record.get("metrics") or {}
    if (record.get("metadata") or {}).get("reasoning_leak"):
        return ["reasoning leak"]
    if "timed out" in failure:
        return ["timeout"]
    if "tool-call limit" in failure:
        return ["call limit"]
    if "truncated" in failure:
        return ["output limit"]
    if failure or not str(record.get("answer") or "").strip():
        return [f"harness/parser error: {(failure or str(record.get('answer_error')))[:60]}"]
    found = [f"gate {g}" for g in ("read_only", "used_hub") if scores.get(g) is not None and scores[g] < 1]
    if scores.get("grounded") is not None and scores["grounded"] < 1:
        found.append("grounded gate")
    if scores.get("reference") is not None and scores["reference"] < REFERENCE_PASS - 1e-9:
        found.append("reference miss")
    quality = scores.get("answer_quality")
    if quality is None:
        found.append("judge error")
    elif quality < QUALITY_PASS - 1e-9:
        criteria = {k.split("/", 1)[1]: v for k, v in metrics.items() if k.startswith("answer_quality/")}
        weakest = min(criteria, key=criteria.get) if criteria else "?"
        found.append(f"judge quality (weakest: {weakest})")
    return found or ["other"]


def describe(records: dict[str, dict[str, Any]], categories: dict[str, str]) -> dict[str, Any]:
    values = list(records.values())
    failed = [r for r in values if not r.get("passed")]
    primary = collections.Counter(reasons(r)[0] for r in failed)
    every = collections.Counter(reason for r in failed for reason in reasons(r))
    per_category: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row, record in records.items():
        cell = per_category[categories.get(row, "?")]
        cell[0] += bool(record.get("passed"))
        cell[1] += 1
    words = [
        len(str(m.get("reasoning_content") or "").split())
        for r in values
        for m in r.get("messages") or []
        if m.get("role") == "assistant"
    ]
    return {
        "n": len(values),
        "passed": sum(bool(r.get("passed")) for r in values),
        "per_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_category.items(), key=lambda kv: -kv[1][1])},
        "failures_primary": dict(primary.most_common()),
        "failures_any": dict(every.most_common()),
        "tool_calls_median": statistics.median(r.get("tool_calls") or 0 for r in values),
        "wall_seconds_mean": round(statistics.mean(r.get("elapsed_seconds") or 0 for r in values), 1),
        "think_words_per_turn_median": statistics.median(words) if words else 0,
        "reasoning_leaks": sum(bool((r.get("metadata") or {}).get("reasoning_leak")) for r in values),
    }


def paired(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(set(a) & set(b))
    only_a = sum(bool(a[r].get("passed")) and not b[r].get("passed") for r in rows)
    only_b = sum(bool(b[r].get("passed")) and not a[r].get("passed") for r in rows)
    return {
        "rows": len(rows),
        "inkling": sum(bool(a[r].get("passed")) for r in rows),
        "other": sum(bool(b[r].get("passed")) for r in rows),
        "inkling_only": only_a,
        "other_only": only_b,
        "mcnemar_exact_p": float(f"{mcnemar_exact(only_a, only_b):.2g}"),
    }


def difficulty(inkling: dict[str, dict[str, Any]], others: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Inkling's pass rate by how many of the Qwen students fail the question (the teacher is left out)."""
    students = [name for name in others if name != "teacher"]
    buckets: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    hard = []
    for row, record in inkling.items():
        fails = sum(not others[s][row].get("passed") for s in students if row in others[s])
        buckets[fails][0] += bool(record.get("passed"))
        buckets[fails][1] += 1
        if not record.get("passed"):
            hard.append(fails)
    return {
        "students": len(students),
        "inkling_passed_by_student_failures": {str(k): f"{v[0]}/{v[1]}" for k, v in sorted(buckets.items())},
        "inkling_failures_student_failures_mean": round(statistics.mean(hard), 2) if hard else None,
        "teacher_also_failed_inkling_failures": sum(
            not others["teacher"][r].get("passed") for r, rec in inkling.items() if not rec.get("passed")
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inkling", default="", help="local episodes JSONL (default: the published eval dataset)")
    args = parser.parse_args(argv)

    from datasets import load_dataset

    categories = {r["question_id"]: r["category"] for r in load_dataset(f"{NS}/hf-hub-questions", split="seed")}
    inkling = first_records(load(args.inkling or INKLING))
    others = {name: first_records(load(source)) for name, source in BASELINES.items()}
    report: dict[str, Any] = {
        "inkling-2-mini-8k": describe(inkling, categories),
        "paired": {name: paired(inkling, other) for name, other in others.items()},
        "difficulty": difficulty(inkling, others),
        "baselines": {name: describe(other, categories) for name, other in others.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
