"""Analyse a model's evaluation on hf-hub-facts-v2 against published Qwen3.5-4B, Inkling-2-Mini and GLM-5.3 records.

`--model` picks the evaluated model (base Inkling-2-Mini by default, or base LFM3-1.7B with thinking on or off); every
other published record in MODELS and BASELINES is a comparison. First record per row everywhere (the students never
rerun; the MCP teacher's reruns are ignored). Paired comparisons are exact McNemar tests on the rows both records
cover. Prints a JSON report.

    uv run --extra hub python examples/hf-hub-qa-mcp/analyze_inkling.py --task mcp
    uv run --extra hub python examples/hf-hub-qa-mcp/analyze_inkling.py --task cli --model lfm3-1.7b
    uv run --extra hub python examples/hf-hub-qa-mcp/analyze_inkling.py --task mcp --model lfm3-1.7b --episodes <jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from typing import Any

from agent_distill.records import mcnemar_exact

NS = "nielsr"
# Base models evaluated out of the box (eval-only runs: examples/hf-hub-qa*/run-inkling*.toml, run-lfm3*.toml).
MODELS = {
    name: {task: (f"{run}-{'hf-hub-qa-mcp' if task == 'mcp' else 'hf-hub-qa'}-v2-glm-53-pi-160-eval",
                  "episodes/base/test.jsonl") for task in ("mcp", "cli")}
    for name, run in {"inkling-2-mini": "inkling-2-mini", "inkling-2-mini-8k": "inkling-2-mini-8k",
                      "lfm3-1.7b": "lfm3-17b", "lfm3-1.7b-nothink": "lfm3-17b-nothink"}.items()
}  # fmt: skip
BASELINES = {
    "mcp": {
        "qwen-base-nothink": ("qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160-eval", "episodes/base-nothink/test.jsonl"),
        "qwen-base-think": ("qwen35-4b-think-hf-hub-qa-mcp-v2-glm-53-pi-160-eval", "episodes/base-think/test.jsonl"),
        "qwen-recipe-a": ("qwen35-4b-think-cfx-a-hf-hub-qa-mcp-v2-glm-53-pi-160-eval", "episodes/sft/test.jsonl"),
        "teacher": ("qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160-eval", "episodes/teacher/test.jsonl"),
    },
    "cli": {
        "qwen-base-nothink": ("qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-grpo-eval", "episodes/base/test.jsonl"),
        "qwen-base-think": ("qwen35-4b-think-hf-hub-qa-v2-glm-53-pi-160-eval", "episodes/base-think/test.jsonl"),
        "qwen-recipe-a": ("qwen35-4b-think-cfx-a-hf-hub-qa-v2-glm-53-pi-160-eval", "episodes/sft/test.jsonl"),
        "qwen-fullcot": ("cursor-qwen35-4b-think-hf-hub-qa-v2-glm-53-pi-160-eval", "episodes/sft/test.jsonl"),
        "teacher": ("qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-grpo-eval", "episodes/teacher/test.jsonl"),
    },
}
THINK_TAG = re.compile(r"</?think>|<\|[a-z_/]+\|>")  # think tags or chat-template special tokens (e.g. Inkling's)
# A tool call the server's tool-call parser did not turn into a structured call (Qwen3-Coder XML, used by Qwen3.5 and
# LFM3, or a bare JSON call), left in the visible text.
RAW_TOOL_CALL = re.compile(r"<tool_call>|<function=|<parameter=|\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"")


def load(source: tuple[str, str] | str) -> list[dict[str, Any]]:
    if isinstance(source, str):
        path = Path(source)
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(f"{NS}/{source[0]}", source[1], repo_type="dataset"))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The first record per row (lowest attempt, then file order)."""
    first: dict[str, dict[str, Any]] = {}
    for record in records:
        row = record["row_id"]
        if row not in first or (record.get("attempt") or 1) < (first[row].get("attempt") or 1):
            first[row] = record
    return first


def reasoning_words(record: dict[str, Any]) -> list[int]:
    """Words of reasoning per assistant turn."""
    return [
        len(str(m.get("reasoning_content") or "").split())
        for m in record.get("messages") or []
        if m.get("role") == "assistant"
    ]


def unparsed_turns(record: dict[str, Any]) -> int:
    """Assistant turns whose visible content (or reasoning) holds a raw, unparsed tool call."""
    return sum(
        bool(RAW_TOOL_CALL.search(str(m.get("content") or "")))
        for m in record.get("messages") or []
        if m.get("role") == "assistant"
    )


UNKNOWN_TOOL = re.compile(r"^Tool \S+ not found")  # Pi's reply to a call of a tool the agent does not have


def unknown_tool_calls(record: dict[str, Any]) -> int:
    """Calls of tools the agent was not given (e.g. a skill's name called as a tool, or an invented MCP tool)."""
    return sum(
        bool(UNKNOWN_TOOL.match(str(m.get("content") or "")))
        for m in record.get("messages") or []
        if m.get("role") == "tool"
    )


def score(scores: dict[str, Any], key: str, default: float = 1) -> float:
    """A rubric score, with unscored (None) treated as the default."""
    value = scores.get(key)
    return default if value is None else value


def failure_mode(record: dict[str, Any], task: str) -> str:
    """One mode per failed episode, in order of precedence."""
    failure = str(record.get("failure") or "")
    scores = record.get("scores") or {}
    if (record.get("metadata") or {}).get("reasoning_leak") or THINK_TAG.search(str(record.get("answer") or "")):
        return "reasoning leaked into answer"
    if "timed out" in failure:
        return "timeout"
    if "tool-call limit" in failure:
        return "call limit"
    if "truncated" in failure:
        return "output token limit"
    if failure or record.get("answer_error"):  # provider, parser or harness errors (every other cut-short episode)
        return f"harness/parser error: {(failure or str(record.get('answer_error')))[:60]}"
    if RAW_TOOL_CALL.search(str(record.get("answer") or "")) or (
        not record.get("tool_calls") and unparsed_turns(record)
    ):
        return "tool call not parsed (raw call in text)"
    if task == "cli" and not score(scores, "read_skill") and score(scores, "fact_correct") < 1:
        return "wrong fact, skill not read"
    if scores.get("fact_correct", 0) is None:  # the live Hub no longer matches the snapshot's gold
        return "label drifted (unscorable)"
    for gate in ("answered", "read_only", "allowed_tools", "used_hub"):
        if score(scores, gate) < 1:
            return f"gate {gate}"
    if score(scores, "fact_correct") < 1:
        return "wrong fact"
    if score(scores, "grounded") < 1:
        return "ungrounded"
    return "other"


def describe(records: dict[str, dict[str, Any]], templates: dict[str, str], task: str) -> dict[str, Any]:
    values = list(records.values())
    passed = sum(bool(r.get("passed")) for r in values)
    per_template: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row, record in records.items():
        cell = per_template[templates.get(row, "?")]
        cell[0] += bool(record.get("passed"))
        cell[1] += 1
    modes = collections.Counter(failure_mode(r, task) for r in values if not r.get("passed"))
    words = [reasoning_words(r) for r in values]
    turns = [w for episode in words for w in episode]
    report: dict[str, Any] = {
        "n": len(values),
        "passed": passed,
        "pass_rate": round(passed / max(1, len(values)), 4),
        "fact_correct": sum(score(r.get("scores") or {}, "fact_correct", 0) >= 1 for r in values),
        "per_template": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_template.items(), key=lambda kv: -kv[1][1])},
        "failure_modes": dict(modes.most_common()),
        "tool_calls_mean": round(statistics.mean(r.get("tool_calls") or 0 for r in values), 2),
        "wall_seconds_mean": round(statistics.mean(r.get("elapsed_seconds") or 0 for r in values), 1),
        "wall_seconds_median": round(statistics.median(r.get("elapsed_seconds") or 0 for r in values), 1),
        "think_words_per_turn_median": statistics.median(turns) if turns else 0,
        "think_words_per_turn_mean": round(statistics.mean(turns), 1) if turns else 0,
        "think_words_per_episode_median": statistics.median(sum(w) for w in words) if words else 0,
        "turns_without_reasoning": sum(w == 0 for w in turns),
        "output_tokens_mean": round(statistics.mean((r.get("usage") or {}).get("output") or 0 for r in values), 1),
        "reasoning_leaks": sum(bool((r.get("metadata") or {}).get("reasoning_leak")) for r in values),
        "think_tags_in_answer": sum(bool(THINK_TAG.search(str(r.get("answer") or ""))) for r in values),
        "episodes_with_unparsed_tool_calls": sum(unparsed_turns(r) > 0 for r in values),
        "episodes_with_reasoning": sum(sum(w) > 0 for w in words),
        "unknown_tool_calls_per_episode": round(statistics.mean(unknown_tool_calls(r) for r in values), 2),
        "episodes_with_unknown_tool_calls": sum(unknown_tool_calls(r) > 0 for r in values),
    }
    if task == "cli":
        report["read_skill"] = sum(score(r.get("scores") or {}, "read_skill", 0) >= 1 for r in values)
        report["hf_errors_per_episode"] = round(
            statistics.mean(score(r.get("scores") or {}, "hf_errors", 0) for r in values), 2
        )
    else:
        calls = sum(score(r.get("scores") or {}, "mcp_calls", 0) for r in values)
        errors = sum(score(r.get("scores") or {}, "mcp_errors", 0) for r in values)
        report["mcp_calls_per_episode"] = round(calls / max(1, len(values)), 2)
        report["failed_mcp_calls"] = round(errors / max(1, calls), 3)
    return report


def paired(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(set(a) & set(b))
    only_a = sum(bool(a[r].get("passed")) and not b[r].get("passed") for r in rows)
    only_b = sum(bool(b[r].get("passed")) and not a[r].get("passed") for r in rows)
    return {
        "rows": len(rows),
        "model": sum(bool(a[r].get("passed")) for r in rows),
        "other": sum(bool(b[r].get("passed")) for r in rows),
        "model_only": only_a,
        "other_only": only_b,
        "mcnemar_exact_p": float(f"{mcnemar_exact(only_a, only_b):.2g}"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mcp", "cli"], required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="inkling-2-mini", help="the evaluated model")
    parser.add_argument("--episodes", "--inkling", default="",
                        help="local episodes JSONL of --model (default: its published eval dataset)")  # fmt: skip
    parser.add_argument("--passed", nargs="*", default=[], help="row ids a rescore turned into passes")
    args = parser.parse_args(argv)

    from datasets import load_dataset

    templates = {r["question_id"]: r["template"] for r in load_dataset(f"{NS}/hf-hub-facts-v2", split="test")}
    model = first_records(load(args.episodes or MODELS[args.model][args.task]))
    for row in args.passed:  # e.g. after `distill rescore` with a fixed rubric (the published records keep old scores)
        model[row] = {**model[row], "passed": True}
    report: dict[str, Any] = {args.model: describe(model, templates, args.task), "paired": {}, "baselines": {}}
    others = {**BASELINES[args.task], **{name: sources[args.task] for name, sources in MODELS.items()}}
    for name, source in others.items():
        if name == args.model:
            continue
        try:
            other = first_records(load(source))
        except Exception as error:  # a model not evaluated (or published) yet
            report["baselines"][name] = f"unavailable: {type(error).__name__}"
            continue
        report["baselines"][name] = describe(other, templates, args.task)
        report["paired"][name] = paired(model, other)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
