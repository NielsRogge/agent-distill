"""Follow-up 2: rewrite open-ended teacher answers to be strictly tool-grounded and re-score with `judged_rubrics.py`.

Reads accepted (`passed=True`, `tool_calls <= max_tool_calls`) episodes from an open-ended traces dataset, keeps every
tool call and tool output unchanged, rewrites ONLY the final assistant answer via HF Inference Providers (GLM-5.3,
billed to `--bill-to`) so every claim is directly supported by the episode's own tool output, re-scores each rewrite
with `open_ended.py`'s rubrics (`judged_rubrics.py` + `gemini-4-hydrogen`), and publishes passing rewrites through
`RecordWriter` + `agent_distill.stages.eval.publish` (Pi sessions + episodes, private until `distill publish`).

Usage:
    env -u UV_PRERELEASE uv run --extra hub python examples/hf-hub-qa/rewrite_grounded_traces.py \\
        --source-repo nielsr/cursor-qwen35-4b-think-hf-hub-qa-open-ended-glm-53-pi-160-traces \\
        --output-repo nielsr/cursor-qwen35-4b-think-hf-hub-qa-open-ended-grounded-glm-53-pi-160-traces \\
        --bill-to <org> --collection nielsr/agent-distillation-6b0445568a39b1d979c52e31
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, InferenceClient, hf_hub_download  # pragma: allowlist secret

from agent_distill.records import RecordWriter, episode_from_record_v2, summarize_records
from agent_distill.rubrics import episode_context
from agent_distill.rubrics.judge import make_judge_runner
from agent_distill.stages.eval import publish as publish_episodes
from agent_distill.task import load_task

HERE = Path(__file__).parent
TOOL_OUT_LIMIT = 4000
TOTAL_LOG_LIMIT = 28000

REWRITE_SYSTEM = """You rewrite final answers for a Hugging Face Hub CLI assistant so that they are 100% tool-grounded.

Strict rules:
1. Every repo id, paper id, number, date, license, download/like count, tag, architecture, author, and description in
   your answer MUST appear directly in the tool outputs below.
2. Remove every claim, description, token count, parameter count, lab affiliation, or extra repo/paper id that is NOT
   present in the tool outputs (even if true from general knowledge).
3. If a tool listing only shows repo ids and basic metadata (likes, downloads, pipeline_tag, tags, license, created_at),
   report ONLY those observed fields — do not invent a "what it does" description for repos whose cards were not read.
4. Preserve all relevant observed repo/paper ids and observed facts so the answer stays complete, accurate, concise,
   and directly answers the user's question.
5. Output ONLY the final answer text (no preamble or explanation of your edits)."""


def format_tool_outputs(tool_events: list[Any]) -> str:
    parts: list[str] = []
    for ev in tool_events:
        inp = getattr(ev, "input", None) or {}
        cmd = str(inp.get("command") or inp.get("path") or json.dumps(inp)[:300])
        if "SKILL.md" in cmd:
            continue
        out = str(getattr(ev, "output", "") or getattr(ev, "error", "") or "")
        if len(out) > TOOL_OUT_LIMIT:
            out = out[:TOOL_OUT_LIMIT] + f"\n... [{len(out) - TOOL_OUT_LIMIT} chars truncated]"
        parts.append(f"$ [{getattr(ev, 'name', '')}] {cmd}\n{out}")
    text = "\n\n".join(parts)
    return text if len(text) <= TOTAL_LOG_LIMIT else text[:TOTAL_LOG_LIMIT] + "\n... [truncated]"


def rewrite_answer(
    client: InferenceClient,
    model: str,
    question: str,
    tool_text: str,
    original_answer: str,
    feedback: str = "",
) -> str:
    import time

    user_msg = (
        f"Question:\n{question}\n\n"
        f"Tool calls and outputs from this session:\n{tool_text}\n\n"
        f"Original answer (may contain unverified parametric claims to remove):\n{original_answer}\n\n"
        "Rewrite the final answer following the strict grounding rules."
    )
    if feedback:
        user_msg += f"\n\nFix these issues from the previous draft:\n{feedback}"
    for attempt in range(4):
        try:
            resp = client.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as exc:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
            print(f"rewrite retry {attempt + 1} after {type(exc).__name__}: {exc}", flush=True)
    return original_answer


def replace_final_assistant_answer(messages: list[dict[str, Any]], new_answer: str) -> list[dict[str, Any]]:
    updated = copy.deepcopy(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") == "assistant" and not updated[idx].get("tool_calls"):
            updated[idx]["content"] = new_answer
            updated[idx].pop("reasoning_content", None)
            return updated
    raise ValueError("no final assistant text message found in episode")


def load_source_records(source_repo: str, max_tool_calls: int) -> list[dict[str, Any]]:
    api = HfApi()
    files = sorted(
        f
        for f in api.list_repo_files(source_repo, repo_type="dataset")
        if f.startswith("episodes/") and f.endswith(".jsonl")
    )
    by_row: dict[str, dict[str, Any]] = {}
    for name in files:
        path = hf_hub_download(source_repo, name, repo_type="dataset")
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("passed"):
                by_row.setdefault(str(rec["row_id"]), rec)
    kept = [
        rec for rec in by_row.values() if (rec.get("tool_calls") or len(rec.get("tool_events") or [])) <= max_tool_calls
    ]
    return sorted(kept, key=lambda r: str(r["row_id"]))


def load_question_rows(dataset_repo: str) -> dict[str, dict[str, Any]]:
    path = hf_hub_download(dataset_repo, "data/train.jsonl", repo_type="dataset")
    return {
        str(row["question_id"]): row
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--questions-repo", default="nielsr/hf-hub-questions-train")
    parser.add_argument("--model", default="zai-org/GLM-5.3")
    parser.add_argument("--bill-to", default="huggingface")  # pragma: allowlist secret
    parser.add_argument("--collection", default="")
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--min-correctness", type=float, default=0.77)  # >= 8/10
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task = load_task(HERE / "open_ended.py")
    records = load_source_records(args.source_repo, args.max_tool_calls)
    if args.limit:
        records = records[: args.limit]
    qrows = load_question_rows(args.questions_repo)
    print(
        f"Loaded {len(records)} source traces (<= {args.max_tool_calls} tool calls) from {args.source_repo}",
        flush=True,
    )
    if args.dry_run:
        return

    client = InferenceClient(bill_to=args.bill_to or None, timeout=180)
    cache_dir = Path(tempfile.mkdtemp(prefix="rewrite-grounded-cache-"))
    judge = make_judge_runner(task.judge, cache_dir=cache_dir, bill_to=args.bill_to or "")

    def process_one(rec: dict[str, Any]) -> tuple[str, Any, Any, dict[str, Any], bool]:
        row_id = str(rec["row_id"])
        row = qrows[row_id]
        ep = episode_from_record_v2(rec, task.answer)
        tool_text = format_tool_outputs(ep.tool_events)
        orig_answer = str(rec.get("answer") or "")
        feedback = ""
        best: tuple[Any, Any, bool] | None = None
        for attempt in range(1, 3):
            rewritten = rewrite_answer(client, args.model, row["question"], tool_text, orig_answer, feedback)
            ep_new = copy.deepcopy(ep)
            ep_new.messages = replace_final_assistant_answer(ep.messages, rewritten)
            ep_new.answer = rewritten
            ep_new.metadata = {**(ep.metadata or {}), "rewritten_grounded": True, "rewrite_attempt": attempt}
            ctx = episode_context(ep_new, row, limits=task.limits, task=task, judge=judge)
            result = task.rubrics.score(ctx, mode="eval")
            corr = result.metrics.get("answer_quality/correctness") or 0.0
            ok = bool(result.passed) and corr >= args.min_correctness - 1e-9
            best = (ep_new, result, ok)
            if ok:
                break
            issues = []
            if result.scores.get("grounded") not in (None, 1.0):
                issues.append(
                    f"grounded={result.scores.get('grounded'):.2f}: "
                    "remove any repo/paper id not verbatim in tool outputs"
                )
            if result.scores.get("reference") is not None and result.scores["reference"] < 2 / 3 - 1e-9:
                issues.append(
                    f"reference={result.scores['reference']:.2f}: include the top matching repo ids from tool outputs"
                )
            if (result.scores.get("answer_quality") or 0) < 2 / 3 - 1e-9 or corr < args.min_correctness - 1e-9:
                issues.append(
                    f"answer_quality={result.scores.get('answer_quality')}, correctness={corr:.2f}: "
                    "remove every claim not explicitly in tool outputs while keeping observed details complete"
                )
            feedback = "; ".join(issues) or "make the answer more strictly grounded in the tool outputs"
        assert best is not None
        return row_id, best[0], best[1], rec, best[2]

    out_dir = Path(tempfile.mkdtemp(prefix="rewrite-grounded-out-"))
    writer = RecordWriter(out_dir, "teacher/train")
    kept = 0
    total = len(records)
    results_all: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(process_one, rec) for rec in records]
        for idx, fut in enumerate(as_completed(futs), 1):
            _row_id, ep_new, res, orig_rec, ok = fut.result()
            if ok:
                kept += 1
            extra = {
                "split": orig_rec.get("split", "train"),
                "label": "teacher",
                "teacher": orig_rec.get("teacher", "glm-5.3"),
                "attempt": 1,
                "cut_short": None,
            }
            # Keep passed=True only when both rubric pass and strict correctness floor are met.
            if not ok:
                res.passed = False
            written = writer.write(ep_new, res, **extra)
            results_all.append(written)
            if idx % 20 == 0 or idx == total:
                print(f"[{idx}/{total}] kept={kept} ({kept / idx:.1%})", flush=True)

    summary = summarize_records(results_all)
    summary["source_repo"] = args.source_repo
    summary["rewritten_grounded"] = True
    summary["kept_grounded"] = kept
    summary_path = out_dir / "summaries" / "teacher" / "train.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    publish_episodes(
        out_dir,
        args.output_repo,
        "pi@1.6.0",
        task.name,
        collection=args.collection,
        note=(
            "hf-hub-qa-open-ended, glm-5.3 -> qwen3.5-4b-think, pi@1.6.0: "
            "strictly tool-grounded rewritten teacher traces"
        ),
    )
    print(f"DONE: {kept}/{total} strictly grounded traces published privately to {args.output_repo}", flush=True)


if __name__ == "__main__":
    main()
