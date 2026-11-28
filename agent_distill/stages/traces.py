"""Traces stage (DESIGN.md §7): the teacher runs the task in the harness; passing episodes become SFT data.

    python -m agent_distill.stages.traces --task task --harness opencode@1.2.4 --teacher glm-5.2 \
        --split train --shard 0/3 --output-dir /traces --output-repo ORG/RUN-traces --bill-to huggingface

The teacher is served by HF Inference Providers (billed to `--bill-to`) through a key-injecting proxy in this
process: the harness gets a per-episode session key, never the HF token. Rows come from the task's train filter;
each row gets up to `--attempts` episodes and stops at its first pass. An episode cut short by the timeout or a
provider error (no answer, often mid-tool-call) is kept in the records but rerun up to `--rerun-cut-short` times
instead of using an attempt. Records are uploaded every `--upload-every` minutes, and a rerun of the same job
resumes from them. An episode is accepted exactly when the task's rubric set passes it (eval mode), the same verdict
the evaluation and the difficulty probe use. Records and Pi sessions go to a private dataset (public only after
`distill publish`), raw harness logs to a separate always-private `-raw` dataset; spend is tracked from the
harness's cost (or the teacher profile's prices) and the stage stops starting episodes at `--budget-usd`.

With `--label teacher --purpose eval --attempts 1 --no-stop-when-passed` the same stage is the teacher's evaluation
baseline, written next to the students' evaluations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from agent_distill.config import load_profile
from agent_distill.harnesses.base import Endpoint, default_cache, load_harness, teacher_endpoint
from agent_distill.reasoning import LeakGuard
from agent_distill.records import RecordWriter, raw_root_for, read_jsonl, summarize_records
from agent_distill.rubrics import ErrorBudget
from agent_distill.rubrics.judge import make_judge_runner
from agent_distill.stages.common import Plan, progress, resume_from_hub, run_episodes, shard
from agent_distill.stages.eval import (
    VLLM_PORT,
    checkpoint,
    publish,
    refuse_public,
    select_rows,
    stop_process,
    vllm_command,
    wait_for_vllm,
)
from agent_distill.task import Limits, load_task


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--teacher", required=True, help="Teacher profile")
    parser.add_argument("--profiles", action="append", type=Path, default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--purpose", choices=("train", "eval"), default="train", help="Which task filter selects rows")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, help="First N rows in total (split across --shard jobs)")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="", help="i/n: this job's disjoint slice of the rows")
    parser.add_argument("--attempts", type=int, help="Episodes per row (default: the teacher profile's)")
    parser.add_argument("--stop-when-passed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--concurrency", type=int, help="Concurrent episodes (default: the teacher profile's)")
    parser.add_argument("--timeout", type=float, help="Episode timeout (default: the teacher profile's)")
    parser.add_argument("--max-tool-calls", type=int, help="Default: the teacher profile's")
    parser.add_argument("--bill-to", default="", help="HF organization billed for Inference Providers")
    parser.add_argument("--budget-usd", type=float, default=0.0, help="Stop starting episodes at this spend (0: none)")
    parser.add_argument("--label", default="teacher")
    parser.add_argument("--output-dir", type=Path, default=Path("/traces"))
    parser.add_argument("--prefix", default="", help="File prefix (default: <label>/<split>[-<shard>])")
    parser.add_argument("--output-repo", default="")
    parser.add_argument("--judge", help="Judge profile for judge rubrics (default: the task's; '' for none)")
    parser.add_argument("--upload-every", type=float, default=10.0,
                        help="Minutes between checkpoint uploads to --output-repo (0: only at the end)")  # fmt: skip
    parser.add_argument("--rerun-cut-short", type=int, default=2,
                        help="Reruns per row of episodes cut short by a timeout or provider error")  # fmt: skip
    parser.add_argument("--raw-output-dir", type=Path, help="Raw harness logs (default: <output-dir>-raw)")
    parser.add_argument("--raw-repo", default="", help="Always-private raw-log dataset (default: <output-repo>-raw)")
    parser.add_argument(
        "--no-raw-upload",
        action="store_true",
        help="Keep raw harness logs in the job only (e.g. when the account's private storage is full)",
    )
    parser.add_argument("--collection", default="", help="Collection to add a newly created --output-repo to")
    parser.add_argument("--collection-note", default="")
    parser.add_argument("--max-error-rate", type=float, default=0.05,
                        help="Stop once more than this share of episodes (after 20) had a rubric error")  # fmt: skip
    parser.add_argument("--max-reasoning-leak-rate", type=float, default=0.2,
                        help="Stop once more than this share of episodes (after 20) leaked reasoning (</think>) into "
                             "their visible content; negative: never stop, only warn and count")  # fmt: skip
    parser.add_argument("--cache", type=Path, default=None, help="Binary cache (default: the per-user cache)")
    parser.add_argument("--vllm-binary", default="vllm", help="vLLM binary for self-served teacher profiles ([serve])")
    parser.add_argument("--no-serve", action="store_true",
                        help="Use an already running vLLM on port 8000 for a self-served teacher profile")  # fmt: skip
    return parser.parse_args(argv)


def trace_rows(task: Any, args: argparse.Namespace) -> list[dict]:
    """`--start`/`--limit` select the run's rows, then `--shard` splits them: the limit is the total over all shards
    (what the pipeline's cost estimate assumes), not a per-shard count."""
    rows = select_rows(
        task, args.split, start=args.start, limit=args.limit, purpose=args.purpose, sample=args.sample, seed=args.seed
    )
    return shard(rows, args.shard) if args.shard else rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = load_task(args.task)
    teacher = load_profile("teachers", args.teacher, args.profiles)
    profile = load_profile("harnesses", args.harness, args.profiles)
    harness = load_harness(profile)
    settings = teacher.traces
    # The task's limits, extended (never shortened) by the teacher profile's: a slow teacher gets more time.
    limits = Limits(
        timeout=args.timeout or max(settings.episode_timeout, task.limits.timeout),
        max_tool_calls=args.max_tool_calls or max(settings.max_tool_calls, task.limits.max_tool_calls or 0),
    )
    raw_dir = args.raw_output_dir or raw_root_for(args.output_dir)
    raw_repo = "" if args.no_raw_upload else args.raw_repo or (f"{args.output_repo}-raw" if args.output_repo else "")
    upload_raw = None if args.no_raw_upload else raw_dir  # raw logs are still written in the job, never uploaded
    if args.output_repo:
        refuse_public(args.output_repo, raw_repo)
    rows = trace_rows(task, args)
    if not rows:
        raise SystemExit(f"no rows selected from {task.dataset.repo}:{args.split}")
    shard_suffix = f"-{args.shard.replace('/', 'of')}" if args.shard else ""
    prefix = args.prefix or f"{args.label}/{args.split}{shard_suffix}"
    if args.output_repo:
        resume_from_hub(args.output_repo, args.output_dir, prefix, raw_repo=raw_repo, raw_dir=raw_dir)
    writer = RecordWriter(args.output_dir, prefix, raw_dir)
    price = (teacher.price.input, teacher.price.output) if teacher.price.input or teacher.price.output else None
    plan = Plan(
        label=args.label,
        attempts=args.attempts or settings.attempts_per_row,
        stop_when_passed=args.stop_when_passed,
        budget_usd=args.budget_usd,
        price=price,
        rerun_cut_short=args.rerun_cut_short,
        judge=make_judge_runner(
            task.judge if args.judge is None else args.judge,
            cache_dir=args.output_dir,
            bill_to=args.bill_to,
            profiles=args.profiles,
        ),
        extra={"split": args.split, "teacher": teacher.name},
    )
    server = None
    if teacher.serve is not None:
        serve = teacher.serve
        endpoint = Endpoint(
            f"http://127.0.0.1:{VLLM_PORT}/v1",
            teacher.model,
            name=args.label,
            context=serve.max_model_len,
            max_output_tokens=serve.max_output_tokens,
        )
        print(
            f"{len(rows)} rows of {task.dataset.repo}:{args.split}; self-served teacher {teacher.model}; "
            f"harness {profile.id}",
            flush=True,
        )
        if not args.no_serve:
            command = vllm_command(teacher, teacher.model, "", args.vllm_binary)
            print(f"starting vLLM: {shlex.join(command)}", flush=True)
            server = subprocess.Popen(command, start_new_session=True)
    else:
        endpoint = teacher_endpoint(teacher, profile, args.bill_to)
        print(
            f"{len(rows)} rows of {task.dataset.repo}:{args.split}; teacher {teacher.routed_model}; harness "
            f"{profile.id}; billed to {args.bill_to or 'the token owner'}",
            flush=True,
        )
    try:
        if server is not None:
            wait_for_vllm(server)
        asyncio.run(
            run_episodes(
                harness,
                task,
                rows,
                endpoint,
                writer,
                plan,
                limits=limits,
                concurrency=args.concurrency or settings.concurrency,
                cache=args.cache or default_cache(),
                checkpoint=checkpoint(
                    args.output_dir,
                    args.output_repo,
                    args.upload_every,
                    profile.id,
                    task.name,
                    raw_dir=upload_raw,
                    raw_repo=raw_repo,
                    collection=args.collection,
                    note=args.collection_note,
                ),
                errors=ErrorBudget(max_rate=args.max_error_rate),
                leaks=LeakGuard(max_rate=args.max_reasoning_leak_rate),
            )
        )
    finally:
        if server is not None:
            stop_process(server)
        finish(args, task, teacher, profile, rows, limits, writer, prefix, raw_dir, raw_repo)


def finish(args, task, teacher, profile, rows, limits, writer, prefix, raw_dir, raw_repo) -> None:
    """Write the summary and publish whatever finished, also when the stage failed part-way."""
    records = read_jsonl(writer.episodes) if writer.episodes.exists() else []
    _, passed, spent = progress(writer, args.label, args.rerun_cut_short)
    summary = summarize_records(records)
    summary.update(
        harness=profile.id,
        teacher=teacher.model,
        split=args.split,
        accepted_rows=len(passed),
        rows=len(rows),
        spent_usd=round(spent, 4),
        limits={"timeout": limits.timeout, "max_tool_calls": limits.max_tool_calls},
    )
    summary_path = args.output_dir / "summaries" / f"{prefix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"accepted {len(passed)}/{len(rows)} rows, spent ${spent:.2f}", flush=True)
    if args.output_repo and writer.episodes.exists():
        upload_raw = None if args.no_raw_upload else raw_dir
        publish(args.output_dir, args.output_repo, profile.id, task.name, raw_dir=upload_raw, raw_repo=raw_repo,
                collection=args.collection, note=args.collection_note)  # fmt: skip


if __name__ == "__main__":
    main()
