"""Traces stage (DESIGN.md §7): the teacher runs the task in the harness; passing episodes become SFT data.

    python -m agent_distill.stages.traces --task task --harness opencode@1.2.4 --teacher glm-5.2 \
        --split train --shard 0/3 --output-dir /traces --output-repo ORG/RUN-traces --bill-to huggingface

The teacher is served by HF Inference Providers (billed to `--bill-to`); its harness process gets the HF token and
nothing else. Rows come from the task's train filter; each row gets up to `--attempts` episodes and stops at its
first pass. An episode cut short by the timeout or a provider error (no answer, often mid-tool-call) is kept in
the records but rerun up to `--rerun-cut-short` times instead of using an attempt. Records are uploaded every
`--upload-every` minutes, and a rerun of the same job resumes from them. An episode is accepted exactly when the
task's rubric set passes it (eval mode), the same verdict the evaluation and the difficulty probe use. Records, raw
logs and Pi sessions go to a private dataset; spend is tracked from the harness's cost (or the teacher profile's
prices) and the stage stops starting episodes at `--budget-usd`.

With `--label teacher --purpose eval --attempts 1 --no-stop-when-passed` the same stage is the teacher's evaluation
baseline, written next to the students' evaluations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from agent_distill.config import load_profile
from agent_distill.harnesses.base import load_harness, teacher_endpoint
from agent_distill.records import RecordWriter, read_jsonl, summarize_records
from agent_distill.rubrics.judge import make_judge_runner
from agent_distill.stages.common import Plan, progress, resume_from_hub, run_episodes
from agent_distill.stages.eval import checkpoint, publish, select_rows
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
    parser.add_argument("--limit", type=int)
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
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "agent-distill-cache")
    return parser.parse_args(argv)


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
    rows = select_rows(
        task,
        args.split,
        start=args.start,
        limit=args.limit,
        purpose=args.purpose,
        sample=args.sample,
        seed=args.seed,
        shard_spec=args.shard,
    )
    if not rows:
        raise SystemExit(f"no rows selected from {task.dataset.repo}:{args.split}")
    shard_suffix = f"-{args.shard.replace('/', 'of')}" if args.shard else ""
    prefix = args.prefix or f"{args.label}/{args.split}{shard_suffix}"
    if args.output_repo:
        resume_from_hub(args.output_repo, args.output_dir, prefix)
    writer = RecordWriter(args.output_dir, prefix)
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
    endpoint = teacher_endpoint(teacher, profile, args.bill_to)
    print(
        f"{len(rows)} rows of {task.dataset.repo}:{args.split}; teacher {teacher.routed_model}; harness "
        f"{profile.id}; billed to {args.bill_to or 'the token owner'}",
        flush=True,
    )
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
            cache=args.cache,
            checkpoint=checkpoint(args.output_dir, args.output_repo, args.upload_every, profile.id, task.name),
        )
    )
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
    if args.output_repo:
        publish(args.output_dir, args.output_repo, profile.id, task.name)


if __name__ == "__main__":
    main()
