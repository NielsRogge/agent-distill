"""Evaluation stage (DESIGN.md §7): a student served by vLLM, evaluated in the run's harness, on HF Jobs.

    python -m agent_distill.stages.eval --task task --harness opencode@1.2.4 --student qwen3.5-9b \
        --model Qwen/Qwen3.5-9B --label base --split test --limit 50 --output-dir /eval --output-repo ORG/RUN-eval

Serves `--model` (and, with `--adapter`, a LoRA on top of it; both labels run on the same rows), runs every
selected row in the pinned harness with credentials stripped from the agent's environment, scores each episode with
the task's rubric set (eval mode), and writes records, Pi sessions and a summary, published to a private dataset
(the publish stage makes it public after a credential scan), and the raw harness logs, published to a separate
always-private `-raw` dataset. Whatever finished is published even when the job fails part-way. An episode whose
harness broke (its shell could not spawn, its server died) is not recorded; the harness restarts and the row is run
again. Ported from url-extraction's `evaluate_vllm.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent_distill.config import Student, load_profile
from agent_distill.harnesses.base import Endpoint, default_cache, load_harness
from agent_distill.records import RecordWriter, raw_root_for, read_jsonl, summarize_records
from agent_distill.rubrics import ErrorBudget
from agent_distill.rubrics.judge import make_judge_runner
from agent_distill.stages.common import Checkpoint, Plan, resume_from_hub, run_episodes, shard
from agent_distill.task import Limits, Task, load_task

VLLM_PORT = 8000
ADAPTER_NAME = "adapter"


# --------------------------------------------------------------------------------------------------
# vLLM
# --------------------------------------------------------------------------------------------------


def vllm_command(
    student: Student, model: str, adapter_path: str, vllm_binary: str = "vllm", *, temperature: float | None = None
) -> list[str]:
    serve = student.serve
    command = [
        shutil.which(vllm_binary) or vllm_binary, "serve", model, "--host", "127.0.0.1", "--port", str(VLLM_PORT),
        "--max-model-len", str(serve.max_model_len), "--gpu-memory-utilization", str(serve.gpu_memory_utilization),
        "--enable-auto-tool-choice", "--tool-call-parser", serve.tool_call_parser,
    ]  # fmt: skip
    if serve.enforce_eager:
        command.append("--enforce-eager")
    if serve.language_model_only:
        command.append("--language-model-only")
    if serve.reasoning_parser:
        command += ["--reasoning-parser", serve.reasoning_parser]
    if adapter_path:
        command += ["--enable-lora", "--max-lora-rank", str(student.lora.rank),
                    "--lora-modules", f"{ADAPTER_NAME}={adapter_path}"]  # fmt: skip
    if temperature is not None:  # the probe samples like the GRPO rollouts; harnesses send no temperature of their own
        command += ["--override-generation-config", json.dumps({"temperature": temperature, "top_p": 1.0, "top_k": -1})]
    command += shlex.split(serve.vllm_extra_args)
    return command


def wait_for_vllm(process: subprocess.Popen, timeout: float = 2400) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with status {process.returncode}")
        with suppress(urllib.error.URLError, TimeoutError, ConnectionError):
            with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5) as response:
                if response.status == 200:
                    return
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy within the startup timeout")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def prepare_text_adapter(source: str, destination: Path, converter: str = "") -> str:
    """Make a trained adapter servable by text-only vLLM (url-extraction `prepare_text_adapter`).

    With `converter` (`module:function(source, destination)` from the student profile) that function rewrites the
    adapter. Otherwise inert vision-tower tensors, which text-only vLLM rejects, are removed.
    """
    from safetensors.torch import load_file, save_file

    source_path = Path(source)
    if converter:
        from agent_distill.hooks import resolve

        return str(resolve(converter)(source_path, destination))
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_path, destination, ignore=shutil.ignore_patterns("adapter_model.safetensors"))
    tensors = load_file(source_path / "adapter_model.safetensors", device="cpu")
    text_tensors = {name: tensor for name, tensor in tensors.items() if ".visual." not in name}
    removed = set(tensors) - set(text_tensors)
    if not text_tensors:
        raise RuntimeError("Expected text LoRA tensors in the trained adapter")
    nonzero_visual_b = [name for name in removed if ".lora_B." in name and bool(tensors[name].count_nonzero().item())]
    if nonzero_visual_b:
        raise RuntimeError("Refusing to remove trained visual LoRA tensors")
    save_file(text_tensors, destination / "adapter_model.safetensors")
    config_path = destination / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["target_modules"] = sorted({name.split(".")[-3] for name in text_tensors})
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return str(destination)


# --------------------------------------------------------------------------------------------------
# Rows and episodes
# --------------------------------------------------------------------------------------------------


def select_rows(
    task: Task,
    split: str,
    *,
    start: int = 0,
    limit: int | None = None,
    row_ids: list[str] | None = None,
    purpose: str = "eval",
    sample: int = 0,
    seed: int = 0,
    shard_spec: str = "",
) -> list[dict]:
    """Rows of a split after the task's filter for `purpose` ("eval", or "train": traces, probe, GRPO); then a
    seeded sample, a shard, and `start`/`limit` (or explicit row ids)."""
    import random

    from datasets import load_dataset

    spec = task.dataset
    rows = [dict(row) for row in load_dataset(spec.repo, split=split, revision=spec.revision)]
    keep = spec.train_filter if purpose == "train" and spec.train_filter is not None else spec.filter
    if keep is not None:
        rows = [row for row in rows if keep(row)]
    if sample:
        rows = random.Random(seed).sample(sorted(rows, key=task.row_id), min(sample, len(rows)))
    if shard_spec:
        rows = shard(rows, shard_spec)
    if row_ids:
        wanted = set(row_ids)
        rows = [row for row in rows if task.row_id(row) in wanted]
        missing = wanted - {task.row_id(row) for row in rows}
        if missing:
            raise ValueError(f"rows not found (or filtered out) in {spec.repo}:{split}: {sorted(missing)}")
        return rows
    return rows[start : start + limit] if limit is not None else rows[start:]


def difficulty(records: list[dict[str, Any]], attempts: int) -> dict[str, Any]:
    """The difficulty probe's result: rows passed on some but not all of `attempts` (the GRPO prompt pool; GRPO
    learns nothing from rows the policy always or never solves)."""
    from collections import defaultdict

    passes: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        passes[str(record["row_id"])].append(bool(record.get("passed")))
    complete = {row: values[:attempts] for row, values in passes.items() if len(values) >= attempts}
    histogram = {str(k): sum(sum(v) == k for v in complete.values()) for k in range(attempts + 1)}
    qualified = sorted(row for row, values in complete.items() if 0 < sum(values) < attempts)
    return {"rows": len(complete), "attempts": attempts, "passes_histogram": histogram, "qualified": qualified}


# --------------------------------------------------------------------------------------------------
# Publishing
# --------------------------------------------------------------------------------------------------


def dataset_card(repo: str, harness_id: str, task: str) -> str:
    return (
        "---\n"
        f"tags: [agent-traces, agent-distill, {harness_id.split('@')[0]}, {harness_id.replace('@', '-')}]\n"
        "configs:\n"
        "- config_name: sessions\n  default: true\n  data_files:\n  - split: train\n    path: sessions/**/*.jsonl\n"
        "- config_name: episodes\n  data_files:\n  - split: train\n    path: episodes/**/*.jsonl\n"
        "---\n\n"
        f"# {repo.split('/')[-1]}\n\n"
        f"Episodes of the agent-distill task `{task}` in `{harness_id}`.\n\n"
        "- `sessions/`: one Pi-format session per episode, opened by the Hub's "
        "[agent-traces viewer](https://huggingface.co/docs/hub/en/agent-traces) (default config).\n"
        "- `episodes/`: one record per episode (answer, rubric scores, reward, tool events, messages).\n"
        "- `summaries/`: evaluation summaries.\n\n"
        "The harness's own raw messages are kept in a separate private dataset for rescoring; they are never "
        "published. Credential-shaped strings are redacted in every file.\n"
    )


def private_dataset(api: Any, repo: str) -> bool:
    """Create `repo` as a private dataset, or check that the existing one is private. Returns True when created.

    `create_repo(private=True, exist_ok=True)` leaves an existing public repo public, so unscanned episodes would be
    uploaded straight into a public dataset: refuse instead.
    """
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.repo_info(repo, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        info, created = api.repo_info(repo, repo_type="dataset"), True
    else:
        created = False
    if not info.private:
        raise SystemExit(
            f"{repo} is public: refusing to upload unscanned episodes into it. Use a new repo name, or make it "
            "private and run `distill run ... --stages publish` after the upload to scan and publish it again."
        )
    return created


def publish(
    output_dir: Path,
    repo: str,
    harness_id: str,
    task: str,
    *,
    raw_dir: Path | None = None,
    raw_repo: str = "",
    collection: str = "",
    note: str = "",
) -> None:
    """Upload the episodes, sessions and summaries to the private `repo` (public only after `distill publish`) and
    the raw logs to the always-private `raw_repo`. A newly created `repo` is added to `collection` at once."""
    from huggingface_hub import HfApi

    api = HfApi()
    if raw_dir is not None and raw_dir.exists() and any(raw_dir.rglob("*.jsonl")):
        if not raw_repo:
            raise SystemExit("raw harness logs need a private --raw-repo; they never go into the episode dataset")
        private_dataset(api, raw_repo)
        api.upload_folder(folder_path=str(raw_dir), repo_id=raw_repo, repo_type="dataset",
                          commit_message=f"agent-distill raw harness logs ({harness_id})")  # fmt: skip
        print(f"published raw logs to {raw_repo} (private, never published)", flush=True)
    created = private_dataset(api, repo)
    if created and collection:
        api.add_collection_item(collection, item_id=repo, item_type="dataset", note=note or f"{task}, {harness_id}",
                                exists_ok=True)  # fmt: skip
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    if "README.md" not in files:
        (output_dir / "README.md").write_text(dataset_card(repo, harness_id, task))
    api.upload_folder(folder_path=str(output_dir), repo_id=repo, repo_type="dataset", ignore_patterns=["raw/**"],
                      commit_message=f"agent-distill evaluation ({harness_id})")  # fmt: skip
    print(f"published {output_dir} to {repo} (private)", flush=True)


def checkpoint(
    output_dir: Path,
    repo: str,
    minutes: float,
    harness_id: str,
    task: str,
    *,
    raw_dir: Path | None = None,
    raw_repo: str = "",
    collection: str = "",
    note: str = "",
) -> Checkpoint | None:
    """Periodic uploads of the records to `repo` and of the raw logs to the private `raw_repo`."""
    if not repo or minutes <= 0:
        return None

    def upload(folder: Path, raw_folder: Path | None) -> None:
        publish(folder, repo, harness_id, task, raw_dir=raw_folder, raw_repo=raw_repo, collection=collection, note=note)

    return Checkpoint(output_dir, upload, minutes * 60, raw_dir=raw_dir)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--harness", required=True, help="Harness profile, e.g. opencode@1.2.4")
    parser.add_argument("--student", required=True, help="Student profile (serving flags)")
    parser.add_argument("--profiles", action="append", type=Path, default=[], help="Extra profile folders")
    parser.add_argument("--model", required=True, help="Model to serve (Hub id or path)")
    parser.add_argument("--label", default="base", help="Label of the served model's episodes")
    parser.add_argument("--adapter", default="", help="LoRA adapter (Hub id or dir) served on top of --model")
    parser.add_argument("--adapter-label", default="adapter")
    parser.add_argument("--adapter-only", action="store_true", help="Skip the served model without the adapter")
    parser.add_argument("--split", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--row-ids", default="", help="Comma-separated row ids (overrides --start/--limit)")
    parser.add_argument("--purpose", choices=("eval", "train"), default="eval", help="Which task filter selects rows")
    parser.add_argument("--sample", type=int, default=0, help="Seeded random sample of this many rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="", help="i/n: this job's disjoint slice of the rows")
    parser.add_argument("--attempts", type=int, default=1, help="Episodes per row (difficulty probe: K)")
    parser.add_argument("--temperature", type=float, help="Serve with this sampling temperature (probe)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-error-rate", type=float, default=0.05,
                        help="Stop once more than this share of episodes (after 20) had a rubric error")  # fmt: skip
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tool-calls", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("/eval"))
    parser.add_argument("--prefix", default="", help="File prefix, e.g. 'base/test' (default: <label>/<split>)")
    parser.add_argument("--output-repo", default="", help="Dataset to publish to (private); '' keeps files local")
    parser.add_argument("--judge", help="Judge profile for judge rubrics (default: the task's; '' for none)")
    parser.add_argument("--bill-to", default="", help="HF organization billed for a router judge")
    parser.add_argument("--upload-every", type=float, default=10.0,
                        help="Minutes between checkpoint uploads to --output-repo (0: only at the end)")  # fmt: skip
    parser.add_argument("--raw-output-dir", type=Path, help="Raw harness logs (default: <output-dir>-raw)")
    parser.add_argument("--raw-repo", default="", help="Always-private raw-log dataset (default: <output-repo>-raw)")
    parser.add_argument("--collection", default="", help="Collection to add a newly created --output-repo to")
    parser.add_argument("--collection-note", default="")
    parser.add_argument("--answer-retry", action="store_true",
                        help="OpenCode 2.x: ask once for the JSON answer when an episode ends without one")  # fmt: skip
    parser.add_argument("--cache", type=Path, default=None, help="Binary cache (default: the per-user cache)")
    parser.add_argument("--vllm-binary", default="vllm")
    parser.add_argument("--no-serve", action="store_true", help="Use an already running vLLM on port 8000")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = load_task(args.task)
    student = load_profile("students", args.student, args.profiles)
    profile = load_profile("harnesses", args.harness, args.profiles)
    harness = load_harness(profile, **({"answer_retry": True} if args.answer_retry else {}))
    cache = args.cache or default_cache()
    raw_dir = args.raw_output_dir or raw_root_for(args.output_dir)
    raw_repo = args.raw_repo or (f"{args.output_repo}-raw" if args.output_repo else "")
    limits = Limits(timeout=args.timeout, max_tool_calls=args.max_tool_calls)
    rows = select_rows(
        task,
        args.split,
        start=args.start,
        limit=args.limit,
        row_ids=[r for r in args.row_ids.split(",") if r] or None,
        purpose=args.purpose,
        sample=args.sample,
        seed=args.seed,
        shard_spec=args.shard,
    )
    if not rows:
        raise SystemExit(f"no rows selected from {task.dataset.repo}:{args.split}")
    print(f"{len(rows)} rows of {task.dataset.repo}:{args.split}; harness {profile.id}; model {args.model}", flush=True)

    adapter_path = ""
    if args.adapter:
        from huggingface_hub import snapshot_download

        snapshot = args.adapter if Path(args.adapter).is_dir() else snapshot_download(args.adapter)
        adapter_path = prepare_text_adapter(snapshot, cache / "text-adapter", student.lora.adapter_converter)
    serve = student.serve
    base_url = f"http://127.0.0.1:{VLLM_PORT}/v1"
    targets: list[tuple[str, Endpoint]] = []
    if not (args.adapter and args.adapter_only):
        endpoint = Endpoint(base_url, args.model, name=args.label, context=serve.max_model_len,
                            max_output_tokens=serve.max_output_tokens)  # fmt: skip
        targets.append((args.label, endpoint))
    if adapter_path:
        endpoint = Endpoint(base_url, ADAPTER_NAME, name=args.adapter_label, context=serve.max_model_len,
                            max_output_tokens=serve.max_output_tokens)  # fmt: skip
        targets.append((args.adapter_label, endpoint))

    judge_name = task.judge if args.judge is None else args.judge
    judge = make_judge_runner(judge_name, cache_dir=args.output_dir, bill_to=args.bill_to, profiles=args.profiles)
    server = None
    if not args.no_serve:
        command = vllm_command(student, args.model, adapter_path, args.vllm_binary, temperature=args.temperature)
        print(f"starting vLLM: {shlex.join(command)}", flush=True)
        server = subprocess.Popen(command, start_new_session=True)
    try:
        if server is not None:
            wait_for_vllm(server)
        errors = ErrorBudget(max_rate=args.max_error_rate)
        for label, endpoint in targets:
            prefix = args.prefix or f"{label}/{args.split}"
            if args.output_repo:
                resume_from_hub(args.output_repo, args.output_dir, prefix, raw_repo=raw_repo, raw_dir=raw_dir)
            writer = RecordWriter(args.output_dir, prefix, raw_dir)
            extra = {"split": args.split, "served_model": args.model if label == args.label else args.adapter}
            plan = Plan(label=label, attempts=args.attempts, extra=extra, judge=judge)
            asyncio.run(
                run_episodes(
                    harness,
                    task,
                    rows,
                    endpoint,
                    writer,
                    plan,
                    limits=limits,
                    concurrency=args.concurrency,
                    cache=cache,
                    checkpoint=checkpoint(
                        args.output_dir,
                        args.output_repo,
                        args.upload_every,
                        profile.id,
                        task.name,
                        raw_dir=raw_dir,
                        raw_repo=raw_repo,
                        collection=args.collection,
                        note=args.collection_note,
                    ),
                    errors=errors,
                )
            )
            summary = summarize_records(read_jsonl(writer.episodes))
            summary.update(harness=profile.id, model=args.model, adapter=args.adapter, split=args.split,
                           limits={"timeout": limits.timeout, "max_tool_calls": limits.max_tool_calls})  # fmt: skip
            summary_path = args.output_dir / "summaries" / f"{prefix}.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if args.attempts > 1:
                summary["difficulty"] = difficulty(read_jsonl(writer.episodes), args.attempts)
                qualified = args.output_dir / "difficulty" / f"{prefix}-qualified-ids.txt"
                qualified.parent.mkdir(parents=True, exist_ok=True)
                qualified.write_text("".join(f"{row_id}\n" for row_id in summary["difficulty"]["qualified"]))
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(json.dumps(summary["labels"], indent=2), flush=True)
    finally:
        if server is not None:
            stop_process(server)
        # Publish whatever finished, also when a label failed part-way (the error still ends the job).
        if args.output_repo and any((args.output_dir / "episodes").rglob("*.jsonl")):
            publish(args.output_dir, args.output_repo, profile.id, task.name, raw_dir=raw_dir, raw_repo=raw_repo,
                    collection=args.collection, note=args.collection_note)  # fmt: skip


if __name__ == "__main__":
    main()
