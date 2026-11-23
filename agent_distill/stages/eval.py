"""Evaluation stage (DESIGN.md §7): a student served by vLLM, evaluated in the run's harness, on HF Jobs.

    python -m agent_distill.stages.eval --task task --harness opencode@1.2.4 --student qwen3.5-9b \
        --model Qwen/Qwen3.5-9B --label base --split test --limit 50 --output-dir /eval --output-repo ORG/RUN-eval

Serves `--model` (and, with `--adapter`, a LoRA on top of it; both labels run on the same rows), runs every
selected row in the pinned harness with credentials stripped from the agent's environment, scores each episode with
the task's rubric set (eval mode), and writes records, raw logs, Pi sessions and a summary, published to a private
dataset (the publish stage makes it public after a credential scan). An episode whose harness broke (e.g. its
shell could not spawn) is not recorded; the harness restarts and the row is run again. Ported from url-extraction's
`evaluate_vllm.py`.
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
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent_distill.config import Student, load_profile
from agent_distill.harnesses.base import Endpoint, load_harness
from agent_distill.harnesses.opencode import harness_failure
from agent_distill.records import RecordWriter, read_jsonl, summarize_records
from agent_distill.rubrics import episode_context
from agent_distill.task import Limits, Task, load_task

VLLM_PORT = 8000
ADAPTER_NAME = "adapter"


# --------------------------------------------------------------------------------------------------
# vLLM
# --------------------------------------------------------------------------------------------------


def vllm_command(student: Student, model: str, adapter_path: str, vllm_binary: str = "vllm") -> list[str]:
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


def select_rows(task: Task, split: str, *, start: int, limit: int, row_ids: list[str] | None = None) -> list[dict]:
    from datasets import load_dataset

    spec = task.dataset
    rows = [dict(row) for row in load_dataset(spec.repo, split=split, revision=spec.revision)]
    if spec.filter is not None:
        rows = [row for row in rows if spec.filter(row)]
    if row_ids:
        wanted = set(row_ids)
        rows = [row for row in rows if task.row_id(row) in wanted]
        missing = wanted - {task.row_id(row) for row in rows}
        if missing:
            raise ValueError(f"rows not found (or filtered out) in {spec.repo}:{split}: {sorted(missing)}")
        return rows
    return rows[start : start + limit]


async def run_label(
    harness: Any,
    task: Task,
    rows: list[dict],
    endpoint: Endpoint,
    label: str,
    writer: RecordWriter,
    *,
    limits: Limits,
    concurrency: int,
    cache: Path,
    max_restarts: int = 3,
    extra: dict[str, Any] | None = None,
) -> None:
    """Run every row not yet recorded for `label`, restarting the harness after a harness failure."""
    for restart in range(max_restarts + 1):
        done = writer.done()
        pending = [row for row in rows if (label, task.row_id(row)) not in done]
        if not pending:
            return
        if restart:
            print(f"[{label}] restarting the harness after a harness failure ({len(pending)} rows left)", flush=True)
        broken = await _run_once(harness, task, pending, endpoint, label, writer, limits, concurrency, cache, extra)
        if not broken:
            return
    raise RuntimeError(f"the harness kept failing for {label} after {max_restarts} restarts")


async def _run_once(
    harness: Any,
    task: Task,
    pending: list[dict],
    endpoint: Endpoint,
    label: str,
    writer: RecordWriter,
    limits: Limits,
    concurrency: int,
    cache: Path,
    extra: dict[str, Any] | None,
) -> bool:
    """Run `pending` on one fresh harness. Returns True when the harness broke (the rest is left for a restart)."""
    broken = asyncio.Event()
    semaphore = asyncio.Semaphore(concurrency)
    async with harness.session([endpoint], cache=cache, task=task) as runner:

        async def run_row(index: int, row: dict) -> None:
            async with semaphore:
                if broken.is_set():
                    return
                episode = await harness.run_episode(runner, task, row, endpoint, limits)
                if failure := harness_failure(episode):
                    print(f"  [{label}] {episode.row_id}: harness failure, not recorded: {failure}", flush=True)
                    broken.set()
                    return
                context = episode_context(episode, row, limits=limits, task=task)
                result = await task.rubrics.ascore(context, mode="eval")
                writer.write(episode, result, harness.native_trace(episode), label=label, **(extra or {}))
                print(
                    f"[{label} {index}/{len(pending)}] {episode.row_id} passed={result.passed} "
                    f"reward={result.reward} tools={episode.tool_calls} failure={episode.failure}",
                    flush=True,
                )

        await asyncio.gather(*(run_row(i, row) for i, row in enumerate(pending, 1)))
        if not broken.is_set():
            await harness.healthy(runner)
    return broken.is_set()


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
        "- `raw/`: the harness's own messages per episode.\n"
        "- `summaries/`: evaluation summaries.\n\n"
        "Credential-shaped strings are redacted in every file.\n"
    )


def publish(output_dir: Path, repo: str, harness_id: str, task: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)  # public only after `distill publish`
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    if "README.md" not in files:
        (output_dir / "README.md").write_text(dataset_card(repo, harness_id, task))
    api.upload_folder(folder_path=str(output_dir), repo_id=repo, repo_type="dataset",
                      commit_message=f"agent-distill evaluation ({harness_id})")  # fmt: skip
    print(f"published {output_dir} to {repo} (private)", flush=True)


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
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tool-calls", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("/eval"))
    parser.add_argument("--prefix", default="", help="File prefix, e.g. 'base/test' (default: <label>/<split>)")
    parser.add_argument("--output-repo", default="", help="Dataset to publish to (private); '' keeps files local")
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "agent-distill-cache")
    parser.add_argument("--vllm-binary", default="vllm")
    parser.add_argument("--no-serve", action="store_true", help="Use an already running vLLM on port 8000")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = load_task(args.task)
    student = load_profile("students", args.student, args.profiles)
    profile = load_profile("harnesses", args.harness, args.profiles)
    harness = load_harness(profile)
    limits = Limits(timeout=args.timeout, max_tool_calls=args.max_tool_calls)
    rows = select_rows(
        task, args.split, start=args.start, limit=args.limit, row_ids=[r for r in args.row_ids.split(",") if r] or None
    )
    if not rows:
        raise SystemExit(f"no rows selected from {task.dataset.repo}:{args.split}")
    print(f"{len(rows)} rows of {task.dataset.repo}:{args.split}; harness {profile.id}; model {args.model}", flush=True)

    adapter_path = ""
    if args.adapter:
        from huggingface_hub import snapshot_download

        snapshot = args.adapter if Path(args.adapter).is_dir() else snapshot_download(args.adapter)
        adapter_path = prepare_text_adapter(snapshot, args.cache / "text-adapter", student.lora.adapter_converter)
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

    server = None
    if not args.no_serve:
        command = vllm_command(student, args.model, adapter_path, args.vllm_binary)
        print(f"starting vLLM: {shlex.join(command)}", flush=True)
        server = subprocess.Popen(command, start_new_session=True)
    try:
        if server is not None:
            wait_for_vllm(server)
        for label, endpoint in targets:
            prefix = args.prefix or f"{label}/{args.split}"
            writer = RecordWriter(args.output_dir, prefix)
            extra = {"split": args.split, "served_model": args.model if label == args.label else args.adapter}
            asyncio.run(
                run_label(
                    harness,
                    task,
                    rows,
                    endpoint,
                    label,
                    writer,
                    limits=limits,
                    concurrency=args.concurrency,
                    cache=args.cache,
                    extra=extra,
                )
            )
            summary = summarize_records(read_jsonl(writer.episodes))
            summary.update(harness=profile.id, model=args.model, adapter=args.adapter, split=args.split,
                           limits={"timeout": limits.timeout, "max_tool_calls": limits.max_tool_calls})  # fmt: skip
            summary_path = args.output_dir / "summaries" / f"{prefix}.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(json.dumps(summary["labels"], indent=2), flush=True)
    finally:
        if server is not None:
            stop_process(server)
    if args.output_repo:
        publish(args.output_dir, args.output_repo, profile.id, task.name)


if __name__ == "__main__":
    main()
