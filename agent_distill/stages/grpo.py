"""GRPO stage (DESIGN.md §6, §7): the student trains in the real harness, rewarded by the task's rubric set.

    python -m agent_distill.stages.grpo --launch --task task --harness opencode@1.2.4 --student qwen3.5-9b \
        --model ORG/RUN-sft-merged --hub-model-id ORG/RUN-grpo-lora --include-ids qualified-ids.txt

Ported from url-extraction's `train_grpo_harness.py`, with the OpenCode-specific session replaced by any harness
adapter:

    harness (unmodified; one process per rollout, its own workspace, no credentials)
      -> OpenEnv capture proxy (API key = capture session id; records engine token ids and logprobs)
      -> vLLM on GPU 0, serving the policy plus every published adapter version by name
    TRL `HarnessAsyncGRPOTrainer` on GPU 1 (or the synchronous `simple` backend, for students TRL cannot pack)
      <- `TrainingTrace` + the rubric set's train-mode reward (the same rubrics as evaluation; only rubrics tagged
         train_only/eval_only differ)
      -> after each step, loads the new adapter into vLLM by name

vLLM runs from its own `uvx` environment because vLLM and OpenEnv pin incompatible `mcp` majors; the trainer talks
to it over HTTP only. The harness runs with the same tool-call watchdog and timeout as evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent-distill-grpo")

VLLM_VERSION = "0.33.0"
POLICY = "policy"
QWEN35_LANGUAGE_LORA_REGEX = (
    r".*language_model.*\.(down_proj|gate_proj|in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|k_proj|o_proj|out_proj|"
    r"q_proj|up_proj|v_proj)$"
)


# --------------------------------------------------------------------------------------------------
# Capture proxy middleware
# --------------------------------------------------------------------------------------------------


class UnconstrainedToolChoice:
    """ASGI middleware: rewrite a forced `tool_choice` to `"auto"` before the capture proxy sees it.

    OpenCode 1.x forces a tool call on every turn when it requests structured output. A forced choice masks logits,
    so the engine's logprobs would not be the policy's and OpenEnv rightly refuses to train on them. Letting the
    policy choose keeps sampling unconstrained; a turn that answers in prose instead of submitting fails the
    harness's structured-output check and earns what the rubrics give a missing answer.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or not scope.get("path", "").endswith("/chat/completions")
        ):
            await self.app(scope, receive, send)
            return
        chunks = []
        more = True
        while more:
            message = await receive()
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        with suppress(ValueError, TypeError):
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("tool_choice") not in (None, "auto", "none"):
                payload["tool_choice"] = "auto"
                body = json.dumps(payload).encode()
        headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        sent = False

        async def replay() -> dict:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app({**scope, "headers": headers}, replay, send)


# --------------------------------------------------------------------------------------------------
# OpenEnv session: one harness rollout through the capture proxy
# --------------------------------------------------------------------------------------------------


class HarnessCaptureSession:
    """A loop-owning OpenEnv session (`TrainableSession`): the harness drives, the capture proxy records."""

    def __init__(self, factory: HarnessCaptureFactory, row: dict[str, Any], episode_id: str) -> None:
        self.factory = factory
        self.row = row
        self.episode_id = episode_id
        self.capture_id: str | None = None
        self.episode: Any = None
        self.result: Any = None
        self.failure: str | None = None

    @property
    def tool_calls(self) -> int:
        return self.episode.tool_calls if self.episode is not None else 0

    # ResourceSession surface ------------------------------------------------------------------
    def initial_messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self.factory.task.render_prompt(self.row)}]

    def list_tools(self) -> list:
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        from openenv.core.harness import ToolResult

        return ToolResult(error="the harness owns its own tool loop")

    def verify(self, transcript: list[dict[str, Any]], final_state: Any | None = None) -> Any:
        from openenv.core.harness import VerifyResult

        if self.failure and self.failure.startswith("infrastructure:"):
            return VerifyResult(env_reward=None, done=True)
        reward = self.result.reward if self.result is not None else None
        return VerifyResult(env_reward=reward, done=True)

    def close(self) -> None:
        if self.capture_id is not None:
            with suppress(Exception):
                self.factory.capture.registry.delete(self.capture_id)

    # Loop-owning extensions ---------------------------------------------------------------------
    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        try:
            asyncio.run(self._run())
        except Exception as exc:  # infrastructure failures make the rollout unscorable
            self.failure = f"infrastructure: {type(exc).__name__}: {exc}"
            logger.warning("rollout %s failed: %s", self.factory.task.row_id(self.row), self.failure)
            return 1
        # A failed rollout (exit code 1) is dropped as unscorable; an invalid capture from a "successful" one would
        # stop the whole trainer, so validate it now.
        try:
            self.fetch_training_trace()
        except Exception as exc:
            self.failure = f"infrastructure: untrainable capture: {exc}"
            logger.warning("rollout %s dropped: %s", self.factory.task.row_id(self.row), exc)
            return 1
        return 0

    def fetch_training_trace(self) -> Any:
        from openenv.core.harness.capture.contract import to_training_trace
        from openenv.core.harness.capture.export import export_session

        session = self.factory.capture.registry.get(self.capture_id)
        if session is None:
            raise ValueError("capture session disappeared")
        level = session.capture_level or self.factory.capture.capture_level
        return to_training_trace(session.graph, export_session(session, capture_level=level))

    def _mint_capture_session(self) -> str:
        import httpx

        factory = self.factory
        payload: dict[str, Any] = {
            "llm_url": factory.llm_url,
            "model": factory.model,
            "max_model_calls": factory.max_model_calls,
            "purpose": "train",
            "metadata": {"row_id": factory.task.row_id(self.row), "episode_id": self.episode_id},
        }
        if factory.sampling:
            payload["sampling"] = factory.sampling
        response = httpx.post(f"{factory.capture_url}/sessions", json=payload, timeout=120)
        response.raise_for_status()
        body = response.json()
        if body.get("rollout_type") != "train":
            raise RuntimeError(f"capture session is not trainable: {body}")
        return str(body["session_id"])

    async def _run(self) -> None:
        from agent_distill.harnesses.base import Endpoint
        from agent_distill.rubrics import episode_context

        factory = self.factory
        self.capture_id = self._mint_capture_session()
        endpoint = Endpoint(
            base_url=f"{factory.capture_url}/v1",
            model=POLICY,
            api_key=self.capture_id,
            context=factory.max_model_len,
            max_output_tokens=factory.max_output_tokens,
            small_model=True,  # OpenCode 1.x title calls go through the capture too, never to another model
        )
        harness, task = factory.harness, factory.task
        async with harness.session([endpoint], cache=factory.cache, task=task) as runner:
            self.episode = await harness.run_episode(runner, task, self.row, endpoint, factory.limits)
        context = episode_context(self.episode, self.row, limits=factory.limits, task=task, judge=factory.judge)
        self.result = await task.rubrics.ascore(context, mode="train")


class HarnessCaptureFactory:
    """`ResourceSessionFactory` for TRL. Picklable: the task, the harness and the capture proxy load lazily in the
    rollout process (tasks hold functions, which do not pickle)."""

    def __init__(
        self,
        *,
        task_dir: str,
        harness: str,
        profiles: list[str],
        rows: list[dict[str, Any]],
        llm_url: str,
        model: str,
        cache: str,
        timeout: float,
        max_tool_calls: int | None,
        sampling: dict[str, Any] | None = None,
        max_model_calls: int = 24,
        max_model_len: int = 65536,
        max_output_tokens: int = 1024,
        judge: str = "",
        bill_to: str = "",
    ) -> None:
        self.task_dir, self.harness_name, self.profiles = task_dir, harness, profiles
        self.judge_name, self.bill_to = judge, bill_to
        self._judge: Any = None
        self.rows = rows
        self.llm_url = llm_url.rstrip("/")
        self.model = model
        self.cache = Path(cache)
        self.timeout, self.max_tool_calls = timeout, max_tool_calls
        self.sampling = sampling
        self.max_model_calls, self.max_model_len, self.max_output_tokens = (
            max_model_calls,
            max_model_len,
            max_output_tokens,
        )
        self._task = self._harness = self._prompts = self._capture = None
        self._capture_url: str | None = None
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.update(_task=None, _harness=None, _prompts=None, _capture=None, _capture_url=None, _lock=None,
                     _judge=None)  # fmt: skip
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @property
    def task(self) -> Any:
        if self._task is None:
            from agent_distill.task import load_task

            self._task = load_task(self.task_dir)
        return self._task

    @property
    def harness(self) -> Any:
        if self._harness is None:
            from agent_distill.config import load_profile
            from agent_distill.harnesses.base import load_harness

            self._harness = load_harness(load_profile("harnesses", self.harness_name, [Path(p) for p in self.profiles]))
        return self._harness

    @property
    def judge(self) -> Any:
        """The judge runner of the reward's judge rubrics (verdicts cached next to the rollout cache)."""
        with self._lock:
            if self._judge is None and (self.judge_name or self.task.judge):
                from agent_distill.rubrics.judge import make_judge_runner

                profiles = [Path(p) for p in self.profiles]
                name = self.judge_name or self.task.judge
                self._judge = make_judge_runner(name, cache_dir=self.cache, bill_to=self.bill_to, profiles=profiles)
            return self._judge

    @property
    def limits(self) -> Any:
        from agent_distill.task import Limits

        return Limits(timeout=self.timeout, max_tool_calls=self.max_tool_calls)

    @property
    def capture(self) -> Any:
        with self._lock:
            if self._capture is None:
                from openenv.core.harness.capture.runner import CaptureServer

                from agent_distill.harnesses.opencode import free_port

                port = free_port()
                capture = CaptureServer(
                    llm_url=self.llm_url,
                    model=self.model,
                    port=port,
                    max_output_tokens=self.max_output_tokens,
                    max_model_calls=self.max_model_calls,
                )
                capture.app.add_middleware(UnconstrainedToolChoice)
                capture.start()
                self._capture, self._capture_url = capture, f"http://127.0.0.1:{port}"
                logger.warning("capture proxy listening on %s -> %s", self._capture_url, self.llm_url)
        return self._capture

    @property
    def capture_url(self) -> str:
        _ = self.capture
        return str(self._capture_url)

    def prompt_rows(self) -> list[dict[str, Any]]:
        return [
            {"prompt": [{"role": "user", "content": self.task.render_prompt(row)}], "row_id": self.task.row_id(row)}
            for row in self.rows
        ]

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> HarnessCaptureSession:
        if self._prompts is None:
            self._prompts = {self.task.render_prompt(row): row for row in self.rows}
        if isinstance(task, list) and task:
            prompt = str(task[-1].get("content") or "")
        elif isinstance(task, dict):
            prompt = str(task.get("content") or task.get("prompt") or "")
        else:
            prompt = str(task)
        row = self._prompts.get(prompt)
        if row is None:
            raise KeyError("prompt does not match any training row")
        if self.llm_url:  # start the capture proxy before the first rollout (tests build sessions without one)
            _ = self.capture
        return HarnessCaptureSession(self, row, episode_id or f"ep-{random.getrandbits(48):x}")


def make_factory(**kwargs: Any) -> HarnessCaptureFactory:
    return HarnessCaptureFactory(**kwargs)


def rollout_reward(outcome: Any) -> float | None:
    """Module-level so TRL can pickle it: the rubric set's train-mode reward (None: unscorable, dropped)."""
    return None if outcome.env_reward is None else float(outcome.env_reward)


# --------------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------------


def training_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The prompt pool: train rows (optionally only the current probe's qualified ones), without excluded tiers, with
    weighted tiers repeated; refuses a pool with fewer distinct rows than the planned steps draw."""
    from agent_distill.stages.eval import select_rows
    from agent_distill.task import load_task

    task = load_task(args.task)
    rows = select_rows(task, args.split, purpose="train")
    include = include_ids(args)
    if include is not None:
        rows = [row for row in rows if task.row_id(row) in include]
    if args.n_prompts:
        rows = random.Random(args.seed).sample(rows, min(args.n_prompts, len(rows)))
    rows = weight_by_tier(rows, args.exclude_tiers, args.tier_weights, args.seed, field=args.tier_field)
    if not rows:
        raise SystemExit("no training rows")
    check_pool(args, len({task.row_id(row) for row in rows}))
    return rows


def parse_tiers(exclude: str, weights: str) -> tuple[set[int], dict[int, int]]:
    """`--exclude-tiers "7"` and `--tier-weights "3:2,5:3"` (tier:repeat factor), as in url-extraction."""
    drop = {int(tier) for tier in exclude.split(",") if tier.strip()}
    repeat: dict[int, int] = {}
    for item in weights.split(","):
        if not item.strip():
            continue
        tier, sep, factor = item.partition(":")
        if not sep or int(factor) < 1:
            raise SystemExit(f"bad --tier-weights entry {item!r} (expected TIER:REPEATS, repeats >= 1)")
        repeat[int(tier)] = int(factor)
    return drop, repeat


def weight_by_tier(
    rows: list[dict[str, Any]], exclude: str, weights: str, seed: int, *, field: str = "difficulty_tier"
) -> list[dict[str, Any]]:
    """url-extraction's `weight_by_tier`: drop rows of excluded difficulty tiers and repeat rows of weighted tiers so
    RL draws them more often (rows without a tier are kept once). The pool is shuffled when either is set."""
    drop, repeat = parse_tiers(exclude, weights)
    if not drop and not repeat:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        tier = row.get(field)
        if tier is not None and int(tier) in drop:
            continue
        out.extend([row] * repeat.get(int(tier) if tier is not None else -1, 1))
    random.Random(seed).shuffle(out)
    counts: dict[str, int] = {}
    for row in out:
        counts[str(row.get(field))] = counts.get(str(row.get(field)), 0) + 1
    print(f"[grpo] prompt pool by {field} (after weighting): {dict(sorted(counts.items()))}", flush=True)
    return out


def prompts_per_step(args: argparse.Namespace) -> int:
    """Distinct prompts one optimizer step draws: `--prompts-per-step` for the simple backend; for TRL, one prompt
    group per gradient-accumulation micro-step (per-device batch 1)."""
    return args.prompts_per_step if args.trainer == "simple" else args.gradient_accumulation_steps


def check_pool(args: argparse.Namespace, distinct: int) -> None:
    """Refuse a pool too small for the planned steps (steps x prompts per step), which trains on the same rows over
    and over (url-extraction: a 136-row probe pool for 150 steps), unless `--allow-small-pool`."""
    needed = args.max_steps * prompts_per_step(args)
    if distinct >= needed:
        print(f"[grpo] prompt pool: {distinct} distinct rows for {needed} prompt draws", flush=True)
        return
    message = (
        f"the prompt pool has {distinct} distinct rows, but {args.max_steps} steps x {prompts_per_step(args)} "
        f"prompts per step draw {needed}"
    )
    if not args.allow_small_pool:
        raise SystemExit(f"{message}; widen the pool (probe more rows) or pass --allow-small-pool")
    print(f"[grpo] WARNING: {message} (--allow-small-pool: rows repeat)", flush=True)


def include_ids(args: argparse.Namespace) -> set[str] | None:
    """Row ids from `--include-ids` files and/or the current probe's qualified-id files in `--include-ids-repo`
    (exactly the `--include-ids-repo-file` names; other probes' results in the same dataset are never read)."""
    ids: set[str] = set()
    found = False
    for path in args.include_ids or []:
        ids |= {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}
        found = True
    if args.include_ids_repo:
        from huggingface_hub import HfApi, hf_hub_download

        wanted = list(args.include_ids_repo_file or [])
        if not wanted:
            raise SystemExit("--include-ids-repo needs --include-ids-repo-file (the current probe's qualified ids)")
        present = set(HfApi().list_repo_files(args.include_ids_repo, repo_type="dataset"))
        missing = [name for name in wanted if name not in present]
        if missing:
            raise SystemExit(f"difficulty-probe results missing in {args.include_ids_repo}: {missing}")
        for name in wanted:
            text = Path(hf_hub_download(args.include_ids_repo, name, repo_type="dataset")).read_text()
            ids |= {line.strip() for line in text.splitlines() if line.strip()}
        print(f"[grpo] {len(ids)} qualified rows from {len(wanted)} probe files", flush=True)
        found = True
    return ids if found else None


def trace_harnesses(repo: str) -> set[str] | None:
    """`harness@version` of every traces-stage summary in `repo` (None: no repo or no summaries)."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        names = [n for n in HfApi().list_repo_files(repo, repo_type="dataset") if n.startswith("summaries/")]
    except RepositoryNotFoundError:
        return None
    found = set()
    for name in names:
        if name.endswith(".json"):
            harness = json.loads(Path(hf_hub_download(repo, name, repo_type="dataset")).read_text()).get("harness")
            if harness:
                found.add(str(harness))
    return found or None


def check_harness(args: argparse.Namespace) -> None:
    """DESIGN.md §6 "No mixing of versions": GRPO runs the student in the harness its SFT traces came from, unless
    `--allow-harness-mismatch` (which still warns)."""
    if not args.traces_repo:
        return
    found = trace_harnesses(args.traces_repo)
    if found is None:
        print(f"[grpo] WARNING: no traces summaries in {args.traces_repo}; cannot check the harness", flush=True)
        return
    if found == {args.harness}:
        return
    message = f"the SFT traces in {args.traces_repo} came from {sorted(found)}, but GRPO runs in {args.harness}"
    if not args.allow_harness_mismatch:
        raise SystemExit(f"{message}; pass --allow-harness-mismatch to train anyway")
    print(f"[grpo] WARNING: {message} (--allow-harness-mismatch)", flush=True)


# --------------------------------------------------------------------------------------------------
# vLLM and the two-GPU launch
# --------------------------------------------------------------------------------------------------


def vllm_command(args: argparse.Namespace, student: Any) -> list[str]:
    serve = student.serve
    launcher = (
        [args.vllm_binary]
        if args.vllm_binary
        else [shutil.which("uvx") or "uvx", "--python", "3.12", "--from", f"vllm=={args.vllm_version}", "vllm"]
    )
    command = [
        *launcher,
        "serve",
        args.model,
        "--served-model-name",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.vllm_port),
        "--max-model-len",
        str(serve.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        serve.tool_call_parser,
        "--return-tokens-as-token-ids",
        "--logprobs-mode",
        "processed_logprobs",
        "--enable-lora",
        "--max-lora-rank",
        str(args.lora_r),
        "--max-loras",
        str(args.max_staleness + 2),
    ]
    if serve.language_model_only:
        command.append("--language-model-only")
    if args.trainer == "trl":  # TRL's merged-weight fallback path; adapter-only trainers never need it
        command += ["--weight-transfer-config", '{"backend":"nccl"}']
    if serve.reasoning_parser:
        command += ["--reasoning-parser", serve.reasoning_parser]
    command += shlex.split(serve.vllm_extra_args) + shlex.split(args.vllm_extra_args)
    return command


def start_vllm(args: argparse.Namespace, student: Any) -> subprocess.Popen:
    command = vllm_command(args, student)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.vllm_device,
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
    }
    print("[launch] " + shlex.join(command), flush=True)
    return subprocess.Popen(command, env=env, start_new_session=True)


def wait_for_vllm(url: str, process: subprocess.Popen, timeout_s: float = 2400) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with status {process.returncode}")
        with suppress(Exception), urllib.request.urlopen(f"{url}/health", timeout=5) as response:
            if response.status == 200:
                return
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy")


# --------------------------------------------------------------------------------------------------
# TRL backend
# --------------------------------------------------------------------------------------------------


def factory_kwargs(args: argparse.Namespace, student: Any, rows: list[dict[str, Any]], sampling: Any = None) -> dict:
    return {
        "task_dir": str(Path(args.task).resolve()),
        "harness": args.harness,
        "profiles": [str(Path(p).resolve()) for p in args.profiles],
        "rows": rows,
        "llm_url": f"http://127.0.0.1:{args.vllm_port}",
        "model": args.model,
        "cache": str(args.cache),
        "timeout": args.timeout,
        "max_tool_calls": args.max_tool_calls,
        "sampling": sampling,
        "max_model_calls": args.max_model_calls,
        "max_model_len": student.serve.max_model_len,
        "max_output_tokens": args.max_completion_length,
        "judge": args.judge,
        "bill_to": args.bill_to,
    }


def lora_targets(args: argparse.Namespace, student: Any) -> Any:
    value = args.lora_target_modules or student.lora.grpo_target_modules
    return value or QWEN35_LANGUAGE_LORA_REGEX


def relocate_lora_dir(trainer: Any, args: argparse.Namespace) -> None:
    """TRL saves every hot-loaded adapter version under <output_dir>/.vllm_lora, and the checkpoint pushes would
    upload them all; keep them outside the output directory (vLLM reads them by absolute path on this machine).
    `_lora_dir` is TRL's private attribute: refuse to train (and push) if a TRL upgrade renamed it."""
    if not hasattr(trainer, "_lora_dir"):
        raise RuntimeError(
            f"{type(trainer).__name__} has no `_lora_dir` (TRL changed it); its checkpoint pushes would upload every "
            "hot-loaded adapter version. Update agent_distill.stages.grpo.relocate_lora_dir for this TRL version."
        )
    trainer._lora_dir = str(Path(args.scratch_dir or tempfile.gettempdir()).resolve() / "agent-distill-vllm-lora")


def train_trl(args: argparse.Namespace, student: Any) -> None:
    from functools import partial

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import AsyncGRPOConfig
    from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer

    rows = training_rows(args)
    factory = partial(make_factory, **factory_kwargs(args, student, rows))
    dataset = Dataset.from_list(factory().prompt_rows())
    print(f"[grpo] {len(dataset)} rows, model {args.model}, harness {args.harness}", flush=True)
    vllm_url = f"http://127.0.0.1:{args.vllm_port}"
    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        score_centering=False,
        max_concurrent_rollouts=args.max_concurrent_rollouts,
        max_staleness=args.max_staleness,
        vllm_server_base_url=vllm_url,
        rollout_worker_timeout=args.timeout + 300,
        optim="adamw_torch",
        dtype="bfloat16",
        bf16=True,
        activation_checkpointing=True,
        activation_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to,
        run_name=args.run_name,
        log_completions=False,
        logging_steps=1,
        seed=args.seed,
        # Push every checkpoint as it is written (latest under `last-checkpoint/`, resumable).
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
        hub_strategy="checkpoint",
        hub_private_repo=True,
    )
    trainer = HarnessAsyncGRPOTrainer(
        model=args.model,
        harness_session_factory=factory,
        train_dataset=dataset,
        args=config,
        processing_class=AutoTokenizer.from_pretrained(args.model),
        rollout_reward_fn=rollout_reward,
        peft_config=LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, target_modules=lora_targets(args, student)
        ),
    )
    relocate_lora_dir(trainer, args)
    trainer.train()
    trainer.save_model(args.output_dir)
    if args.hub_model_id:
        trainer.model.push_to_hub(args.hub_model_id, private=True)
        print(f"[grpo] pushed the adapter to {args.hub_model_id} (private)", flush=True)


# --------------------------------------------------------------------------------------------------
# Simple synchronous backend (students TRL's tree-packed async trainer cannot run, e.g. Inkling 2)
# --------------------------------------------------------------------------------------------------


def rollout_chains(trace: Any, max_tokens: int) -> tuple[list[tuple[list[int], list[int], list[float]]], int]:
    """Fold a rollout's engine calls into as few forward passes as possible.

    A call whose prompt extends the previous call's prompt + completion continues that sequence; otherwise a new
    sequence starts (e.g. when the harness drops earlier reasoning from the history). Returns
    `(input_ids, trained_positions, behaviour_logprobs)` per sequence, plus the number of trained tokens dropped
    because their sequence exceeds `max_tokens`.
    """
    chains: list[tuple[list[int], list[int], list[float]]] = []
    ids: list[int] = []
    positions: list[int] = []
    old: list[float] = []
    dropped = 0

    def flush() -> None:
        nonlocal dropped
        if positions:
            if len(ids) <= max_tokens:
                chains.append((list(ids), list(positions), list(old)))
            else:
                dropped += len(positions)

    for turn in trace.turns:
        prompt, completion = list(turn.prompt_token_ids), list(turn.completion_token_ids)
        if not (ids and prompt[: len(ids)] == ids):
            flush()
            positions, old = [], []
        ids = prompt + completion
        mask = turn.loss_mask[len(prompt) :]
        for offset, (keep, logprob) in enumerate(zip(mask, turn.per_token_logps, strict=False)):
            if keep:
                positions.append(len(prompt) + offset)
                old.append(float(logprob))
    flush()
    return chains, dropped


def group_advantages(rewards: list[float]) -> list[float] | None:
    """Per-group normalized advantages; None when the group carries no signal (all rewards equal)."""
    if len(rewards) < 2:
        return None
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    if std < 1e-6:
        return None
    return [(value - mean) / (std + 1e-4) for value in rewards]


def train_simple(args: argparse.Namespace, student: Any) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    import torch
    import transformers
    from huggingface_hub import HfApi
    from openenv.core.harness.capture.upstream import training_sampling
    from peft import LoraConfig, get_peft_model

    from agent_distill.hooks import resolve

    rows = training_rows(args)
    sampling = training_sampling({"temperature": args.temperature, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0})
    factory = make_factory(**factory_kwargs(args, student, rows, sampling))
    print(f"[simple] {len(rows)} rows, model {args.model}, output {args.output_dir}", flush=True)

    device = torch.device("cuda")
    if student.sft.train_patch:
        resolve(student.sft.train_patch).enable(chunk_size=args.attention_chunk_size)
    model_class = getattr(transformers, student.sft.model_class or "AutoModelForCausalLM")
    model = model_class.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, target_modules=lora_targets(args, student)
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi() if args.hub_model_id else None
    if api:
        api.create_repo(args.hub_model_id, private=True, exist_ok=True)
    vllm_url = f"http://127.0.0.1:{args.vllm_port}"
    converter = resolve(student.lora.adapter_converter) if student.lora.adapter_converter else None

    def publish_adapter(step: int) -> str:
        """Save the adapter, convert it for vLLM if the profile says so, hot-load it as `policy-v{step}`."""
        raw = output / f"checkpoint-{step}"
        model.save_pretrained(raw)
        served = converter(raw, output / "served" / f"policy-v{step}") if converter else raw
        name = f"policy-v{step}"
        httpx.post(
            f"{vllm_url}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(served)}, timeout=600
        ).raise_for_status()
        stale = f"policy-v{step - (args.max_staleness + 1) * args.sync_every}"
        with suppress(Exception):
            httpx.post(f"{vllm_url}/v1/unload_lora_adapter", json={"lora_name": stale}, timeout=120)
        return name

    def run_rollout(row: dict[str, Any]) -> dict[str, Any] | None:
        session = factory.create([{"role": "user", "content": factory.task.render_prompt(row)}])
        started = time.monotonic()
        try:
            if session.wait_for_completion():
                return None
            trace = session.fetch_training_trace()
            verdict = session.verify([])
            if verdict.env_reward is None:
                return None
            return {
                "reward": float(verdict.env_reward),
                "trace": trace,
                "tool_calls": session.tool_calls,
                "seconds": time.monotonic() - started,
            }
        except Exception as exc:  # unscorable rollouts are dropped, never fatal
            logger.warning("rollout %s dropped: %s", factory.task.row_id(row), exc)
            return None
        finally:
            session.close()

    def logprobs_of(ids: list[int], positions: list[int]) -> Any:
        input_ids = torch.tensor([ids], device=device)
        keep = torch.tensor([p - 1 for p in positions], device=device)
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=keep).logits[0].float()
        if args.temperature != 1.0:
            logits = logits / args.temperature
        targets = torch.tensor([ids[p] for p in positions], device=device)
        return torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None])[:, 0]

    metrics_path = output / "metrics.jsonl"
    cursor = 0
    random.Random(args.seed).shuffle(rows)
    policy = args.model
    with ThreadPoolExecutor(max_workers=args.max_concurrent_rollouts) as pool:
        for step in range(1, args.max_steps + 1):
            started = time.monotonic()
            batch = [rows[(cursor + i) % len(rows)] for i in range(args.prompts_per_step)]
            cursor += args.prompts_per_step
            factory.model = policy
            futures = [
                (i, pool.submit(run_rollout, row)) for i, row in enumerate(batch) for _ in range(args.num_generations)
            ]
            groups: dict[int, list[dict[str, Any]]] = {}
            for index, future in futures:
                result = future.result()
                if result is not None:
                    groups.setdefault(index, []).append(result)
            rewards = [r["reward"] for group in groups.values() for r in group]
            samples: list[tuple[float, list[tuple[list[int], list[int], list[float]]]]] = []
            dropped = 0
            for group in groups.values():
                advantages = group_advantages([r["reward"] for r in group])
                if advantages is None:
                    continue
                for result, advantage in zip(group, advantages, strict=True):
                    chains, lost = rollout_chains(result["trace"], args.max_train_tokens)
                    dropped += lost
                    samples.append((advantage, chains))
            total_tokens = sum(len(positions) for _, chains in samples for _, positions, _ in chains)
            stats = {"ratio_sum": 0.0, "clipped": 0, "loss": 0.0}
            model.train()
            grad_norm = 0.0
            if total_tokens:
                optimizer.zero_grad(set_to_none=True)
                for advantage, chains in samples:
                    for ids, positions, old in chains:
                        offload = (
                            torch.autograd.graph.save_on_cpu(pin_memory=True)
                            if args.offload_activations
                            else nullcontext()
                        )
                        with offload:
                            new = logprobs_of(ids, positions)
                        ratio = torch.exp(new - torch.tensor(old, device=device))
                        clipped = ratio.clamp(1 - args.epsilon, 1 + args.epsilon_high)
                        loss = -torch.minimum(ratio * advantage, clipped * advantage).sum() / total_tokens
                        loss.backward()
                        with torch.no_grad():
                            stats["loss"] += float(loss)
                            stats["ratio_sum"] += float(ratio.sum())
                            stats["clipped"] += int((ratio != clipped).sum())
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                optimizer.step()
                scheduler.step()
            if total_tokens and step % args.sync_every == 0:
                policy = publish_adapter(step)
            finished = [r for group in groups.values() for r in group]
            record = {
                "step": step,
                "reward_mean": sum(rewards) / len(rewards) if rewards else None,
                "rollouts": len(rewards),
                "trained_tokens": total_tokens,
                "dropped_tokens": dropped,
                "importance_ratio_mean": stats["ratio_sum"] / total_tokens if total_tokens else None,
                "clip_fraction": stats["clipped"] / total_tokens if total_tokens else None,
                "loss": stats["loss"],
                "grad_norm": grad_norm,
                "policy": policy,
                "tool_calls_mean": sum(r["tool_calls"] for r in finished) / len(finished) if finished else None,
                "groups_with_signal": sum(
                    1 for group in groups.values() if group_advantages([r["reward"] for r in group]) is not None
                ),
                "step_seconds": round(time.monotonic() - started, 1),
            }
            print("[simple] " + json.dumps(record), flush=True)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if api and step % args.save_steps == 0:
                raw = output / f"checkpoint-{step}"
                if not raw.exists():
                    model.save_pretrained(raw)
                api.upload_folder(
                    repo_id=args.hub_model_id,
                    folder_path=str(raw),
                    path_in_repo=f"checkpoint-{step}",
                    allow_patterns=["adapter_*", "README.md"],
                )
    model.save_pretrained(output)
    if api:
        api.upload_folder(
            repo_id=args.hub_model_id,
            folder_path=str(output),
            allow_patterns=["adapter_*", "README.md", "metrics.jsonl"],
        )
        print(f"[simple] pushed the adapter to {args.hub_model_id} (private)", flush=True)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--launch", action="store_true", help="Start vLLM on --vllm-device, then train on --train-device"
    )
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--profiles", action="append", type=Path, default=[])
    parser.add_argument("--model", required=True, help="Starting policy (the merged SFT checkpoint)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--include-ids", action="append", type=Path, help="Only these row ids (files, one per line)")
    parser.add_argument("--include-ids-repo", default="", help="Dataset holding the difficulty probe's results")
    parser.add_argument("--include-ids-repo-file", action="append",
                        help="A qualified-ids file of the current probe in that dataset (repeatable)")  # fmt: skip
    parser.add_argument("--n-prompts", type=int)
    parser.add_argument("--exclude-tiers", default="", help="Difficulty tiers to drop, e.g. '7'")
    parser.add_argument("--tier-weights", default="", help="Repeat factors per tier, e.g. '3:2,4:2,5:3,6:2'")
    parser.add_argument("--tier-field", default="difficulty_tier", help="Row field holding the difficulty tier")
    parser.add_argument("--allow-small-pool", action="store_true",
                        help="Train even with fewer distinct rows than steps x prompts per step")  # fmt: skip
    parser.add_argument("--traces-repo", default="", help="The SFT traces (their harness@version must match)")
    parser.add_argument("--allow-harness-mismatch", action="store_true")
    parser.add_argument("--trainer", choices=("trl", "simple"), default="trl")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=16)
    parser.add_argument("--max-staleness", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-target-modules", default="")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-model-calls", type=int, default=24)
    parser.add_argument("--timeout", "--episode-timeout", dest="timeout", type=float, default=180.0,
                        help="Episode timeout in seconds (as in evaluation)")  # fmt: skip
    parser.add_argument("--max-tool-calls", type=int, default=15, help="Tool-call limit (as in evaluation)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-device", default="0")
    parser.add_argument("--train-device", default="1")
    parser.add_argument("--vllm-binary", default="")
    parser.add_argument("--vllm-version", default=VLLM_VERSION)
    parser.add_argument("--vllm-extra-args", default="")
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "agent-distill-cache")
    parser.add_argument("--judge", default="", help="Judge profile of the reward's judge rubrics (default: the task's)")
    parser.add_argument("--bill-to", default="", help="HF organization billed for a router judge")
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--output-dir", default="/grpo")
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    # simple backend
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--sync-every", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=0.28)
    parser.add_argument("--max-train-tokens", type=int, default=49152)
    parser.add_argument("--attention-chunk-size", type=int, default=1024)
    parser.add_argument("--offload-activations", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from agent_distill.config import load_profile

    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    student = load_profile("students", args.student, args.profiles)
    if not args.launch:
        (train_simple if args.trainer == "simple" else train_trl)(args, student)
        return
    # Refuse before any GPU time is spent: a harness other than the traces', or a prompt pool too small.
    check_harness(args)
    training_rows(args)
    # Install the pinned harness once, before any rollout starts many at a time.
    from agent_distill.harnesses.base import load_harness

    load_harness(load_profile("harnesses", args.harness, args.profiles)).install(args.cache)
    vllm = start_vllm(args, student)
    try:
        wait_for_vllm(f"http://127.0.0.1:{args.vllm_port}", vllm)
        print("[launch] vLLM ready", flush=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.train_device}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        child = [arg for arg in argv if arg != "--launch"]
        code = subprocess.call([sys.executable, "-m", "agent_distill.stages.grpo", *child], env=env)
        if code:
            raise SystemExit(code)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(vllm.pid), signal.SIGTERM)


if __name__ == "__main__":
    main()
