"""`distill run`: a run.toml's stages as HF Jobs (DESIGN.md §7).

    distill run examples/paper-urls/run.toml --stages eval --dry-run
    distill run examples/paper-urls/run.toml --stages eval --eval-model base=Qwen/Qwen3.5-9B --splits test
    distill run examples/paper-urls/run.toml --stages publish

Stages: `traces` (teacher episodes, CPU jobs), `sft`, `merge` (SFT adapter into the student), `probe` (difficulty
probe of the merged SFT policy), `eval` (base / sft / grpo / teacher, or explicit models) and `publish`; `grpo`
arrives in M3b. With several stages
each waits for the previous one's jobs; a single stage runs detached unless `--wait`. `--dry-run` prints every job
(image, flavor, command) and the cost estimate without staging code or launching anything. Every job's worst-case
cost (all jobs running to their timeout) is checked against `[budget]` before launch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_distill import jobs
from agent_distill.config import BUILTIN_PROFILES, UV_IMAGE, RunConfig, load_profile, load_run, slugify
from agent_distill.task import Task, load_task

STAGES = ("traces", "sft", "merge", "probe", "grpo", "eval", "publish")
IMPLEMENTED = ("traces", "sft", "merge", "probe", "eval", "publish")
# Expected minutes per episode-slot for the cost estimate (url-extraction evaluations: about 1 min per episode).
EVAL_MINUTES_PER_EPISODE = 1.5
VLLM_STARTUP_MINUTES = 15


@dataclass
class Run:
    config: RunConfig
    task: Task
    task_dir: Path
    name: str
    namespace: str

    def repo(self, suffix: str) -> str:
        return self.config.repos.get(suffix) or f"{self.namespace}/{self.name}-{suffix}"


def make_run(config: RunConfig) -> Run:
    task_dir = config.path(config.task)
    task = load_task(task_dir)
    harness = load_profile("harnesses", config.harness, config.profile_paths)
    parts = [config.student, task.name, config.teacher, harness.id]
    name = config.name or slugify("-".join(part for part in parts if part))
    namespace = config.hub.namespace
    if not namespace:
        from huggingface_hub import whoami

        namespace = whoami()["name"]
    return Run(config, task, task_dir if task_dir.is_dir() else task_dir.parent, name, namespace)


def profile_files(run: Run) -> dict[str, Path]:
    """The student, harness and teacher profile files a job needs (staged under profiles/<kind>/)."""
    files = {}
    kinds = [("students", run.config.student), ("harnesses", run.config.harness)]
    if run.config.teacher:
        kinds.append(("teachers", run.config.teacher))
    for kind, name in kinds:
        for folder in [*(p / kind for p in run.config.profile_paths), BUILTIN_PROFILES / kind]:
            if (folder / f"{name}.toml").exists():
                files[f"profiles/{kind}/{name}.toml"] = folder / f"{name}.toml"
                break
    return files


# --------------------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------------------


# Stage environments (url-extraction's pinned PEP 723 dependencies); a student profile's `deps` replace them.
SFT_DEPS = [
    "accelerate==1.14.0",
    "bitsandbytes>=0.49.0",
    "datasets>=4.0.0",
    "peft==0.20.0",
    "torch>=2.7.0",
    "transformers==5.14.1",
    "trl==1.9.2",
]
MERGE_DEPS = ["accelerate==1.14.0", "peft==0.20.0", "safetensors>=0.5.0", "torch>=2.7.0", "transformers==5.14.1"]
TRACE_MINUTES_PER_EPISODE = 1.5


def student_profile(run: Run) -> Any:
    return load_profile("students", run.config.student, run.config.profile_paths)


def teacher_profile(run: Run) -> Any:
    if not run.config.teacher:
        raise SystemExit("this stage needs a teacher (`teacher = ...` in the run.toml)")
    return load_profile("teachers", run.config.teacher, run.config.profile_paths)


def harness_id(run: Run) -> str:
    return load_profile("harnesses", run.config.harness, run.config.profile_paths).id


def common_args(run: Run) -> list[str]:
    return [
        "--task",
        "/work/task",
        "--harness",
        jobs.q(harness_id(run)),
        "--profiles",
        "/work/profiles",
        "--cache",
        "/work/cache",
    ]


def eval_models(run: Run) -> dict[str, str]:
    """label -> "model[+adapter]": explicit `[eval] models`, else the run's variants."""
    settings = run.config.eval
    if settings.models:
        return dict(settings.models)
    student = student_profile(run)
    known = {
        "base": student.model,
        "sft": f"{student.model}+{run.repo('lora')}",
        "grpo": f"{run.repo('sft-merged')}+{run.repo('grpo-lora')}",
    }
    return {variant: known[variant] for variant in settings.variants if variant in known}


def served_job(
    run: Run,
    rev: str,
    stage: str,
    name: str,
    model: str,
    adapter: str,
    label: str,
    args: list[str],
    expected_minutes: float,
) -> jobs.Job:
    """A job serving `model` (+ `adapter`) with vLLM and running the eval stage's episode loop."""
    student = student_profile(run)
    serve = student.serve
    deps = list(serve.deps)
    vllm_binary: list[str] = []
    if serve.vllm == "image":
        vllm_binary = ["--vllm-binary", '"$(command -v vllm)"']
    else:
        deps.append(f"vllm=={serve.vllm_version}")
    full = [*common_args(run), "--student", jobs.q(run.config.student), "--model", jobs.q(model), *vllm_binary, *args]
    if adapter:
        full += ["--adapter", jobs.q(adapter), "--adapter-label", jobs.q(label), "--adapter-only"]
    else:
        full += ["--label", jobs.q(label)]
    command = "\n".join(
        [
            jobs.prelude(run.repo("code"), rev, run.task.job_setup),
            jobs.python_module("agent_distill.stages.eval", full, deps=deps),
            f"echo {stage.upper()}_DONE",
        ]
    )
    return jobs.Job(
        stage,
        name[:90],
        serve.image,
        student.probe.flavor or serve.flavor if stage == "probe" else serve.flavor,
        (run.config.probe.timeout or student.probe.timeout) if stage == "probe" else serve.timeout,
        command,
        expected_minutes=expected_minutes,
    )


def eval_jobs(run: Run, rev: str) -> list[jobs.Job]:
    settings = run.config.eval
    built = []
    for label, spec in eval_models(run).items():
        model, _, adapter = spec.partition("+")
        for split in settings.splits:
            args = [
                "--split",
                jobs.q(split),
                "--limit",
                str(settings.limit),
                "--concurrency",
                str(settings.concurrency),
                "--timeout",
                str(settings.timeout),
                "--max-tool-calls",
                str(settings.max_tool_calls),
                "--output-dir",
                "/eval",
                "--output-repo",
                jobs.q(run.repo("eval")),
            ]
            minutes = VLLM_STARTUP_MINUTES + settings.limit * EVAL_MINUTES_PER_EPISODE / max(1, settings.concurrency)
            name = f"{run.name}-eval-{slugify(label)}-{slugify(split)}"
            built.append(served_job(run, rev, "eval", name, model, adapter, label, args, minutes))
    if "teacher" in settings.variants and not settings.models:
        built += [teacher_eval_job(run, rev, split) for split in settings.splits]
    return built


def teacher_job(run: Run, rev: str, stage: str, name: str, args: list[str], minutes: float) -> jobs.Job:
    teacher = teacher_profile(run)
    full = [*common_args(run), "--teacher", jobs.q(run.config.teacher), *args]
    if run.config.hub.bill_to:
        full += ["--bill-to", jobs.q(run.config.hub.bill_to)]
    command = "\n".join(
        [
            jobs.prelude(run.repo("code"), rev, run.task.job_setup),
            jobs.python_module("agent_distill.stages.traces", full),
            f"echo {stage.upper()}_DONE",
        ]
    )
    return jobs.Job(
        stage, name[:90], UV_IMAGE, teacher.traces.flavor, teacher.traces.timeout, command, expected_minutes=minutes
    )


def teacher_eval_job(run: Run, rev: str, split: str) -> jobs.Job:
    """The teacher's baseline under the students' evaluation settings (one attempt, same limits and rows)."""
    settings = run.config.eval
    teacher = teacher_profile(run)
    args = [
        "--split",
        jobs.q(split),
        "--purpose",
        "eval",
        "--limit",
        str(settings.limit),
        "--attempts",
        "1",
        "--no-stop-when-passed",
        "--concurrency",
        str(teacher.traces.baseline_concurrency),
        "--timeout",
        str(settings.timeout),
        "--max-tool-calls",
        str(settings.max_tool_calls),
        "--label",
        "teacher",
        "--output-dir",
        "/eval",
        "--output-repo",
        jobs.q(run.repo("eval")),
    ]
    if run.config.budget.traces_usd:
        args += ["--budget-usd", str(run.config.budget.traces_usd)]
    minutes = settings.limit * TRACE_MINUTES_PER_EPISODE / max(1, teacher.traces.baseline_concurrency)
    return teacher_job(run, rev, "eval", f"{run.name}-eval-teacher-{slugify(split)}", args, minutes)


def trace_jobs(run: Run, rev: str) -> list[jobs.Job]:
    teacher = teacher_profile(run)
    settings = run.config.traces
    shards = settings.shards or teacher.traces.shards
    budget = run.config.budget.traces_usd
    built = []
    for index in range(shards):
        args = [
            "--split",
            jobs.q(settings.split),
            "--shard",
            f"{index}/{shards}",
            "--output-dir",
            "/traces",
            "--output-repo",
            jobs.q(run.repo("traces")),
        ]
        if settings.limit:
            args += ["--limit", str(settings.limit)]
        if settings.attempts:
            args += ["--attempts", str(settings.attempts)]
        if budget:
            args += ["--budget-usd", f"{budget / shards:.4f}"]
        rows = (settings.limit or 1000) / shards
        minutes = rows * TRACE_MINUTES_PER_EPISODE / max(1, teacher.traces.concurrency)
        built.append(teacher_job(run, rev, "traces", f"{run.name}-traces-{index}", args, minutes))
    return built


def sft_jobs(run: Run, rev: str) -> list[jobs.Job]:
    student = student_profile(run)
    args = [
        "--task",
        "/work/task",
        "--harness",
        jobs.q(harness_id(run)),
        "--profiles",
        "/work/profiles",
        "--student",
        jobs.q(run.config.student),
        "--traces-repo",
        jobs.q(run.repo("traces")),
        "--hub-model-id",
        jobs.q(run.repo("lora")),
        *[jobs.q(a) for a in run.config.sft.args],
    ]
    deps = student.sft.deps or SFT_DEPS
    command = "\n".join(
        [
            jobs.prelude(run.repo("code"), rev),
            jobs.python_module("agent_distill.stages.sft", args, deps=deps),
            "echo SFT_DONE",
        ]
    )
    return [
        jobs.Job(
            "sft",
            f"{run.name}-sft"[:90],
            student.sft.image,
            student.sft.flavor,
            run.config.sft.timeout or student.sft.timeout,
            command,
            expected_minutes=jobs.timeout_minutes(run.config.sft.timeout or student.sft.timeout) / 2,
        )
    ]


def merge_jobs(
    run: Run, rev: str, adapter_suffix: str = "lora", output_suffix: str = "sft-merged", base: str = ""
) -> list[jobs.Job]:
    student = student_profile(run)
    merge = student.merge
    deps = list(merge.deps) if merge.replace_deps else [*MERGE_DEPS, *merge.deps]
    args = [
        "--student",
        jobs.q(run.config.student),
        "--profiles",
        "/work/profiles",
        "--adapter",
        jobs.q(run.repo(adapter_suffix)),
        "--hub-model-id",
        jobs.q(run.repo(output_suffix)),
        "--note",
        jobs.q(f"agent-distill run {run.name}: task {run.task.name}, {harness_id(run)}."),
    ]
    if base:
        args += ["--base-model", jobs.q(base)]
    command = "\n".join(
        [
            jobs.prelude(run.repo("code"), rev),
            jobs.python_module("agent_distill.stages.merge", args, deps=deps),
            "echo MERGE_DONE",
        ]
    )
    return [
        jobs.Job(
            "merge",
            f"{run.name}-merge-{output_suffix}"[:90],
            merge.image,
            merge.flavor,
            merge.timeout,
            command,
            expected_minutes=jobs.timeout_minutes(merge.timeout) / 3,
        )
    ]


def probe_jobs(run: Run, rev: str) -> list[jobs.Job]:
    """The difficulty probe: the merged SFT policy, K attempts per train row at the GRPO sampling temperature."""
    student = student_profile(run)
    settings, probe = run.config.probe, student.probe
    shards = settings.shards or probe.shards
    k = settings.k or probe.k
    sample = settings.sample or probe.sample
    built = []
    for index in range(shards):
        args = [
            "--split",
            jobs.q(settings.split),
            "--purpose",
            "train",
            "--sample",
            str(sample),
            "--seed",
            "0",
            "--shard",
            f"{index}/{shards}",
            "--attempts",
            str(k),
            "--temperature",
            str(probe.temperature),
            "--concurrency",
            str(probe.concurrency),
            "--timeout",
            str(probe.episode_timeout),
            "--max-tool-calls",
            str(run.config.eval.max_tool_calls),
            "--output-dir",
            "/eval",
            "--prefix",
            f"probe/{settings.split}-{index}of{shards}",
            "--output-repo",
            jobs.q(run.repo("eval")),
            *probe.args,
        ]
        minutes = VLLM_STARTUP_MINUTES + sample / shards * k * EVAL_MINUTES_PER_EPISODE / max(1, probe.concurrency)
        built.append(
            served_job(
                run, rev, "probe", f"{run.name}-probe-{index}", run.repo("sft-merged"), "", "probe", args, minutes
            )
        )
    return built


def artifacts(run: Run) -> list[tuple[str, str, str]]:
    """(repo, type, collection note) of everything a run publishes."""
    pair = " -> ".join(p for p in (run.config.teacher, run.config.student) if p)
    prefix = f"{run.task.name}, {pair}, {harness_id(run)}"
    return [
        (run.repo("traces"), "dataset", f"{prefix}: teacher traces with agent-trace sessions"),
        (run.repo("lora"), "model", f"{prefix}: student SFT LoRA adapter"),
        (run.repo("sft-merged"), "model", f"{prefix}: student SFT merged checkpoint"),
        (run.repo("grpo-lora"), "model", f"{prefix}: student GRPO LoRA adapter (from the SFT merge)"),
        (run.repo("grpo-merged"), "model", f"{prefix}: student SFT + GRPO merged checkpoint"),
        (run.repo("eval"), "dataset", f"{prefix}: evaluation episodes (and the difficulty probe)"),
    ]


def credential_files(api: Any, repo: str, files: list[str]) -> list[str]:
    """Text files of a dataset repo holding unredacted credential-shaped strings (streamed line by line)."""
    from huggingface_hub import hf_hub_download

    from agent_distill.credentials import CREDENTIAL_RE

    leaks = []
    for name in files:
        if not name.endswith((".jsonl", ".json", ".md", ".txt", ".csv")):
            continue
        with open(hf_hub_download(repo, name, repo_type="dataset"), errors="ignore") as handle:
            if any(match for line in handle for match in CREDENTIAL_RE.findall(line) if "REDACTED" not in match):
                leaks.append(name)
    return leaks


def publish(run: Run, *, dry_run: bool) -> None:
    """Credential-scan datasets, make models and datasets public, add everything to the collection; the code repo
    must stay private. Nothing is ever deleted."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    collection = run.config.hub.collection
    for repo, kind, note in artifacts(run):
        try:
            info = api.repo_info(repo, repo_type=kind)
        except RepositoryNotFoundError:
            print(f"skip {repo}: not created yet", flush=True)
            continue
        if kind == "dataset":
            files = api.list_repo_files(repo, repo_type="dataset")
            if not any(name.startswith("sessions/") for name in files):
                raise SystemExit(f"{repo} has no agent-trace sessions; it stays private")
            leaks = credential_files(api, repo, files)
            if leaks:
                raise SystemExit(f"{repo} contains credential-shaped strings in {leaks[:5]}; it stays private")
        if dry_run:
            print(f"[dry-run] {repo}: scan passed; private={info.private} -> public; collection note: {note}")
            continue
        if info.private:
            api.update_repo_settings(repo, repo_type=kind, private=False)
        if collection:
            api.add_collection_item(collection, item_id=repo, item_type=kind, note=note, exists_ok=True)
        print(f"published {repo} ({kind})" + (f" in {collection}" if collection else ""), flush=True)
    try:
        if not api.repo_info(run.repo("code")).private:
            raise SystemExit(f"{run.repo('code')} must stay private")
    except RepositoryNotFoundError:
        pass


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="distill run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path, help="run.toml")
    parser.add_argument("--stages", help=f"Comma-separated subset of {','.join(STAGES)} (default: the run's)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for a single stage's jobs")
    parser.add_argument("--over-budget", action="store_true", help="Launch even if the worst case exceeds [budget]")
    parser.add_argument("--splits", help="Override [eval] splits (comma-separated)")
    parser.add_argument("--limit", type=int, help="Override [eval] limit")
    parser.add_argument(
        "--eval-model",
        action="append",
        default=[],
        metavar="LABEL=MODEL[+ADAPTER]",
        help="Evaluate this model under LABEL (repeatable; replaces [eval] models)",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="Override any run.toml key")
    return parser.parse_args(argv)


def overrides(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError:
            values[key] = raw
    if args.stages:
        values["stages"] = [s for s in args.stages.split(",") if s]
    if args.splits:
        values["eval.splits"] = [s for s in args.splits.split(",") if s]
    if args.limit:
        values["eval.limit"] = args.limit
    if args.eval_model:
        values["eval.models"] = dict(item.split("=", 1) for item in args.eval_model)
    return values


def check_budget(run: Run, built: list[jobs.Job], *, over_budget: bool) -> None:
    cost = jobs.estimate(built)
    cap = run.config.budget.gpu_usd
    print(
        f"cost estimate for {len(built)} jobs: expected ${cost['expected_usd']}, worst case "
        f"${cost['worst_case_usd']} (every job to its timeout); cap {'$' + str(cap) if cap else 'none'}"
    )
    if cap and cost["worst_case_usd"] > cap and not over_budget:
        raise SystemExit(f"worst case ${cost['worst_case_usd']} exceeds [budget] gpu_usd = {cap}; pass --over-budget")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_run(args.run, overrides(args))
    stages = config.stages
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}")
    pending = [s for s in stages if s not in IMPLEMENTED]
    if pending:
        raise SystemExit(f"stages {pending} are not implemented yet (milestone M3b)")
    run = make_run(config)
    namespace = config.hub.bill_to
    print(
        f"run {run.name}: task {run.task.name}, student {config.student}, harness {config.harness}, "
        f"jobs billed to {namespace or 'your account'}"
    )
    job_stages = [s for s in stages if s != "publish"]
    rev = ""
    if job_stages:
        manifest = {
            "run": dataclasses.asdict(config),
            "name": run.name,
            "git_commit": jobs.git_commit(),
            "task": run.task.name,
            "task_version": run.task.version,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        extra = []
        if not args.dry_run:
            import shutil
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="agent-distill-profiles-"))
            for target, source in profile_files(run).items():
                (tmp / target).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, tmp / target)
            extra = [tmp / "profiles"]
        rev = jobs.stage_code(run.repo("code"), run.task_dir, extra, dry_run=args.dry_run, manifest=manifest)
    for stage in stages:
        if stage == "publish":
            publish(run, dry_run=args.dry_run)
            continue
        built = {"traces": trace_jobs, "sft": sft_jobs, "merge": merge_jobs, "probe": probe_jobs, "eval": eval_jobs}[
            stage
        ](run, rev)
        check_budget(run, built, over_budget=args.over_budget)
        if args.dry_run:
            jobs.describe(built, namespace)
            continue
        ids = jobs.launch(built, namespace)
        if len(stages) > 1 or args.wait:
            jobs.wait(ids, namespace)
    return 0
