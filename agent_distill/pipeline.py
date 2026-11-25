"""`distill run`: a run.toml's stages as HF Jobs (DESIGN.md §7).

    distill run examples/paper-urls/run.toml --stages eval --dry-run
    distill run examples/paper-urls/run.toml --stages eval --eval-model base=Qwen/Qwen3.5-9B --splits test
    distill run examples/paper-urls/run.toml --stages publish

Stages: `traces` (teacher episodes, CPU jobs), `sft`, `merge` (SFT adapter into the student), `probe` (difficulty
probe of the merged SFT policy), `grpo` (GRPO in the harness, then its merge), `eval` (base / sft / grpo / teacher,
or explicit models) and `publish`. With several stages
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
# Stages that only redo part of another one (not in a run's default list): `grpo-merge` reruns GRPO's merge.
PARTIAL_STAGES = ("grpo-merge",)
IMPLEMENTED = STAGES + PARTIAL_STAGES
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


def default_name(config: RunConfig, task: Task) -> str:
    """`<student>-<task>-<teacher>-<harness>`, slugified (DESIGN.md §7)."""
    harness = load_profile("harnesses", config.harness, config.profile_paths)
    parts = [config.student, task.name, config.teacher, harness.id]
    return slugify("-".join(part for part in parts if part))


def make_run(config: RunConfig) -> Run:
    task_dir = config.path(config.task)
    task = load_task(task_dir)
    name = config.name or default_name(config, task)
    namespace = config.hub.namespace
    if not namespace:
        from huggingface_hub import whoami

        namespace = whoami()["name"]
    return Run(config, task, task_dir if task_dir.is_dir() else task_dir.parent, name, namespace)


def episode_extras(run: Run) -> str:
    """The wheel extras of stages that run episodes: `hub` plus the task's own (e.g. `browser`)."""
    return ",".join(dict.fromkeys(["hub", *run.task.extras]))


def judge_name(run: Run) -> str:
    return run.task.judge if run.config.judge is None else run.config.judge


def judge_secrets(run: Run) -> tuple[str, ...]:
    """The judge's key variable when it is not the HF token every job already has (e.g. GEMINI_API_KEY)."""
    name = judge_name(run)
    if not name:
        return ()
    profile = load_profile("judges", name, run.config.profile_paths)
    key = profile.key_env or ("GEMINI_API_KEY" if profile.api == "gemini" else "HF_TOKEN")
    return () if key == "HF_TOKEN" else (key,)


def judge_args(run: Run, *, bill_to: bool = True) -> list[str]:
    """`--judge` (always explicit in jobs) and, for a judge, `--bill-to` (router judges bill like teachers)."""
    name = judge_name(run)
    args = ["--judge", jobs.q(name)]
    if name and bill_to and run.config.hub.bill_to:
        args += ["--bill-to", jobs.q(run.config.hub.bill_to)]
    return args


def profile_files(run: Run) -> dict[str, Path]:
    """The student, harness, teacher and judge profile files a job needs (staged under profiles/<kind>/)."""
    files = {}
    kinds = [("students", run.config.student), ("harnesses", run.config.harness)]
    if run.config.teacher:
        kinds.append(("teachers", run.config.teacher))
    if judge_name(run):
        kinds.append(("judges", judge_name(run)))
    for kind, name in kinds:
        for folder in [*(p / kind for p in run.config.profile_paths), BUILTIN_PROFILES / kind]:
            if (folder / f"{name}.toml").exists():
                files[f"profiles/{kind}/{name}.toml"] = folder / f"{name}.toml"
                break
    return files


def requires_files(run: Run) -> dict[str, Path]:
    """The student profile's `requires` (hook modules: `.py` files or package folders), staged at the top of each
    job's working directory so `module:function` hooks import. Looked up next to the run.toml, then in the project
    profile folders, then next to the student profile; refuses the run when one is missing."""
    student = student_profile(run)
    profile = profile_files(run).get(f"profiles/students/{run.config.student}.toml")
    roots = [Path(run.config.base_dir), *run.config.profile_paths, *([profile.parent] if profile else [])]
    found: dict[str, Path] = {}
    missing = []
    for entry in student.requires:
        path = next((root / entry for root in roots if (root / entry).exists()), None)
        if path is None:
            missing.append(entry)
            continue
        if path.name in found or path.name in ("task", "profiles", "manifest.json"):
            raise SystemExit(f"student {student.name}: `requires` entry {entry!r} clashes with another staged file")
        found[path.name] = path
    if missing:
        raise SystemExit(
            f"student {student.name} requires {missing}, not found in {[str(r) for r in roots]}; copy the hook "
            "modules into the project (next to the run.toml)"
        )
    return found


def defines(path: Path, name: str) -> bool:
    """Whether a staged Python module (file or package `__init__.py`) defines `name` at top level."""
    import ast

    source = path / "__init__.py" if path.is_dir() else path
    if not source.is_file():
        return False
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.name == name:
            return True
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return True
        if isinstance(node, ast.ImportFrom | ast.Import) and any(
            (alias.asname or alias.name.split(".")[0]) == name for alias in node.names
        ):
            return True
    return False


def check_hooks(run: Run, staged: dict[str, Path]) -> None:
    """Every `module[:function]` hook of the student profile resolves from the staged `requires` files or the
    agent_distill package; otherwise the run (and its dry run) refuses before anything is launched."""
    import importlib.util

    student = student_profile(run)
    problems = []
    for key, spec in student.hooks().items():
        module, _, attribute = spec.partition(":")
        top, _, rest = module.partition(".")
        local = staged.get(f"{top}.py") if not rest else None
        local = local or staged.get(top)
        if local is not None:
            target = local
            for part in rest.split(".") if rest else []:
                target = target / part if (target / part).is_dir() else target / f"{part}.py"
            if not target.exists():
                problems.append(f"{key} = {spec!r}: {module} is not in the staged {local.name}")
            elif attribute and not defines(target, attribute):
                problems.append(f"{key} = {spec!r}: {target.name} defines no {attribute!r}")
            continue
        if top == "agent_distill":
            try:
                found = importlib.util.find_spec(module) is not None
            except ModuleNotFoundError:
                found = False
            if found:
                continue
        problems.append(f"{key} = {spec!r}: module {top!r} is neither in `requires` nor in the agent_distill package")
    if problems:
        raise SystemExit(f"student {student.name}: unresolved hooks:\n  " + "\n  ".join(problems))


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
GRPO_DEPS = [
    "trl[kernels]==1.18.1",
    "openenv==0.14.0",
    "transformers==5.23.0",
    "torch==2.15.0",
    "peft==0.21.3",
    "flash-linear-attention==0.6.0",
    "datasets>=4.0.0",
    "aiohttp>=3.13.3",
    "requests",
    "safetensors>=0.4.0",
]
# url-extraction's GRPO recipe (student profiles append their own arguments; the last occurrence of a flag wins).
GRPO_RECIPE = [
    "--num-generations",
    "8",
    "--max-steps",
    "100",
    "--save-steps",
    "50",
    "--gradient-accumulation-steps",
    "2",
    "--report-to",
    "none",
]
TRACE_MINUTES_PER_EPISODE = 1.5


def student_profile(run: Run) -> Any:
    return load_profile("students", run.config.student, run.config.profile_paths)


def teacher_profile(run: Run) -> Any:
    if not run.config.teacher:
        raise SystemExit("this stage needs a teacher (`teacher = ...` in the run.toml)")
    return load_profile("teachers", run.config.teacher, run.config.profile_paths)


def harness_id(run: Run) -> str:
    return load_profile("harnesses", run.config.harness, run.config.profile_paths).id


def job_task(run: Run) -> str:
    """The task as jobs see it: the staged task folder, or the task file within it when the run names a file
    (e.g. `task = "judged.py"` for a second task in the same folder)."""
    path = run.config.path(run.config.task)
    return "/work/task" if path.is_dir() else f"/work/task/{path.name}"


def common_args(run: Run) -> list[str]:
    return [
        "--task",
        job_task(run),
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


def output_args(run: Run, suffix: str) -> list[str]:
    """Where an episode stage's raw logs go (the always-private `-raw` sibling of its dataset, never the dataset
    itself), and the collection a newly created dataset joins at once."""
    raw = "raw" if suffix == "traces" else f"{suffix}-raw"
    args = ["--raw-repo", jobs.q(run.repo(raw))]
    if run.config.hub.collection:
        args += ["--collection", jobs.q(run.config.hub.collection), "--collection-note",
                 jobs.q(collection_note(run, suffix))]  # fmt: skip
    return args


def model_collection_args(run: Run, suffix: str) -> list[str]:
    """The collection a model job's pushed repo joins at once (models are public from creation)."""
    if not run.config.hub.collection:
        return []
    return ["--collection", jobs.q(run.config.hub.collection), "--collection-note",
            jobs.q(collection_note(run, suffix))]  # fmt: skip


def image_vllm(student: Any) -> list[str]:
    """`--vllm-binary` for students whose image ships its own vLLM (`[serve] vllm = "image"`), else nothing."""
    return ["--vllm-binary", '"$(command -v vllm)"'] if student.serve.vllm == "image" else []


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
    if serve.vllm != "image":
        deps.append(f"vllm=={serve.vllm_version}")
    full = [*common_args(run), "--student", jobs.q(run.config.student), "--model", jobs.q(model),
            *image_vllm(student), *args, *output_args(run, "eval"),
            *judge_args(run)]  # fmt: skip
    if adapter:
        full += ["--adapter", jobs.q(adapter), "--adapter-label", jobs.q(label), "--adapter-only"]
    else:
        full += ["--label", jobs.q(label)]
    command = "\n".join(
        [
            jobs.prelude(run.repo("code"), rev, run.task.job_setup, run.task.commands),
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
        secret_envs=judge_secrets(run),
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
            if settings.answer_retry:
                args.append("--answer-retry")
            minutes = VLLM_STARTUP_MINUTES + settings.limit * EVAL_MINUTES_PER_EPISODE / max(1, settings.concurrency)
            name = f"{run.name}-eval-{slugify(label)}-{slugify(split)}"
            built.append(served_job(run, rev, "eval", name, model, adapter, label, args, minutes))
    if "teacher" in settings.variants and not settings.models:
        built += [teacher_eval_job(run, rev, split) for split in settings.splits]
    return built


def teacher_job(run: Run, rev: str, stage: str, name: str, args: list[str], minutes: float) -> jobs.Job:
    teacher = teacher_profile(run)
    full = [*common_args(run), "--teacher", jobs.q(run.config.teacher), *args,
            *output_args(run, "traces" if stage == "traces" else "eval"),
            *judge_args(run, bill_to=False)]  # fmt: skip
    if run.config.hub.bill_to:
        full += ["--bill-to", jobs.q(run.config.hub.bill_to)]
    command = "\n".join(
        [
            jobs.prelude(run.repo("code"), rev, run.task.job_setup, run.task.commands),
            jobs.python_module("agent_distill.stages.traces", full),
            f"echo {stage.upper()}_DONE",
        ]
    )
    return jobs.Job(
        stage,
        name[:90],
        UV_IMAGE,
        teacher.traces.flavor,
        teacher.traces.timeout,
        command,
        expected_minutes=minutes,
        secret_envs=judge_secrets(run),
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
        job_task(run),
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
        *model_collection_args(run, "lora"),
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
        *model_collection_args(run, output_suffix),
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


def grpo_jobs(run: Run, rev: str) -> list[jobs.Job]:
    """GRPO from the merged SFT checkpoint in the run's harness (two GPUs: vLLM + trainer)."""
    student = student_profile(run)
    grpo, settings = student.grpo, run.config.grpo
    args = [
        *common_args(run),
        "--student",
        jobs.q(run.config.student),
        "--model",
        jobs.q(run.repo("sft-merged")),
        "--hub-model-id",
        jobs.q(run.repo("grpo-lora")),
        *model_collection_args(run, "grpo-lora"),
        "--trainer",
        grpo.backend,
        "--lora-r",
        str(student.lora.rank),
        "--timeout",
        str(run.config.eval.timeout),
        "--max-tool-calls",
        str(run.config.eval.max_tool_calls),
        "--traces-repo",
        jobs.q(run.repo("traces")),
        *image_vllm(student),
        *([] if student.serve.vllm == "image" else ["--vllm-version", jobs.q(student.serve.vllm_version)]),
        *GRPO_RECIPE,
        *judge_args(run),
        *[jobs.q(a) for a in grpo.args],
        *[jobs.q(a) for a in settings.args],
        "--launch",
    ]
    if settings.use_probe:  # the qualified ids of this run's probe (its configured split and shards) only
        probe_files = [f"difficulty/{prefix}-qualified-ids.txt" for prefix in probe_prefixes(run)]
        args[-1:-1] = ["--include-ids-repo", jobs.q(run.repo("eval"))]
        args[-1:-1] = [arg for name in probe_files for arg in ("--include-ids-repo-file", jobs.q(name))]
    deps = grpo.deps or GRPO_DEPS
    command = "\n".join(
        [
            jobs.prelude(run.repo("code"), rev, run.task.job_setup, run.task.commands),
            jobs.python_module("agent_distill.stages.grpo", args, deps=deps),
            "echo GRPO_DONE",
        ]
    )
    timeout = settings.timeout or grpo.timeout
    return [
        jobs.Job(
            "grpo",
            f"{run.name}-grpo"[:90],
            grpo.image,
            grpo.flavor,
            timeout,
            command,
            expected_minutes=jobs.timeout_minutes(timeout) * 0.75,
            secret_envs=judge_secrets(run),
        )
    ]


def grpo_merge_jobs(run: Run, rev: str) -> list[jobs.Job]:
    return merge_jobs(run, rev, "grpo-lora", "grpo-merged", base=run.repo("sft-merged"))


def probe_prefixes(run: Run) -> list[str]:
    """The probe jobs' record prefixes; each writes `difficulty/<prefix>-qualified-ids.txt` in the eval dataset."""
    shards = run.config.probe.shards or student_profile(run).probe.shards
    return [f"probe/{run.config.probe.split}-{index}of{shards}" for index in range(shards)]


def probe_jobs(run: Run, rev: str) -> list[jobs.Job]:
    """The difficulty probe: the merged SFT policy, K attempts per train row at the GRPO sampling temperature."""
    student = student_profile(run)
    settings, probe = run.config.probe, student.probe
    prefixes = probe_prefixes(run)
    shards = len(prefixes)
    k = settings.k or probe.k
    sample = settings.sample or probe.sample
    built = []
    for index, prefix in enumerate(prefixes):
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
            prefix,
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


def collection_note(run: Run, suffix: str) -> str:
    """The collection note of the run's `suffix` repo (as in `artifacts`)."""
    return next(text for repo, _, text in artifacts(run) if repo == run.repo(suffix))


def private_repos(run: Run) -> list[tuple[str, str]]:
    """(repo, type) of the repos that stay private forever: the code snapshots and the raw harness logs."""
    return [(run.repo("code"), "model"), (run.repo("raw"), "dataset"), (run.repo("eval-raw"), "dataset")]


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
    """Credential-scan datasets, add everything to the collection, then make models and datasets public. Code and
    raw-log repos must stay private and are never touched; a dataset holding raw logs is never made public. Nothing
    is ever deleted."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    collection = run.config.hub.collection
    for repo, kind in private_repos(run):
        try:
            if not api.repo_info(repo, repo_type=kind).private:
                raise SystemExit(f"{repo} must stay private; make it private and revoke any token it may hold")
        except RepositoryNotFoundError:
            pass
    for repo, kind, note in artifacts(run):
        if repo.endswith("-raw") or repo in {r for r, _ in private_repos(run)}:
            raise SystemExit(f"{repo} holds raw logs or code and is never published")
        try:
            info = api.repo_info(repo, repo_type=kind)
        except RepositoryNotFoundError:
            print(f"skip {repo}: not created yet", flush=True)
            continue
        if kind == "dataset":
            files = api.list_repo_files(repo, repo_type="dataset")
            if any(name.startswith("raw/") for name in files):
                raise SystemExit(f"{repo} contains raw harness logs (raw/); it stays private")
            if not any(name.startswith("sessions/") for name in files):
                raise SystemExit(f"{repo} has no agent-trace sessions; it stays private")
            leaks = credential_files(api, repo, files)
            if leaks:
                raise SystemExit(f"{repo} contains credential-shaped strings in {leaks[:5]}; it stays private")
        if dry_run:
            print(f"[dry-run] {repo}: scan passed; collection note: {note}; private={info.private} -> public")
            continue
        if collection:  # in the collection before it becomes public
            api.add_collection_item(collection, item_id=repo, item_type=kind, note=note, exists_ok=True)
        if info.private:
            api.update_repo_settings(repo, repo_type=kind, private=False)
        print(f"published {repo} ({kind})" + (f" in {collection}" if collection else ""), flush=True)


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


def stage_waves(stage: str, run: Run, rev: str) -> list[list[jobs.Job]]:
    """A stage's jobs in waves; each wave waits for the previous one (GRPO, then its merge)."""
    if stage == "grpo":
        return [grpo_jobs(run, rev), grpo_merge_jobs(run, rev)]
    builders = {"traces": trace_jobs, "sft": sft_jobs, "merge": merge_jobs, "probe": probe_jobs, "eval": eval_jobs,
                "grpo-merge": grpo_merge_jobs}  # fmt: skip
    return [builders[stage](run, rev)]


# `distill smoke RUN.toml`: the run's whole recipe on a handful of rows and steps, under its own name.
SMOKE = {
    "stages": ["traces", "sft", "merge", "probe", "grpo", "eval"],
    "traces.limit": 8,
    "traces.shards": 1,
    "traces.attempts": 1,
    "sft.args": ["--max-steps", "2", "--save-steps", "1"],
    "sft.timeout": "1h",
    "probe.sample": 4,
    "probe.k": 2,
    "probe.shards": 1,
    "probe.timeout": "1h",
    "grpo.args": ["--max-steps", "2", "--save-steps", "1", "--num-generations", "2", "--max-concurrent-rollouts", "4"],
    "grpo.timeout": "2h",
    "grpo.use_probe": False,
    "eval.limit": 5,
    "eval.concurrency": 5,
    "eval.variants": ["base", "sft", "grpo", "teacher"],
    "eval.models": {},  # evaluate what the smoke run trained, never a run.toml's explicit models
    "repos": {},  # the smoke run's own repos, never a run.toml's overrides
    "budget.traces_usd": 2,
}


def smoke_name(config: RunConfig) -> str:
    """The run's name (explicit or default) with a `-smoke` suffix, so a smoke run never writes into (or resumes
    from) the real run's repos."""
    name = config.name or default_name(config, load_task(config.path(config.task)))
    return name if name.endswith("-smoke") else f"{name}-smoke"


def smoke_main(argv: list[str] | None = None) -> int:
    """`distill smoke RUN.toml [--dry-run]`: the smoke variant of a run (name suffix `-smoke`)."""
    import sys

    items = list(sys.argv[1:] if argv is None else argv)
    if not items or items[0].startswith("-"):
        raise SystemExit("usage: distill smoke RUN.toml [--dry-run] [--over-budget] [--set KEY=JSON ...]")
    sets = [f"{key}={json.dumps(value)}" for key, value in {**SMOKE, "name": smoke_name(load_run(items[0]))}.items()]
    return main([items[0], *(arg for pair in sets for arg in ("--set", pair)), *items[1:]])


def apply_judge_budget(run: Run, built: list[jobs.Job]) -> None:
    """Split `[budget] judge_usd` evenly over a stage's judging jobs; each job's judge client stops at its share."""
    cap = run.config.budget.judge_usd
    judging = [job for job in built if "--judge " in job.command and "--judge ''" not in job.command]
    name = judge_name(run)
    if name and judging:
        profile = load_profile("judges", name, run.config.profile_paths)
        if cap and profile.api == "gemini" and not (profile.price.input or profile.price.output):
            raise SystemExit(f"[budget] judge_usd needs a price in the {name} judge profile")
        print(f"judge {name}: cap {'$' + format(cap / len(judging), '.2f') + ' per job' if cap else 'none'} "
              f"({len(judging)} judging jobs)")  # fmt: skip
    for job in judging:
        if cap:
            job.env["AGENT_DISTILL_JUDGE_BUDGET_USD"] = f"{cap / len(judging):.4f}"


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
    unknown = [s for s in stages if s not in IMPLEMENTED]
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
    hook_files: dict[str, Path] = {}
    if any(s != "traces" for s in job_stages):  # student stages: hook modules staged and resolvable, or refuse
        hook_files = requires_files(run)
        check_hooks(run, hook_files)
    if job_stages:
        manifest = {"run": dataclasses.asdict(config), "name": run.name, "git_commit": jobs.git_commit(),
                    "task": run.task.name, "task_version": run.task.version,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "residual_risks": list(jobs.RESIDUAL_RISKS)}  # fmt: skip
        if args.dry_run:
            for risk in jobs.RESIDUAL_RISKS:
                print(f"[dry-run] residual risk: {risk}")
        extra = []
        if not args.dry_run:
            import shutil
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="agent-distill-profiles-"))
            for target, source in profile_files(run).items():
                (tmp / target).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, tmp / target)
            extra = [tmp / "profiles", *hook_files.values()]  # hook modules land in each job's working directory
        elif hook_files:
            print(f"[dry-run] would stage the student's hook modules {sorted(hook_files)}")
        rev = jobs.stage_code(run.repo("code"), run.task_dir, extra, dry_run=args.dry_run, manifest=manifest,
                              extras=tuple(episode_extras(run).split(",")))  # fmt: skip
    for stage in stages:
        if stage == "publish":
            publish(run, dry_run=args.dry_run)
            continue
        waves = stage_waves(stage, run, rev)
        apply_judge_budget(run, [job for wave in waves for job in wave])
        check_budget(run, [job for wave in waves for job in wave], over_budget=args.over_budget)
        for index, wave in enumerate(waves):
            if args.dry_run:
                jobs.describe(wave, namespace)
                continue
            ids = jobs.launch(wave, namespace)
            if len(stages) > 1 or args.wait or index < len(waves) - 1:
                jobs.wait(ids, namespace)
    return 0
