"""`distill run`: a run.toml's stages as HF Jobs (DESIGN.md §7).

    distill run examples/paper-urls/run.toml --stages eval --dry-run
    distill run examples/paper-urls/run.toml --stages eval --eval-model base=Qwen/Qwen3.5-9B --splits test
    distill run examples/paper-urls/run.toml --stages publish

M2 implements `eval` and `publish`; `traces`, `sft`, `merge`, `probe` and `grpo` arrive in M3. With several stages
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
from agent_distill.config import BUILTIN_PROFILES, RunConfig, load_profile, load_run, slugify
from agent_distill.task import Task, load_task

STAGES = ("traces", "sft", "merge", "probe", "grpo", "eval", "publish")
IMPLEMENTED = ("eval", "publish")
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
    """The student and harness profile files a job needs (staged under profiles/<kind>/)."""
    files = {}
    for kind, name in (("students", run.config.student), ("harnesses", run.config.harness)):
        for folder in [*(p / kind for p in run.config.profile_paths), BUILTIN_PROFILES / kind]:
            if (folder / f"{name}.toml").exists():
                files[f"profiles/{kind}/{name}.toml"] = folder / f"{name}.toml"
                break
    return files


# --------------------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------------------


def eval_models(run: Run) -> dict[str, str]:
    student = load_profile("students", run.config.student, run.config.profile_paths)
    return run.config.eval.models or {"base": student.model}


def eval_jobs(run: Run, rev: str) -> list[jobs.Job]:
    config = run.config
    settings = config.eval
    student = load_profile("students", config.student, config.profile_paths)
    harness = load_profile("harnesses", config.harness, config.profile_paths)
    serve = student.serve
    deps = list(serve.deps)
    vllm_binary: list[str] = []
    if serve.vllm == "image":
        vllm_binary = ["--vllm-binary", '"$(command -v vllm)"']
    else:
        deps.append(f"vllm=={serve.vllm_version}")
    built = []
    for label, spec in eval_models(run).items():
        model, _, adapter = spec.partition("+")
        for split in settings.splits:
            args = [
                "--task", "/work/task", "--harness", jobs.q(harness.id), "--student", jobs.q(config.student),
                "--profiles", "/work/profiles", "--split", jobs.q(split), "--limit", str(settings.limit),
                "--concurrency", str(settings.concurrency), "--timeout", str(settings.timeout),
                "--max-tool-calls", str(settings.max_tool_calls), "--output-dir", "/eval",
                "--output-repo", jobs.q(run.repo("eval")), "--cache", "/work/cache", "--model", jobs.q(model),
                *vllm_binary,
            ]  # fmt: skip
            if adapter:
                args += ["--adapter", jobs.q(adapter), "--adapter-label", jobs.q(label), "--adapter-only"]
            else:
                args += ["--label", jobs.q(label)]
            command = "\n".join(
                [
                    jobs.prelude(run.repo("code"), rev, run.task.job_setup),
                    jobs.python_module("agent_distill.stages.eval", args, deps=deps),
                    "echo EVAL_DONE",
                ]
            )
            minutes = VLLM_STARTUP_MINUTES + settings.limit * EVAL_MINUTES_PER_EPISODE / max(1, settings.concurrency)
            built.append(
                jobs.Job(
                    "eval",
                    f"{run.name}-eval-{slugify(label)}-{slugify(split)}"[:90],
                    serve.image,
                    serve.flavor,
                    serve.timeout,
                    command,
                    expected_minutes=minutes,
                )
            )
    return built


def artifacts(run: Run) -> list[tuple[str, str, str]]:
    """(repo, type, collection note) of everything a run publishes (M2: the evaluation dataset)."""
    pair = " -> ".join(p for p in (run.config.teacher, run.config.student) if p)
    harness = load_profile("harnesses", run.config.harness, run.config.profile_paths).id
    return [(run.repo("eval"), "dataset", f"{run.task.name}, {pair}, {harness}: evaluation episodes")]


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
    parser = argparse.ArgumentParser(prog="distill run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("run", type=Path, help="run.toml")
    parser.add_argument("--stages", help=f"Comma-separated subset of {','.join(STAGES)} (default: the run's)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for a single stage's jobs")
    parser.add_argument("--over-budget", action="store_true", help="Launch even if the worst case exceeds [budget]")
    parser.add_argument("--splits", help="Override [eval] splits (comma-separated)")
    parser.add_argument("--limit", type=int, help="Override [eval] limit")
    parser.add_argument("--eval-model", action="append", default=[], metavar="LABEL=MODEL[+ADAPTER]",
                        help="Evaluate this model under LABEL (repeatable; replaces [eval] models)")  # fmt: skip
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
    print(f"cost estimate for {len(built)} jobs: expected ${cost['expected_usd']}, worst case "
          f"${cost['worst_case_usd']} (every job to its timeout); cap {'$' + str(cap) if cap else 'none'}")  # fmt: skip
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
        raise SystemExit(f"stages {pending} are not implemented yet (milestone M3)")
    run = make_run(config)
    namespace = config.hub.bill_to
    print(f"run {run.name}: task {run.task.name}, student {config.student}, harness {config.harness}, "
          f"jobs billed to {namespace or 'your account'}")  # fmt: skip
    job_stages = [s for s in stages if s != "publish"]
    rev = ""
    if job_stages:
        manifest = {"run": dataclasses.asdict(config), "name": run.name, "git_commit": jobs.git_commit(),
                    "task": run.task.name, "task_version": run.task.version,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
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
        built = {"eval": eval_jobs}[stage](run, rev)
        check_budget(run, built, over_budget=args.over_budget)
        if args.dry_run:
            jobs.describe(built, namespace)
            continue
        ids = jobs.launch(built, namespace)
        if len(stages) > 1 or args.wait:
            jobs.wait(ids, namespace)
    return 0
