"""HF Jobs: code staging, job commands, launching, waiting, cost estimates (DESIGN.md §7, §8).

Code staging: a wheel of the current checkout, a hash-pinned `requirements.txt` exported from `uv.lock` (plus the
same pins without hashes as `constraints.txt`, for reference) and the task folder go to the run's private
`<run>-code` model repo under `code/<rev>/`. Every job downloads them, installs the locked dependencies with
`--require-hashes`, then the wheel with `--no-deps`, so jobs that need nothing else run exactly the locked versions
and no GitHub credential ever enters a job.

A job's runtime stack (`vllm==<version>`, or a student profile's pinned training packages such as
`transformers==5.14.1`) is authoritative for that job: it is installed afterwards together with the wheel, so the
resolver may move a locked package the stack needs at another version (e.g. `huggingface-hub` below 2.0 for
`transformers` 5.14) while still honouring the package's own requirements. Under the lock's constraints those stacks
are unsatisfiable. Jobs run in the `[hub] bill_to` namespace with the caller's token as the only secret.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HF_CLI = "uvx --from 'huggingface-hub>=1.13' hf"
REPO_ROOT = Path(__file__).resolve().parent.parent
JOB_EXTRAS = ("hub",)  # the package extras every job installs (from uv.lock)
VENV_PYTHON = "/work/venv/bin/python"
# Printed by every dry run and recorded in the run manifest (DESIGN.md §6, credentials).
RESIDUAL_RISKS = (
    "the job's own HF token (its HF_TOKEN secret) is in the stage process's environment and token cache. Pi agents "
    "run as their own unprivileged users and cannot read it (agent_distill.harnesses.isolation); OpenCode agents, "
    "and Pi agents in a stage that is not root or has AGENT_DISTILL_AGENT_USERS=off, run as the stage's user and "
    "could read it from /proc/<pid>/environ, although the harness's own environment is stripped (teacher keys reach "
    "the harness only as key-proxy session keys)",
)


@dataclass
class Job:
    stage: str
    name: str
    image: str
    flavor: str
    timeout: str  # e.g. "4h"
    command: str
    env: dict[str, str] = field(default_factory=dict)
    # Environment variables passed from the launcher's environment as job secrets, besides HF_TOKEN (e.g. a Gemini
    # judge's GEMINI_API_KEY). Names only; values are read at launch and never printed.
    secret_envs: tuple[str, ...] = ()
    expected_minutes: float = 0.0  # for the cost estimate (the timeout bounds the worst case)


def q(value: Any) -> str:
    return shlex.quote(str(value))


def timeout_minutes(timeout: str) -> float:
    unit = timeout[-1]
    value = float(timeout[:-1]) if unit in "smhd" else float(timeout)
    return value * {"s": 1 / 60, "m": 1, "h": 60, "d": 1440}.get(unit, 1 / 60)


def build_wheel(out_dir: Path) -> Path:
    """`uv build --wheel` of this checkout (its pyproject) into `out_dir`."""
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out_dir), str(REPO_ROOT)], check=True,
                   capture_output=True)  # fmt: skip
    wheels = sorted(out_dir.glob("agent_distill-*.whl"))
    if not wheels:
        raise RuntimeError("uv build produced no wheel")
    return wheels[-1]


def export_requirements(out_dir: Path, extras: tuple[str, ...] = JOB_EXTRAS) -> tuple[Path, Path]:
    """`requirements.txt` (hash-pinned) and `constraints.txt` (the same pins, no hashes) from this checkout's
    uv.lock, for the package's dependencies and `extras` (without the package itself and the dev group)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-header", "--format", "requirements-txt",
            "--project", str(REPO_ROOT), *(f"--extra={extra}" for extra in extras)]  # fmt: skip
    requirements, constraints = out_dir / "requirements.txt", out_dir / "constraints.txt"
    subprocess.run([*base, "--output-file", str(requirements)], check=True, capture_output=True)
    subprocess.run([*base, "--no-hashes", "--output-file", str(constraints)], check=True, capture_output=True)
    if "--hash=sha256:" not in requirements.read_text():
        raise RuntimeError("uv export wrote requirements without hashes")
    return requirements, constraints


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()  # fmt: skip
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stage_code(
    code_repo: str,
    task_dir: Path,
    extra: list[Path],
    *,
    dry_run: bool,
    manifest: dict[str, Any],
    extras: tuple[str, ...] = JOB_EXTRAS,
) -> str:
    """Upload the wheel, the hash-pinned requirements (the package plus `extras`, e.g. a task's `browser`), the task
    folder, extra files and the run manifest under code/<rev>/; returns <rev>."""
    rev = time.strftime("%Y%m%d-%H%M%S")
    if dry_run:
        print(f"[dry-run] would stage the agent-distill wheel, requirements.txt (uv export --frozen --no-dev "
              f"--extra {','.join(extras)}, hash-pinned), {task_dir} and {len(extra)} files to {code_repo} "
              f"(private) under code/{rev}/")  # fmt: skip
        return rev
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(code_repo, private=True, exist_ok=True)  # code snapshots stay private
    if not api.repo_info(code_repo).private:
        raise SystemExit(f"{code_repo} must be private")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_wheel(root)
        export_requirements(root, extras)
        shutil.copytree(task_dir, root / "task", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in extra:
            if path.is_dir():
                shutil.copytree(path, root / path.name)
            else:
                shutil.copy2(path, root / path.name)
        (root / "manifest.json").write_text(json.dumps({**manifest, "rev": rev}, indent=2, sort_keys=True))
        api.upload_folder(folder_path=tmp, path_in_repo=f"code/{rev}", repo_id=code_repo,
                          commit_message=f"Stage agent-distill code {rev}")  # fmt: skip
    print(f"staged code to {code_repo}@code/{rev}", flush=True)
    return rev


def prelude(code_repo: str, rev: str, setup: list[str] = (), commands: Sequence[str] = ()) -> str:
    """Shell lines every job starts with: uv, the staged code, the task's own setup commands, then a check that the
    commands the task's agents need are on the PATH (Task.commands), so a job image without them fails at once."""
    return "\n".join(
        [
            "set -euo pipefail",
            "unset UV_OVERRIDE UV_INDEX_STRATEGY UV_PRERELEASE",
            "command -v uvx >/dev/null || python3 -m pip install -q uv",
            'export PATH="$HOME/.local/bin:$PATH"',
            "nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true",
            # uv's Python outside root's home: isolated agents run the stage's Python for task tools (`./preview`).
            'export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/uv/python}"',
            "mkdir -p /work && cd /work",
            f"{HF_CLI} download {q(code_repo)} --include 'code/{rev}/*' --local-dir /work/stage >/dev/null",
            f"cp -r /work/stage/code/{rev}/. /work/",
            "WHEEL=$(ls /work/agent_distill-*.whl)",
            "uv venv -q --python 3.12 --python-preference only-managed /work/venv",
            f"uv pip install -q --python {VENV_PYTHON} --require-hashes -r /work/requirements.txt",
            f'uv pip install -q --python {VENV_PYTHON} --no-deps "$WHEEL"',
            *setup,
            *([require(commands)] if commands else []),
        ]
    )


def require(commands: Sequence[str]) -> str:
    """A shell line that fails the job, naming the missing ones, unless every command is on the PATH."""
    names = " ".join(q(command) for command in commands)
    return (
        f'missing=(); for c in {names}; do command -v "$c" >/dev/null || missing+=("$c"); done; '
        'if [ ${#missing[@]} -gt 0 ]; then echo "this job image lacks commands the task needs: ${missing[*]}" >&2; '
        "exit 1; fi"
    )


def python_module(module: str, args: list[str], *, deps: list[str] = ()) -> str:
    """Run `python -m module` in the job's locked environment (see `prelude`), after installing `deps` (the job's
    runtime stack outside uv.lock, e.g. vLLM or a student's pinned transformers/TRL). The stack is resolved together
    with the wheel, not under the lock's constraints: its pins win where they conflict with the lock, and the
    package's own requirements still hold."""
    lines = []
    if deps:
        packages = " ".join(q(dep) for dep in deps)
        lines.append(f'uv pip install -q --python {VENV_PYTHON} "$WHEEL" {packages}')
    lines.append(f"{VENV_PYTHON} -m {module} {' '.join(args)}")
    return "\n".join(lines)


def hardware_prices() -> dict[str, float]:
    """USD per minute of each Jobs flavor (from the Hub)."""
    from huggingface_hub import HfApi

    return {hw.name: float(hw.unit_cost_usd) for hw in HfApi().list_jobs_hardware() if hw.unit_label == "minute"}


def estimate(jobs: list[Job], prices: dict[str, float] | None = None) -> dict[str, float]:
    """Expected and worst-case (every job runs to its timeout) USD of GPU/CPU time."""
    prices = prices if prices is not None else hardware_prices()
    expected = worst = 0.0
    for job in jobs:
        price = prices.get(job.flavor)
        if price is None:
            raise SystemExit(f"unknown Jobs flavor {job.flavor!r}")
        expected += price * (job.expected_minutes or timeout_minutes(job.timeout))
        worst += price * timeout_minutes(job.timeout)
    return {"expected_usd": round(expected, 2), "worst_case_usd": round(worst, 2)}


def describe(jobs: list[Job], namespace: str) -> None:
    for job in jobs:
        print(f"\n[dry-run] job {job.name}: image={job.image} flavor={job.flavor} timeout={job.timeout} "
              f"namespace={namespace or '(your account)'} env={job.env} "
              f"secrets={','.join(('HF_TOKEN', *job.secret_envs))}")  # fmt: skip
        print(job.command)


def launch(jobs: list[Job], namespace: str) -> list[str]:
    from huggingface_hub import HfApi, get_token

    api, token = HfApi(), get_token()
    if not token:
        raise SystemExit("no Hugging Face token (set HF_TOKEN or run `hf auth login`)")
    missing = sorted({name for job in jobs for name in job.secret_envs if not os.environ.get(name)})
    if missing:
        raise SystemExit(f"jobs need secrets missing from this environment: {', '.join(missing)}")
    ids = []
    for job in jobs:
        kwargs: dict[str, Any] = {"namespace": namespace} if namespace else {}
        secrets = {"HF_TOKEN": token, **{name: os.environ[name] for name in job.secret_envs}}
        info = api.run_job(image=job.image, command=["bash", "-c", job.command], env=job.env or None,
                           secrets=secrets, flavor=job.flavor, timeout=job.timeout, name=job.name,
                           **kwargs)  # fmt: skip
        print(f"launched {job.name}: {info.id} ({info.url})", flush=True)
        ids.append(info.id)
    return ids


def wait(ids: list[str], namespace: str) -> None:
    from huggingface_hub import HfApi

    if not ids:
        return
    kwargs: dict[str, Any] = {"namespace": namespace} if namespace else {}
    infos = HfApi().wait_for_job(ids, poll_interval=60, **kwargs)
    infos = infos if isinstance(infos, list) else [infos]
    failed = [(info.id, info.status.stage) for info in infos if "COMPLETED" not in str(info.status.stage).upper()]
    if failed:
        raise SystemExit(f"jobs did not complete: {failed}")
