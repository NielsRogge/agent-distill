"""HF Jobs: code staging, job commands, launching, waiting, cost estimates (DESIGN.md §7, §8).

Code staging: a wheel of the current checkout plus the task folder go to the run's private `<run>-code` model repo
under `code/<rev>/`; every job downloads them and installs the wheel, so no GitHub credential ever enters a job.
Jobs run in the `[hub] bill_to` namespace with the caller's token as the only secret.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HF_CLI = "uvx --from 'huggingface-hub>=1.13' hf"
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Job:
    stage: str
    name: str
    image: str
    flavor: str
    timeout: str  # e.g. "4h"
    command: str
    env: dict[str, str] = field(default_factory=dict)
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


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()  # fmt: skip
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stage_code(code_repo: str, task_dir: Path, extra: list[Path], *, dry_run: bool, manifest: dict[str, Any]) -> str:
    """Upload the wheel, the task folder, extra files and the run manifest under code/<rev>/; returns <rev>."""
    rev = time.strftime("%Y%m%d-%H%M%S")
    if dry_run:
        print(f"[dry-run] would stage the agent-distill wheel, {task_dir} and {len(extra)} files to {code_repo} "
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


def prelude(code_repo: str, rev: str, setup: list[str] = ()) -> str:
    """Shell lines every job starts with: uv, the staged code, the task's own setup commands."""
    return "\n".join(
        [
            "set -euo pipefail",
            "unset UV_OVERRIDE UV_INDEX_STRATEGY UV_PRERELEASE",
            "command -v uvx >/dev/null || python3 -m pip install -q uv",
            'export PATH="$HOME/.local/bin:$PATH"',
            "nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true",
            "mkdir -p /work && cd /work",
            f"{HF_CLI} download {q(code_repo)} --include 'code/{rev}/*' --local-dir /work/stage >/dev/null",
            f"cp -r /work/stage/code/{rev}/. /work/",
            "WHEEL=$(ls /work/agent_distill-*.whl)",
            *setup,
        ]
    )


def python_module(module: str, args: list[str], *, extras: str = "hub", deps: list[str] = ()) -> str:
    """Run `python -m module` in a fresh uv environment with the staged wheel (and `deps`)."""
    withs = " ".join(f"--with {q(dep)}" for dep in deps)
    return (
        f'uv run --no-project --python 3.12 --python-preference only-managed --with "$WHEEL[{extras}]" {withs} '
        f"python -m {module} {' '.join(args)}"
    )


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
              f"namespace={namespace or '(your account)'} env={job.env} secrets=HF_TOKEN")  # fmt: skip
        print(job.command)


def launch(jobs: list[Job], namespace: str) -> list[str]:
    from huggingface_hub import HfApi, get_token

    api, token = HfApi(), get_token()
    if not token:
        raise SystemExit("no Hugging Face token (set HF_TOKEN or run `hf auth login`)")
    ids = []
    for job in jobs:
        kwargs: dict[str, Any] = {"namespace": namespace} if namespace else {}
        info = api.run_job(image=job.image, command=["bash", "-c", job.command], env=job.env or None,
                           secrets={"HF_TOKEN": token}, flavor=job.flavor, timeout=job.timeout, name=job.name,
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
