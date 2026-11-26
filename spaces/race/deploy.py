#!/usr/bin/env python3
"""Deploy the race Space and its student endpoint, from race.toml.

    uv run spaces/race/deploy.py endpoint [--dry-run] [--update]   # the student's vLLM Inference Endpoint
    uv run spaces/race/deploy.py space [--dry-run]                  # the (private) Space

`endpoint` creates the managed Inference Endpoint in `[endpoint] namespace`: the student profile's base model
served by the profile's vLLM version with the `eval` stage's serving arguments (config.endpoint_args), every race's
adapter loaded as a LoRA named after the race, one replica at most and scale-to-zero after
`[endpoint] scale_to_zero_minutes`. `--update` brings an existing endpoint to that configuration.

`space` uploads the bundle (spaces/race, the agent_distill package, the races' task folders, pyproject and uv.lock),
sets the Space's variables (`ENDPOINT_URL`, and nothing secret) and requests cpu-basic. The Space is created private
and never made public here. Its `HF_TOKEN` secret is set by its owner in the Space settings; it needs
`inference.endpoints.infer.write` on the endpoint's namespace and Inference Providers calls billed to the races'
`[hub] bill_to` (`inference.serverless.write` there).

The token of the deploying process comes from `HF_TOKEN` (or the local login) and is never printed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import config as race_config  # noqa: E402

API_ROOT = "https://api.endpoints.huggingface.cloud/v2/endpoint"


def endpoint_payload(cfg: race_config.SpaceConfig, revision: str) -> dict[str, Any]:
    settings, student = cfg.endpoint, cfg.student()
    return {
        "name": settings.name,
        "type": "authenticated",
        "provider": {"vendor": settings.vendor, "region": settings.region},
        "compute": {
            "accelerator": "gpu",
            "instanceType": settings.instance_type,
            "instanceSize": settings.instance_size,
            "scaling": {
                "minReplica": 0,
                "maxReplica": 1,
                "scaleToZeroTimeout": settings.scale_to_zero_minutes,
                "measure": {"pendingRequests": 1.0},
            },
        },
        "model": {
            "repository": student.model,
            "revision": revision,
            "task": "text-generation",
            "framework": "pytorch",
            "image": {
                "vLLM": {
                    "url": f"vllm/vllm-openai:v{student.serve.vllm_version}",
                    "healthRoute": "/health",
                    "port": 8000,
                    "tensorParallelSize": 1,
                    "dataParallelSize": 1,
                    "maxNumSeqs": settings.max_num_seqs,
                }
            },
            "args": cfg.endpoint_args(),
            "env": {},
            "secrets": {},
        },
        "tags": ["agent-distill", "race-space"],
    }


def token() -> str:
    from huggingface_hub import get_token

    value = get_token()
    if not value:
        raise SystemExit("no Hugging Face token (HF_TOKEN or `hf auth login`)")
    return value


def deploy_endpoint(cfg: race_config.SpaceConfig, *, dry_run: bool, update: bool) -> None:
    import httpx
    from huggingface_hub import HfApi

    student = cfg.student()
    settings = cfg.endpoint
    if dry_run:
        payload = endpoint_payload(cfg, "<current revision>")
        print(json.dumps({k: v for k, v in payload.items() if k != "model"}, indent=2))
        print(f"model {student.model} on vllm/vllm-openai:v{student.serve.vllm_version}")
        print("vllm args: " + shlex.join(payload["model"]["args"]))
        print(f"~${settings.price_per_hour:.2f}/h while up; scales to zero after {settings.scale_to_zero_minutes} min")
        return
    headers = {"Authorization": f"Bearer {token()}"}
    url = f"{API_ROOT}/{settings.namespace}/{settings.name}"
    revision = HfApi().model_info(student.model).sha
    with httpx.Client(headers=headers, timeout=60) as client:
        existing = client.get(url)
        if existing.status_code == 200:
            if update:
                desired = endpoint_payload(cfg, revision)
                response = client.put(url, json={"model": desired["model"], "compute": desired["compute"]})
                response.raise_for_status()
                status = response.json().get("status") or {}
                print({"name": settings.name, "state": status.get("state"), "url": status.get("url"), "updated": True})
                return
            status = existing.json().get("status") or {}
            print({"name": settings.name, "state": status.get("state"), "url": status.get("url"), "existing": True})
            return
        if existing.status_code != 404:
            existing.raise_for_status()
        response = client.post(f"{API_ROOT}/{settings.namespace}", json=endpoint_payload(cfg, revision))
        if response.is_error:
            raise SystemExit(f"endpoint creation failed: {response.status_code} {response.text[:500]}")
        status = response.json().get("status") or {}
        print({"name": settings.name, "state": status.get("state"), "url": status.get("url")})


# --------------------------------------------------------------------------------------------------
# Space
# --------------------------------------------------------------------------------------------------

SPACE_CARD = """---
title: {title}
emoji: 🏁
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
pinned: false
short_description: A distilled 4B agent races its teacher in the same harness
---

# {title}

Built from [agent-distill](https://github.com/NielsRogge/agent-distill) `spaces/race` (see its README for the
design, configuration and security model). Races: {races}.

The student runs on the Inference Endpoint `{namespace}/{endpoint}` (one {gpu}, scales to zero after {idle} idle
minutes; the first race after that waits for a cold start). The teacher runs on Hugging Face Inference Providers.

- `HF_TOKEN` (secret): calls the student endpoint (`inference.endpoints.infer.write` on `{namespace}`) and
  Inference Providers for the teacher, billed to `{bill_to}` (`inference.serverless.write` there). It never reaches
  an agent: `start.sh` moves it into a one-time file and out of every environment, and model calls go through the
  library's in-process key proxy.
- `ENDPOINT_URL` (variable): the student endpoint's URL (set by `deploy.py space`).
"""

DOCKERFILE = """FROM python:3.12-slim-bookworm

# Everything is installed as root and stays read-only to agents, so an agent can change neither the app nor the tools
# of a later race. The app runs as root: the Pi adapter then gives every agent its own UID (agent isolation).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \\
    XDG_CACHE_HOME=/opt/cache \\
    HF_HUB_DISABLE_UPDATE_CHECK=1 \\
    PYTHONUNBUFFERED=1
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \\
      ca-certificates curl xz-utils procps >/dev/null && rm -rf /var/lib/apt/lists/* \\
 && pip install --no-cache-dir uv==0.9.5
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY agent_distill ./agent_distill
# The locked dependencies (hash-pinned, as in HF Jobs), then the package itself.
RUN uv export --frozen --no-dev --no-emit-project --extra hub --extra space -o /tmp/requirements.txt -q \\
 && uv venv /opt/app --python "$(command -v python3)" -q \\
 && uv pip install --python /opt/app/bin/python --require-hashes -r /tmp/requirements.txt -q \\
 && uv pip install --python /opt/app/bin/python --no-deps . -q
COPY examples ./examples
COPY spaces ./spaces
# Each race's task setup (its `job_setup`: e.g. the pinned hf CLI) and the pinned harness, installed at build time.
ENV PATH=/opt/app/bin:$PATH
RUN python spaces/race/setup.py > /tmp/setup.sh && bash -euo pipefail /tmp/setup.sh \\
 && python spaces/race/setup.py --install-harnesses \\
 && touch /opt/cache/agent-distill/.pi-install.lock && chmod 666 /opt/cache/agent-distill/.pi-install.lock \\
 && chmod -R a+rX /opt /app

# Runtime state (episodes, caches) lives under /tmp. No HF_HOME: agents inherit the app's environment, and one
# pointing into the app's private home makes their `hf` CLI fail on its token file. Each agent gets its own HOME (and
# so its own, empty Hugging Face cache) from the Pi adapter.
ENV HOME=/tmp/home
EXPOSE 7860
WORKDIR /app/spaces/race
CMD ["bash", "start.sh"]
"""


def task_folders(cfg: race_config.SpaceConfig) -> list[Path]:
    folders = []
    for race in cfg.races:
        folder = race.config.path(race.config.task).resolve()
        folders.append(folder if folder.is_dir() else folder.parent)
    return list(dict.fromkeys(folders))


def build_bundle(cfg: race_config.SpaceConfig, bundle: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(ROOT / "agent_distill", bundle / "agent_distill", ignore=ignore)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, bundle / name)
    shutil.copytree(HERE, bundle / "spaces" / "race", ignore=ignore)
    for folder in task_folders(cfg):
        relative = folder.relative_to(ROOT)
        shutil.copytree(folder, bundle / relative, ignore=ignore)
    first = cfg.races[0].config
    card = SPACE_CARD.format(
        title=cfg.title, races=", ".join(f"{r.name} (`{r.run.removeprefix('../../')}`)" for r in cfg.races),
        namespace=cfg.endpoint.namespace, endpoint=cfg.endpoint.name, gpu=cfg.endpoint.instance_type,
        idle=cfg.endpoint.scale_to_zero_minutes, bill_to=first.hub.bill_to or "your account",
    )  # fmt: skip
    (bundle / "README.md").write_text(card)
    (bundle / "Dockerfile").write_text(DOCKERFILE)


def deploy_space(cfg: race_config.SpaceConfig, *, dry_run: bool, hardware: str) -> None:
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        build_bundle(cfg, bundle)
        files = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file())
        if dry_run:
            print(f"{len(files)} files for {cfg.space} (private, {hardware}):")
            print("\n".join(f"  {f}" for f in files if not f.startswith("agent_distill/")))
            print("  agent_distill/... (" + str(sum(f.startswith("agent_distill/") for f in files)) + " files)")
            return
        api = HfApi()
        endpoint = api.get_inference_endpoint(cfg.endpoint.name, namespace=cfg.endpoint.namespace)
        if not endpoint.url:
            raise SystemExit(f"endpoint {endpoint.name} has no URL yet (status {endpoint.status})")
        api.create_repo(cfg.space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
        if not api.space_info(cfg.space).private:
            raise SystemExit(f"{cfg.space} is public; Spaces stay private (AGENTS.md). Make it private first.")
        api.add_space_variable(cfg.space, "ENDPOINT_URL", endpoint.url)
        api.upload_folder(repo_id=cfg.space, repo_type="space", folder_path=str(bundle),
                          commit_message="Deploy the race Space from agent-distill spaces/race")  # fmt: skip
        api.request_space_hardware(cfg.space, hardware)
        print(f"https://huggingface.co/spaces/{cfg.space} (private, {hardware}); student endpoint {endpoint.url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["endpoint", "space"])
    parser.add_argument("--config", type=Path, default=race_config.CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update", action="store_true", help="bring an existing endpoint to the configuration")
    parser.add_argument("--hardware", default="cpu-basic")
    args = parser.parse_args()
    cfg = race_config.load(args.config)
    if args.what == "endpoint":
        deploy_endpoint(cfg, dry_run=args.dry_run, update=args.update)
    else:
        deploy_space(cfg, dry_run=args.dry_run, hardware=args.hardware)


if __name__ == "__main__":
    main()
