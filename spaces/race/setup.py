"""Build-time setup of the Space image, from race.toml (run by the Dockerfile as root).

    python setup.py                       # print each race task's `job_setup` as one shell script (and save its PATH)
    python setup.py --install-harnesses   # install every race's pinned harness into the cache

The task's `job_setup` is what every HF Job of that task runs first (e.g. the pinned `hf` CLI), so the agents here get
exactly the tools they had in training and evaluation.
"""

from __future__ import annotations

import sys

import config as race_config

from agent_distill.harnesses.base import default_cache, load_harness

PATH_FILE = "/opt/race.path"


def main() -> None:
    cfg = race_config.load()
    if "--install-harnesses" in sys.argv:
        for profile in {race.config.harness: race.harness() for race in cfg.races}.values():
            print(f"installed {profile.id}: {load_harness(profile).install(default_cache())}", file=sys.stderr)
        return
    seen: set[str] = set()
    for race in cfg.races:
        task = race.task()
        if task.name in seen or not task.job_setup:
            continue
        seen.add(task.name)
        print(f"# {task.name}\n" + "\n".join(task.job_setup))
    # The PATH the setup built (e.g. the pinned CLI's bin directory), for start.sh: agents inherit it.
    print(f'printf "%s" "$PATH" > {PATH_FILE}')


if __name__ == "__main__":
    main()
