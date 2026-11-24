"""Guards of the training stages: built-in profiles' arguments parse, hook modules are staged and resolvable, the
smoke run trains and evaluates its own models, the GRPO prompt pool (probe scope, tiers, size) and harness checks."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_distill import jobs, pipeline
from agent_distill.config import available_profiles, load_profile, load_run
from agent_distill.stages import eval as eval_stage
from agent_distill.stages import grpo, sft, traces

RUN_TOML = Path("examples/paper-urls/run.toml")
SMOKE_TOML = Path("examples/paper-urls/run-smoke.toml")
TASK_DIR = Path("examples/paper-urls").resolve()


def module_args(command: str, module: str) -> list[str]:
    """The arguments a job command passes to `python -m module`."""
    line = next(line for line in command.splitlines() if f"python -m {module} " in line)
    return shlex.split(line.split(f"python -m {module} ", 1)[1])


def make_run(path: Path = SMOKE_TOML, **overrides):
    return pipeline.make_run(load_run(path, overrides))


# --------------------------------------------------------------------------------------------------
# Profiles' arguments parse (finding: Inkling GRPO exited 2 on --episode-timeout / --exclude-tiers / --tier-weights)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("student", available_profiles("students"))
def test_every_builtin_student_profiles_stage_arguments_parse(student):
    run = make_run(student=student)
    (job,) = pipeline.grpo_jobs(run, "R")
    args = grpo.parse_args(module_args(job.command, "agent_distill.stages.grpo"))
    assert args.launch and args.student == student and args.traces_repo == run.repo("traces")
    profile = load_profile("students", student)
    # SFT: the profile's arguments first (as sft.main does), then the job's
    (sft_job,) = pipeline.sft_jobs(run, "R")
    sft.parse_args([*profile.sft.args, *module_args(sft_job.command, "agent_distill.stages.sft")])
    for probe_job in pipeline.probe_jobs(run, "R"):
        eval_stage.parse_args(module_args(probe_job.command, "agent_distill.stages.eval"))
    # the smoke run's GRPO arguments come last
    grpo.parse_args([*module_args(job.command, "agent_distill.stages.grpo"), *pipeline.SMOKE["grpo.args"]])


def test_inkling_grpo_arguments_end_to_end():
    run = make_run(RUN_TOML, student="inkling-2-mini")  # no run-level [grpo] args
    (job,) = pipeline.grpo_jobs(run, "R")
    args = grpo.parse_args(module_args(job.command, "agent_distill.stages.grpo"))
    assert args.trainer == "simple" and args.lora_r == 64 and args.prompts_per_step == 2
    assert args.timeout == 300  # --episode-timeout (the profile's) after the recipe's --timeout
    assert args.exclude_tiers == "7" and args.tier_weights == "0:3,2:2,3:3,4:3,5:3,6:3"
    assert args.vllm_extra_args == "--max-num-batched-tokens=16384" and args.num_generations == 4
    assert args.vllm_binary == "$(command -v vllm)"  # the image's own vLLM, as in served_job
    assert '--vllm-binary "$(command -v vllm)"' in job.command
    assert args.include_ids_repo == run.repo("eval")
    assert args.include_ids_repo_file == [f"difficulty/probe/train-{i}of4-qualified-ids.txt" for i in range(4)]
    assert grpo.parse_tiers(args.exclude_tiers, args.tier_weights) == ({7}, {0: 3, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3})


def test_pep723_vllm_students_pin_the_profile_version_in_grpo():
    (job,) = pipeline.grpo_jobs(make_run(), "R")
    args = grpo.parse_args(module_args(job.command, "agent_distill.stages.grpo"))
    assert args.vllm_binary == "" and args.vllm_version == load_profile("students", "qwen3.5-4b").serve.vllm_version


# --------------------------------------------------------------------------------------------------
# Hook modules (`requires`)
# --------------------------------------------------------------------------------------------------


def project(tmp_path: Path, *, hooks: dict[str, str], requires: list[str], merge: str = "") -> Path:
    for name, source in hooks.items():
        (tmp_path / name).write_text(source)
    students = tmp_path / "profiles" / "students"
    students.mkdir(parents=True)
    profile = f'name = "hooked"\nmodel = "org/m"\nrequires = {json.dumps(requires)}\n'
    profile += '[lora]\nadapter_converter = "hook_mod:convert"\n[sft]\ntrain_patch = "hook_patch"\n'
    if merge:
        profile += f'[merge]\nmethod = "{merge}"\n'
    (students / "hooked.toml").write_text(profile)
    run_toml = tmp_path / "run.toml"
    run_toml.write_text(
        f'task = "{TASK_DIR}"\nteacher = "glm-5.3"\nstudent = "hooked"\nharness = "opencode@2.3.2"\n'
        'name = "hooked-run"\n[hub]\nnamespace = "me"\n'
    )
    return run_toml


HOOKS = {"hook_mod.py": "def convert(source, destination):\n    return destination\n", "hook_patch.py": "X = 1\n"}


def test_inkling_profile_lists_every_hook_module():
    student = load_profile("students", "inkling-2-mini")
    staged = {Path(entry).stem for entry in student.requires}
    assert {spec.partition(":")[0] for spec in student.hooks().values()} <= staged
    assert student.hooks()["merge.method"] == "merge_adapter:merge_inkling"


def test_requires_are_resolved_and_hooks_checked(tmp_path):
    run_toml = project(tmp_path, hooks=HOOKS, requires=["hook_mod.py", "hook_patch.py"], merge="hook_mod:merge")
    run = pipeline.make_run(load_run(run_toml))
    files = pipeline.requires_files(run)
    assert files == {name: tmp_path / name for name in HOOKS}
    with pytest.raises(SystemExit, match=r"merge.method = 'hook_mod:merge': hook_mod.py defines no 'merge'"):
        pipeline.check_hooks(run, files)
    (tmp_path / "hook_mod.py").write_text(HOOKS["hook_mod.py"] + "def merge(args, snapshot, token):\n    pass\n")
    pipeline.check_hooks(run, files)
    with pytest.raises(SystemExit, match="neither in `requires` nor in the agent_distill package"):
        pipeline.check_hooks(run, {"hook_mod.py": files["hook_mod.py"]})  # hook_patch not staged


def test_missing_requires_refuse_the_dry_run(tmp_path, monkeypatch):
    run_toml = project(tmp_path, hooks={}, requires=["hook_mod.py", "hook_patch.py"])
    monkeypatch.setattr(jobs, "estimate", lambda built: {"expected_usd": 0, "worst_case_usd": 0})
    with pytest.raises(SystemExit, match=r"requires \['hook_mod.py', 'hook_patch.py'\], not found"):
        pipeline.main([str(run_toml), "--stages", "merge", "--dry-run"])
    pipeline.main([str(run_toml), "--stages", "traces", "--dry-run"])  # teacher-only stages need no student hooks


def test_inkling_run_refuses_until_its_hook_modules_are_in_the_project(monkeypatch):
    monkeypatch.setattr(jobs, "estimate", lambda built: {"expected_usd": 0, "worst_case_usd": 0})
    with pytest.raises(SystemExit, match=r"merge_adapter\.py"):
        pipeline.main([str(SMOKE_TOML), "--set", 'student="inkling-2-mini"', "--stages", "sft", "--dry-run"])


def test_requires_are_staged_with_the_job_code(tmp_path, monkeypatch):
    run_toml = project(tmp_path, hooks=HOOKS, requires=["hook_mod.py", "hook_patch.py"])
    staged = {}

    def stage_code(code_repo, task_dir, extra, *, dry_run, manifest, extras=()):
        staged["extra"] = {path.name: path for path in extra}
        staged["profiles"] = sorted(str(p.relative_to(extra[0])) for p in extra[0].rglob("*.toml"))
        return "REV"

    monkeypatch.setattr(jobs, "stage_code", stage_code)
    monkeypatch.setattr(jobs, "estimate", lambda built: {"expected_usd": 0, "worst_case_usd": 0})
    monkeypatch.setattr(jobs, "launch", lambda wave, namespace: ["id"])
    pipeline.main([str(run_toml), "--stages", "merge"])
    assert staged["extra"]["hook_mod.py"] == tmp_path / "hook_mod.py"
    assert staged["extra"]["hook_patch.py"] == tmp_path / "hook_patch.py"
    assert "students/hooked.toml" in staged["profiles"]


# --------------------------------------------------------------------------------------------------
# distill smoke
# --------------------------------------------------------------------------------------------------


def smoke_config(path: Path, monkeypatch) -> pipeline.RunConfig:
    seen = {}
    monkeypatch.setattr(pipeline, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    pipeline.smoke_main([str(path), "--dry-run"])
    args = pipeline.parse_args(seen["argv"])
    assert args.dry_run
    return load_run(args.run, pipeline.overrides(args))


def test_smoke_evaluates_its_own_models_under_its_own_name(monkeypatch):
    config = smoke_config(RUN_TOML, monkeypatch)  # run.toml has a name and [eval.models]
    assert config.name == "qwen35-9b-paper-urls-parity-opencode-124-smoke"
    run = pipeline.make_run(config)
    assert pipeline.eval_models(run) == {
        "base": "Qwen/Qwen3.5-9B",
        "sft": f"Qwen/Qwen3.5-9B+{run.repo('lora')}",
        "grpo": f"{run.repo('sft-merged')}+{run.repo('grpo-lora')}",
    }
    assert all(repo.startswith("nielsr/qwen35-9b-paper-urls-parity-opencode-124-smoke-") for repo, *_ in
               pipeline.artifacts(run))  # fmt: skip
    assert pipeline.smoke_name(load_run(SMOKE_TOML)) == load_run(SMOKE_TOML).name  # already a smoke name


def test_smoke_without_a_name_gets_the_default_name_with_the_suffix(tmp_path, monkeypatch):
    text = RUN_TOML.read_text().replace('name = "qwen35-9b-paper-urls-parity-opencode-124"\n', "")
    run_toml = tmp_path / "run.toml"
    run_toml.write_text(text.replace('task = "."', f'task = "{TASK_DIR}"') + '\n[repos]\neval = "org/real-eval"\n')
    config = smoke_config(run_toml, monkeypatch)
    assert config.name == "qwen35-9b-paper-urls-kimi-k31-pro-opencode-124-smoke"
    assert config.repos == {} and config.eval.models == {}


# --------------------------------------------------------------------------------------------------
# GRPO prompt pool and harness checks
# --------------------------------------------------------------------------------------------------


def grpo_args(*extra: str) -> argparse.Namespace:
    return grpo.parse_args(["--task", "t", "--harness", "opencode@2.3.2", "--student", "qwen3.5-4b", "--model", "m",
                            *extra])  # fmt: skip


def test_tier_exclusion_and_weights():
    rows = [{"id": i, "difficulty_tier": tier} for i, tier in enumerate([1, 1, 3, 5, 7, None])]
    pool = grpo.weight_by_tier(rows, "7", "3:2,5:3", seed=0)
    counts = {}
    for row in pool:
        counts[row["id"]] = counts.get(row["id"], 0) + 1
    assert counts == {0: 1, 1: 1, 2: 2, 3: 3, 5: 1}  # tier 7 dropped, untiered kept once
    assert grpo.weight_by_tier(rows, "", "", seed=0) is rows
    with pytest.raises(SystemExit, match="bad --tier-weights"):
        grpo.parse_tiers("", "3")
    assert len(grpo.weight_by_tier(rows, "1", "", seed=0, field="other_tier")) == 6  # no tier: nothing dropped


def test_pool_size_guard(capsys):
    args = grpo_args("--trainer", "simple", "--max-steps", "150", "--prompts-per-step", "2")
    with pytest.raises(SystemExit, match=r"136 distinct rows, but 150 steps x 2 prompts per step draw 300"):
        grpo.check_pool(args, 136)
    grpo.check_pool(args, 300)
    grpo.check_pool(grpo_args("--trainer", "simple", "--max-steps", "150", "--prompts-per-step", "2",
                              "--allow-small-pool"), 136)  # fmt: skip
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="draw 200"):  # TRL: one prompt group per accumulation step
        grpo.check_pool(grpo_args("--max-steps", "100", "--gradient-accumulation-steps", "2"), 136)


def test_training_rows_apply_probe_ids_tiers_and_the_guard(monkeypatch):
    rows = [{"paper_id": i, "difficulty_tier": i % 4} for i in range(1, 9)]
    monkeypatch.setattr(eval_stage, "select_rows", lambda task, split, purpose: list(rows))
    monkeypatch.setattr(grpo, "include_ids", lambda args: {"1", "2", "3", "4", "5", "6"})
    args = grpo_args("--trainer", "simple", "--max-steps", "2", "--prompts-per-step", "2", "--exclude-tiers", "3",
                     "--tier-weights", "2:2")  # fmt: skip
    args.task = TASK_DIR
    pool = grpo.training_rows(args)
    assert sorted(row["paper_id"] for row in pool) == [1, 2, 2, 4, 5, 6, 6]
    args.max_steps = 3  # 3 x 2 = 6 draws > 5 distinct rows
    with pytest.raises(SystemExit, match="5 distinct rows"):
        grpo.training_rows(args)


def test_include_ids_read_only_the_current_probe(tmp_path, monkeypatch):
    import huggingface_hub

    files = {
        "difficulty/probe/train-0of2-qualified-ids.txt": "1\n2\n",
        "difficulty/probe/train-1of2-qualified-ids.txt": "3\n",
        "difficulty/probe/train-0of4-qualified-ids.txt": "99\n",  # a stale probe with other shards
    }

    class Api:
        def list_repo_files(self, repo, repo_type):
            return list(files)

    def download(repo, name, repo_type):
        path = tmp_path / name.replace("/", "_")
        path.write_text(files[name])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    wanted = [name for name in files if "of2" in name]
    args = grpo_args("--include-ids-repo", "org/eval", *(a for n in wanted for a in ("--include-ids-repo-file", n)))
    assert grpo.include_ids(args) == {"1", "2", "3"}
    with pytest.raises(SystemExit, match="needs --include-ids-repo-file"):
        grpo.include_ids(grpo_args("--include-ids-repo", "org/eval"))
    with pytest.raises(SystemExit, match="missing"):
        grpo.include_ids(grpo_args("--include-ids-repo", "org/eval", "--include-ids-repo-file", "difficulty/x.txt"))


def test_grpo_refuses_another_harness_than_the_traces(monkeypatch, capsys):
    monkeypatch.setattr(grpo, "trace_harnesses", lambda repo: {"opencode@1.2.4"})
    with pytest.raises(SystemExit, match=r"came from \['opencode@1.2.4'\], but GRPO runs in opencode@2.3.2"):
        grpo.check_harness(grpo_args("--traces-repo", "org/traces"))
    grpo.check_harness(grpo_args("--traces-repo", "org/traces", "--allow-harness-mismatch"))
    assert "WARNING" in capsys.readouterr().out
    monkeypatch.setattr(grpo, "trace_harnesses", lambda repo: {"opencode@2.3.2"})
    grpo.check_harness(grpo_args("--traces-repo", "org/traces"))
    monkeypatch.setattr(grpo, "trace_harnesses", lambda repo: None)
    grpo.check_harness(grpo_args("--traces-repo", "org/traces"))
    assert "cannot check" in capsys.readouterr().out


def test_lora_dir_relocation_fails_loudly_without_trls_private_attribute(tmp_path):
    args = grpo_args("--scratch-dir", str(tmp_path))
    trainer = SimpleNamespace(_lora_dir="/grpo/.vllm_lora")
    grpo.relocate_lora_dir(trainer, args)
    assert trainer._lora_dir == str(tmp_path.resolve() / "agent-distill-vllm-lora")
    with pytest.raises(RuntimeError, match="_lora_dir"):
        grpo.relocate_lora_dir(SimpleNamespace(), args)


# --------------------------------------------------------------------------------------------------
# Traces: --limit is the total over the shards
# --------------------------------------------------------------------------------------------------


def test_traces_limit_is_split_across_shards(monkeypatch):
    rows = [{"paper_id": i} for i in range(20)]

    def select_rows(task, split, *, start, limit, purpose, sample, seed):
        return rows[start : start + limit] if limit is not None else rows[start:]

    monkeypatch.setattr(traces, "select_rows", select_rows)
    base = ["--task", "t", "--harness", "h", "--teacher", "x", "--limit", "8"]
    picked = [traces.trace_rows(None, traces.parse_args([*base, "--shard", f"{i}/3"])) for i in range(3)]
    assert sorted(row["paper_id"] for shard in picked for row in shard) == list(range(8))
    run = make_run(**{"traces.shards": 3})
    assert all("--limit 8" in job.command and "--shard" in job.command for job in pipeline.trace_jobs(run, "R"))
