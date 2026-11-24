"""Profiles (teachers, students, harnesses) and run configuration (DESIGN.md §2).

One TOML file per profile. Built-in profiles live in `agent_distill/profiles/<kind>/`; a project's own
`profiles/<kind>/` folder (next to its run.toml, or `--profiles DIR`) adds or shadows them. Fields left out take
the defaults below, which describe Qwen3.5 on H200 (url-extraction's defaults).

A run is a `run.toml`:

    task = "examples/paper-urls"
    teacher = "glm-5.2"
    student = "qwen3.5-9b"
    harness = "opencode@1.2.4"

    [hub]
    namespace = "nielsr"
    bill_to = "huggingface"
    collection = "nielsr/..."

    [eval]
    splits = ["test"]
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUILTIN_PROFILES = Path(__file__).resolve().parent / "profiles"
QWEN35_TEXT_LORA_TARGETS = (
    "down_proj,gate_proj,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj"
)
DEFAULT_TRAIN_IMAGE = "huggingface/trl"
UV_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"


# --------------------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------------------


@dataclass
class TraceSettings:
    concurrency: int = 4
    attempts_per_row: int = 1
    accept_partial: bool = True
    episode_timeout: float = 180
    max_tool_calls: int = 20
    shards: int = 3
    flavor: str = "cpu-upgrade"
    timeout: str = "12h"
    baseline_concurrency: int = 4


@dataclass
class Price:
    input: float = 0.0  # USD per 1M tokens
    output: float = 0.0


@dataclass
class Teacher:
    name: str
    model: str  # Hub id
    provider: str = ""  # Inference Providers pin ("" = automatic); also the `:provider` suffix of the model id
    opencode_model: str = ""  # OpenCode PROVIDER/MODEL (default: huggingface/<model>[:provider])
    opencode_model_entry: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)  # e.g. ["image_input"]
    price: Price = field(default_factory=Price)
    notes: str = ""
    traces: TraceSettings = field(default_factory=TraceSettings)

    @property
    def routed_model(self) -> str:
        return f"{self.model}:{self.provider}" if self.provider else self.model

    @property
    def opencode_id(self) -> str:
        return self.opencode_model or f"huggingface/{self.routed_model}"


@dataclass
class ServeSettings:
    image: str = DEFAULT_TRAIN_IMAGE
    vllm: str = "pep723"  # "pep723": install vllm==vllm_version; "image": use the image's `vllm`
    vllm_version: str = "0.33.0"
    flavor: str = "h200"
    timeout: str = "4h"
    deps: list[str] = field(default_factory=list)
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str = ""
    enforce_eager: bool = True
    language_model_only: bool = True
    max_model_len: int = 65536
    max_output_tokens: int = 1024
    gpu_memory_utilization: float = 0.85
    vllm_extra_args: str = ""


@dataclass
class LoraSettings:
    rank: int = 32
    sft_target_modules: str = QWEN35_TEXT_LORA_TARGETS
    grpo_target_modules: str = ""
    adapter_converter: str = ""


@dataclass
class SFTSettings:
    image: str = DEFAULT_TRAIN_IMAGE
    flavor: str = "h200"
    timeout: str = "8h"
    deps: list[str] = field(default_factory=list)
    model_class: str = ""
    train_patch: str = ""
    keep_reasoning: bool = False
    args: list[str] = field(
        default_factory=lambda: [
            "--no-4bit",
            "--max-length",
            "32768",
            "--epochs",
            "1",
            "--full-trajectory",
            "--save-steps",
            "50",
        ]
    )


@dataclass
class MergeSettings:
    method: str = "peft"
    image: str = DEFAULT_TRAIN_IMAGE
    flavor: str = "a100-large"
    timeout: str = "2h"
    deps: list[str] = field(default_factory=lambda: ["torchvision", "pillow"])
    replace_deps: bool = False


@dataclass
class ProbeSettings:
    flavor: str = ""
    timeout: str = "5h"
    shards: int = 4
    sample: int = 300
    k: int = 4
    concurrency: int = 8
    temperature: float = 0.8
    episode_timeout: float = 180
    args: list[str] = field(default_factory=list)


@dataclass
class GRPOSettings:
    image: str = DEFAULT_TRAIN_IMAGE
    flavor: str = "h200x2"
    timeout: str = "8h"
    backend: str = "trl"
    deps: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=lambda: ["--max-concurrent-rollouts", "32"])


@dataclass
class Student:
    name: str
    model: str
    notes: str = ""
    requires: list[str] = field(default_factory=list)
    serve: ServeSettings = field(default_factory=ServeSettings)
    lora: LoraSettings = field(default_factory=LoraSettings)
    sft: SFTSettings = field(default_factory=SFTSettings)
    merge: MergeSettings = field(default_factory=MergeSettings)
    probe: ProbeSettings = field(default_factory=ProbeSettings)
    grpo: GRPOSettings = field(default_factory=GRPOSettings)

    def hooks(self) -> dict[str, str]:
        """Profile field -> the `module[:function]` hook it names (DESIGN.md §2): the modules must be staged with
        the job code (`requires`) or ship in the agent_distill package."""
        named = {
            "lora.adapter_converter": self.lora.adapter_converter,
            "sft.train_patch": self.sft.train_patch,
            "merge.method": "" if self.merge.method == "peft" else self.merge.method,
        }
        return {key: value for key, value in named.items() if value}


@dataclass
class Download:
    url: str = ""  # may contain {version} and {arch} (x64 / arm64)
    sha256: dict[str, str] = field(default_factory=dict)  # arch -> sha256 of the download


@dataclass
class HarnessProfile:
    name: str  # e.g. "opencode"
    version: str  # exact pinned version
    adapter: str  # module:Class implementing agent_distill.harnesses.base.Harness
    answer_mode: str = "native"  # native | submit_tool | final_json
    download: Download = field(default_factory=Download)
    notes: str = ""

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


def slugify(name: str) -> str:
    return "".join(char for char in name.lower().replace(".", "").replace("@", "-") if char.isalnum() or char == "-")


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        default = fields[key].default_factory() if fields[key].default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value, f"{where}.{key}")
        else:
            kwargs[key] = value
    return cls(**kwargs)


KINDS = {"teachers": Teacher, "students": Student, "harnesses": HarnessProfile}


def profile_dirs(kind: str, extra: list[Path] | None = None) -> list[Path]:
    """Search order: project folders first (they shadow built-ins), then the built-in profiles."""
    return [*(Path(p) / kind for p in extra or []), BUILTIN_PROFILES / kind]


def load_profile(kind: str, name: str, extra: list[Path] | None = None) -> Any:
    """Load a profile by name (`qwen3.5-9b`, `opencode@1.2.4`) or path (`my/profile.toml`)."""
    cls = KINDS[kind]
    if name.endswith(".toml"):
        candidates = [Path(name)]
    else:
        candidates = [folder / f"{name}.toml" for folder in profile_dirs(kind, extra)]
    for path in candidates:
        if path.exists():
            with path.open("rb") as handle:
                return _build(cls, tomllib.load(handle), str(path))
    available = ", ".join(available_profiles(kind, extra)) or "none"
    raise FileNotFoundError(f"No {kind[:-1]} profile {name!r} (available: {available})")


def available_profiles(kind: str, extra: list[Path] | None = None) -> list[str]:
    return sorted({path.stem for folder in profile_dirs(kind, extra) for path in folder.glob("*.toml")})


# --------------------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------------------


@dataclass
class HubSettings:
    namespace: str = ""
    bill_to: str = ""  # X-HF-Bill-To for Inference Providers and --namespace for Jobs ("" = your account)
    collection: str = ""


@dataclass
class EvalSettings:
    splits: list[str] = field(default_factory=lambda: ["test"])
    limit: int = 50  # first N rows of each split after the task's filter
    concurrency: int = 8
    timeout: float = 180
    max_tool_calls: int = 15
    # Which models to evaluate: variants of this run (base, sft, grpo, teacher) ...
    variants: list[str] = field(default_factory=lambda: ["base", "sft", "grpo", "teacher"])
    # ... or explicit models, label -> "model[+adapter]" (replaces `variants`)
    models: dict[str, str] = field(default_factory=dict)


@dataclass
class TracesRun:
    split: str = "train"
    limit: int | None = None  # first N train rows after the task's train filter, in total over the shards (None: all)
    shards: int | None = None  # default: the teacher profile's
    attempts: int | None = None  # default: the teacher profile's


@dataclass
class SFTRun:
    args: list[str] = field(default_factory=list)  # appended after the student profile's [sft] args
    timeout: str = ""  # default: the student profile's


@dataclass
class GRPORun:
    args: list[str] = field(default_factory=list)  # appended after the recipe defaults and the profile's args
    timeout: str = ""  # default: the student profile's
    use_probe: bool = True  # train on the difficulty probe's qualified rows (difficulty/ in the eval dataset)


@dataclass
class ProbeRun:
    split: str = "train"
    sample: int | None = None  # default: the student profile's [probe] values
    k: int | None = None
    shards: int | None = None
    timeout: str = ""  # default: the student profile's


@dataclass
class Budget:
    traces_usd: float = 0.0  # 0 = no cap
    judge_usd: float = 0.0
    gpu_usd: float = 0.0


@dataclass
class RunConfig:
    task: str
    student: str
    harness: str
    teacher: str = ""
    name: str = ""
    stages: list[str] = field(default_factory=lambda: ["eval"])
    hub: HubSettings = field(default_factory=HubSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)
    traces: TracesRun = field(default_factory=TracesRun)
    sft: SFTRun = field(default_factory=SFTRun)
    probe: ProbeRun = field(default_factory=ProbeRun)
    grpo: GRPORun = field(default_factory=GRPORun)
    budget: Budget = field(default_factory=Budget)
    repos: dict[str, str] = field(default_factory=dict)  # override any artifact repo by suffix: {"eval": "..."}
    profiles: list[str] = field(default_factory=list)  # extra profile folders, relative to the run.toml
    base_dir: str = "."  # where the run.toml lives (relative paths resolve against it)

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else Path(self.base_dir) / path

    @property
    def profile_paths(self) -> list[Path]:
        return [self.path(p) for p in self.profiles] + [Path(self.base_dir) / "profiles"]


def load_run(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Read a run.toml and apply `overrides` (dotted keys, e.g. {"eval.splits": ["validation"]})."""
    path = Path(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    for dotted, value in (overrides or {}).items():
        target = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            target = target.setdefault(key, {})
        target[leaf] = value
    data.setdefault("base_dir", str(path.resolve().parent))
    return _build(RunConfig, data, str(path))
