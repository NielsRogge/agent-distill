"""The race Space's configuration (race.toml): which races it offers, each driven by a run config.

Shared by the Space (core.py) and the deploy script (deploy.py). Everything task-, model- and harness-specific comes
from the library: the run config names the task, the teacher and student profiles and the pinned harness; the student
profile's `[serve]` settings decide how the endpoint serves it, exactly as the `eval` stage serves it.
"""

from __future__ import annotations

import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_distill.config import HarnessProfile, RunConfig, Student, Teacher, load_profile, load_run
from agent_distill.harnesses.base import Endpoint, teacher_endpoint
from agent_distill.stages.eval import vllm_command
from agent_distill.task import Limits, Task, load_task

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "race.toml"


@dataclass
class EndpointSettings:
    name: str
    namespace: str
    vendor: str = "aws"
    region: str = "us-east-1"
    instance_type: str = "nvidia-l4"
    instance_size: str = "x1"
    price_per_hour: float = 0.80
    scale_to_zero_minutes: int = 15
    max_num_seqs: int = 4


@dataclass
class Race:
    id: str
    name: str
    run: str
    adapter: str
    student_label: str
    split: str = "test"
    limit: int | None = None
    tools_note: str = ""
    # The row column a typed question goes into (the task's prompt renders the row); typed questions are not graded.
    question_field: str = "question"
    # Loaded from the run config.
    config: RunConfig = field(init=False, repr=False)

    def load(self, base: Path) -> Race:
        self.config = load_run(base / self.run)
        return self

    @property
    def run_dir(self) -> Path:
        return Path(self.config.base_dir)

    def task(self) -> Task:
        return load_task(self.config.path(self.config.task))

    def teacher(self) -> Teacher:
        return load_profile("teachers", self.config.teacher, self.config.profile_paths)

    def student(self) -> Student:
        return load_profile("students", self.config.student, self.config.profile_paths)

    def harness(self) -> HarnessProfile:
        return load_profile("harnesses", self.config.harness, self.config.profile_paths)

    def limits(self) -> Limits:
        """The run's evaluation limits: the race runs under the conditions the published numbers were measured in."""
        return Limits(timeout=self.config.eval.timeout, max_tool_calls=self.config.eval.max_tool_calls)

    def teacher_endpoint(self) -> Endpoint:
        return teacher_endpoint(self.teacher(), self.harness(), self.config.hub.bill_to)

    def student_endpoint(self, endpoint_url: str, key_env: str = "HF_TOKEN") -> Endpoint:
        """The student on the Inference Endpoint, by its adapter's served name, with the profile's serving limits.
        The endpoint needs the token: the harness reaches it through the library's key proxy, never directly."""
        serve = self.student().serve
        return Endpoint(f"{endpoint_url.rstrip('/')}/v1", self.id, key_env=key_env, name=self.student_label,
                        context=serve.max_model_len, max_output_tokens=serve.max_output_tokens,
                        headers={"X-Scale-Up-Timeout": "600"})  # fmt: skip


@dataclass
class SpaceConfig:
    title: str
    lede: str
    space: str
    endpoint: EndpointSettings
    races: list[Race]

    def race(self, race_id: str) -> Race:
        return next(r for r in self.races if r.id == race_id)

    def student(self) -> Student:
        """The one student profile the endpoint serves; every race must share it (one base model, one LoRA each)."""
        students = {r.config.student for r in self.races}
        if len(students) != 1:
            raise ValueError(f"one endpoint serves one student profile; the races use {sorted(students)}")
        return self.races[0].student()

    def endpoint_args(self) -> list[str]:
        """vLLM arguments for the endpoint: the `eval` stage's serving command for the student profile (reasoning
        parser, tool parser, context, evaluation sampling), with every race's adapter as a named LoRA module."""
        student = self.student()
        placeholder = "\0adapter"
        command = vllm_command(student, student.model, placeholder)
        args = command[3:]  # drop `vllm serve <model>`: the endpoint serves its own model folder
        for flag in ("--host", "--port"):
            index = args.index(flag)
            del args[index : index + 2]
        index = args.index("--lora-modules")
        args[index + 1 : index + 2] = [f"{r.id}={r.adapter}" for r in self.races]
        if any(placeholder in arg for arg in args):  # pragma: no cover - the placeholder must be fully replaced
            raise RuntimeError(shlex.join(args))
        return [*args, "--max-loras", str(len(self.races))]


def load(path: Path = CONFIG) -> SpaceConfig:
    data: dict[str, Any] = tomllib.loads(Path(path).read_text())
    races = [Race(**race).load(Path(path).parent) for race in data.pop("races")]
    endpoint = EndpointSettings(**data.pop("endpoint"))
    return SpaceConfig(endpoint=endpoint, races=races, **data)
