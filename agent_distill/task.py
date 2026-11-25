"""Task definition (DESIGN.md §3) and loading a task from its `task.py`."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_distill.rubrics import RubricSet

Row = dict[str, Any]


@dataclass(frozen=True)
class Limits:
    timeout: float = 180.0
    max_tool_calls: int | None = 15
    max_model_calls: int | None = None


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    splits: dict[str, str | list[str]] = field(default_factory=lambda: {"train": "train", "eval": ["test"]})
    filter: Callable[[Row], bool] | None = None  # evaluation rows
    train_filter: Callable[[Row], bool] | None = None  # rows for traces, the probe and GRPO (default: `filter`)
    revision: str | None = None
    # Label datasets `distill rescore` reads rows from, in precedence order (the first dataset holding a row id
    # wins); default: `repo` alone. E.g. v1 labels first, then a later dataset for rows v1 lacks (test_hard).
    label_repos: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileAnswer:
    """The answer is a file the agent writes in its workspace (e.g. `index.html`)."""

    path: str


@dataclass(frozen=True)
class TextAnswer:
    """The answer is the agent's final message, in natural language: no submit tool, the run ends when the agent
    stops calling tools (Pi only, so far)."""


@dataclass
class Task:
    name: str
    dataset: DatasetSpec
    prompt: Callable[[Row], str] | str
    answer: Any  # a pydantic model class, a FileAnswer or a TextAnswer
    rubrics: RubricSet
    row_id: Callable[[Row], str]
    system_prompt: str | Path | None = None
    tools: Sequence[str] = ("shell", "read", "search")
    setup: Callable[[Row, Path], None] | None = None
    network: bool = True
    limits: Limits = field(default_factory=Limits)
    version: str = "1"
    # Harness-specific system prompts, keyed by "name@version", "name@major" or "name" (most specific wins), e.g.
    # {"opencode@2": Path("system_prompt_v2.txt")}. `{schema}` is replaced by the answer's JSON schema.
    system_prompts: dict[str, str | Path] = field(default_factory=dict)
    # Shell commands every job of this task runs first (tools the agent's prompt relies on, e.g. the `hf` CLI).
    job_setup: list[str] = field(default_factory=list)
    # The judge profile its judge rubrics use unless a run picks another (`--judge`), e.g. "qwen3.8-27b".
    judge: str = ""
    # Agent Skills (directories holding a SKILL.md) the agent may load, e.g. a pinned `hf-cli` skill. Only harnesses
    # with skill support pass them (Pi: `--skill`); everything else is discovered from nowhere (`--no-skills`).
    skills: tuple[Path, ...] = ()
    # agent-distill extras the task's episode stages need beyond `hub`, e.g. ("browser",) for the `preview` tool.
    extras: tuple[str, ...] = ()
    # The limits a stored record ran with when the record does not say (`distill rescore`), from its file name,
    # e.g. url-extraction's teacher trace files ran with 20 tool calls, its evaluations with 15. None: `limits`.
    record_limits: Callable[[str], Limits | None] | None = None

    def render_prompt(self, row: Row) -> str:
        if callable(self.prompt):
            return self.prompt(row)
        return self.prompt.format(**row)

    def system_prompt_text(self) -> str | None:
        return _text(self.system_prompt)

    def answer_schema(self) -> dict[str, Any] | None:
        """The answer's JSON schema (None for file answers), without the model docstring."""
        if isinstance(self.answer, (FileAnswer, TextAnswer)) or not hasattr(self.answer, "model_json_schema"):
            return None
        return self.answer.model_json_schema()

    def system_prompt_for(self, harness: str, version: str, answer_mode: str) -> str | None:
        """The system prompt a harness sends.

        A harness-specific prompt (`system_prompts`) wins. Otherwise the task's prompt, plus, for harnesses that
        receive the answer as a JSON-only final message (`final_json`), the answer contract with the schema.
        """
        major = version.split(".")[0]
        for key in (f"{harness}@{version}", f"{harness}@{major}", harness):
            if key in self.system_prompts:
                return self._with_schema(_text(self.system_prompts[key]) or "")
        prompt = self.system_prompt_text()
        schema = self.answer_schema()
        if answer_mode == "final_json" and schema is not None:
            contract = FINAL_JSON_CONTRACT.replace("{schema}", _schema_json(schema))
            return f"{prompt}\n\n{contract}" if prompt else contract
        return prompt

    def _with_schema(self, text: str) -> str:
        schema = self.answer_schema()
        return text.replace("{schema}", _schema_json(schema)) if schema is not None else text


FINAL_JSON_CONTRACT = (
    "When you are done, reply with only one JSON object and no other text. "
    "It is validated against this JSON Schema: {schema}"
)


def _schema_json(schema: dict[str, Any]) -> str:
    return json.dumps({key: value for key, value in schema.items() if key != "description"}, separators=(",", ":"))


def _text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.read_text().strip()
    return value


def _task_package(folder: Path) -> str:
    """Register a task folder as a package with a unique name (its folder name may contain a hyphen)."""
    package = "agent_distill_tasks._" + hashlib.sha1(str(folder).encode()).hexdigest()[:12]
    if "agent_distill_tasks" not in sys.modules:
        namespace = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("agent_distill_tasks", None))
        namespace.__path__ = []
        sys.modules["agent_distill_tasks"] = namespace
    if package not in sys.modules:
        init = folder / "__init__.py"
        spec = importlib.util.spec_from_file_location(package, init, submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[package] = module
        if init.exists():
            spec.loader.exec_module(module)  # type: ignore[union-attr]
    return package


def task_module(path: str | Path, name: str = "task") -> Any:
    """Import a module of a task folder (e.g. its `rubrics`), so siblings can import each other relatively."""
    path = Path(path).resolve()
    folder = path if path.is_dir() else path.parent
    return importlib.import_module(f"{_task_package(folder)}.{name}")


def load_task(path: str | Path, attribute: str = "task") -> Task:
    """Import `task` from a task folder's `task.py` (or the given file)."""
    path = Path(path).resolve()
    module = task_module(path, "task" if path.is_dir() else path.stem)
    task = getattr(module, attribute, None)
    if not isinstance(task, Task):
        raise TypeError(f"{path} does not define a Task named {attribute!r}")
    return task
