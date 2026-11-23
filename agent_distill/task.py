"""Task definition (DESIGN.md §3) and loading a task from its `task.py`."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
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
    filter: Callable[[Row], bool] | None = None
    revision: str | None = None


@dataclass(frozen=True)
class FileAnswer:
    """The answer is a file the agent writes in its workspace (e.g. `index.html`)."""

    path: str


@dataclass
class Task:
    name: str
    dataset: DatasetSpec
    prompt: Callable[[Row], str] | str
    answer: Any  # a pydantic model class or a FileAnswer
    rubrics: RubricSet
    row_id: Callable[[Row], str]
    system_prompt: str | Path | None = None
    tools: Sequence[str] = ("shell", "read", "search")
    setup: Callable[[Row, Path], None] | None = None
    network: bool = True
    limits: Limits = field(default_factory=Limits)
    version: str = "1"

    def render_prompt(self, row: Row) -> str:
        if callable(self.prompt):
            return self.prompt(row)
        return self.prompt.format(**row)

    def system_prompt_text(self) -> str | None:
        if self.system_prompt is None:
            return None
        if isinstance(self.system_prompt, Path):
            return self.system_prompt.read_text().strip()
        return self.system_prompt


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
