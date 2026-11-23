"""Code named by profiles and configs as `module:function` (or a bare module), resolved at run time.

Profiles keep model quirks out of the stage code: a student that needs its adapter renamed for vLLM, a training
patch, or a custom merge names the function instead of the stages testing for the model family. The module must
be staged with the job (`requires` in the student profile).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def resolve(spec: str) -> Callable[..., Any] | Any:
    """`"pkg.module:attr"` -> the attribute; `"pkg.module"` -> the module."""
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Hook {spec!r} is not available in this checkout ({exc}); stage the module the profile requires"
        ) from exc
    if not attribute:
        return module
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise RuntimeError(f"Hook {spec!r}: {module_name} has no attribute {attribute!r}") from exc
