"""judged.py with the efficiency variant of the hf-cli skill (`skills-efficient/hf-cli`).

That skill is the pinned, generated `skills/hf-cli/SKILL.md` plus `skills-efficient/ADDENDUM.md`, which covers
batching lookups, `--expand` and filters that agents otherwise look up with `--help`. Everything else (questions,
rubrics, judge, limits) is judged.py's. It is a separate task so that published hf-hub-qa runs stay comparable;
the run config sets the tool-call cap. Run it with `task = "judged_efficient.py"`.
"""

from __future__ import annotations

import dataclasses

from .judged import task as judged
from .task import HERE

task = dataclasses.replace(judged, name="hf-hub-qa-judged-efficient", skills=(HERE / "skills-efficient" / "hf-cli",))

__all__ = ["task"]
