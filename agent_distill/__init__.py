"""agent-distill: distill frontier agents into small open models inside a fixed agent harness.

See docs/DESIGN.md for the design and the milestone plan.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agent-distill")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = ["__version__"]
