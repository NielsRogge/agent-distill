"""The `distill` command line.

Commands not implemented yet report the milestone that implements them (docs/DESIGN.md §14) and exit with
status 3 (`EXIT_NOT_IMPLEMENTED`), distinct from argparse's usage-error status 2.
"""

from __future__ import annotations

import argparse
import sys

from agent_distill import __version__

DESCRIPTION = "The `distill` command line."  # not __doc__: `python -OO` strips docstrings
EXIT_NOT_IMPLEMENTED = 3  # argparse uses 2 for usage errors

# command -> (help, milestone that implements it)
COMMANDS: dict[str, tuple[str, str]] = {
    "run": ("Run pipeline stages for a run.toml (traces, sft, merge, probe, grpo, eval, publish)", "M2-M3"),
    "rescore": ("Recompute rubric scores of published episode records", "M1"),
    "publish": ("Credential-scan and publish a run's Hub artifacts", "M2"),
    "smoke": ("Two-episode end-to-end check on HF Jobs", "M3"),
    "rubrics": ("Draft and calibrate rubrics", "M7"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distill", description=DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"agent-distill {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (help_text, _) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text, description=help_text)
    return parser


# Implemented commands parse their own arguments.
IMPLEMENTED = {
    "rescore": "agent_distill.rescore:main",
    "run": "agent_distill.pipeline:main",
    "smoke": "agent_distill.pipeline:smoke_main",
}


def main(argv: list[str] | None = None) -> int:
    items = list(sys.argv[1:] if argv is None else argv)
    if items and items[0] in IMPLEMENTED:
        from agent_distill.hooks import resolve

        return int(resolve(IMPLEMENTED[items[0]])(items[1:]) or 0)
    parser = build_parser()
    args, _unparsed = parser.parse_known_args(items)  # stubs accept and ignore their future arguments
    if args.command is None:
        parser.print_help()
        return 0
    milestone = COMMANDS[args.command][1]
    message = f"distill {args.command}: not implemented yet (milestone {milestone}, see docs/DESIGN.md §14)"
    print(message, file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    raise SystemExit(main())
