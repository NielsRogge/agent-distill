import subprocess
import sys

import pytest

import agent_distill
from agent_distill.cli import COMMANDS, EXIT_NOT_IMPLEMENTED, main


def test_version_is_set():
    assert agent_distill.__version__


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_unimplemented_commands_exit_distinct_code(command, capsys):
    assert EXIT_NOT_IMPLEMENTED != 2  # argparse's usage-error status
    assert main([command, "--anything"]) == EXIT_NOT_IMPLEMENTED
    assert "not implemented yet" in capsys.readouterr().err


def test_version_flag():
    result = subprocess.run([sys.executable, "-m", "agent_distill.cli", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith("agent-distill ")


def test_help_works_without_docstrings():
    result = subprocess.run(
        [sys.executable, "-OO", "-m", "agent_distill.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
