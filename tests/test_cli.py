import subprocess
import sys

import pytest

import agent_distill
from agent_distill.cli import COMMANDS, IMPLEMENTED, main


def test_version_is_set():
    assert agent_distill.__version__


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


@pytest.mark.parametrize("command", sorted(set(COMMANDS) - set(IMPLEMENTED)))
def test_unimplemented_commands_exit_2(command, capsys):
    assert main([command, "--anything"]) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_version_flag():
    result = subprocess.run([sys.executable, "-m", "agent_distill.cli", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith("agent-distill ")


def test_rescore_has_its_own_parser(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["rescore", "--help"])
    assert exit_info.value.code == 0
    assert "--task" in capsys.readouterr().out
