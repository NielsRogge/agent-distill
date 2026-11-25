"""Agent isolation (agent_distill.harnesses.isolation): agents as their own unprivileged users."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agent_distill.harnesses import isolation

as_root = pytest.mark.skipif(os.geteuid() != 0, reason="switching users needs root")


def test_read_agent_file_refuses_symlinks_and_escapes(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "token"
    secret.write_text("hf_secret")
    (workspace / "index.html").write_text("<html>ok</html>")
    assert isolation.read_agent_file(workspace / "index.html", workspace) == "<html>ok</html>"
    (workspace / "link.html").symlink_to(secret)
    assert isolation.read_agent_file(workspace / "link.html", workspace) is None
    (workspace / "dir").symlink_to(tmp_path)  # a symlinked directory on the way
    assert isolation.read_agent_file(workspace / "dir" / "token", workspace) is None
    os.mkfifo(workspace / "fifo")  # would block a plain read forever
    assert isolation.read_agent_file(workspace / "fifo", workspace) is None
    assert isolation.read_agent_file(workspace / "missing.html", workspace) is None


def test_switch_off_and_non_root(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv(isolation.SWITCH_ENV, "off")
    assert not isolation.enabled()
    monkeypatch.delenv(isolation.SWITCH_ENV)
    assert isolation.enabled()
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert not isolation.enabled()
    with isolation.agent_user() as uid:
        assert uid is None


def test_grant_read_refuses_the_stage_users_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / "cache").mkdir()
    with pytest.raises(RuntimeError, match="home"):
        isolation.grant_read([tmp_path / "cache"])


@as_root
def test_concurrent_agents_get_distinct_uids():
    with isolation.agent_user() as first, isolation.agent_user() as second:
        assert first is not None and second is not None and first != second
        assert first >= isolation.UID_BASE and second >= isolation.UID_BASE


@as_root
def test_agent_user_cannot_touch_the_stage():
    base = Path(tempfile.mkdtemp(prefix="agent-distill-isolation-test-"))  # in /tmp, as episode directories are
    base.chmod(0o755)
    outputs = base / "eval"
    (outputs / "episodes").mkdir(parents=True)
    (outputs / "episodes" / "test.jsonl").write_text("{}\n")
    isolation.private(outputs)
    episode = base / "episode"
    (episode / "workspace").mkdir(parents=True)
    with isolation.agent_user() as uid:
        assert uid is not None
        isolation.give(episode, uid)
        script = (
            f"import os, pathlib, shutil\n"
            f"shutil.rmtree({str(outputs)!r}, ignore_errors=True)\n"
            f"try:\n    open('/proc/1/environ').read(); print('environ')\nexcept OSError: pass\n"
            f"try:\n    os.listdir('/root'); print('root')\nexcept OSError: pass\n"
            f"pathlib.Path({str(episode / 'workspace' / 'index.html')!r}).write_text('mine')\n"
            f"print(os.getuid())\n"
        )
        done = subprocess.run([sys.executable, "-I", "-c", script], user=uid, group=uid, extra_groups=[],
                              capture_output=True, text=True, timeout=60, cwd=episode / "workspace",
                              env={"PATH": os.environ.get("PATH", ""), "HOME": str(episode)})  # fmt: skip
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == [str(uid)]  # neither the stage's environment nor /root was readable
    assert (outputs / "episodes" / "test.jsonl").read_text() == "{}\n"  # the stage's records survived
    assert (episode / "workspace" / "index.html").read_text() == "mine"
    shutil.rmtree(base)


@as_root
def test_leftover_processes_are_killed(tmp_path):
    with isolation.agent_user() as uid:
        process = subprocess.Popen(["sleep", "600"], user=uid, group=uid, extra_groups=[], start_new_session=True)
    assert process.wait(timeout=10) == -9


def test_record_writer_keeps_its_outputs_from_agents(tmp_path, monkeypatch):
    from agent_distill.records import RecordWriter

    monkeypatch.setattr(isolation, "enabled", lambda: True)
    writer = RecordWriter(tmp_path / "eval", "sft/test", raw_root=tmp_path / "eval-raw")
    for root in (tmp_path / "eval", tmp_path / "eval-raw"):
        assert root.stat().st_mode & 0o777 == 0o700  # an agent user can neither list nor delete what is inside
    assert writer.episodes.parent.is_relative_to(tmp_path / "eval")
