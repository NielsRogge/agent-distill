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


def test_reads_through_a_pinned_root_refuse_a_swapped_directory(tmp_path):
    episode = tmp_path / "episode"
    (episode / "workspace").mkdir(parents=True)
    (episode / "workspace" / "index.html").write_text("mine")
    root_fd = isolation.open_root(episode)
    try:
        path = episode / "workspace" / "index.html"
        assert isolation.read_agent_file(path, episode, root_fd=root_fd) == "mine"
        assert isolation.list_agent_dir(episode / "workspace", episode, root_fd=root_fd) == ["index.html"]
        # The agent swaps its workspace for a symlink to somewhere else, with a same-named file there.
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "index.html").write_text("hf_secret")
        (episode / "workspace").rename(episode / "old")
        (episode / "workspace").symlink_to(outside)
        assert isolation.read_agent_file(path, episode, root_fd=root_fd) is None
        assert isolation.list_agent_dir(episode / "workspace", episode, root_fd=root_fd) == []
        assert isolation.read_agent_file(tmp_path / "outside" / "index.html", episode, root_fd=root_fd) is None
        assert isolation.list_agent_dir(tmp_path / "outside", episode, root_fd=root_fd) == []
    finally:
        os.close(root_fd)


def test_mode_and_warning_when_agents_share_the_stage_user(monkeypatch, capsys):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(isolation, "_warned", False)
    assert isolation.mode() == "stage-user (not root)"
    isolation.preflight()  # nothing to check, but it says so
    assert "agent isolation is off" in capsys.readouterr().err
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv(isolation.SWITCH_ENV, "off")
    assert isolation.mode() == "stage-user (switched off)"


def test_preflight_fails_clearly_without_the_capabilities(monkeypatch):
    monkeypatch.setattr(isolation, "enabled", lambda: True)

    def refuse(path, uid):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(isolation, "give", refuse)
    monkeypatch.setattr(isolation, "kill_user", lambda uid: None)
    with pytest.raises(RuntimeError, match=r"CAP_SETUID.*AGENT_DISTILL_AGENT_USERS=off"):
        isolation.preflight()


@as_root
def test_preflight_passes_as_root():
    isolation.preflight()


def test_system_directories_are_never_opened_up(monkeypatch):
    with pytest.raises(RuntimeError, match="refusing"):
        isolation.grant_read([Path("/usr")])
    from agent_distill.harnesses.pi import task_tool_paths

    monkeypatch.setattr(sys, "base_prefix", "/usr")
    assert Path("/usr") not in task_tool_paths()


@as_root
def test_a_process_that_escaped_the_process_group_is_killed():
    with isolation.agent_user() as uid:
        # setsid: a new session and process group, so killing the agent's group would miss it.
        subprocess.run(["setsid", "-f", "sleep", "600"], user=uid, group=uid, extra_groups=[], check=True)
        assert any(p.stat().st_uid == uid for p in Path("/proc").iterdir() if p.name.isdigit())
    survivors = [p for p in Path("/proc").iterdir() if p.name.isdigit() and _uid(p) == uid]
    assert not [p for p in survivors if _state(p) != "Z"]  # killed (a zombie awaits its reaper at most)


def _uid(proc: Path) -> int | None:
    try:
        return proc.stat().st_uid
    except OSError:
        return None


def _state(proc: Path) -> str:
    try:
        return (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return "gone"


def test_browser_home_is_private_to_this_process():
    from agent_distill.browser.sandbox import browser_home

    home = Path(browser_home())
    assert home.name.startswith("agent-distill-browser-home-") and home.name != "agent-distill-browser-home"
    assert home.stat().st_uid == os.getuid() and home.stat().st_mode & 0o777 == 0o700


@as_root
def test_agent_can_read_sees_parent_directories():
    base = Path(tempfile.mkdtemp(prefix="agent-distill-isolation-test-"))
    try:
        (base / "open").mkdir()
        (base / "open" / "SKILL.md").write_text("---\nname: x\n---\n")
        base.chmod(0o755)
        assert isolation.agent_can_read(base / "open" / "SKILL.md")
        base.chmod(0o700)  # a private parent: the agent cannot reach the skill
        assert not isolation.agent_can_read(base / "open" / "SKILL.md")
        isolation.grant_read([base / "open"])  # grants traversal of the parents too
        assert isolation.agent_can_read(base / "open" / "SKILL.md")
    finally:
        shutil.rmtree(base)


@as_root
def test_agent_can_run_checks_the_path_as_an_agent(monkeypatch):
    base = Path(tempfile.mkdtemp(prefix="agent-distill-isolation-test-"))
    try:
        (base / "bin").mkdir()
        tool = base / "bin" / "private-tool-xyz"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
        monkeypatch.setenv("PATH", f"{base / 'bin'}:/usr/bin:/bin")
        assert isolation.agent_can_run("sh") and not isolation.agent_can_run("no-such-command-xyz")
        base.chmod(0o755)
        assert isolation.agent_can_run("private-tool-xyz")
        base.chmod(0o700)  # on root's PATH, but under a private directory: not the agent's
        assert shutil.which("private-tool-xyz") and not isolation.agent_can_run("private-tool-xyz")
    finally:
        shutil.rmtree(base)
