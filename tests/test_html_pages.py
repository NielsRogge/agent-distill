from __future__ import annotations

import asyncio
import os
import shutil
import sys

import pytest

from agent_distill.browser.preview import PageReport
from agent_distill.episode import Episode, ToolEvent
from agent_distill.rubrics import episode_context
from agent_distill.rubrics.judge import JudgeRunner
from agent_distill.task import FileAnswer, load_task

TASK = load_task("examples/html-pages")
RUBRICS = sys.modules[TASK.rubrics["page_ok"].fn.__module__]
ROW = {"concept_id": "c1", "title": "Tide Clock", "idea": "Tides tell time", "category": "art"}
REPLY = "visual_quality: 7\ncreativity: 9\nconcept_fit: 8\ninteractivity: 6\nSCORE: 8"


def report(**overrides) -> PageReport:
    page = PageReport(path="index.html", size_bytes=10, pixel_stddev=40, animated=True, changed_by_interaction=True,
                      screenshots={"desktop": b"d", "desktop_after": b"a", "phone": b"p"})  # fmt: skip
    for key, value in overrides.items():
        setattr(page, key, value)
    return page


class FakeJudge:
    model = "fake"

    def __init__(self, reply=REPLY):
        self.reply, self.calls = reply, []

    async def complete(self, prompt, images):
        self.calls.append((prompt, images))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def score(monkeypatch, page: PageReport, judge: FakeJudge, events=(), answer="<html lang=en><title>x</title></html>"):
    async def fake_inspect(html, **kw):
        return page

    monkeypatch.setattr(RUBRICS, "inspect_html", fake_inspect)
    episode = Episode(task="html-pages", row_id="c1", harness="pi", harness_version="1.6.0", model="m",
                      answer=answer, answer_mode="file", tool_events=list(events))  # fmt: skip
    context = episode_context(episode, ROW, limits=TASK.limits, task=TASK, judge=JudgeRunner(judge))
    return TASK.rubrics.score(context, mode="eval")


def test_task_shape():
    assert TASK.answer == FileAnswer("index.html") and TASK.judge == "gemini-4-hydrogen" and TASK.extras == ("browser",)
    assert "./preview" in TASK.system_prompt_for("pi", "1.6.0", "submit_tool")
    assert "Tide Clock" in TASK.render_prompt(ROW)


def test_setup_writes_the_preview_tool(tmp_path):
    TASK.setup(ROW, tmp_path)
    tool = tmp_path / "preview"
    assert os.access(tool, os.X_OK) and "agent_distill.browser.preview" in tool.read_text()


def test_good_page_is_judged_and_passes(monkeypatch):
    judge = FakeJudge()
    events = [
        ToolEvent(kind="shell", name="bash", input={"command": "./preview"}),
        ToolEvent(kind="submit", name="submit"),
    ]
    result = score(monkeypatch, report(), judge, events)
    assert result.scores["page_ok"] == 1.0 and result.scores["page_quality"] == pytest.approx(7 / 9)
    assert result.passed and result.reward == pytest.approx(7 / 9)
    assert result.metrics["page_quality/creativity"] == pytest.approx(8 / 9)
    assert result.scores["previewed"] == 1.0 and result.scores["accessibility_basics"] == pytest.approx(2 / 3 + 1 / 3)
    ((prompt, images),) = judge.calls
    assert "Tide Clock" in prompt and images == [b"d", b"a", b"p"]


def test_broken_page_is_not_judged(monkeypatch):
    judge = FakeJudge()
    result = score(monkeypatch, report(network_requests=["https://cdn.example/x.js"]), judge)
    assert result.scores["page_ok"] == 0.0 and result.metrics["page_ok/self_contained"] == 0.0
    assert result.scores["page_quality"] is None and result.reward == 0.0 and not result.passed and not judge.calls


def test_low_score_fails_and_judge_errors_are_unscorable(monkeypatch):
    assert not score(monkeypatch, report(), FakeJudge("SCORE: 6")).passed
    failed = score(monkeypatch, report(), FakeJudge(RuntimeError("HTTP 402")))
    assert failed.scores["page_quality"] is None and failed.reward is None and not failed.passed
    assert "HTTP 402" in failed.errors["page_quality"]


def test_submit_without_preview_is_recorded(monkeypatch):
    result = score(monkeypatch, report(), FakeJudge(), [ToolEvent(kind="submit", name="submit")])
    assert result.scores["previewed"] == 0.0


def chromium() -> str | None:
    return os.environ.get("AGENT_DISTILL_CHROMIUM") or shutil.which("chromium")


@pytest.mark.skipif(not chromium(), reason="needs a Chromium (AGENT_DISTILL_CHROMIUM)")
def test_preview_on_real_pages(tmp_path):
    pytest.importorskip("playwright")
    from agent_distill.browser.preview import inspect_page

    good = tmp_path / "good.html"
    good.write_text("<!doctype html><title>t</title><body style='background:linear-gradient(red,blue);height:100vh'>"
                    "<button id=b>0</button><script>b.onclick=()=>b.textContent++</script>")  # fmt: skip
    bad = tmp_path / "bad.html"
    bad.write_text("<script src='https://cdn.example/x.js'></script><div style='width:3000px;height:20px'></div>"
                   "<script>document.onclick=()=>missing()</script>")  # fmt: skip
    ok = asyncio.run(inspect_page(good))
    assert all(ok.gates().values()) and set(ok.screenshots) == {"desktop", "desktop_after", "phone"}
    broken = asyncio.run(inspect_page(bad)).gates()
    assert not broken["self_contained"] and not broken["no_errors"] and not broken["no_overflow"]
    assert not broken["not_blank"]
