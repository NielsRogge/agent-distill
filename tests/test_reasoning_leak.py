"""Reasoning leakage: detection, the stage guard, record summaries and the non-thinking student profiles."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from agent_distill.config import load_profile
from agent_distill.episode import Episode
from agent_distill.reasoning import LeakGuard, ReasoningLeak, leaked
from agent_distill.records import RecordWriter, read_jsonl, summarize_records
from agent_distill.rubrics import RubricResult, TooManyErrors
from agent_distill.stages.common import Plan, run_episodes
from agent_distill.stages.eval import vllm_command
from agent_distill.task import Limits, load_task

TASK = load_task("examples/paper-urls")


def episode(row_id: str = "1", content: str = "The answer is 42.", answer: object = None) -> Episode:
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": content}]
    return Episode(task="paper-urls", row_id=row_id, harness="pi", harness_version="1.6.0", model="m",
                   messages=messages, answer=answer, usage={"cost": 0.0})  # fmt: skip


@pytest.mark.parametrize(("content", "answer", "leak"), [
    ("The answer is 42.", None, False),
    ("Let me count the files... 27 of them.\n</think>\n\nThe repo has **27** files.", None, True),
    ("<think>\nhmm\n</think>\nanswer", None, True),
    ("<thinking>draft</thinking> final", None, True),
    ("A plain answer.", "reasoning ...</think> final", True),
    ("Use the `think` tool or write <b>bold</b>.", None, False),
])  # fmt: skip
def test_leaked(content, answer, leak):
    assert leaked(episode(content=content, answer=answer)) is leak


def test_guard_stops_above_the_rate_after_enough_episodes():
    guard = LeakGuard(max_rate=0.2, min_episodes=20)
    for _ in range(19):
        guard.record(True)  # too few episodes yet
    with pytest.raises(ReasoningLeak, match="20 of 20 episodes leaked reasoning"):
        guard.record(True)
    assert issubclass(ReasoningLeak, TooManyErrors)  # the stages' stop-and-publish path handles it
    ok = LeakGuard(max_rate=0.2, min_episodes=20)
    for i in range(50):
        ok.record(i % 10 == 0)  # 10%: never stops
    assert ok.leaks == 5
    never = LeakGuard(max_rate=-1, min_episodes=1)
    for _ in range(30):
        never.record(True)
    assert never.leaks == 30


class LeakyHarness:
    """Episodes whose assistant content carries reasoning for the rows in `leaky`."""

    def __init__(self, leaky: set[str]) -> None:
        self.leaky = leaky

    def session(self, endpoints, cache, task):
        @contextlib.asynccontextmanager
        async def runner():
            yield None

        return runner()

    async def healthy(self, runner) -> None:
        return None

    def native_trace(self, ep):
        return None

    async def run_episode(self, runner, task, row, endpoint, limits):
        row_id = task.row_id(row)
        text = "thinking...\n</think>\nanswer" if row_id in self.leaky else "answer"
        return episode(row_id, content=text)


def run(tmp_path, monkeypatch, leaky: set[str], n: int, guard: LeakGuard | None):
    async def ascore(context, mode="eval"):
        return RubricResult(scores={}, reward=1.0, passed=True)

    monkeypatch.setattr(TASK.rubrics, "ascore", ascore)
    monkeypatch.setattr("agent_distill.stages.common.episode_context", lambda ep, row, **kw: ep)
    writer = RecordWriter(tmp_path, "base/test")
    rows = [{"paper_id": i} for i in range(1, n + 1)]
    asyncio.run(run_episodes(LeakyHarness(leaky), TASK, rows, None, writer, Plan(label="base"), limits=Limits(),
                             concurrency=1, cache=tmp_path, leaks=guard))  # fmt: skip
    return read_jsonl(writer.episodes)


def test_stage_marks_counts_and_warns(tmp_path, monkeypatch, capsys):
    records = run(tmp_path, monkeypatch, leaky={"2"}, n=3, guard=LeakGuard())
    assert [r["metadata"]["reasoning_leak"] for r in records] == [False, True, False]
    assert summarize_records(records)["labels"]["base"]["reasoning_leaks"] == 1
    assert "2: WARNING reasoning leaked" in capsys.readouterr().out


def test_stage_stops_when_most_episodes_leak(tmp_path, monkeypatch):
    with pytest.raises(ReasoningLeak):
        run(tmp_path, monkeypatch, leaky={str(i) for i in range(1, 31)}, n=30, guard=LeakGuard(min_episodes=5))
    records = read_jsonl(RecordWriter(tmp_path, "base/test").episodes)
    assert 5 <= len(records) < 30  # recorded what ran, then stopped starting episodes


@pytest.mark.parametrize("name", ["qwen3.5-4b-nothink", "qwen3.5-9b-nothink"])
def test_nothink_profiles_serve_with_thinking_off(name):
    student = load_profile("students", name)
    base = load_profile("students", name.removesuffix("-nothink"))
    assert student.model == base.model and student.lora == base.lora and student.sft == base.sft
    command = vllm_command(student, student.model, "")
    i = command.index("--default-chat-template-kwargs")
    assert command[i + 1] == '{"enable_thinking": false}'
    assert "--default-chat-template-kwargs" not in vllm_command(base, base.model, "")
