from __future__ import annotations

import asyncio

import pytest

from agent_distill.episode import Episode, ToolEvent
from agent_distill.rubrics import (
    JudgeCache,
    JudgeRunner,
    Rubric,
    RubricSet,
    bonus_if,
    episode_context,
    from_callable,
    from_verifiers,
    gate,
    judge,
    metric,
    penalize,
    process,
    programmatic,
    weighted,
)


def make_episode(**overrides) -> Episode:
    base = dict(
        task="t",
        row_id="1",
        harness="pi",
        harness_version="1.6.0",
        model="m",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
        tool_events=[ToolEvent(kind="shell", name="bash", input={"command": "curl https://x.io/a"}, success=True)],
        answer="a",
    )
    return Episode(**{**base, **overrides})


def context(**extra):
    return episode_context(make_episode(), {"answer": "a", "id": 1}, **extra)


def test_injection_by_name_and_kwargs():
    @programmatic
    def exact(row, answer):
        return answer == row["answer"]

    @programmatic(weight=2.0, name="everything")
    def takes_all(**kwargs):
        return 1.0 if {"episode", "row", "tool_events"} <= set(kwargs) else 0.0

    result = RubricSet([exact, takes_all]).score(context())
    assert result.scores == {"exact": 1.0, "everything": 1.0}
    assert takes_all.weight == 2.0 and takes_all.name == "everything"


def test_unknown_argument_is_rejected_at_definition():
    with pytest.raises(TypeError, match="injectable"):

        @programmatic
        def bad(prompt):  # not injectable (that is the verifiers name; use from_verifiers)
            return 1.0


def test_async_none_metrics_and_errors():
    @programmatic
    async def unscorable():
        return None

    @metric
    def with_metrics():
        return 0.5, {"calls": 3}

    @process
    def broken(tool_events):
        raise RuntimeError("boom")

    result = RubricSet([unscorable, with_metrics, broken]).score(context())
    assert result.scores == {"unscorable": None, "with_metrics": 0.5, "broken": 0.0}
    assert result.metrics == {"with_metrics/calls": 3.0}
    assert "RuntimeError: boom" in result.errors["broken"]
    assert result.reward == 0.0  # weighted mean of non-metric, non-None scores: only `broken`


def test_default_reward_is_weighted_mean_and_passed_uses_threshold():
    a = Rubric(fn=lambda: 1.0, name="a", weight=3.0)
    b = Rubric(fn=lambda: 0.0, name="b", weight=1.0)
    result = RubricSet([a, b], threshold=0.75).score(context())
    assert result.reward == pytest.approx(0.75) and result.passed


def test_combinators():
    scores = {"ok": 1.0, "bad": 0.0, "x": 0.8, "none": None}
    assert gate("bad", "x")(scores) == 0.0
    assert gate("ok", "x")(scores) == 0.8
    assert gate("none", "x")(scores) == 0.8  # an unscorable gate passes
    assert penalize("bad", 0.5, "x")(scores) == pytest.approx(0.4)
    assert penalize("ok", 0.5, "x")(scores) == 0.8
    assert bonus_if(0.1, "ok", when_value_at_least=1.0)(scores) == pytest.approx(1.1)
    assert bonus_if(0.1, "x", when_value_at_least=1.0)(scores) == 0.8
    assert bonus_if(0.1, "x", when="ok")(scores) == pytest.approx(0.9)
    assert weighted({"x": 1, "none": 5})(scores) == 0.8
    assert weighted({"none": 1})(scores) is None
    with pytest.raises(ValueError):
        bonus_if(0.1, "x")


def test_train_only_and_eval_only_modes():
    calls = []

    @programmatic(train_only=True)
    def guard():
        return 0.0

    async def fake_judge(prompt, images=None):
        calls.append(prompt)
        return "score: 1"

    @judge("rate {answer}", eval_only=True)
    def quality(answer):
        return {"answer": answer}

    @programmatic
    def base():
        return 1.0

    rubrics = RubricSet([guard, quality, base], reward=gate("guard", weighted({"base": 1, "quality": 1})))
    train = rubrics.score(context(judge=fake_judge), mode="train")
    assert train.reward == 0.0 and "quality" not in train.scores and not calls  # eval-only judge skipped in RL
    evaluation = rubrics.score(context(judge=fake_judge), mode="eval")
    assert evaluation.scores["guard"] == 0.0  # still computed and reported
    assert evaluation.reward == 1.0  # but neutral in the eval reward
    assert calls == ["rate a"]
    with pytest.raises(ValueError):
        Rubric(fn=lambda: 1, name="x", train_only=True, eval_only=True)


class FakeClient:
    model = "judge-model"

    def __init__(self, reply="Reasoning... Score: 4"):
        self.reply, self.calls = reply, 0

    async def complete(self, prompt, images):
        self.calls += 1
        return self.reply


def test_judge_scaling_images_and_cache(tmp_path):
    @judge("Concept: {concept}", scale=(1, 5))
    def visual(row):
        return {"concept": row["answer"], "images": [b"png-bytes"]}

    client = FakeClient()
    cache_path = tmp_path / "verdicts.jsonl"
    runner = JudgeRunner(client, JudgeCache(cache_path))
    rubrics = RubricSet([visual])
    first = rubrics.score(context(judge=runner))
    second = rubrics.score(context(judge=runner))
    assert first.scores["visual"] == pytest.approx(0.75) == second.scores["visual"]
    assert client.calls == 1 and runner.calls == 1
    # The cache persists: a new runner reuses the verdict.
    fresh = JudgeRunner(FakeClient(), JudgeCache(cache_path))
    assert rubrics.score(context(judge=fresh)).scores["visual"] == pytest.approx(0.75) and fresh.calls == 0


def test_judge_unparseable_reply_scores_zero_with_error():
    @judge("x")
    def j():
        return {}

    result = RubricSet([j]).score(context(judge=JudgeRunner(FakeClient("no number here"))))
    assert result.scores["j"] == 0.0 and "no score" in result.errors["j"]


def test_from_callable_accepts_tinker_reward_fn():
    async def reward_fn(messages):
        return float(messages[-1]["content"] == "a"), {"turns": len(messages)}

    result = RubricSet([from_callable(reward_fn, name="tinker")]).score(context())
    assert result.scores["tinker"] == 1.0 and result.metrics["tinker/turns"] == 3.0


def test_from_verifiers_classic_signature():
    def correct_answer(completion, answer, **kwargs):
        assert kwargs["prompt"][-1]["content"] == "q"
        return 1.0 if completion[-1]["content"] == answer else 0.0

    def info_only(info):
        return info["id"]

    rubrics = RubricSet([from_verifiers(correct_answer), from_verifiers(info_only)])
    result = rubrics.score(context())
    assert result.scores == {"correct_answer": 1.0, "info_only": 1.0}


def test_tool_event_accessors():
    shell = ToolEvent(kind="shell", name="bash", input={"command": "curl -s https://x.io/p?q=1 | head"})
    fetch = ToolEvent(kind="fetch", name="webfetch", input={"url": "https://y.io"})
    read = ToolEvent(kind="read", name="read", input={"command": "not a shell"})
    assert shell.command.startswith("curl") and shell.url == "https://x.io/p?q=1"
    assert fetch.url == "https://y.io" and fetch.command == ""
    assert read.command == "" and read.url == ""
    with pytest.raises(ValueError):
        ToolEvent(kind="browse", name="x")


def test_ascore_inside_event_loop():
    @programmatic
    def one():
        return 1

    async def main():
        return await RubricSet([one]).ascore(context())

    assert asyncio.run(main()).reward == 1.0
