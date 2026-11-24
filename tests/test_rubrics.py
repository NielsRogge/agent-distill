from __future__ import annotations

import asyncio

import pytest

from agent_distill.episode import Episode, ToolEvent
from agent_distill.rubrics import (
    JudgeCache,
    JudgeRunner,
    JudgeSpec,
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
    assert result.scores == {"unscorable": None, "with_metrics": 0.5, "broken": None}
    assert result.metrics == {"with_metrics/calls": 3.0}
    assert "RuntimeError: boom" in result.errors["broken"]
    # A rubric error makes the episode unscorable (no reward, not passed), never a fake 0.
    assert result.errored and result.reward is None and not result.passed


def test_metric_errors_do_not_make_the_episode_unscorable():
    @programmatic
    def ok():
        return 1.0

    @metric
    def broken_metric():
        raise RuntimeError("boom")

    result = RubricSet([ok, broken_metric]).score(context())
    assert not result.errored and result.reward == 1.0 and result.passed


@pytest.mark.parametrize("value", [float("nan"), 1.5, -0.1])
def test_scores_outside_unit_interval_are_errors(value):
    bad = Rubric(fn=lambda: value, name="bad")
    result = RubricSet([bad]).score(context())
    assert result.scores["bad"] is None and "must be in [0, 1]" in result.errors["bad"]
    assert result.reward is None


def test_inactive_rubrics_drop_out_of_the_default_reward():
    a = Rubric(fn=lambda: 0.5, name="a")
    b = Rubric(fn=lambda: 0.0, name="b", eval_only=True)
    rubrics = RubricSet([a, b])
    assert rubrics.score(context(), mode="train").reward == pytest.approx(0.5)  # not (0.5 + neutral 1) / 2
    assert rubrics.score(context(), mode="eval").reward == pytest.approx(0.25)


def test_error_budget_stops_after_too_many_unscorable_episodes():
    from agent_distill.rubrics import ErrorBudget, RubricResult, TooManyErrors

    ok = RubricResult(scores={}, reward=1.0, passed=True)
    bad = RubricResult(scores={}, reward=None, passed=False, errors={"j": "HTTP 402"})
    budget = ErrorBudget(max_rate=0.1, min_episodes=5)
    for _ in range(4):
        budget.record(bad)  # below min_episodes: no decision yet
    with pytest.raises(TooManyErrors, match="HTTP 402"):
        budget.record(ok)
    calm = ErrorBudget(max_rate=0.5, min_episodes=2)
    for result in (ok, bad, ok, ok):
        calm.record(result)


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


def test_judge_unparseable_reply_is_unscorable_and_not_cached(tmp_path):
    @judge("x")
    def j():
        return {}

    cache = JudgeCache(tmp_path / "verdicts.jsonl")
    client = FakeClient("no number here")
    result = RubricSet([j]).score(context(judge=JudgeRunner(client, cache)))
    assert result.scores["j"] is None and "no score" in result.errors["j"] and result.reward is None
    assert not (tmp_path / "verdicts.jsonl").exists()
    # The next attempt asks again (and caches the parseable reply).
    client.reply = "Score: 1"
    assert RubricSet([j]).score(context(judge=JudgeRunner(client, cache))).scores["j"] == 1.0
    assert client.calls == 2


def test_cached_reply_that_no_longer_parses_is_asked_again(tmp_path):
    @judge("x")
    def j():
        return {}

    path = tmp_path / "verdicts.jsonl"
    stale = JudgeCache(path)
    stale.put(JudgeCache.key("judge-model", "x", []), "garbage")
    client = FakeClient("Score: 1")
    assert RubricSet([j]).score(context(judge=JudgeRunner(client, JudgeCache(path)))).scores["j"] == 1.0
    assert client.calls == 1


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Overall score: 6 for layout... SCORE: 9", 0.9),
        ("**SCORE:** 8", 0.8),
        ("Score = 7.5", 0.75),
        ("final **Score**: 3", 0.3),
    ],
)
def test_judge_score_parsing_takes_the_last_score(reply, expected):
    from agent_distill.rubrics.judge import parse_score

    assert parse_score(reply, JudgeSpec(prompt="", scale=(0, 10))) == pytest.approx(expected)


def test_empty_judge_reply_is_not_cached(tmp_path):
    import asyncio

    client = FakeClient("")
    runner = JudgeRunner(client, JudgeCache(tmp_path / "verdicts.jsonl"))
    assert asyncio.run(runner("p")) == "" and asyncio.run(runner("p")) == ""
    assert client.calls == 2 and not (tmp_path / "verdicts.jsonl").exists()


def test_judge_criteria_become_metrics():
    @judge("x", scale=(1, 10), criteria=("creativity", "polish"))
    def j():
        return {}

    reply = "creativity: 10\npolish = 4\nSCORE: 7"
    result = RubricSet([j]).score(context(judge=JudgeRunner(FakeClient(reply))))
    assert result.scores["j"] == 6 / 9 and result.metrics == {"j/creativity": 1.0, "j/polish": 3 / 9}


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


def test_criteria_take_the_last_value_and_allow_markdown():
    from agent_distill.rubrics.judge import JudgeSpec, parse_criteria

    spec = JudgeSpec(prompt="", scale=(1.0, 10.0), criteria=("visual_quality", "creativity"))
    reply = "visual_quality: 3 at first glance... on reflection\n**visual_quality:** 8\ncreativity: **6**"
    found = parse_criteria(reply, spec)
    assert found["visual_quality"] == pytest.approx(7 / 9) and found["creativity"] == pytest.approx(5 / 9)


def test_judge_budget_stops_calls_and_needs_a_price(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from agent_distill.rubrics import clients

    price = SimpleNamespace(input=1.5, output=7.5)
    profile = SimpleNamespace(name="g", model="g", api="gemini", key_env="", max_tokens=10, price=price)
    client = clients.judge_client(profile, budget_usd=0.01)
    client.usage.add(1000, 2000, client.price)  # $0.0165
    with pytest.raises(clients.JudgeError, match="budget reached"):
        asyncio.run(client.complete("p", []))
    monkeypatch.setenv(clients.JUDGE_BUDGET_ENV, "5")
    assert clients.judge_client(profile).usage.budget_usd == 5.0
    unpriced = SimpleNamespace(**{**vars(profile), "price": SimpleNamespace(input=0.0, output=0.0)})
    with pytest.raises(ValueError, match="no price"):
        clients.judge_client(unpriced)


def test_jobs_get_a_share_of_the_judge_budget():
    from pathlib import Path

    from agent_distill import pipeline
    from agent_distill.config import load_run

    run = pipeline.make_run(load_run(Path("examples/html-pages/run.toml"), {"budget.judge_usd": 60}))
    built = pipeline.trace_jobs(run, "R")
    pipeline.apply_judge_budget(run, built)
    assert len(built) == 6 and all(job.env["AGENT_DISTILL_JUDGE_BUDGET_USD"] == "10.0000" for job in built)
