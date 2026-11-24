"""GRPO plumbing that runs without GPUs or OpenEnv: rollout folding, advantages, the tool-choice middleware, the
picklable session factory and the reward of a finished rollout (the rubric set's train mode)."""

from __future__ import annotations

import asyncio
import json
import pickle
from types import SimpleNamespace

import pytest
from test_paper_urls import CASES, TASK

from agent_distill.harnesses.opencode_v1_reader import episode_from_messages
from agent_distill.rubrics import episode_context
from agent_distill.stages import grpo
from agent_distill.task import Limits


def turn(prompt, completion, mask, logps):
    return SimpleNamespace(prompt_token_ids=prompt, completion_token_ids=completion, loss_mask=mask,
                           per_token_logps=logps)  # fmt: skip


def test_rollout_chains_fold_extending_turns_and_split_the_rest():
    trace = SimpleNamespace(turns=[
        turn([1, 2], [3, 4], [0, 0, 1, 1], [-0.1, -0.2]),
        turn([1, 2, 3, 4, 5], [6], [0, 0, 0, 0, 0, 1], [-0.3]),  # extends the first sequence
        turn([9], [8, 7], [0, 1, 0], [-0.4, -0.5]),  # a new sequence (history rewritten)
    ])  # fmt: skip
    chains, dropped = grpo.rollout_chains(trace, max_tokens=100)
    assert dropped == 0
    assert chains == [([1, 2, 3, 4, 5, 6], [2, 3, 5], [-0.1, -0.2, -0.3]), ([9, 8, 7], [1], [-0.4])]
    chains, dropped = grpo.rollout_chains(trace, max_tokens=4)
    assert chains == [([9, 8, 7], [1], [-0.4])] and dropped == 3


def test_group_advantages():
    assert grpo.group_advantages([1.0, 1.0]) is None
    assert grpo.group_advantages([0.5]) is None
    advantages = grpo.group_advantages([0.0, 1.0])
    assert advantages[0] < 0 < advantages[1] and advantages[0] == pytest.approx(-advantages[1])


def test_unconstrained_tool_choice_rewrites_forced_choices():
    seen = {}

    async def app(scope, receive, send):
        message = await receive()
        seen["body"] = json.loads(message["body"])
        seen["length"] = dict(scope["headers"])[b"content-length"]

    body = json.dumps({"tool_choice": "required", "messages": []}).encode()
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive():
        return next(messages)

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [(b"content-length", b"1")]}
    asyncio.run(grpo.UnconstrainedToolChoice(app)(scope, receive, None))
    assert seen["body"]["tool_choice"] == "auto"
    assert int(seen["length"]) == len(json.dumps(seen["body"]).encode())


def factory(**overrides):
    rows = [CASES["correct_and_complete"][0], dict(CASES["correct_and_complete"][0], paper_id=2, arxiv_id="2601.1")]
    kwargs = dict(task_dir="examples/paper-urls", harness="opencode@2.3.2", profiles=[], rows=rows, llm_url="",
                  model="m", cache="/tmp/cache", timeout=180, max_tool_calls=15)  # fmt: skip
    return grpo.make_factory(**{**kwargs, **overrides})


def test_factory_pickles_without_the_task_and_maps_prompts_to_rows():
    original = factory()
    assert original.task.name == "paper-urls"  # loaded lazily
    restored = pickle.loads(pickle.dumps(original))
    assert restored._task is None and restored._lock is not None
    prompts = restored.prompt_rows()
    assert [p["row_id"] for p in prompts] == ["1", "2"]
    session = restored.create(prompts[1]["prompt"])
    assert session.row["paper_id"] == 2 and session.initial_messages() == prompts[1]["prompt"]
    with pytest.raises(KeyError):
        restored.create([{"role": "user", "content": "unknown"}])


def test_session_reward_is_the_rubric_sets_train_reward():
    row, messages, structured = CASES["correct_and_complete"]
    session = factory().create([{"role": "user", "content": TASK.render_prompt(row)}])
    session.episode = episode_from_messages(messages, answer_model=TASK.answer, task=TASK.name, row_id="1",
                                            model="m", structured=structured)  # fmt: skip
    session.result = TASK.rubrics.score(episode_context(session.episode, row, limits=Limits(max_tool_calls=15)),
                                        mode="train")  # fmt: skip
    assert session.tool_calls == 2
    assert grpo.rollout_reward(SimpleNamespace(env_reward=session.result.reward)) == pytest.approx(
        1.0 + 0.1 * (1 - 2 / 15)
    )
    assert grpo.rollout_reward(SimpleNamespace(env_reward=None)) is None


def test_vllm_command_for_rollouts():
    from agent_distill.config import load_profile

    args = grpo.parse_args(["--task", "t", "--harness", "opencode@1.2.4", "--student", "qwen3.5-4b", "--model", "m"])
    command = grpo.vllm_command(args, load_profile("students", "qwen3.5-4b"))
    assert "--return-tokens-as-token-ids" in command and "processed_logprobs" in command
    assert "--weight-transfer-config" in command and "--enable-lora" in command and "--language-model-only" in command
    assert command[:5] == ["uvx", "--python", "3.12", "--from", "vllm==0.33.0"] or command[0].endswith("uvx")
