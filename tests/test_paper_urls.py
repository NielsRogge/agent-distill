"""The paper-URL rubrics reproduce url-extraction's scorer and GRPO reward (ported from its tests/test_parity.py).

Each case is an OpenCode 1.2.4 session; it is read with `opencode_v1_reader` and scored by the example's rubric
set. Expected values are url-extraction's (fields, attempt_complete, reward with protocol weight 0.5 and the
tool-efficiency bonus).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_distill.harnesses.opencode_v1_reader import episode_from_messages, episode_from_record
from agent_distill.rubrics import episode_context
from agent_distill.task import Limits, load_task, task_module

TASK = load_task("examples/paper-urls")
RUBRICS_MOD = task_module("examples/paper-urls", "rubrics")
verify_fields = RUBRICS_MOD.verify_fields
incomplete_reason = RUBRICS_MOD.incomplete_reason
TOOL_BUDGET, W_TOOL_EFFICIENCY = RUBRICS_MOD.TOOL_BUDGET, RUBRICS_MOD.W_TOOL_EFFICIENCY

ARXIV = "2601.22060"
GITHUB = "https://github.com/Osilly/Vision-DeepResearch"
PAGE = "https://osilly.github.io/Vision-DeepResearch/"
MAX_TOOL_CALLS = 15


def row(**overrides: Any) -> dict[str, Any]:
    base = {
        "paper_id": 1,
        "arxiv_id": ARXIV,
        "title": "Vision-DeepResearch",
        "abstract": "abstract",
        "missing_github": True,
        "missing_project_page": True,
        "known_github_url": None,
        "known_project_page_url": None,
        "github_label_quality": "gold",
        "project_page_label_quality": "silver",
        "reference_github_url": GITHUB,
        "reference_project_page_url": PAGE,
    }
    return {**base, **overrides}


def tool(name: str, tool_input: dict[str, Any], output: str = "", status: str = "completed") -> dict[str, Any]:
    call = f"call-{abs(hash((name, json.dumps(tool_input)))) % 10**8}"
    return {"type": "tool", "id": call, "callID": call, "tool": name,
            "state": {"status": status, "input": tool_input, "output": output}}  # fmt: skip


INFO = tool("bash", {"command": f"hf papers info {ARXIV} --format json"}, f'{{"github_repo": "{GITHUB}"}}')
READ = tool("bash", {"command": f"hf papers read {ARXIV}"}, "paper text")


def answer(github: str | None, page: str | None) -> dict[str, Any]:
    return {"reasoning": "trusted HF metadata", "github_url": github, "project_page_url": page}


def session(parts: list[dict[str, Any]], structured: dict[str, Any] | None) -> tuple[list[dict[str, Any]], Any]:
    if structured is not None:
        parts = [*parts, tool("StructuredOutput", structured, "Structured output submitted successfully")]
    messages = [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "prompt"}]},
        {"info": {"role": "assistant"}, "parts": parts},
    ]
    return messages, structured


CASES = {
    "correct_and_complete": (row(), *session([INFO], answer(GITHUB, PAGE))),
    "verified_null_page": (
        row(project_page_label_quality="verified_null", reference_project_page_url=None),
        *session([INFO, READ], answer(GITHUB, None)),
    ),
    "wrong_github": (row(), *session([INFO], answer("https://github.com/someone/fork", PAGE))),
    "no_answer": (row(), *session([INFO, {"type": "text", "text": "I could not decide."}], None)),
    "json_in_final_text": (
        row(),
        *session([INFO, {"type": "text", "text": "```json\n" + json.dumps(answer(GITHUB, PAGE)) + "\n```"}], None),
    ),
    "quoted_null_is_null": (
        row(project_page_label_quality="verified_null", reference_project_page_url=None),
        *session([INFO, READ], answer(GITHUB, "null")),
    ),
    "alias_accepted": (
        row(reference_github_url="https://github.com/osilly/new-name", reference_github_aliases=[GITHUB]),
        *session([INFO], answer(GITHUB, PAGE)),
    ),
    "metadata_skipped": (row(), *session([READ], answer(GITHUB, PAGE))),
    "page_fetched_after_github": (
        row(),
        *session([INFO, tool("webfetch", {"url": PAGE}, "page")], answer(GITHUB, PAGE)),
    ),
    "tool_call_limit": (
        row(),
        *session(
            [INFO] + [tool("bash", {"command": f"echo {i}"}, "ok") for i in range(MAX_TOOL_CALLS)], answer(GITHUB, PAGE)
        ),
    ),
    "only_github_requested": (
        row(missing_project_page=False, known_project_page_url=PAGE),
        *session([INFO], answer(GITHUB, None)),
    ),
    "unverified_label_unscored": (
        row(project_page_label_quality="unknown"),
        *session([INFO], answer(GITHUB, "https://example.org/guess")),
    ),
}

EXPECTED = {  # (fields, attempt_complete)
    "correct_and_complete": ({"github": True, "project_page": True}, True),
    "verified_null_page": ({"github": True, "project_page": True}, True),
    "wrong_github": ({"github": False, "project_page": True}, True),
    "no_answer": ({"github": False, "project_page": False}, False),
    "json_in_final_text": ({"github": True, "project_page": True}, True),
    "quoted_null_is_null": ({"github": True, "project_page": True}, True),
    "alias_accepted": ({"github": True, "project_page": True}, True),
    "metadata_skipped": ({"github": True, "project_page": True}, False),
    "page_fetched_after_github": ({"github": True, "project_page": True}, False),
    "tool_call_limit": ({"github": True, "project_page": True}, False),
    "only_github_requested": ({"github": True, "project_page": None}, True),
    "unverified_label_unscored": ({"github": True, "project_page": None}, True),
}


def episode(case: str):
    data, messages, structured = CASES[case]
    return data, episode_from_messages(
        messages,
        answer_model=TASK.answer,
        task=TASK.name,
        row_id=str(data["paper_id"]),
        model="p/m",
        structured=structured,
    )


def scored(case: str, mode: str = "eval"):
    data, ep = episode(case)
    context = episode_context(ep, data, limits=Limits(max_tool_calls=MAX_TOOL_CALLS))
    return ep, TASK.rubrics.score(context, mode=mode)


def url_extraction_reward(fields: dict[str, bool | None], complete: bool, tool_calls: int) -> float:
    """url-extraction's `rollout_reward` from the evaluator's record (its tests' `reward_from_record`)."""
    scoredf = [value for value in fields.values() if value is not None]
    reward = (sum(scoredf) / len(scoredf) if scoredf else 0.0) * (1 - 0.5 * (not complete))
    if reward >= 1.0:
        reward += W_TOOL_EFFICIENCY * max(0.0, 1 - tool_calls / TOOL_BUDGET)
    return reward


@pytest.mark.parametrize("case", sorted(CASES))
def test_fields_and_completeness_match_url_extraction(case: str) -> None:
    ep, result = scored(case)
    data = CASES[case][0]
    fields, complete = EXPECTED[case]
    assert verify_fields(data, ep.answer) == fields
    assert (result.scores["protocol"] == 1.0) is complete
    assert (incomplete_reason(data, ep.answer, ep.tool_events, MAX_TOOL_CALLS) is None) is complete


@pytest.mark.parametrize("case", sorted(CASES))
def test_train_reward_matches_url_extraction(case: str) -> None:
    ep, result = scored(case, mode="train")
    fields, complete = EXPECTED[case]
    assert result.reward == pytest.approx(url_extraction_reward(fields, complete, ep.tool_calls))


@pytest.mark.parametrize("case", sorted(CASES))
def test_passed_is_the_rl_success_criterion(case: str) -> None:
    _, result = scored(case)
    fields, complete = EXPECTED[case]
    scoredf = [v for v in fields.values() if v is not None]
    assert result.passed is (complete and bool(scoredf) and all(scoredf) and case != "no_answer")


def test_reward_values() -> None:
    assert scored("correct_and_complete", "train")[1].reward == pytest.approx(
        1.0 + W_TOOL_EFFICIENCY * (1 - 2 / TOOL_BUDGET)
    )
    assert scored("wrong_github", "train")[1].reward == pytest.approx(0.5)
    assert scored("no_answer", "train")[1].reward == 0.0
    assert scored("metadata_skipped", "train")[1].reward == pytest.approx(0.5)
    assert scored("tool_call_limit", "train")[1].reward == pytest.approx(0.5)


def test_shortcut_guard_is_the_only_train_eval_difference() -> None:
    data, messages, structured = CASES["correct_and_complete"]
    messages = json.loads(json.dumps(messages))
    messages[1]["parts"].insert(0, tool("bash", {"command": "hf download nielsr/paper-url-extraction-v1"}, "ok"))
    ep = episode_from_messages(
        messages, answer_model=TASK.answer, task="t", row_id="1", model="m", structured=structured
    )
    context = episode_context(ep, data, limits=Limits(max_tool_calls=MAX_TOOL_CALLS))
    train, evaluation = TASK.rubrics.score(context, mode="train"), TASK.rubrics.score(context, mode="eval")
    assert train.reward == 0.0
    assert evaluation.scores["all_fields_correct"] == 1.0 and evaluation.passed
    assert evaluation.scores["no_shortcut"] == 0.0  # still reported in evaluation
    for name in ("url_correctness", "all_fields_correct", "protocol", "tool_efficiency"):
        assert train.scores[name] == evaluation.scores[name]


def test_no_scoreable_field_is_unscorable() -> None:
    data, ep = episode("correct_and_complete")
    data = dict(data, github_label_quality="unknown", project_page_label_quality="unknown")
    result = TASK.rubrics.score(episode_context(ep, data, limits=Limits(max_tool_calls=15)), mode="train")
    assert result.scores["url_correctness"] is None and result.reward is None and not result.passed


def test_stored_record_round_trip() -> None:
    """A url-extraction record (findings + raw messages) reads back to the same answer and tool events."""
    _, messages, _ = CASES["json_in_final_text"]
    record = {"paper_id": 1, "arxiv_id": ARXIV, "model": "p/m", "findings": None, "opencode_messages": messages,
              "trajectory": [{"role": "system", "content": "sys"}], "evaluation_model": "adapter"}  # fmt: skip
    ep = episode_from_record(record, answer_model=TASK.answer, task=TASK.name, row_id="1")
    assert ep.answer is not None and ep.answer.github_url == GITHUB and ep.answer_mode == "final_json"
    assert [event.kind for event in ep.tool_events] == ["shell"]
    assert ep.messages[0] == {"role": "system", "content": "sys"}
    assert ep.metadata["evaluation_model"] == "adapter"


def test_prompt_matches_url_extraction() -> None:
    assert TASK.render_prompt(row(known_project_page_url="https://x.io")) == "\n".join(
        [
            f"Paper ID: {ARXIV}",
            "Title: Vision-DeepResearch",
            "Abstract: abstract",
            "Find only: GitHub repository, project page",
            "Known GitHub URL: none",
            "Known project page URL: https://x.io/",
        ]
    )


def test_rescore_and_summarize_records() -> None:
    from agent_distill.rescore import rescore, summarize

    records = []
    for case in ("correct_and_complete", "wrong_github", "no_answer"):
        _, messages, structured = CASES[case]
        records.append({"paper_id": 1, "arxiv_id": ARXIV, "model": "p/m", "findings": structured,
                        "opencode_messages": messages, "evaluation_model": "adapter"})  # fmt: skip
    items = rescore(records, TASK, {"1": row()}, limits=Limits(max_tool_calls=15))
    summary = summarize(items)["adapter"]
    assert summary["episodes"] == 3 and summary["passed"] == 1
    metrics = summary["metric_sums"]
    field_accuracy = (metrics["url_correctness/github_correct"] + metrics["url_correctness/project_page_correct"]) / (
        metrics["url_correctness/github_scored"] + metrics["url_correctness/project_page_scored"]
    )
    assert field_accuracy == pytest.approx(3 / 6)
