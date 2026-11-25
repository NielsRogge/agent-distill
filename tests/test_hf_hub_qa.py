"""The hf-hub-qa example: the facts task (task.py), the judged task (judged.py), seed questions and rubrics (no
network: the reference CLI, the Hub lookups and the judge are faked)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from agent_distill.episode import Episode, ToolEvent
from agent_distill.rubrics import JudgeRunner, episode_context, grounding
from agent_distill.task import TextAnswer, load_task

TASK = load_task("examples/hf-hub-qa/judged.py")  # the judged questions; FACTS is the folder's main task
FACTS = load_task("examples/hf-hub-qa")
REPLY = "correctness: 9\nrelevance: 9\ncompleteness: 8\nclarity: 9\nSCORE: 9"
TRENDING = [{"id": f"org/model-{i}"} for i in range(10)]


def rubrics_module():
    """The judged rubrics module, as load_task imported it."""
    return next(m for name, m in sys.modules.items() if name.endswith("judged_rubrics") and hasattr(m, "reference"))


def facts_module():
    return next(m for name, m in sys.modules.items() if name.endswith(".rubrics") and hasattr(m, "fact_correct"))


class FakeJudge:
    model = "fake"

    def __init__(self, reply=REPLY):
        self.reply, self.calls = reply, []

    async def complete(self, prompt, images):
        self.calls.append(prompt)
        return self.reply


def shell(command: str, output: str = "", success: bool = True) -> ToolEvent:
    return ToolEvent(kind="shell", name="bash", input={"command": command}, success=success, output=output)


def score(monkeypatch, answer, events, row, judge=None, listing=None):
    module = rubrics_module()
    monkeypatch.setattr(module, "run_hf", lambda argv: listing if listing is not None else TRENDING)
    monkeypatch.setattr(grounding, "exists_on_hub", lambda repo_id: repo_id in {"BAAI/bge-m3"})
    episode = Episode(task="hf-hub-qa", row_id=row["question_id"], harness="pi", harness_version="1.6.0", model="m",
                      answer=answer, answer_mode="text", tool_events=list(events))  # fmt: skip
    judge = judge or FakeJudge()
    context = episode_context(episode, row, task=TASK, judge=JudgeRunner(judge))
    return TASK.rubrics.score(context, mode="eval"), judge


def row(question="List the top trending models.", reference=None):
    return {"question_id": "q1", "category": "trending", "question": question,
            "reference": json.dumps(reference) if reference else ""}  # fmt: skip


LISTING = json.dumps(TRENDING)
TRENDING_REF = {"ids": ["models", "list", "--sort", "trending_score", "--limit", "10"], "key": "id", "k": 5}


def test_task_shape():
    assert isinstance(TASK.answer, TextAnswer) and TASK.tools == ("shell", "read") and TASK.network
    assert TASK.skills and (TASK.skills[0] / "SKILL.md").read_text().startswith("---\nname: hf-cli")
    assert "huggingface_hub==2.4.0" in " ".join(TASK.job_setup)
    assert "read-only" in TASK.system_prompt_for("pi", "1.6.0", "text")


def test_seed_questions_have_ids_categories_and_valid_references():
    sys.path.insert(0, "examples/hf-hub-qa")
    try:
        questions = importlib.import_module("questions")
    finally:
        sys.path.pop(0)
    rows = questions.rows()
    assert len(rows) >= 50 and len({r["question_id"] for r in rows}) == len(rows)
    assert len({r["category"] for r in rows}) == 8
    for r in rows:
        if r["reference"]:
            spec = json.loads(r["reference"])
            assert ("ids" in spec and spec["k"] >= 1) or ("fact" in spec and spec["path"])


def test_cited_ids(monkeypatch):
    module = grounding  # shared with the MCP twin (agent_distill.rubrics.grounding)
    monkeypatch.setattr(module, "exists_on_hub", lambda repo_id: repo_id == "BAAI/bge-m3")
    text = ("Try `Qwen/Qwen3.5-9B` or [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), see paper "
            "2205.02340. Use PEFT/LoRA in F16/BF16, or BAAI/bge-m3 for retrieval, and/or more.")  # fmt: skip
    assert module.cited_ids(text) == {"Qwen/Qwen3.5-9B", "HuggingFaceFW/fineweb", "2205.02340", "BAAI/bge-m3"}
    assert module.cited_ids("An invented `Made/Up-Model-7B`.") == {"Made/Up-Model-7B"}
    org_pages = (
        "See [`Qwen`](https://huggingface.co/Qwen/models) and [its datasets](https://huggingface.co/Qwen/datasets)."
    )
    assert module.cited_ids(org_pages) == set()
    shorthand = "Try `NeoQuasar/Kronos-mini/small`, `Kronos-Tokenizer-base/2k` or `mmproj-Qwen3.5-9B-bf16/f16.gguf`."
    assert module.cited_ids(shorthand) == set()


def test_help_on_a_write_command_is_read_only(monkeypatch):
    events = [shell("hf auth login --help", "usage"), shell("hf models list --format json", LISTING)]
    result, _ = score(monkeypatch, "`org/model-0` is trending.", events, row())
    assert result.scores["read_only"] == 1.0


def test_good_grounded_answer_passes(monkeypatch):
    answer = "Trending now: `org/model-0`, `org/model-1`, `org/model-2` and `org/model-3`."
    events = [shell("hf models list --sort trending_score --limit 10 --format json", LISTING)]
    result, judge = score(monkeypatch, answer, events, row(reference=TRENDING_REF))
    assert result.scores["answered"] == result.scores["read_only"] == result.scores["used_hub"] == 1.0
    assert result.scores["grounded"] == 1.0 and abs(result.scores["reference"] - 0.8) < 1e-9
    assert result.passed and result.reward > 0.8
    assert "org/model-0" in judge.calls[0] and "hf models list" in judge.calls[0]


def test_invented_ids_fail_grounding(monkeypatch):
    answer = "Trending: `org/model-0`, `org/model-1`, `org/model-2`, `org/model-3` and `Made/Up-Model-7B`."
    events = [shell("hf models list --sort trending_score --format json", LISTING)]
    result, _ = score(monkeypatch, answer, events, row(reference=TRENDING_REF))
    assert result.scores["grounded"] == 0.8 and not result.passed


def test_answer_from_memory_or_writes_score_zero(monkeypatch):
    memory, _ = score(monkeypatch, "Llama is popular.", [], row())
    assert memory.scores["used_hub"] == 0.0 and memory.reward == 0.0 and not memory.passed
    events = [shell("hf models list --format json", LISTING), shell("hf repos create me/x", "error", success=False)]
    writes, _ = score(monkeypatch, "`org/model-0` is trending.", events, row())
    assert writes.scores["read_only"] == 0.0 and writes.reward == 0.0
    cut, judge = score(monkeypatch, None, [shell("hf models list", LISTING)], row())
    assert cut.scores["answered"] == 0.0 and cut.reward == 0.0 and not judge.calls  # no judge spend


def test_fact_reference(monkeypatch):
    ref = {"fact": ["models", "info", "Qwen/Qwen3.5-9B"], "path": "card_data.license"}
    events = [shell("hf models info Qwen/Qwen3.5-9B --format json", '{"card_data": {"license": "apache-2.0"}}')]
    info = {"card_data": {"license": "apache-2.0"}}
    good, _ = score(
        monkeypatch, "`Qwen/Qwen3.5-9B` is licensed Apache 2.0.", events, row("License?", ref), listing=info
    )
    bad, _ = score(monkeypatch, "Qwen/Qwen3.5-9B is MIT licensed.", events, row("License?", ref), listing=info)
    assert good.scores["reference"] == 1.0 and bad.scores["reference"] == 0.0 and not bad.passed


# --- The facts task (task.py): deterministic, the same rubrics as the MCP twin with the CLI's gates ---------------

FACT_ROW = {"question_id": "f1", "question": "What is the license of Qwen/Qwen3.5-9B?", "kind": "category",
            "gold": json.dumps("apache-2.0"), "extra": json.dumps({"hedge": ["mit", "apache-2.0", "llama3"]}),
            "live": json.dumps({"type": "model", "id": "Qwen/Qwen3.5-9B", "field": "license"})}  # fmt: skip
INFO = shell(
    "hf models info Qwen/Qwen3.5-9B --format json", '{"id": "Qwen/Qwen3.5-9B", "card_data": {"license": "apache-2.0"}}'
)


def score_fact(monkeypatch, answer, events, live="apache-2.0"):
    module = facts_module()
    module.live.cache_clear()
    monkeypatch.setattr(module, "live_value", lambda spec: live)
    monkeypatch.setattr(grounding, "exists_on_hub", lambda repo_id: False)
    episode = Episode(task="hf-hub-qa", row_id="f1", harness="pi", harness_version="1.6.0", model="m",
                      answer=answer, answer_mode="text", tool_events=list(events))  # fmt: skip
    return FACTS.rubrics.score(episode_context(episode, FACT_ROW, task=FACTS), mode="eval")


def test_facts_task_shape():
    assert FACTS.dataset.repo == "nielsr/hf-hub-facts-v2" and FACTS.version == "2"
    assert FACTS.dataset.splits["eval"] == ["test", "validation"]
    assert FACTS.limits.max_tool_calls == 15 and FACTS.limits.timeout == 300  # the MCP twin's limits
    assert FACTS.judge == "" and FACTS.skills == TASK.skills and FACTS.job_setup == TASK.job_setup


def test_right_fact_passes_with_the_efficiency_bonus(monkeypatch):
    result = score_fact(monkeypatch, "`Qwen/Qwen3.5-9B` is licensed Apache 2.0.", [INFO])
    assert result.scores["fact_correct"] == 1.0 and result.scores["grounded"] == 1.0 and result.passed
    assert abs(result.reward - (1.0 + 0.1 * (1 - 1 / 20))) < 1e-9


def test_wrong_fact_fails_and_drift_is_unscorable(monkeypatch):
    wrong = score_fact(monkeypatch, "It is MIT licensed.", [INFO])
    assert wrong.scores["fact_correct"] == 0.0 and not wrong.passed
    drifted = score_fact(monkeypatch, "It is MIT licensed.", [INFO], live="mit")  # the Hub changed since the snapshot
    assert drifted.scores["fact_correct"] is None


def test_fact_gates(monkeypatch):
    memory = score_fact(monkeypatch, "Apache 2.0.", [])
    assert memory.scores["used_hub"] == 0.0 and memory.reward == 0.0 and not memory.passed
    write = score_fact(monkeypatch, "Apache 2.0.", [INFO, shell("hf repos create me/x", "error", success=False)])
    assert write.scores["read_only"] == 0.0 and write.reward == 0.0


def test_jobs_get_the_task_file_a_run_names(tmp_path):
    from agent_distill import pipeline
    from agent_distill.config import load_run

    folder = pipeline.make_run(load_run(Path("examples/hf-hub-qa/run.toml"), {}))
    judged = pipeline.make_run(load_run(Path("examples/hf-hub-qa/run-judged-pilot.toml"), {}))
    assert pipeline.job_task(folder) == "/work/task" and folder.task.name == "hf-hub-qa"
    assert pipeline.job_task(judged) == "/work/task/judged.py" and judged.task.name == "hf-hub-qa-judged"
