"""The cross-format SFT recipe (CoT-proxy paper, arXiv 2611.33148): empty-think targets, visible-text stripping,
CoT-proxy summaries and their acceptance rules, the annotate retry loop, the KL-replay update plan, the thinking-on
student profile and the pipeline's annotate / replay / sft jobs."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from agent_distill import cot_proxy, pipeline
from agent_distill.config import load_profile, load_run
from agent_distill.reasoning import leaked
from agent_distill.stages import annotate, replay, sft

SMOKE = Path("examples/paper-urls/run-smoke.toml")


class QwenLikeTokenizer:
    """Renders like Qwen3.5's chat template for a single-question agent episode (every assistant turn after the
    question gets a think block, empty without reasoning); one token per character."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        out = []
        for m in messages:
            if m["role"] in ("system", "user"):
                out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            elif m["role"] == "assistant":
                reasoning = (m.get("reasoning_content") or "").strip()
                text = f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{m.get('content') or ''}"
                for call in m.get("tool_calls") or []:
                    text += (
                        f"<tool_call>{call['function']['name']}:{json.dumps(call['function']['arguments'])}</tool_call>"
                    )
                out.append(text + "<|im_end|>\n")
            else:
                out.append(f"<|im_start|>user\n<tool_response>{m['content']}</tool_response><|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n<think>\n")
        return "".join(out)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def call(id_, name="bash", command="hf models info x"):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps({"command": command})}}


RECORD = {
    "row_id": "r1",
    "passed": True,
    "harness": "pi",
    "harness_version": "1.6.0",
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is the license?"},
        {"role": "assistant", "content": "Let me look it up.", "reasoning_content": "I should call hf.",
         "tool_calls": [call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "license: mit"},
        {"role": "assistant", "content": "The license is **MIT**.", "reasoning_content": "It says mit."},
    ],
}  # fmt: skip


def supervised_texts(example: dict) -> list[str]:
    return [example["prompt"][a:b] for a, b in json.loads(example["spans"])]


# --------------------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------------------


def test_native_target_supervises_the_think_close_but_empty_target_does_not():
    tok = QwenLikeTokenizer()
    messages = sft.trajectory(RECORD, keep_reasoning=False)
    (native,) = sft.render_trajectory_examples([messages], tok, [])
    assert all(text.startswith("\n</think>") for text in supervised_texts(native))  # today's action-only recipe
    (empty,) = sft.render_trajectory_examples([messages], tok, [], "empty")
    texts = supervised_texts(empty)
    assert texts[0].startswith("Let me look it up.<tool_call>bash") and texts[1].startswith("The license is")
    assert all("think>" not in text for text in texts) and all(text.endswith("<|im_end|>\n") for text in texts)
    # the empty block is in the context of every turn, history included
    assert empty["prompt"].count("<think>\n\n</think>\n\n") == 2


def test_strip_visible_text_keeps_only_calls_and_the_final_answer():
    messages = sft.trajectory(RECORD, keep_reasoning=False, strip_visible_text=True)
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert assistant[0]["content"] == "" and assistant[0]["tool_calls"]
    assert assistant[1]["content"] == "The license is **MIT**."  # TextAnswer: the final message is the answer


def test_summaries_go_before_the_action_and_are_supervised():
    tok = QwenLikeTokenizer()
    summaries = {("r1", 0): "Plan to read the license with hf."}  # turn 1 was rejected: an empty block
    messages = sft.trajectory(RECORD, keep_reasoning=False, strip_visible_text=True, summaries=summaries)
    (example,) = sft.render_trajectory_examples([messages], tok, [], "empty")
    first, second = supervised_texts(example)
    assert first.startswith("<summary>Plan to read the license with hf.</summary><tool_call>bash")
    assert second.startswith("<summary></summary>\n\nThe license is **MIT**.")


def test_empty_target_refuses_reasoning_or_a_template_without_think_blocks():
    tok = QwenLikeTokenizer()
    with pytest.raises(ValueError, match="reasoning in its think block"):
        sft.render_trajectory_examples([sft.trajectory(RECORD, keep_reasoning=True)], tok, [], "empty")

    class NoThink(QwenLikeTokenizer):
        def apply_chat_template(self, *a, **k):
            return super().apply_chat_template(*a, **k).replace("<think>\n", "").replace("\n</think>\n\n", "")

    with pytest.raises(ValueError, match="renders no think block"):
        sft.render_trajectory_examples([sft.trajectory(RECORD, keep_reasoning=False)], NoThink(), [], "empty")


def test_per_turn_examples_put_the_empty_block_in_the_prompt():
    rows = sft.render_examples([sft.trajectory(RECORD, keep_reasoning=False)], QwenLikeTokenizer(), [], "empty")
    assert all(row["prompt"].endswith("<think>\n\n</think>\n\n") for row in rows)
    assert rows[0]["completion"].startswith("Let me look it up.")


def test_whole_turn_truncation_keeps_complete_turns_only():
    tok = QwenLikeTokenizer()
    (example,) = sft.render_trajectory_examples([sft.trajectory(RECORD, keep_reasoning=False)], tok, [], "empty")
    first_end = json.loads(example["spans"])[0][1]
    (row,) = sft.tokenize_examples([example], tok, first_end + 5, whole_turns=True)
    assert len(row["input_ids"]) == first_end and sum(row["completion_mask"]) == len(supervised_texts(example)[0])
    (cut,) = sft.tokenize_examples([example], tok, first_end + 5)
    assert len(cut["input_ids"]) == first_end + 5  # the old rule cuts inside the second turn


# --------------------------------------------------------------------------------------------------
# CoT-proxies
# --------------------------------------------------------------------------------------------------

THOUGHT = (
    "The overview gives rounded parameter counts, so I should read the safetensors metadata of each of the "
    "three models with hf models info and sum the totals. I will run the three lookups in parallel now."
)


@pytest.mark.parametrize(
    ("summary", "reason"),
    [
        ("Overview counts are rounded. Plan to sum the exact safetensors totals.", None),
        ("Version 3.5 is out, e.g. on the Hub. Plan to check it.", None),  # decimal and abbreviation: two sentences
        ("One. Two. Three.", "more than two sentences"),
        ("N/A", "empty or placeholder"),
        ("```bash\nhf models info x\n```", "code fence"),
        ("- read the metadata", "list marker"),
        ("<think>sum</think>", "special token or tag"),
        ("word " * 61, "too long"),
    ],
)
def test_summary_acceptance_rules(summary, reason):
    assert cot_proxy.rejection(summary, THOUGHT) == reason


def test_a_summary_may_not_be_longer_than_the_thought_or_unfinished():
    assert cot_proxy.rejection("Plan to read the metadata now.", "Read it.") == "longer than the reasoning"
    assert cot_proxy.rejection("Plan to read it.", THOUGHT, finished=False) == "unfinished"


def test_annotate_retries_rejected_drafts_with_the_next_seed_then_gives_up():
    items = annotate.turns([RECORD, {**RECORD, "row_id": "r2"}])
    assert [(i["row_id"], i["turn"]) for i in items] == [("r1", 0), ("r1", 1), ("r2", 0), ("r2", 1)]
    items.append({"row_id": "r3", "turn": 0, "thought": ""})  # no reasoning: never sent, empty summary
    seen = []

    def generate(conversations, seed):
        seen.append((seed, len(conversations)))
        assert all(
            c[1]["content"].startswith("Reasoning:\n") and "license" not in c[1]["content"] for c in conversations
        )
        return [("Plan to call hf." if seed == 1 else "One. Two. Three.", True) for _ in conversations]

    results = {(r["row_id"], r["turn"]): r for r in annotate.annotate(items, generate)}
    assert seen == [(0, 4), (1, 4), (2, 2)]  # turn 1's thought is shorter than the draft: rejected three times
    assert results[("r1", 0)]["summary"] == "Plan to call hf." and results[("r1", 0)]["attempts"] == 2
    assert results[("r1", 0)]["rejections"] == ["more than two sentences"]
    assert results[("r3", 0)] == {**results[("r3", 0)], "summary": "", "attempts": 0}
    assert results[("r1", 1)]["rejections"][1] == "longer than the reasoning"
    assert cot_proxy.summaries_by_turn(list(results.values()))[("r1", 1)] == ""  # rendered as <summary></summary>


def test_annotator_sees_only_the_thought():
    system, user = cot_proxy.messages(THOUGHT)
    assert system["content"].startswith("Summarize the writer's reasoning in at most two short sentences.")
    assert user["content"] == f"Reasoning:\n{THOUGHT}"


# --------------------------------------------------------------------------------------------------
# KL replay
# --------------------------------------------------------------------------------------------------


def test_update_plan_mixes_17_teacher_and_3_replay_samples_for_8_passes():
    plan = sft.update_plan(599, 1000, epochs=8, teacher_per_update=17, replay_per_update=3, seed=0)
    assert len(plan) == 282 and all(len(r) == 3 for _, r in plan)
    assert all(len(t) == 17 for t, _ in plan[:-1]) and len(plan[-1][0]) == 599 * 8 - 17 * 281
    counts = [0] * 599
    for teacher, _ in plan:
        for index in teacher:
            counts[index] += 1
    assert set(counts) == {8}
    replayed = [i for _, r in plan for i in r]
    assert len(replayed) == 846 and len(set(replayed)) == 846  # without replacement until the pool runs out


def test_replay_rows_supervise_the_completion_only():
    row = sft.replay_row({"prompt_ids": [1, 2, 3], "completion_ids": [4, 5]}, 10)
    assert row == {"input_ids": [1, 2, 3, 4, 5], "completion_mask": [0, 0, 0, 1, 1]}
    assert sft.supervised_positions(row["completion_mask"]) == [2, 3]  # logits that predict tokens 4 and 5
    assert sft.replay_row({"prompt_ids": [1, 2, 3], "completion_ids": [4]}, 3) is None


def test_replay_prompts_are_out_of_domain_roots():
    assert replay.prompt_of("openr1-math", {"problem": " 2+2? "}) == "2+2?"
    root = {"role": "prompter", "parent_id": None, "lang": "en", "text": "Write a haiku."}
    assert replay.prompt_of("oasst", root) == "Write a haiku."
    assert replay.prompt_of("oasst", {**root, "parent_id": "p"}) is None
    assert replay.prompt_of("oasst", {**root, "lang": "de"}) is None


def test_sft_arguments_of_the_paper_recipe_parse():
    args = sft.parse_args(["--task", "t", "--harness", "pi@1.6.0", "--student", "qwen3.5-4b-think", "--traces", "x",
                           "--think-target", "empty", "--strip-visible-text", "--replay-repo", "o/r",
                           "--summaries-repo", "o/s", "--lora-rank", "16", "--lora-alpha", "32", "--lora-dropout", "0",
                           "--learning-rate", "3e-5", "--warmup-steps", "2", "--epochs", "8"])  # fmt: skip
    assert (args.think_target, args.teacher_per_update, args.replay_per_update, args.max_grad_norm) == (
        "empty",
        17,
        3,
        1.0,
    )


# --------------------------------------------------------------------------------------------------
# Profile and pipeline
# --------------------------------------------------------------------------------------------------


def test_thinking_profile_serves_with_the_reasoning_parser():
    from agent_distill.stages.eval import vllm_command

    student = load_profile("students", "qwen3.5-4b-think")
    command = vllm_command(student, student.model, "")
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert "enable_thinking" not in " ".join(command)
    generation = json.loads(command[command.index("--override-generation-config") + 1])
    assert generation == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert student.serve.max_output_tokens == 8192 and not student.sft.keep_reasoning
    assert student.sft.deps == [*pipeline.SFT_DEPS, "flash-linear-attention==0.6.0"]  # the default stack + fla


def test_parsed_reasoning_is_not_a_leak():
    class Episode:
        messages = ({"role": "assistant", "content": "MIT", "reasoning_content": "<think>is it mit?</think>"},)
        answer = "MIT"

    assert not leaked(Episode())


def test_pipeline_builds_annotate_replay_and_recipe_sft_jobs():
    run = pipeline.make_run(load_run(SMOKE, {"sft.replay": True, "sft.summaries": True, "student": "qwen3.5-4b-think",
                                              "repos": {"replay": "nielsr/qwen35-4b-replay"},
                                              "sft.args": ["--think-target", "empty"]}))  # fmt: skip
    (job,) = pipeline.sft_jobs(run, "R")
    args = shlex.split(next(line for line in job.command.splitlines() if "agent_distill.stages.sft" in line))
    assert args[args.index("--replay-repo") + 1] == "nielsr/qwen35-4b-replay"
    assert args[args.index("--summaries-repo") + 1].endswith("-smoke-summaries")
    assert args.index("--summaries-repo") < args.index("--think-target")  # the run's own args come last
    (ann,) = pipeline.annotate_jobs(run, "R")
    assert "agent_distill.stages.annotate" in ann.command and "Qwen/Qwen3-4B-Instruct-2507" in ann.command
    assert "vllm==0.33.0" in ann.command and ann.flavor == "h200"
    (rep,) = pipeline.replay_jobs(run, "R")
    assert "--output-repo nielsr/qwen35-4b-replay" in rep.command and "--size 1000 --heldout 100" in rep.command
    assert "--student qwen3.5-4b-think" in rep.command
    for built in (job, ann, rep):
        assert "HF_TOKEN" not in built.command
    assert {run.repo("replay"), run.repo("summaries")} == pipeline.sessionless(run)


# --------------------------------------------------------------------------------------------------
# Tool definitions SFT renders (found while checking the phase-1 runs: MCP SFT saw only a phantom `submit`)
# --------------------------------------------------------------------------------------------------


def test_text_answer_tasks_render_no_submit_tool():
    from agent_distill.harnesses.tools import harness_tools
    from agent_distill.task import load_task

    task = load_task(Path("examples/hf-hub-qa"))
    assert [t["function"]["name"] for t in harness_tools("pi@1.6.0", task)] == ["bash", "read"]


def test_mcp_tools_are_rendered_from_the_snapshot_as_pi_declares_them(tmp_path):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from agent_distill.harnesses.tools import harness_tools
    from agent_distill.task import TextAnswer

    snapshot = {"tools": [
        {"name": "hub_repo_details", "description": " Details. ", "inputSchema": {"properties": {"repo_ids": {}}}},
        {"name": "hf-fs", "title": "Files", "inputSchema": {"type": "object"}},
        {"name": "hf_whoami", "description": "not allowed", "inputSchema": {}},
    ]}  # fmt: skip

    @dataclass
    class Server:
        name: str = "hf"
        tools: tuple = ("hub_repo_details", "hf-fs")

        def snapshot_data(self):
            return snapshot

    task = SimpleNamespace(tools=(), answer=TextAnswer(), mcp_servers=(Server(),), answer_schema=lambda: None)
    tools = harness_tools("pi@1.6.0", task)
    assert [t["function"]["name"] for t in tools] == ["mcp__hf__hub_repo_details", "mcp__hf__hf_fs"]
    assert tools[0]["function"]["description"] == "Details."
    assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {"repo_ids": {}}}
    assert tools[1]["function"]["description"] == "Files" and tools[1]["function"]["parameters"]["properties"] == {}
