"""SFT stage (DESIGN.md §7): LoRA SFT of the student on the teacher's accepted trajectories.

    python -m agent_distill.stages.sft --task task --harness opencode@1.2.4 --student qwen3.5-9b \
        --traces-repo ORG/RUN-traces --hub-model-id ORG/RUN-lora

Ported from url-extraction's `train_sft.py`. Accepted records (`passed`, one per row) of one `harness@version` are
rendered with the student's chat template and the exact tool definitions the harness sends; the loss covers the
assistant turns only (exact completion masks from character offsets). With `--full-trajectory` one example holds a
whole trajectory with every assistant turn supervised, when the template renders each turn's prompt as a prefix of
the full conversation (checked per trajectory, otherwise per-turn examples). LoRA settings, the model class, a
training patch and whether the teacher's reasoning is distilled come from the student profile.

**Cross-format recipe** (Kopiczko et al., "CoT-Proxy", arXiv 2611.33148; docs/DESIGN.md §7, "Cross-format SFT"):

- `--think-target empty`: every assistant turn, history included, is rendered with an empty think block
  (`<think>\n\n</think>\n\n`) that is in the context but not supervised; the loss starts after it. A trajectory longer
  than `--max-length` is cut to its longest prefix of complete turns. The student is then served with thinking on
  (e.g. the `qwen3.5-4b-think` profile) and reasons in its own words before acting.
- `--strip-visible-text`: a turn that calls tools keeps only its calls (the teacher's prose around them is dropped).
  A turn without tool calls is the final natural-language answer of a `TextAnswer` task and keeps its text.
- `--summaries-repo`: CoT-proxies from the `annotate` stage. Each turn becomes `<think></think><summary>S</summary>`
  then its action, with the summary supervised; a turn whose drafts were all rejected gets `<summary></summary>`.
- `--replay-repo`: KL replay (the `replay` stage's pool of the initial student's own out-of-domain generations). Each
  update mixes `--teacher-per-update` teacher trajectories with `--replay-per-update` replay samples; the loss is the
  cross-entropy on teacher tokens plus KL(p_init || p_theta) on replay tokens, each normalised by its own token count,
  with p_init the same model with the LoRA adapter disabled. A custom loop replaces TRL's SFTTrainer for this.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from agent_distill.config import QWEN35_TEXT_LORA_TARGETS, load_profile
from agent_distill.cot_proxy import summary_block
from agent_distill.hub_collection import join_collection
from agent_distill.records import read_jsonl
from agent_distill.task import load_task

MAX_TOOL_CONTENT_CHARS = 8_000
SUBMIT_TOOLS = {"structuredoutput", "submit"}


# --------------------------------------------------------------------------------------------------
# Records -> chat trajectories
# --------------------------------------------------------------------------------------------------


def load_accepted(paths: list[Path], *, harness: str, allow_mismatch: bool = False) -> list[dict[str, Any]]:
    """Accepted records (first per row) from episode JSONL files, all from `harness` (`name@version`).

    DESIGN.md §6 "No mixing of versions": records from another `harness@version` refuse the run, unless
    `allow_mismatch` (`--allow-harness-mismatch`), which keeps them and warns."""
    by_row: dict[str, dict[str, Any]] = {}
    other: dict[str, int] = {}
    for path in paths:
        for record in read_jsonl(path):
            if not record.get("passed"):
                continue
            harness_id = f"{record.get('harness')}@{record.get('harness_version')}"
            if harness_id != harness:
                other[harness_id] = other.get(harness_id, 0) + 1
            by_row.setdefault(str(record["row_id"]), record)
    if other:
        message = f"accepted traces from other harnesses than {harness}: {dict(sorted(other.items()))}"
        if not allow_mismatch:
            raise SystemExit(f"{message}; pass --allow-harness-mismatch to train on them anyway")
        print(f"WARNING: {message} (--allow-harness-mismatch: training on them)", flush=True)
    if not by_row:
        raise ValueError(f"no accepted {harness} trajectories in {[str(p) for p in paths]}")
    return list(by_row.values())


def chat_message(message: dict[str, Any], *, keep_reasoning: bool) -> dict[str, Any]:
    """A record message as the chat template expects it (no timestamps, usage or error flags)."""
    keep = {"role", "content", "tool_calls", "tool_call_id"}
    if keep_reasoning:
        keep.add("reasoning_content")
    result = {key: copy.deepcopy(value) for key, value in message.items() if key in keep}
    if result.get("role") == "tool":
        content = str(result.get("content") or "")
        if len(content) > MAX_TOOL_CONTENT_CHARS:
            result["content"] = content[:6_000] + "\n[tool output compacted for SFT]\n" + content[-2_000:]
    for call in result.get("tool_calls") or []:
        arguments = call.get("function", {}).get("arguments")
        if isinstance(arguments, str):
            call["function"]["arguments"] = json.loads(arguments)
    if result.get("role") == "assistant" and not result.get("content"):
        result["content"] = ""
    return result


def trajectory(
    record: dict[str, Any],
    *,
    keep_reasoning: bool,
    strip_visible_text: bool = False,
    summaries: dict[tuple[str, int], str] | None = None,
) -> list[dict[str, Any]]:
    """The episode's messages up to the answer: a structured-output submission ends the trajectory.

    `strip_visible_text` drops the prose of every turn that calls tools (its action is the calls); a turn without tool
    calls (the final answer of a `TextAnswer` task) keeps its text. `summaries` ({(row_id, assistant turn index):
    summary}) puts a CoT-proxy block before each turn's action."""
    result: list[dict[str, Any]] = []
    submit_id: str | None = None
    turn = -1
    row_id = str(record.get("row_id"))
    for message in record.get("messages") or []:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            continue
        entry = chat_message(message, keep_reasoning=keep_reasoning)
        if entry["role"] == "assistant":
            turn += 1
            if strip_visible_text and entry.get("tool_calls"):
                entry["content"] = ""
            if summaries is not None:
                block = summary_block(summaries.get((row_id, turn), ""))
                entry["content"] = f"{block}\n\n{entry['content']}" if entry["content"].strip() else block
        result.append(entry)
        for call in message.get("tool_calls") or []:
            if str(call.get("function", {}).get("name", "")).lower() in SUBMIT_TOOLS:
                submit_id = call.get("id")
        if submit_id and message.get("role") == "tool" and message.get("tool_call_id") == submit_id:
            break
    return result


# --------------------------------------------------------------------------------------------------
# Rendering and tokenization (url-extraction train_sft.py)
# --------------------------------------------------------------------------------------------------


THINK_CLOSE = "</think>"


def supervised_start(prompt: str, through: str, think_target: str = "native") -> int:
    """Where the supervised part of an assistant turn starts in `through` (the conversation rendered through it).

    `native`: right after the generation prompt, as the template renders it. `empty`: after the turn's empty think
    block, which stays in the context but is not supervised (the paper's prefilled `<think></think>`). The template
    must render that block for this turn, and it must be empty (reasoning is never supervised in this target)."""
    if think_target == "native":
        return len(prompt)
    if think_target != "empty":
        raise ValueError(f"unknown think target {think_target!r}")
    segment = through[len(prompt) :]
    close = segment.find(THINK_CLOSE)
    if close < 0:
        raise ValueError("--think-target empty: the chat template renders no think block for an assistant turn")
    if segment[:close].replace("<think>", "").strip():
        raise ValueError("--think-target empty: an assistant turn has reasoning in its think block (keep_reasoning?)")
    start = len(prompt) + close + len(THINK_CLOSE)
    while start < len(through) and through[start] in "\n ":
        start += 1
    return start


def has_inline_reasoning(messages: list[dict]) -> bool:
    """Does a turn's visible text carry a think block (`<think>` / `</think>` in the content), which the template
    would turn into reasoning? Such a trajectory cannot get the empty-think target; the caller skips and counts it."""
    from agent_distill.reasoning import MARKERS

    return any(m.get("role") == "assistant" and MARKERS.search(str(m.get("content") or "")) for m in messages)


def render_trajectory_examples(
    trajectories: list[list[dict]], tokenizer: Any, tools: list[dict], think_target: str = "native"
) -> list[dict]:
    """One example per trajectory with every assistant turn supervised, when the template is prefix-stable."""
    examples: list[dict[str, Any]] = []
    fallback: list[list[dict]] = []
    inline = 0
    for messages in trajectories:
        full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        spans: list[list[int]] = []
        stable = True
        if think_target == "empty" and has_inline_reasoning(messages):
            inline += 1
            continue
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prompt = tokenizer.apply_chat_template(
                messages[:index], tools=tools, tokenize=False, add_generation_prompt=True
            )
            through = tokenizer.apply_chat_template(
                messages[: index + 1], tools=tools, tokenize=False, add_generation_prompt=False
            )
            if not (full.startswith(prompt) and full.startswith(through) and through.startswith(prompt)):
                stable = False
                break
            start = supervised_start(prompt, through, think_target)
            if through[start:].strip():
                spans.append([start, len(through)])
        if not stable:
            fallback.append(messages)
            continue
        if spans:
            examples.append({"prompt": full, "completion": "", "spans": json.dumps(spans)})
    if inline:
        print(f"skipped {inline} trajectories whose visible text carries a think block (--think-target empty)",
              flush=True)  # fmt: skip
    if fallback:
        print(f"{len(fallback)} trajectories are not prefix-stable; using per-turn examples for them", flush=True)
        examples += [{**row, "spans": ""} for row in render_examples(fallback, tokenizer, tools, think_target)]
    print(f"rendered {len(examples)} examples from {len(trajectories)} trajectories", flush=True)
    return examples


def render_examples(
    trajectories: list[list[dict]], tokenizer: Any, tools: list[dict], think_target: str = "native"
) -> list[dict]:
    """One example per assistant turn: the history as prompt, the turn as completion."""
    examples: list[dict[str, str]] = []
    for messages in trajectories:
        if think_target == "empty" and has_inline_reasoning(messages):
            continue
        history: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "assistant":
                history.append(message)
                continue
            prompt = tokenizer.apply_chat_template(history, tools=tools, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                [*history, message], tools=tools, tokenize=False, add_generation_prompt=False
            )
            if not full.startswith(prompt):
                raise ValueError("the chat template cannot isolate the assistant completion")
            start = supervised_start(prompt, full, think_target)  # the prefilled empty think block joins the prompt
            completion = full[start:]
            if completion.strip():
                examples.append({"prompt": full[:start], "completion": completion})
            history.append(message)
    if not examples:
        raise ValueError("the accepted trajectories contain no assistant turns")
    return examples


def tokenize_examples(
    examples: list[dict], tokenizer: Any, max_length: int, *, whole_turns: bool = False
) -> list[dict[str, list[int]]]:
    """Exact completion masks from character offsets (no BPE boundary drift). With `whole_turns`, a trajectory longer
    than `max_length` is cut to its longest prefix of complete supervised turns (the paper's rule); otherwise it is cut
    at `max_length`, possibly inside a turn."""
    rows: list[dict[str, list[int]]] = []
    for example in examples:
        text = example["prompt"] + example["completion"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = list(encoded["input_ids"])
        if example.get("spans"):
            spans = json.loads(example["spans"])
            offsets = encoded["offset_mapping"]
            mask = [int(any(end > a and start < b for a, b in spans)) for start, end in offsets]
            if len(input_ids) > max_length and whole_turns:
                cut = 0
                for a, b in spans:  # the end of the last span whose tokens all fit
                    last = max((i for i, (start, end) in enumerate(offsets) if end > a and start < b), default=-1)
                    if last >= max_length:
                        break
                    cut = last + 1
                input_ids, mask = input_ids[:cut], mask[:cut]
            elif len(input_ids) > max_length:  # keep the supervised turns that fit
                input_ids, mask = input_ids[:max_length], mask[:max_length]
            if any(mask):
                rows.append({"input_ids": input_ids, "completion_mask": mask})
            continue
        boundary = len(example["prompt"])
        mask = [int(end > boundary) for _, end in encoded["offset_mapping"]]
        if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
            input_ids.append(tokenizer.eos_token_id)
            mask.append(1)
        if len(input_ids) <= max_length:
            rows.append({"input_ids": input_ids, "completion_mask": mask})
    if not rows:
        raise ValueError("no tokenized example fits the max length")
    return rows


def lora_target_modules(value: str) -> Any:
    """Comma-separated module names, or a regex (anything with regex metacharacters)."""
    value = value or QWEN35_TEXT_LORA_TARGETS
    if any(char in value for char in "\\()[]|*+?^$"):
        return value
    return [name.strip() for name in value.split(",") if name.strip()]


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--profiles", action="append", type=Path, default=[])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", nargs="+", type=Path, help="Local episode JSONL files")
    source.add_argument("--traces-repo", help="Dataset repo; reads episodes/**/*.jsonl (or --traces-files)")
    parser.add_argument("--traces-files", nargs="*", help="Files in --traces-repo")
    parser.add_argument("--allow-harness-mismatch", action="store_true")
    parser.add_argument("--model", help="Student checkpoint (default: the profile's model)")
    parser.add_argument("--output", type=Path, default=Path("/sft/adapter"))
    parser.add_argument("--hub-model-id", default="", help="Push the adapter (a public model repo)")
    parser.add_argument("--collection", default="", help="Collection the pushed model joins")
    parser.add_argument("--collection-note", default="", help="Its one-line note in the collection")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (smoke runs)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, help="Default 2e-4; 3e-5 with the cross-format recipe")
    parser.add_argument("--epochs", type=float, help="Default 2; 8 with the cross-format recipe")
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--full-trajectory", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attention-chunk-size", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Render and tokenize only; print statistics")
    recipe = parser.add_argument_group("cross-format recipe (CoT-proxy paper; see the module docstring)")
    recipe.add_argument(
        "--think-target",
        choices=["native", "empty"],
        default="native",
        help="empty: an unsupervised empty think block before every assistant turn's target",
    )
    recipe.add_argument(
        "--strip-visible-text",
        action="store_true",
        help="Drop the prose of turns that call tools (the final answer turn keeps its text)",
    )
    recipe.add_argument("--summaries-repo", default="", help="CoT-proxy summaries (the annotate stage's dataset)")
    recipe.add_argument("--summaries-file", default="summaries/summaries.jsonl")
    recipe.add_argument("--replay-repo", default="", help="KL replay pool (the replay stage's dataset)")
    recipe.add_argument("--replay-file", default="replay/pool.jsonl")
    recipe.add_argument("--heldout-file", default="replay/heldout.jsonl", help="Drift probe samples (KL, reported)")
    recipe.add_argument("--teacher-per-update", type=int, default=17)
    recipe.add_argument("--replay-per-update", type=int, default=3)
    recipe.add_argument("--lora-alpha", type=int, help="Default: twice the rank")
    recipe.add_argument("--lora-dropout", type=float, help="Default 0.05; 0 with the cross-format recipe")
    recipe.add_argument("--warmup-steps", type=int, help="Linear warmup, then constant (default 0; 2 with the recipe)")
    recipe.add_argument(
        "--min-summary-coverage",
        type=float,
        default=0.8,
        help="Refuse summaries covering fewer of the turns with reasoning",
    )
    recipe.add_argument("--max-grad-norm", type=float, default=1.0)
    recipe.add_argument("--seed", type=int, default=0)
    return resolve_defaults(parser.parse_args(argv))


# Defaults by recipe: the cross-format recipe takes the paper's hyperparameters (its Table 10) unless set explicitly.
LEGACY_DEFAULTS = {"learning_rate": 2e-4, "epochs": 2.0, "lora_dropout": 0.05, "warmup_steps": 0, "lora_rank": None}
PAPER_DEFAULTS = {"learning_rate": 3e-5, "epochs": 8.0, "lora_dropout": 0.0, "warmup_steps": 2, "lora_rank": 16}


def cross_format(args: argparse.Namespace) -> bool:
    return args.think_target == "empty" or bool(args.replay_repo) or bool(args.summaries_repo)


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the unset hyperparameters with the recipe's defaults, and refuse inconsistent recipe flags."""
    defaults = PAPER_DEFAULTS if cross_format(args) else LEGACY_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.summaries_repo and args.think_target != "empty":
        raise SystemExit("CoT-proxy summaries go after an empty think block: pass --think-target empty")
    if args.replay_repo and args.batch_size != 1:
        raise SystemExit("the KL-replay loop runs one sample at a time (an update is --teacher-per-update + "
                         "--replay-per-update samples); --batch-size must be 1")  # fmt: skip
    return args


def main(argv: list[str] | None = None) -> None:
    import sys

    args = parse_args(argv)
    student = load_profile("students", args.student, args.profiles)
    if student.sft.args:  # profile arguments first, the command line's after (the last occurrence wins)
        args = parse_args([*student.sft.args, *(sys.argv[1:] if argv is None else argv)])
    task = load_task(args.task)
    model_id = args.model or student.model
    paths = list(args.traces or [])
    if args.traces_repo:
        from huggingface_hub import HfApi, hf_hub_download

        names = args.traces_files or [
            name
            for name in HfApi().list_repo_files(args.traces_repo, repo_type="dataset")
            if name.startswith("episodes/") and name.endswith(".jsonl")
        ]
        paths = [Path(hf_hub_download(args.traces_repo, name, repo_type="dataset")) for name in names]
    records = load_accepted(paths, harness=args.harness, allow_mismatch=args.allow_harness_mismatch)
    keep_reasoning = student.sft.keep_reasoning
    if args.think_target == "empty" and keep_reasoning:
        raise SystemExit("--think-target empty supervises no reasoning; the student profile sets keep_reasoning")
    summaries = None
    if args.summaries_repo:
        summaries = load_summaries(args.summaries_repo, args.summaries_file, records, args.min_summary_coverage)
    trajectories = [
        trajectory(record, keep_reasoning=keep_reasoning, strip_visible_text=args.strip_visible_text,
                   summaries=summaries)
        for record in records
    ]  # fmt: skip
    from agent_distill.harnesses.tools import harness_tools

    tools = harness_tools(args.harness, task)
    print(
        f"{len(trajectories)} accepted {args.harness} trajectories; tools {[t['function']['name'] for t in tools]}; "
        f"reasoning {'kept' if keep_reasoning else 'dropped'}; think target {args.think_target}; visible text "
        f"{'stripped' if args.strip_visible_text else 'kept'}; CoT-proxies {'on' if summaries is not None else 'off'}"
        f"; KL replay {'on' if args.replay_repo else 'off'}",
        flush=True,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    render = render_trajectory_examples if args.full_trajectory else render_examples
    examples = render(trajectories, tokenizer, tools, args.think_target)
    rows = tokenize_examples(examples, tokenizer, args.max_length, whole_turns=args.think_target == "empty")
    supervised = sum(sum(row["completion_mask"]) for row in rows)
    print(f"{len(rows)} tokenized examples, {supervised} supervised tokens", flush=True)
    replay = heldout = None
    if args.replay_repo:
        replay = load_replay(args.replay_repo, args.replay_file, args.max_length)
        heldout = load_replay(args.replay_repo, args.heldout_file, args.max_length) if args.heldout_file else []
        print(f"replay pool: {len(replay)} samples, {sum(sum(r['completion_mask']) for r in replay)} tokens; "
              f"{len(heldout)} held out for the drift probe", flush=True)  # fmt: skip
    if args.dry_run:
        return
    if replay is not None:
        per_turn = sum(1 for example in examples if not example.get("spans"))
        if per_turn:
            print(f"WARNING: {per_turn} of {len(examples)} teacher samples are single turns, not trajectories "
                  f"(per-turn fallback or no --full-trajectory): an update's {args.teacher_per_update} teacher "
                  "samples are then not all whole trajectories as in the paper", flush=True)  # fmt: skip
        train_kl_replay(args, student, model_id, tokenizer, rows, replay, heldout or [])
    else:
        train(args, student, model_id, tokenizer, rows)


def load_summaries(
    repo: str, filename: str, records: list[dict[str, Any]], min_coverage: float
) -> dict[tuple[str, int], str]:
    """The annotate stage's summaries, checked against the trace records SFT uses (`cot_proxy.check_summaries`)."""
    from huggingface_hub import hf_hub_download

    from agent_distill.cot_proxy import check_summaries, summaries_by_turn

    annotations = list(read_jsonl(Path(hf_hub_download(repo, filename, repo_type="dataset"))))
    stats = check_summaries(records, annotations, min_coverage=min_coverage)
    print(f"CoT-proxy summaries from {repo}: {stats['summarised']} of {stats['turns_with_reasoning']} turns with "
          f"reasoning ({stats['coverage']:.1%}); the rest get <summary></summary>", flush=True)  # fmt: skip
    return summaries_by_turn(annotations)


def replay_row(sample: dict[str, Any], max_length: int) -> dict[str, list[int]] | None:
    """A replay sample (prompt and completion token ids, as the initial student generated them) -> a training row
    whose mask covers the completion. Cut to `max_length`; None if no completion token fits."""
    prompt, completion = list(sample["prompt_ids"]), list(sample["completion_ids"])
    input_ids = (prompt + completion)[:max_length]
    mask = ([0] * len(prompt) + [1] * len(completion))[:max_length]
    return {"input_ids": input_ids, "completion_mask": mask} if any(mask) else None


def load_replay(repo: str, filename: str, max_length: int) -> list[dict[str, list[int]]]:
    from huggingface_hub import hf_hub_download

    samples = read_jsonl(Path(hf_hub_download(repo, filename, repo_type="dataset")))
    return [row for sample in samples if (row := replay_row(sample, max_length)) is not None]


def update_plan(
    n_teacher: int, n_replay: int, *, epochs: float, teacher_per_update: int, replay_per_update: int, seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """(teacher indices, replay indices) of every update: `epochs` passes over the shuffled teacher samples (passes are
    counted on teacher samples), `teacher_per_update` per update plus `replay_per_update` replay samples drawn
    without replacement from the reshuffled pool."""
    import math
    import random

    rng = random.Random(seed)
    order: list[int] = []
    for _ in range(math.ceil(epochs)):
        passes = list(range(n_teacher))
        rng.shuffle(passes)
        order += passes
    order = order[: round(n_teacher * epochs)]
    pool: list[int] = []
    plan = []
    for start in range(0, len(order), teacher_per_update):
        replay = []
        for _ in range(replay_per_update if n_replay else 0):
            if not pool:
                pool = list(range(n_replay))
                rng.shuffle(pool)
            replay.append(pool.pop())
        plan.append((order[start : start + teacher_per_update], replay))
    return plan


def supervised_positions(mask: list[int]) -> list[int]:
    """Positions whose logits predict a supervised token (the token after each position)."""
    return [index - 1 for index in range(1, len(mask)) if mask[index]]


def kl_from_logits(policy_logits: Any, init_logits: Any, chunk: int = 1024) -> Any:
    """Sum over positions of KL(p_init || p_theta) over the full vocabulary, in chunks of positions (fp32 math)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def part(policy: Any, init: Any) -> Any:
        init_logp = F.log_softmax(init.float(), dim=-1)
        return (init_logp.exp() * (init_logp - F.log_softmax(policy.float(), dim=-1))).sum()

    total = policy_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, policy_logits.shape[0], chunk):
        piece = (policy_logits[start : start + chunk], init_logits[start : start + chunk])
        total = total + (checkpoint(part, *piece, use_reentrant=False) if policy_logits.requires_grad else part(*piece))
    return total


def ce_from_logits(logits: Any, targets: Any, chunk: int = 1024) -> Any:
    """Sum of the cross-entropy of `targets` under `logits` (fp32 math, in chunks of positions)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def part(piece: Any, target: Any) -> Any:
        return F.cross_entropy(piece.float(), target, reduction="sum")

    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, logits.shape[0], chunk):
        piece = (logits[start : start + chunk], targets[start : start + chunk])
        total = total + (checkpoint(part, *piece, use_reentrant=False) if logits.requires_grad else part(*piece))
    return total


def train_kl_replay(
    args: argparse.Namespace,
    student: Any,
    model_id: str,
    tokenizer: Any,
    rows: list[dict],
    replay: list[dict],
    heldout: list[dict],
) -> None:
    """LoRA SFT with KL replay (paper eq. 1): per update, CE on the teacher samples' supervised tokens / their count
    plus KL(p_init || p_theta) on the replay samples' tokens / their count. p_init is the base model with the adapter
    disabled (one extra no-grad forward per replay sample, no second copy of the weights). Samples run one at a time
    with gradient accumulation; logits are computed only at supervised positions."""
    import json as _json

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    if not args.no_4bit:
        raise SystemExit("KL replay trains bf16 base weights; pass --no-4bit")
    if args.save_total_limit:
        print("note: --save-total-limit does not apply to the KL-replay loop (each save overwrites the adapter and "
              "pushes a new commit)", flush=True)  # fmt: skip
    print(
        "note: --gradient-accumulation does not apply to the KL-replay loop (an update accumulates "
        f"{args.teacher_per_update} teacher + {args.replay_per_update} replay samples)",
        flush=True,
    )
    if student.sft.train_patch:
        from agent_distill.hooks import resolve

        resolve(student.sft.train_patch).enable(chunk_size=args.attention_chunk_size)
    model_class = getattr(transformers, student.sft.model_class or "AutoModelForCausalLM")
    device = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for local smoke tests of the loop
    model = model_class.from_pretrained(model_id, dtype=dtype, trust_remote_code=True).to(device)
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config.use_cache = False
    rank = args.lora_rank or student.lora.rank
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=args.lora_alpha or rank * 2, lora_dropout=args.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=lora_target_modules(student.lora.sft_target_modules),
    ))  # fmt: skip
    for parameter in model.parameters():  # fp32 adapter weights over bf16 base weights
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    warmup = max(0, args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / (warmup + 1)))
    plan = update_plan(len(rows), len(replay), epochs=args.epochs, teacher_per_update=args.teacher_per_update,
                       replay_per_update=args.replay_per_update, seed=args.seed)  # fmt: skip
    if args.max_steps > 0:
        plan = plan[: args.max_steps]
    print(f"{len(plan)} updates: {args.teacher_per_update} teacher + {args.replay_per_update} replay samples each, "
          f"{args.epochs} passes over {len(rows)} teacher samples", flush=True)  # fmt: skip

    def forward(row: dict, positions: list[int]) -> Any:
        ids = torch.tensor([row["input_ids"]], device=device)
        keep = torch.tensor(positions, device=device)
        return model(input_ids=ids, logits_to_keep=keep).logits[0]

    def replay_kl(row: dict, *, grad: bool) -> tuple[Any, int]:
        positions = supervised_positions(row["completion_mask"])
        with torch.no_grad(), model.disable_adapter():
            init = forward(row, positions)
        with torch.set_grad_enabled(grad):
            return kl_from_logits(forward(row, positions), init), len(positions)

    def drift() -> float:
        """Held-out KL(p_init || p_theta) per token (the paper's drift probe)."""
        total, tokens = 0.0, 0
        for row in heldout:
            value, count = replay_kl(row, grad=False)
            total, tokens = total + float(value), tokens + count
        return total / max(1, tokens)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    for step, (teacher_ids, replay_ids) in enumerate(plan, start=1):
        n_teacher = sum(len(supervised_positions(rows[i]["completion_mask"])) for i in teacher_ids)
        n_replay = sum(len(supervised_positions(replay[i]["completion_mask"])) for i in replay_ids)
        ce_total = kl_total = 0.0
        for index in teacher_ids:
            row = rows[index]
            positions = supervised_positions(row["completion_mask"])
            targets = torch.tensor([row["input_ids"][p + 1] for p in positions], device=device)
            loss = ce_from_logits(forward(row, positions), targets) / max(1, n_teacher)
            loss.backward()
            ce_total += float(loss.detach())
        for index in replay_ids:
            value, _ = replay_kl(replay[index], grad=True)
            loss = value / max(1, n_replay)
            loss.backward()
            kl_total += float(loss.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        entry = {"step": step, "ce": ce_total, "kl": kl_total, "grad_norm": grad_norm,
                 "lr": scheduler.get_last_lr()[0], "teacher_tokens": n_teacher, "replay_tokens": n_replay}  # fmt: skip
        history.append(entry)
        print(_json.dumps(entry), flush=True)
        if args.save_steps and step % args.save_steps == 0 and step < len(plan):
            save_adapter(model, tokenizer, output, args, {"step": step, "history": history}, final=False)
    report = {"steps": len(plan), "history": history, "teacher_per_update": args.teacher_per_update,
              "replay_per_update": args.replay_per_update, "epochs": args.epochs}  # fmt: skip
    if heldout:
        model.eval()
        report["heldout_kl_per_token"] = drift()
        print(f"drift probe: held-out KL(p_init || p_theta) = {report['heldout_kl_per_token']:.6f} nats/token "
              f"over {len(heldout)} samples", flush=True)  # fmt: skip
    save_adapter(model, tokenizer, output, args, report, final=True)


def save_adapter(model: Any, tokenizer: Any, output: Path, args: argparse.Namespace, report: dict, *,
                 final: bool) -> None:  # fmt: skip
    """Save the adapter and its training report; push them (a public model repo, in the collection) if asked."""
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "kl_replay_report.json").write_text(json.dumps(report, indent=1))
    if not args.hub_model_id:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    created = not api.repo_exists(args.hub_model_id)
    api.create_repo(args.hub_model_id, private=False, exist_ok=True)  # models are public from the start
    if created:
        join_collection(args.hub_model_id, "model", args.collection, args.collection_note)
    message = "final adapter" if final else f"checkpoint at update {report.get('step')}"
    api.upload_folder(repo_id=args.hub_model_id, folder_path=str(output), commit_message=message)
    print(f"pushed the adapter ({message}) to {args.hub_model_id}", flush=True)


def train(args: argparse.Namespace, student: Any, model_id: str, tokenizer: Any, rows: list[dict]) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quantization = None
    if not args.no_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    model: Any = model_id
    model_init_kwargs: dict[str, Any] | None = {"torch_dtype": dtype, "trust_remote_code": True}
    if student.sft.model_class:
        if quantization is not None:
            raise ValueError("a profile model_class loads full-precision weights; pass --no-4bit")
        import transformers

        if student.sft.train_patch:
            from agent_distill.hooks import resolve

            resolve(student.sft.train_patch).enable(chunk_size=args.attention_chunk_size)
        model = getattr(transformers, student.sft.model_class).from_pretrained(model_id, dtype=dtype)
        for config in (model.config, getattr(model.config, "text_config", None)):
            if config is not None:
                config.use_cache = False
        model_init_kwargs = None
    rank = args.lora_rank or student.lora.rank
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(rows),
        peft_config=LoraConfig(
            r=rank,
            lora_alpha=args.lora_alpha or rank * 2,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules(student.lora.sft_target_modules),
        ),
        quantization_config=quantization,
        args=SFTConfig(
            output_dir=str(args.output),
            max_length=args.max_length,
            completion_only_loss=True,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type="constant_with_warmup" if cross_format(args) else "linear",
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            bf16=not args.fp16,
            fp16=args.fp16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            report_to="none",
            push_to_hub=bool(args.hub_model_id),
            hub_model_id=args.hub_model_id or None,
            hub_private_repo=False,  # models are public from the start (no episode data)
            hub_strategy="checkpoint" if args.hub_model_id else "every_save",
            model_init_kwargs=model_init_kwargs,
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    if args.hub_model_id:
        trainer.push_to_hub()
        print(f"pushed the adapter to {args.hub_model_id}", flush=True)
        join_collection(args.hub_model_id, "model", args.collection, args.collection_note)


if __name__ == "__main__":
    main()
