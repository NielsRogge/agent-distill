"""SFT stage (DESIGN.md §7): LoRA SFT of the student on the teacher's accepted trajectories.

    python -m agent_distill.stages.sft --task task --harness opencode@1.2.4 --student qwen3.5-9b \
        --traces-repo ORG/RUN-traces --hub-model-id ORG/RUN-lora

Ported from url-extraction's `train_sft.py`. Accepted records (`passed`, one per row) of one `harness@version` are
rendered with the student's chat template and the exact tool definitions the harness sends; the loss covers the
assistant turns only (exact completion masks from character offsets). With `--full-trajectory` one example holds a
whole trajectory with every assistant turn supervised, when the template renders each turn's prompt as a prefix of
the full conversation (checked per trajectory, otherwise per-turn examples). LoRA settings, the model class, a
training patch and whether the teacher's reasoning is distilled come from the student profile.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from agent_distill.config import QWEN35_TEXT_LORA_TARGETS, load_profile
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


def trajectory(record: dict[str, Any], *, keep_reasoning: bool) -> list[dict[str, Any]]:
    """The episode's messages up to the answer: a structured-output submission ends the trajectory."""
    result: list[dict[str, Any]] = []
    submit_id: str | None = None
    for message in record.get("messages") or []:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            continue
        result.append(chat_message(message, keep_reasoning=keep_reasoning))
        for call in message.get("tool_calls") or []:
            if str(call.get("function", {}).get("name", "")).lower() in SUBMIT_TOOLS:
                submit_id = call.get("id")
        if submit_id and message.get("role") == "tool" and message.get("tool_call_id") == submit_id:
            break
    return result


# --------------------------------------------------------------------------------------------------
# Rendering and tokenization (url-extraction train_sft.py)
# --------------------------------------------------------------------------------------------------


def render_trajectory_examples(trajectories: list[list[dict]], tokenizer: Any, tools: list[dict]) -> list[dict]:
    """One example per trajectory with every assistant turn supervised, when the template is prefix-stable."""
    examples: list[dict[str, Any]] = []
    fallback: list[list[dict]] = []
    for messages in trajectories:
        full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        spans: list[list[int]] = []
        stable = True
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
            if through[len(prompt) :].strip():
                spans.append([len(prompt), len(through)])
        if not stable:
            fallback.append(messages)
            continue
        if spans:
            examples.append({"prompt": full, "completion": "", "spans": json.dumps(spans)})
    if fallback:
        print(f"{len(fallback)} trajectories are not prefix-stable; using per-turn examples for them", flush=True)
        examples += [{**row, "spans": ""} for row in render_examples(fallback, tokenizer, tools)]
    print(f"rendered {len(examples)} examples from {len(trajectories)} trajectories", flush=True)
    return examples


def render_examples(trajectories: list[list[dict]], tokenizer: Any, tools: list[dict]) -> list[dict]:
    """One example per assistant turn: the history as prompt, the turn as completion."""
    examples: list[dict[str, str]] = []
    for messages in trajectories:
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
            completion = full[len(prompt) :]
            if completion.strip():
                examples.append({"prompt": prompt, "completion": completion})
            history.append(message)
    if not examples:
        raise ValueError("the accepted trajectories contain no assistant turns")
    return examples


def tokenize_examples(examples: list[dict], tokenizer: Any, max_length: int) -> list[dict[str, list[int]]]:
    """Exact completion masks from character offsets (no BPE boundary drift)."""
    rows: list[dict[str, list[int]]] = []
    for example in examples:
        text = example["prompt"] + example["completion"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = list(encoded["input_ids"])
        if example.get("spans"):
            spans = json.loads(example["spans"])
            mask = [int(any(end > a and start < b for a, b in spans)) for start, end in encoded["offset_mapping"]]
            if len(input_ids) > max_length:  # keep the supervised turns that fit
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
    parser.add_argument("--hub-model-id", default="", help="Push the adapter (private until publish)")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (smoke runs)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--full-trajectory", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attention-chunk-size", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Render and tokenize only; print statistics")
    return parser.parse_args(argv)


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
    trajectories = [trajectory(record, keep_reasoning=keep_reasoning) for record in records]
    from agent_distill.harnesses.tools import harness_tools

    tools = harness_tools(args.harness, task)
    print(
        f"{len(trajectories)} accepted {args.harness} trajectories; tools {[t['function']['name'] for t in tools]}; "
        f"reasoning {'kept' if keep_reasoning else 'dropped'}",
        flush=True,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    render = render_trajectory_examples if args.full_trajectory else render_examples
    rows = tokenize_examples(render(trajectories, tokenizer, tools), tokenizer, args.max_length)
    supervised = sum(sum(row["completion_mask"]) for row in rows)
    print(f"{len(rows)} tokenized examples, {supervised} supervised tokens", flush=True)
    if args.dry_run:
        return
    train(args, student, model_id, tokenizer, rows)


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
            lora_alpha=rank * 2,
            lora_dropout=0.05,
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
            hub_private_repo=True,  # models go public through `distill publish`
            hub_strategy="checkpoint" if args.hub_model_id else "every_save",
            model_init_kwargs=model_init_kwargs,
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    if args.hub_model_id:
        trainer.push_to_hub()
        print(f"pushed the adapter to {args.hub_model_id} (private)", flush=True)


if __name__ == "__main__":
    main()
