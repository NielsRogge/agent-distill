"""Replay stage: the KL-replay pool of the initial student's own generations on out-of-domain prompts.

    python -m agent_distill.stages.replay --student qwen3.5-4b-think --output-repo ORG/qwen35-4b-replay \
        --size 1000 --heldout 100

The CoT-proxy paper (arXiv 2611.33148, §3 and Appendix B) regularises SFT toward the initial student with a KL term on
1,000 of the initial student's generations on prompts from OpenR1-Math and OpenAssistant (mathematics, chat and
instruction following), plus a disjoint 100 held out to measure drift. The pool must come from outside the target
environment: in-domain replay holds the policy in place (the paper's Table 4). It depends only on the student, so one
pool serves every task distilled into that student (`repos.replay` in a run.toml points to it).

The student generates with its chat template's default (thinking on for Qwen3.5) at the paper's sampling settings
(temperature 0.6, top-p 0.95, top-k 20), offline in vLLM. Each sample keeps the exact prompt and completion token ids;
completions are capped at `--max-tokens` (a KL target needs no end-of-sequence). Output, in a dataset repo created
private and joined to the collection (`distill run --stages publish` scans it and makes it public):
`replay/pool.jsonl`, `replay/heldout.jsonl` and `replay/manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from agent_distill.config import load_profile
from agent_distill.credentials import record_line
from agent_distill.hub_collection import join_collection

# (dataset, config, split, extractor): out-of-domain prompt sources, as in the paper.
SOURCES = {
    "openr1-math": ("open-r1/OpenR1-Math-220k", "default", "train"),
    "oasst": ("OpenAssistant/oasst1", None, "train"),
}


def prompt_of(source: str, row: dict[str, Any]) -> str | None:
    """The user prompt of a source row, or None to skip it (OpenAssistant: English conversation roots only)."""
    if source == "openr1-math":
        return str(row.get("problem") or "").strip() or None
    if source == "oasst":
        if row.get("role") != "prompter" or row.get("parent_id") is not None or row.get("lang") != "en":
            return None
        return str(row.get("text") or "").strip() or None
    raise ValueError(f"unknown replay source {source!r}")


def sample_prompts(total: int, seed: int, sources: list[str]) -> list[tuple[str, str]]:
    """`total` distinct (source, prompt) pairs, split evenly over `sources`, shuffled."""
    from datasets import load_dataset

    rng = random.Random(seed)
    per = [total // len(sources) + (1 if index < total % len(sources) else 0) for index in range(len(sources))]
    picked: list[tuple[str, str]] = []
    for source, count in zip(sources, per, strict=True):
        repo, config, split = SOURCES[source]
        stream = load_dataset(repo, config, split=split, streaming=True).shuffle(seed=seed, buffer_size=20_000)
        seen: set[str] = set()
        for row in stream:
            prompt = prompt_of(source, row)
            if prompt and prompt not in seen and len(prompt) < 4000:
                seen.add(prompt)
                picked.append((source, prompt))
                if len(seen) >= count:
                    break
    rng.shuffle(picked)
    return picked


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True)
    parser.add_argument("--profiles", action="append", type=Path, default=[])
    parser.add_argument("--model", help="Default: the student profile's model (the initial student)")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--heldout", type=int, default=100)
    parser.add_argument("--sources", default="openr1-math,oasst")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("/replay"))
    parser.add_argument("--output-repo", default="", help="Dataset repo (created private)")
    parser.add_argument("--collection", default="")
    parser.add_argument("--collection-note", default="")
    parser.add_argument("--dry-run", action="store_true", help="Sample and render the prompts only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    student = load_profile("students", args.student, args.profiles)
    model_id = args.model or student.model
    sources = [s for s in args.sources.split(",") if s]
    prompts = sample_prompts(args.size + args.heldout, args.seed, sources)
    print(f"{len(prompts)} prompts from {sources}", flush=True)

    from transformers import AutoTokenizer

    revision = student.revision_of(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    rendered = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        for _, prompt in prompts
    ]
    prompt_ids = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in rendered]
    if args.dry_run:
        print(rendered[0], flush=True)
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    serve = student.serve
    # Custom model code is trusted only at the profile's pinned revision (the reviewed code, as in evaluation serving).
    llm = LLM(model=model_id, revision=revision, tokenizer_revision=revision, trust_remote_code=bool(revision),
              max_model_len=max(len(ids) for ids in prompt_ids) + args.max_tokens + 16,
              gpu_memory_utilization=serve.gpu_memory_utilization, enforce_eager=serve.enforce_eager,
              seed=args.seed, **({"language_model_only": True} if serve.language_model_only else {}))  # fmt: skip
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                            max_tokens=args.max_tokens, seed=args.seed)  # fmt: skip
    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], params)
    samples = []
    for (source, prompt), ids, output in zip(prompts, prompt_ids, outputs, strict=True):
        completion = output.outputs[0]
        samples.append({"source": source, "prompt": prompt, "prompt_ids": list(ids),
                        "completion_ids": list(completion.token_ids), "completion": completion.text,
                        "finish_reason": completion.finish_reason, "model": model_id})  # fmt: skip
    pool, heldout = samples[: args.size], samples[args.size :]
    out = args.output_dir / "replay"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pool", pool), ("heldout", heldout)):
        with (out / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(record_line(row))
    manifest = {"model": model_id, "student": args.student, "sources": sources, "size": len(pool),
                "heldout": len(heldout), "max_tokens": args.max_tokens, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed,
                "finished": sum(s["finish_reason"] == "stop" for s in samples),
                "completion_tokens": sum(len(s["completion_ids"]) for s in samples)}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest), flush=True)
    if args.output_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        created = not api.repo_exists(args.output_repo, repo_type="dataset")
        api.create_repo(args.output_repo, repo_type="dataset", private=True, exist_ok=True)
        if created:
            join_collection(args.output_repo, "dataset", args.collection, args.collection_note)
        api.upload_folder(repo_id=args.output_repo, repo_type="dataset", folder_path=str(args.output_dir),
                          commit_message=f"replay pool of {model_id}")  # fmt: skip
        print(f"uploaded the replay pool to {args.output_repo} (private until `--stages publish`)", flush=True)


if __name__ == "__main__":
    main()
