"""Annotate stage: CoT-proxy summaries of the teacher's reasoning, one per assistant turn of the accepted traces.

    python -m agent_distill.stages.annotate --traces-repo ORG/RUN-traces --harness pi@1.6.0 \
        --model Qwen/Qwen3-4B-Instruct-2507 --output-repo ORG/RUN-summaries

The CoT-proxy paper (arXiv 2611.33148, Appendix A and Table 6): a small instruct model summarises each turn's thought
in at most two sentences, seeing only that thought. Qwen3-4B-Instruct worked as well as the teacher itself at about 1%
of the cost. Drafts that break the acceptance rules (`agent_distill.cot_proxy.rejection`) are retried with the next
seed, up to three attempts; a turn whose drafts are all rejected, or that has no reasoning, gets an empty summary.

The annotator runs offline in vLLM on the job's GPU. Output (a dataset repo created private and joined to the
collection; `distill run --stages publish` scans it and makes it public): `summaries/summaries.jsonl`, one line per
assistant turn of every accepted trajectory, `{row_id, turn, summary, attempts, rejections, thought_chars}`, plus
`summaries/manifest.json`. `turn` is the index among the record's assistant messages, as SFT counts them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_distill import cot_proxy
from agent_distill.credentials import record_line
from agent_distill.hub_collection import join_collection

DEFAULT_ANNOTATOR = "Qwen/Qwen3-4B-Instruct-2507"


def turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every assistant turn of the accepted records: {row_id, turn, thought}."""
    found = []
    for record in records:
        index = -1
        for message in record.get("messages") or []:
            if message.get("role") != "assistant":
                continue
            index += 1
            found.append({"row_id": str(record["row_id"]), "turn": index,
                          "thought": str(message.get("reasoning_content") or "").strip()})  # fmt: skip
    return found


def annotate(items: list[dict[str, Any]], generate: Any, attempts: int = cot_proxy.ATTEMPTS) -> list[dict[str, Any]]:
    """Summaries for `items` (with thoughts). `generate(conversations, seed) -> [(text, finished)]` drafts one summary
    per conversation; rejected drafts are retried with the next seed."""
    results = {
        (i["row_id"], i["turn"]): {"row_id": i["row_id"], "turn": i["turn"], "summary": "", "attempts": 0,
                                   "rejections": [], "thought_chars": len(i["thought"])}
        for i in items
    }  # fmt: skip
    pending = [item for item in items if item["thought"]]
    for seed in range(attempts):
        if not pending:
            break
        drafts = generate([cot_proxy.messages(item["thought"]) for item in pending], seed)
        retry = []
        for item, (text, finished) in zip(pending, drafts, strict=True):
            result = results[(item["row_id"], item["turn"])]
            result["attempts"] += 1
            reason = cot_proxy.rejection(text, item["thought"], finished=finished)
            if reason is None:
                result["summary"] = text.strip()
            else:
                result["rejections"].append(reason)
                retry.append(item)
        pending = retry
    return list(results.values())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", nargs="+", type=Path, help="Local episode JSONL files")
    source.add_argument("--traces-repo", help="Dataset repo; reads episodes/**/*.jsonl")
    parser.add_argument("--harness", required=True)
    parser.add_argument("--allow-harness-mismatch", action="store_true")
    parser.add_argument("--model", default=DEFAULT_ANNOTATOR)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256, help="A summary is at most 400 characters")
    parser.add_argument("--output-dir", type=Path, default=Path("/annotate"))
    parser.add_argument("--output-repo", default="")
    parser.add_argument("--collection", default="")
    parser.add_argument("--collection-note", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from agent_distill.stages.sft import load_accepted

    args = parse_args(argv)
    paths = list(args.traces or [])
    if args.traces_repo:
        from huggingface_hub import HfApi, hf_hub_download

        names = [n for n in HfApi().list_repo_files(args.traces_repo, repo_type="dataset")
                 if n.startswith("episodes/") and n.endswith(".jsonl")]  # fmt: skip
        paths = [Path(hf_hub_download(args.traces_repo, name, repo_type="dataset")) for name in names]
    records = load_accepted(paths, harness=args.harness, allow_mismatch=args.allow_harness_mismatch)
    items = turns(records)
    print(f"{len(items)} assistant turns in {len(records)} accepted trajectories, "
          f"{sum(bool(i['thought']) for i in items)} with reasoning", flush=True)  # fmt: skip

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=32768, gpu_memory_utilization=0.85)

    def generate(conversations: list[list[dict[str, str]]], seed: int) -> list[tuple[str, bool]]:
        params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                max_tokens=args.max_tokens, seed=seed)  # fmt: skip
        outputs = llm.chat(conversations, params)
        return [(o.outputs[0].text, o.outputs[0].finish_reason == "stop") for o in outputs]

    results = annotate(items, generate)
    accepted = sum(bool(r["summary"]) for r in results)
    with_thought = sum(r["thought_chars"] > 0 for r in results)
    manifest = {"annotator": args.model, "traces": args.traces_repo or [str(p) for p in paths], "turns": len(results),
                "with_reasoning": with_thought, "accepted": accepted, "empty_after_rejections": with_thought - accepted,
                "retries": sum(max(0, r["attempts"] - 1) for r in results),
                "mean_summary_words": sum(len(r["summary"].split()) for r in results) / max(1, accepted),
                "prompt": cot_proxy.SYSTEM_PROMPT}  # fmt: skip
    print(json.dumps({k: v for k, v in manifest.items() if k != "prompt"}), flush=True)
    out = args.output_dir / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summaries.jsonl").open("w") as handle:
        for result in results:
            handle.write(record_line(result))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if args.output_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        created = not api.repo_exists(args.output_repo, repo_type="dataset")
        api.create_repo(args.output_repo, repo_type="dataset", private=True, exist_ok=True)
        if created:
            join_collection(args.output_repo, "dataset", args.collection, args.collection_note)
        api.upload_folder(repo_id=args.output_repo, repo_type="dataset", folder_path=str(args.output_dir),
                          commit_message=f"CoT-proxy summaries by {args.model}")  # fmt: skip
        print(f"uploaded the summaries to {args.output_repo} (private until `--stages publish`)", flush=True)


if __name__ == "__main__":
    main()
