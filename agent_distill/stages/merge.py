"""Merge stage (DESIGN.md §7): a LoRA adapter merged into its base model, published as a standalone checkpoint.

    python -m agent_distill.stages.merge --student qwen3.5-9b --adapter ORG/RUN-lora --hub-model-id ORG/RUN-sft-merged

Ported from url-extraction's `merge_adapter.py`. The default merges with PEFT (Qwen3.5, kept with its processor so
vLLM can boot from the merged repo alone). A student profile can name another merge method (`[merge] method =
"module:function(args, adapter_snapshot, token) -> output_dir"`), e.g. a merge straight into a quantized checkpoint's
BF16 attention tensors. The merged model is created public (models hold no episode data; private storage is capped).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from agent_distill.config import load_profile
from agent_distill.hub_collection import join_collection


def adapter_files(snapshot: Path) -> list[Path]:
    index = snapshot / "adapter_model.safetensors.index.json"
    if index.exists():
        return [snapshot / name for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    single = snapshot / "adapter_model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no safetensors adapter weights in {snapshot}")
    return [single]


def audit_visual_lora(snapshot: Path) -> dict[str, int]:
    """Refuse an adapter with trained vision-tower outputs (text-only students never train them)."""
    from safetensors import safe_open

    counts = {"total": 0, "visual": 0, "visual_nonzero_lora_b": 0}
    for path in adapter_files(snapshot):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                counts["total"] += 1
                if ".visual." not in name:
                    continue
                counts["visual"] += 1
                if ".lora_B." in name and bool(handle.get_tensor(name).count_nonzero().item()):
                    counts["visual_nonzero_lora_b"] += 1
    if counts["visual_nonzero_lora_b"]:
        raise RuntimeError("the adapter has trained visual LoRA outputs; refusing an unreviewed merge")
    return counts


def model_card(base_model: str, adapter: str, note: str) -> str:
    return (
        f"---\nbase_model: {base_model}\nlibrary_name: transformers\npipeline_tag: text-generation\n"
        "tags:\n- peft\n- merged\n- agent-distill\n---\n\n"
        f"# {base_model.split('/')[-1]} + {adapter.split('/')[-1]}\n\n"
        f"[{adapter}](https://huggingface.co/{adapter}) merged into [{base_model}](https://huggingface.co/{base_model}).\n"
        f"\n{note}\n"
    )


def peft_merge(args: argparse.Namespace, snapshot: Path) -> Path:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    configured = str(PeftConfig.from_pretrained(str(snapshot)).base_model_name_or_path).rstrip("/")
    if configured != args.base_model.rstrip("/"):
        raise ValueError(f"the adapter expects {configured}, not {args.base_model}")
    print(f"adapter audit: {json.dumps(audit_visual_lora(snapshot), sort_keys=True)}", flush=True)
    output = args.output or Path(tempfile.mkdtemp(prefix="agent-distill-merged-"))
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, str(snapshot)).merge_and_unload(safe_merge=True, progressbar=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(output)
    # vLLM initializes Qwen3.5's processor even for text-only requests: keep it so the merged repo boots alone.
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(output)
    return output


def publish_merged(output: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(output), repo_id=repo, repo_type="model", commit_message="agent-distill: merged checkpoint"
    )
    info: Any = api.model_info(repo, files_metadata=True)
    local = {p.name: p.stat().st_size for p in output.iterdir() if p.is_file()}
    remote = {s.rfilename: s.size for s in info.siblings}
    missing = [name for name, size in local.items() if remote.get(name) != size]
    if missing:
        raise RuntimeError(f"upload did not land for {missing}")
    print(f"published {repo} at {info.sha} ({len(local)} files verified)", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True)
    parser.add_argument("--profiles", action="append", type=Path, default=[])
    parser.add_argument("--base-model", help="Default: the student profile's model")
    parser.add_argument("--adapter", required=True, help="Adapter repo or directory")
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--note", default="Trained with agent-distill.")
    parser.add_argument("--collection", default="", help="Collection the pushed model joins")
    parser.add_argument("--collection-note", default="", help="Its one-line note in the collection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    student = load_profile("students", args.student, args.profiles)
    args.base_model = args.base_model or student.model
    from huggingface_hub import snapshot_download

    snapshot = Path(args.adapter) if Path(args.adapter).is_dir() else Path(snapshot_download(args.adapter))
    if student.merge.method == "peft":
        output = peft_merge(args, snapshot)
    else:
        from agent_distill.hooks import resolve

        output = Path(resolve(student.merge.method)(args, snapshot, os.environ.get("HF_TOKEN")))
    (output / "README.md").write_text(model_card(args.base_model, args.adapter, args.note))
    if args.hub_model_id:
        publish_merged(output, args.hub_model_id)
        join_collection(args.hub_model_id, "model", args.collection, args.collection_note)
    else:
        print(f"merged checkpoint in {output}", flush=True)


if __name__ == "__main__":
    main()
