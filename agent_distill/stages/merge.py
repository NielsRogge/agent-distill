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


# --------------------------------------------------------------------------------------------------
# Base tensors the merged model class does not load (e.g. Qwen3.5's multi-token-prediction head, `mtp.*`)
# --------------------------------------------------------------------------------------------------
#
# `save_pretrained` writes only the tensors of the loaded model class. Qwen3.5's checkpoints also hold an MTP head
# (`mtp.*`, vLLM's `qwen3_5_mtp` speculative decoding) that transformers never loads, so merged checkpoints lost it.
# These helpers read and write the safetensors format directly (an 8-byte little-endian header length, a JSON header,
# then the raw bytes), so restored tensors are byte-identical copies in their original dtype and the stage needs
# neither torch nor safetensors for them.

INDEX = "model.safetensors.index.json"
EXTRA_SHARD = "model-base-extra.safetensors"
WRAPPER_PREFIXES = ("base_model.model.", "model.", "language_model.")


def _header(path: Path) -> tuple[dict[str, Any], int]:
    """A safetensors file's header (without `__metadata__`) and the offset where its data starts."""
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def checkpoint_tensors(folder: Path) -> dict[str, str]:
    """Tensor name -> safetensors file name of a checkpoint folder (sharded with an index, or single-file)."""
    folder = Path(folder)
    if (folder / INDEX).exists():
        return dict(json.loads((folder / INDEX).read_text())["weight_map"])
    files = sorted(p.name for p in folder.glob("*.safetensors"))
    return {name: file for file in files for name in _header(folder / file)[0]}


def _module(name: str) -> str:
    """A tensor or adapter name as a module path without wrapper prefixes: the base's
    `model.language_model.layers.0.mlp.down_proj.weight` and the adapter's
    `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight` both become `layers.0.mlp.down_proj`."""
    for suffix in (".lora_A.weight", ".lora_B.weight", ".weight", ".bias"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    stripped = True
    while stripped:
        stripped = False
        for prefix in WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name, stripped = name[len(prefix) :], True
    return name


def adapter_modules(snapshot: Path) -> set[str]:
    """The module paths an adapter's LoRA tensors target (`_module` form)."""
    return {_module(name) for path in adapter_files(snapshot) for name in _header(path)[0] if ".lora_" in name}


def restore_base_tensors(base: Path, output: Path, targeted: set[str] = frozenset()) -> list[str]:
    """Copy into the merged checkpoint at `output` every tensor of the base checkpoint at `base` that the merged
    checkpoint lacks, unchanged, as one extra shard listed in the index. Returns the copied names.

    Tensors whose module an adapter targets are never copied (their merged value would be wrong), and nothing is
    copied when the merged checkpoint names its tensors differently from the base (it has names the base lacks), since
    "missing" then cannot be told from "renamed". Config fields named after a restored top-level module (e.g.
    `mtp_num_hidden_layers` for `mtp.*`) that the merged config.json lacks are copied from the base's.
    """
    base_map, merged_map = checkpoint_tensors(base), checkpoint_tensors(output)
    if renamed := sorted(set(merged_map) - set(base_map)):
        print(f"base tensors not restored: the merged checkpoint names {len(renamed)} tensors the base lacks "
              f"(e.g. {renamed[0]}), so its naming differs", flush=True)  # fmt: skip
        return []
    missing = sorted(set(base_map) - set(merged_map))
    skipped = [name for name in missing if _module(name) in targeted]
    if skipped:
        print(f"base tensors not restored, as the adapter targets their modules: {skipped}", flush=True)
    missing = [name for name in missing if name not in skipped]
    if not missing:
        return []
    by_file: dict[str, list[str]] = {}
    for name in missing:
        by_file.setdefault(base_map[name], []).append(name)
    entries: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for file, names in by_file.items():
        header, start = _header(base / file)
        with (base / file).open("rb") as handle:
            for name in names:
                begin, end = header[name]["data_offsets"]
                handle.seek(start + begin)
                data = handle.read(end - begin)
                entries[name] = {"dtype": header[name]["dtype"], "shape": header[name]["shape"],
                                 "data_offsets": [offset, offset + len(data)]}  # fmt: skip
                chunks.append(data)
                offset += len(data)
    blob = json.dumps({"__metadata__": {"format": "pt"}, **entries}, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # keep the data 8-byte aligned, as the safetensors writer does
    with (output / EXTRA_SHARD).open("wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)
        for chunk in chunks:
            handle.write(chunk)
    index_path = output / INDEX
    index = (
        json.loads(index_path.read_text()) if index_path.exists() else {"metadata": {}, "weight_map": dict(merged_map)}
    )
    index["weight_map"].update(dict.fromkeys(missing, EXTRA_SHARD))
    index["weight_map"] = dict(sorted(index["weight_map"].items()))
    if "total_size" in index.get("metadata", {}):
        index["metadata"]["total_size"] = int(index["metadata"]["total_size"]) + offset
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    _restore_config_fields(base, output, {name.split(".")[0] for name in missing})
    print(f"restored {len(missing)} base tensors the merged model class does not load ({offset:,} bytes, "
          f"{EXTRA_SHARD}): {missing}", flush=True)  # fmt: skip
    return missing


def _restore_config_fields(base: Path, output: Path, prefixes: set[str]) -> None:
    """Copy config fields named after a restored module (`mtp_num_hidden_layers` for `mtp.*`) that the merged
    config.json lacks, at the top level and in `text_config`."""
    if not (base / "config.json").exists() or not (output / "config.json").exists():
        return
    source, merged = json.loads((base / "config.json").read_text()), json.loads((output / "config.json").read_text())
    added = []
    for section in (None, "text_config"):
        src = source.get(section, {}) if section else source
        dst = merged.setdefault(section, {}) if section and section in source else merged
        if not isinstance(src, dict) or not isinstance(dst, dict):
            continue
        for key, value in src.items():
            if key not in dst and any(key.startswith(f"{prefix}_") for prefix in prefixes):
                dst[key] = value
                added.append(f"{section}.{key}" if section else key)
    if added:
        (output / "config.json").write_text(json.dumps(merged, indent=2) + "\n")
        print(f"restored config fields: {added}", flush=True)


def base_snapshot(base_model: str) -> Path:
    """The base checkpoint's folder: a local directory, or its Hub snapshot (weights and configs only; already cached
    by `from_pretrained`)."""
    if Path(base_model).is_dir():
        return Path(base_model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base_model, allow_patterns=["*.safetensors", INDEX, "config.json"]))


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
    # Base tensors the model class never loaded (Qwen3.5's MTP head), copied unchanged so vLLM can still use them.
    restore_base_tensors(base_snapshot(args.base_model), output, adapter_modules(snapshot))
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
