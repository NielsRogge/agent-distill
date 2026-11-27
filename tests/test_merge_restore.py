"""The merge stage restores base tensors the merged model class does not load (Qwen3.5's MTP head), unchanged.
CPU only, no downloads: tiny safetensors checkpoints written by hand."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from agent_distill.stages.merge import (
    EXTRA_SHARD,
    INDEX,
    _header,
    adapter_modules,
    checkpoint_tensors,
    restore_base_tensors,
)

DTYPE_BYTES = {"BF16": 2, "F32": 4, "F16": 2}


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, data, offset = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        data += raw
        offset += len(raw)
    blob = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def tensor(dtype: str, shape: list[int], seed: int) -> tuple[str, list[int], bytes]:
    count = 1
    for dim in shape:
        count *= dim
    return dtype, shape, bytes((seed * 7 + i) % 256 for i in range(count * DTYPE_BYTES[dtype]))


def read(path: Path, name: str) -> tuple[str, list[int], bytes]:
    header, start = _header(path)
    begin, end = header[name]["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(start + begin)
        return header[name]["dtype"], header[name]["shape"], handle.read(end - begin)


def make_base(folder: Path) -> dict[str, tuple[str, list[int], bytes]]:
    """A sharded base: language-model layers in shard 1, the MTP head (BF16 and F32) in shard 2."""
    folder.mkdir()
    shard1 = {
        "model.language_model.layers.0.mlp.down_proj.weight": tensor("BF16", [4, 3], 1),
        "model.language_model.layers.0.input_layernorm.weight": tensor("BF16", [3], 2),
        "model.visual.blocks.0.attn.qkv.weight": tensor("BF16", [2, 2], 3),
    }
    shard2 = {
        "mtp.layers.0.mlp.down_proj.weight": tensor("BF16", [4, 3], 4),
        "mtp.fc.weight": tensor("F32", [3, 6], 5),
        "mtp.norm.weight": tensor("BF16", [3], 6),
    }
    write_safetensors(folder / "model-00001-of-00002.safetensors", shard1)
    write_safetensors(folder / "model-00002-of-00002.safetensors", shard2)
    weight_map = {**dict.fromkeys(shard1, "model-00001-of-00002.safetensors"),
                  **dict.fromkeys(shard2, "model-00002-of-00002.safetensors")}  # fmt: skip
    (folder / INDEX).write_text(json.dumps({"metadata": {"total_size": 1000}, "weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 3, "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False},
    }))  # fmt: skip
    return {**shard1, **shard2}


def make_merged(folder: Path, base: dict, *, sharded: bool) -> None:
    """What `save_pretrained` writes: every base tensor except the MTP head (merged values differ), and a config
    without the MTP fields."""
    folder.mkdir()
    kept = {name: tensor(dtype, shape, 99) for name, (dtype, shape, _) in base.items() if not name.startswith("mtp.")}
    write_safetensors(folder / "model.safetensors", kept)
    if sharded:
        weight_map = dict.fromkeys(kept, "model.safetensors")
        (folder / INDEX).write_text(json.dumps({"metadata": {"total_size": 500}, "weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": 3}}))


def make_adapter(folder: Path, modules: list[str]) -> Path:
    folder.mkdir()
    tensors = {}
    for module in modules:
        tensors[f"base_model.model.{module}.lora_A.weight"] = tensor("F32", [2, 3], 7)
        tensors[f"base_model.model.{module}.lora_B.weight"] = tensor("F32", [4, 2], 8)
    write_safetensors(folder / "adapter_model.safetensors", tensors)
    return folder


def test_restores_missing_base_tensors_byte_for_byte(tmp_path: Path) -> None:
    for sharded in (True, False):
        base_dir, merged_dir = tmp_path / f"base-{sharded}", tmp_path / f"merged-{sharded}"
        base = make_base(base_dir)
        make_merged(merged_dir, base, sharded=sharded)
        adapter = make_adapter(tmp_path / f"adapter-{sharded}", ["model.layers.0.mlp.down_proj"])
        copied = restore_base_tensors(base_dir, merged_dir, adapter_modules(adapter))
        assert copied == sorted(n for n in base if n.startswith("mtp."))
        merged_map = checkpoint_tensors(merged_dir)
        assert set(merged_map) == set(base)  # the merged checkpoint now names every base tensor
        for name in copied:  # identical dtype, shape and bytes
            assert merged_map[name] == EXTRA_SHARD
            assert read(merged_dir / EXTRA_SHARD, name) == base[name]
        # The merged (not copied) tensors are untouched.
        name = "model.language_model.layers.0.mlp.down_proj.weight"
        assert read(merged_dir / "model.safetensors", name)[2] != base[name][2]
        index = json.loads((merged_dir / INDEX).read_text())
        if sharded:
            assert index["metadata"]["total_size"] == 500 + 4 * 3 * 2 + 3 * 6 * 4 + 3 * 2
        config = json.loads((merged_dir / "config.json").read_text())
        assert config["text_config"]["mtp_num_hidden_layers"] == 1
        assert config["text_config"]["mtp_use_dedicated_embeddings"] is False
        assert config["text_config"]["hidden_size"] == 3


def test_nothing_to_restore_changes_nothing(tmp_path: Path) -> None:
    base_dir, merged_dir = tmp_path / "base", tmp_path / "merged"
    base = {k: v for k, v in make_base(base_dir).items()}
    merged_dir.mkdir()
    write_safetensors(merged_dir / "model.safetensors", base)
    assert restore_base_tensors(base_dir, merged_dir) == []
    assert not (merged_dir / EXTRA_SHARD).exists() and not (merged_dir / INDEX).exists()


def test_never_restores_a_module_the_adapter_targets(tmp_path: Path) -> None:
    """An adapter that trained the MTP head's projection: its base value would be wrong in the merged model."""
    base_dir, merged_dir = tmp_path / "base", tmp_path / "merged"
    base = make_base(base_dir)
    make_merged(merged_dir, base, sharded=True)
    adapter = make_adapter(tmp_path / "adapter", ["mtp.layers.0.mlp.down_proj"])
    copied = restore_base_tensors(base_dir, merged_dir, adapter_modules(adapter))
    assert copied == ["mtp.fc.weight", "mtp.norm.weight"]
    assert "mtp.layers.0.mlp.down_proj.weight" not in checkpoint_tensors(merged_dir)


def test_renamed_checkpoints_are_left_alone(tmp_path: Path) -> None:
    """A merged checkpoint whose naming differs from the base: missing cannot be told from renamed."""
    base_dir, merged_dir = tmp_path / "base", tmp_path / "merged"
    make_base(base_dir)
    merged_dir.mkdir()
    write_safetensors(
        merged_dir / "model.safetensors", {"model.layers.0.mlp.down_proj.weight": tensor("BF16", [4, 3], 1)}
    )
    assert restore_base_tensors(base_dir, merged_dir) == []
    assert not (merged_dir / EXTRA_SHARD).exists()
