"""Facts about Hugging Face Hub repos and papers, read live from the Hub API: what fact-question datasets freeze as gold
values, and what their rubrics re-read to tell a row that drifted (the Hub changed since the snapshot) from a wrong
answer. Reads are anonymous by default, as the agent sees the Hub.

A row's `live` spec is `{"type": "model" | "base" | "dataset" | "space" | "paper", "id": ..., "field": ...}`:

- model fields: `license` (a string, or a sorted list for a dual license), `pipeline`, `library`, `params`
  (safetensors total), `base` (the single base model, or None), `arxiv` (sorted linked arXiv ids);
- `base`: a field of the model's base model (e.g. its license);
- dataset: `license`; space: `sdk`; paper: `title` or `first_author`;
- `paper_of`: `title` or `first_author` of the one paper a model links (its single `arxiv:` tag);
- `files`: how many files of a repo (`repo_type`, default model) end with `field` (e.g. ".gguf"), in every folder;
- `sum`: the sum of a model field (`params`) over `ids` (a list of model ids).
"""

from __future__ import annotations

import re
from typing import Any

EXPAND = ["cardData", "pipeline_tag", "library_name", "safetensors", "gated", "tags"]


def text(value: Any) -> str:
    """A value as one line of text (runs of whitespace collapsed)."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def card(info: Any) -> dict[str, Any]:
    """A repo's card metadata as a plain dict."""
    data = getattr(info, "card_data", None)
    return data.to_dict() if hasattr(data, "to_dict") else dict(data or {})


def single(value: Any) -> str | None:
    """A card field naming exactly one value (a string, or a one-item list), else None."""
    if isinstance(value, list):
        return value[0] if len(value) == 1 and isinstance(value[0], str) else None
    return value if isinstance(value, str) and value else None


def license_value(value: Any) -> str | list[str]:
    """A card's `license`: one id as a string; several (a dual license) as a sorted list; "" when missing."""
    if isinstance(value, list):
        values = sorted({text(v) for v in value if text(v)})
        return values[0] if len(values) == 1 else values if values else ""
    return text(value)


def model_fields(info: Any) -> dict[str, Any]:
    data = card(info)
    return {
        "license": license_value(data.get("license")),
        "pipeline": text(info.pipeline_tag),
        "library": text(info.library_name),
        "params": info.safetensors.total if getattr(info, "safetensors", None) else None,
        "base": single(data.get("base_model")),
        "arxiv": sorted({t.split(":", 1)[1] for t in info.tags or [] if t.startswith("arxiv:")}),
    }


def live_value(spec: dict[str, Any], hub: Any = None, token: bool | str | None = False) -> Any:
    """The current value of a fact (`spec` as above), anonymous unless `token` says otherwise."""
    if hub is None:
        from huggingface_hub import HfApi

        hub = HfApi()
    kind, ident, field = spec["type"], spec.get("id"), spec["field"]
    if kind == "sum":
        values = [model_fields(hub.model_info(i, expand=EXPAND, token=token)).get(field) for i in spec["ids"]]
        return None if any(v is None for v in values) else sum(values)
    if kind == "files":
        files = hub.list_repo_files(ident, repo_type=spec.get("repo_type", "model"), token=token)
        return sum(1 for name in files if name.lower().endswith(field))
    if kind == "model":
        return model_fields(hub.model_info(ident, expand=EXPAND, token=token)).get(field)
    if kind == "base":
        base = model_fields(hub.model_info(ident, expand=EXPAND, token=token))["base"]
        return model_fields(hub.model_info(base, expand=EXPAND, token=token)).get(field) if base else None
    if kind == "dataset":
        return license_value(card(hub.dataset_info(ident, expand=["cardData"], token=token)).get("license"))
    if kind == "space":
        return text(hub.space_info(ident, expand=["sdk"], token=token).sdk)
    if kind == "paper_of":
        papers = model_fields(hub.model_info(ident, expand=EXPAND, token=token))["arxiv"]
        if len(papers) != 1:
            return None
        kind, ident = "paper", papers[0]
    if kind == "paper":
        info = hub.paper_info(ident)
        return text(info.title) if field == "title" else text(info.authors[0].name if info.authors else "")
    raise ValueError(f"unknown live spec {spec!r}")
