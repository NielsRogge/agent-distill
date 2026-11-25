"""Facts about Hugging Face Hub repos and papers, read live from the Hub API: what fact-question datasets freeze as gold
values, and what their rubrics re-read to tell a row that drifted (the Hub changed since the snapshot) from a wrong
answer. Reads are anonymous by default, as the agent sees the Hub.

A row's `live` spec is `{"type": "model" | "base" | "dataset" | "space" | "paper", "id": ..., "field": ...}`:

- model fields: `license`, `pipeline`, `library`, `params` (safetensors total), `base` (the single base model, or
  None), `arxiv` (sorted linked arXiv ids);
- `base`: a field of the model's base model (e.g. its license);
- dataset: `license`; space: `sdk`; paper: `title` or `first_author`.
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


def model_fields(info: Any) -> dict[str, Any]:
    data = card(info)
    return {
        "license": text(data.get("license")),
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
    kind, ident, field = spec["type"], spec["id"], spec["field"]
    if kind == "model":
        return model_fields(hub.model_info(ident, expand=EXPAND, token=token)).get(field)
    if kind == "base":
        base = model_fields(hub.model_info(ident, expand=EXPAND, token=token))["base"]
        return model_fields(hub.model_info(base, expand=EXPAND, token=token)).get(field) if base else None
    if kind == "dataset":
        return text(card(hub.dataset_info(ident, expand=["cardData"], token=token)).get("license"))
    if kind == "space":
        return text(hub.space_info(ident, expand=["sdk"], token=token).sdk)
    if kind == "paper":
        info = hub.paper_info(ident)
        return text(info.title) if field == "title" else text(info.authors[0].name if info.authors else "")
    raise ValueError(f"unknown live spec {spec!r}")
