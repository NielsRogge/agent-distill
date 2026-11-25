"""Add a newly created Hub repo to the run's collection, with its one-line note (AGENTS.md, "Hub artifacts")."""

from __future__ import annotations


def join_collection(repo: str, kind: str, collection: str, note: str = "") -> None:
    """Add `repo` (a "model" or "dataset") to `collection`; a no-op without a collection, idempotent otherwise."""
    if not collection:
        return
    from huggingface_hub import HfApi

    HfApi().add_collection_item(collection, item_id=repo, item_type=kind, note=note[:500], exists_ok=True)
    print(f"added {repo} to {collection}", flush=True)
