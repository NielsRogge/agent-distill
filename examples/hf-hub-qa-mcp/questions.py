"""Build the Hub fact questions (`nielsr/hf-hub-facts`): one exact, frozen answer per question.

Entities are sampled from the Hub (the most downloaded models of 20 pipeline tags, the papers those models link,
the most downloaded datasets, the most liked Spaces), and every gold value is read from the Hub API when the dataset
is built, then frozen with `snapshot_date`. Splits are disjoint by entity: a question's split is a hash of the entity
it is about, so test asks about repos and papers never seen in training.

Templates (answer kinds are checked by facts.py):

| template | question about | kind |
|---|---|---|
| model_license, model_pipeline, model_library | a model | category (with the field's vocabulary as hedge) |
| model_params | a model with safetensors metadata | number (±1%) |
| model_base | a model with exactly one base model | repo |
| base_license | the base model of a model (two hops) | category |
| model_paper | a model whose tags link arXiv papers | arxiv |
| paper_title, paper_first_author | a paper linked from a sampled model | title, first_author |
| dataset_license | a dataset | category (hedge) |
| space_sdk | a Space | category (hedge) |

Every row also stores a `canary`: a wrong answer (another entity's value for the same template) that the verifier
must reject, and `live`: how to re-read the value from the Hub, so the rubric can tell a drifted row (the Hub changed
since the snapshot) from a wrong answer.

    uv run python examples/hf-hub-qa-mcp/questions.py --out /tmp/facts           # build and validate, no upload
    uv run python examples/hf-hub-qa-mcp/questions.py --out /tmp/facts --push nielsr/hf-hub-facts \\
        --collection nielsr/agent-distillation-6b0445568a39b1d979c52e31          # private; scanned; then public
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent_distill.hub_facts import EXPAND, card, license_value, model_fields, text
from agent_distill.rubrics import facts

PIPELINE_TAGS = [
    "text-generation", "image-text-to-text", "automatic-speech-recognition", "text-to-image", "feature-extraction",
    "sentence-similarity", "image-classification", "object-detection", "text-classification", "token-classification",
    "translation", "text-to-speech", "image-segmentation", "fill-mask", "zero-shot-classification",
    "depth-estimation", "audio-classification", "image-to-image", "summarization", "question-answering",
]  # fmt: skip
MODELS_PER_TAG = 25
DATASETS = 150
SPACES = 150
TEMPLATES_PER_MODEL = 2
SPLITS = (("train", 70), ("validation", 78), ("test", 100))  # cumulative percent by entity hash
UNKNOWN = {"", "other", "unknown", "none"}


def known(value: Any) -> bool:
    """A gold value worth asking about: set, and not a placeholder (every value, for a dual license)."""
    values = value if isinstance(value, list) else [value]
    return bool(values) and all(str(v).strip() and str(v).lower() not in UNKNOWN for v in values)


QUESTIONS = {
    "model_license": "What is the license of the model {id} on the Hugging Face Hub?",
    "model_pipeline": "What task (pipeline tag) is the Hugging Face model {id} for?",
    "model_library": "Which library is the Hugging Face model {id} meant to be used with?",
    "model_params": "How many parameters does {id} have, according to its safetensors metadata on the Hub?",
    "model_base": "What is the base model of {id} on the Hugging Face Hub?",
    "base_license": "{id} was made from a base model. What is the license of that base model?",
    "model_paper": "Which arXiv paper is linked to the Hugging Face model {id}? Give its arXiv id.",
    "paper_title": "What is the title of the paper with arXiv id {id} on Hugging Face Papers?",
    "paper_first_author": "Who is the first author of the paper {id} on Hugging Face Papers?",
    "dataset_license": "What is the license of the dataset {id} on the Hugging Face Hub?",
    "space_sdk": "Which SDK does the Hugging Face Space {id} use?",
}
KINDS = {
    "model_license": "category", "model_pipeline": "category", "model_library": "category", "model_params": "number",
    "model_base": "repo", "base_license": "category", "model_paper": "arxiv", "paper_title": "title",
    "paper_first_author": "first_author", "dataset_license": "category", "space_sdk": "category",
}  # fmt: skip
HEDGED = {"model_license": "license", "model_pipeline": "pipeline", "model_library": "library",
          "dataset_license": "license", "space_sdk": "sdk"}  # fmt: skip


def api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def split_of(entity: str) -> str:
    bucket = int(hashlib.sha256(entity.encode()).hexdigest(), 16) % 100
    return next(name for name, upto in SPLITS if bucket < upto)


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------


def candidates(hub: Any, log: Any = print) -> list[dict[str, Any]]:
    """Every possible question (before sampling), with its gold value and live spec."""
    out: list[dict[str, Any]] = []

    def add(template: str, entity: str, ident: str, gold: Any, live: dict[str, Any], **extra: Any) -> None:
        out.append({"template": template, "entity": entity, "subject": ident, "gold": gold, "live": live,
                    "extra": extra})  # fmt: skip

    models: dict[str, dict[str, Any]] = {}
    for tag in PIPELINE_TAGS:
        for info in hub.list_models(pipeline_tag=tag, sort="downloads", limit=MODELS_PER_TAG, expand=EXPAND):
            if not info.private:
                models.setdefault(info.id, model_fields(info))
        log(f"models: {tag} done ({len(models)} total)")
    papers: set[str] = set()
    base_cache: dict[str, dict[str, Any] | None] = {}
    for ident, f in models.items():
        entity = f"model:{ident}"
        for field, template in (("license", "model_license"), ("pipeline", "model_pipeline"),
                                ("library", "model_library")):  # fmt: skip
            if known(f[field]):
                add(template, entity, ident, f[field], {"type": "model", "id": ident, "field": field})
        if f["params"]:
            add("model_params", entity, ident, int(f["params"]), {"type": "model", "id": ident, "field": "params"})
        if f["base"] and f["base"] != ident:
            add("model_base", entity, ident, f["base"], {"type": "model", "id": ident, "field": "base"})
            if f["base"] not in base_cache:
                try:
                    base_cache[f["base"]] = model_fields(hub.model_info(f["base"], expand=EXPAND))
                except Exception:
                    base_cache[f["base"]] = None
            base = base_cache[f["base"]]
            if base and known(base["license"]) and base["license"] != f["license"]:
                add("base_license", entity, ident, base["license"], {"type": "base", "id": ident, "field": "license"})
        if 1 <= len(f["arxiv"]) <= 3:
            add("model_paper", entity, ident, f["arxiv"], {"type": "model", "id": ident, "field": "arxiv"})
            papers.update(f["arxiv"])
    log(f"{len(models)} models, {len(papers)} linked papers")
    for paper in sorted(papers):
        try:
            info = hub.paper_info(paper)
        except Exception:
            continue
        authors = [text(a.name) for a in info.authors or [] if text(a.name)]
        title = text(info.title)
        if title:
            add("paper_title", f"paper:{paper}", paper, title, {"type": "paper", "id": paper, "field": "title"})
        if authors:
            add("paper_first_author", f"paper:{paper}", paper, authors[0],
                {"type": "paper", "id": paper, "field": "first_author"}, authors=authors)  # fmt: skip
    for info in hub.list_datasets(sort="downloads", limit=DATASETS, expand=["cardData"]):
        license_ = license_value(card(info).get("license"))
        if known(license_) and not info.private:
            add("dataset_license", f"dataset:{info.id}", info.id, license_,
                {"type": "dataset", "id": info.id, "field": "license"})  # fmt: skip
    for info in hub.list_spaces(sort="likes", limit=SPACES, expand=["sdk"]):
        if info.sdk and not info.private:
            add(
                "space_sdk",
                f"space:{info.id}",
                info.id,
                text(info.sdk),
                {"type": "space", "id": info.id, "field": "sdk"},
            )
    return out


def finalize(cands: list[dict[str, Any]], snapshot_date: str, seed: int = 0) -> list[dict[str, Any]]:
    """Sample at most TEMPLATES_PER_MODEL templates per model; add hedge vocabularies, canaries and splits."""
    rng = random.Random(seed)
    vocab: dict[str, set[str]] = defaultdict(set)
    for c in cands:
        if c["template"] in HEDGED:
            values = c["gold"] if isinstance(c["gold"], list) else [c["gold"]]
            vocab[HEDGED[c["template"]]].update(str(v) for v in values)
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_entity[c["entity"]].append(c)
    chosen = []
    for entity in sorted(by_entity):
        group = by_entity[entity]
        if entity.startswith("model:"):
            rng.shuffle(group)
            group = sorted(group[:TEMPLATES_PER_MODEL], key=lambda c: c["template"])
        chosen.extend(group)
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chosen:
        by_template[c["template"]].append(c)
    rows = []
    for c in chosen:
        kind = KINDS[c["template"]]
        extra = dict(c["extra"])
        if c["template"] in HEDGED:
            extra["hedge"] = sorted(vocab[HEDGED[c["template"]]])
        wrong = [o["gold"] for o in by_template[c["template"]]
                 if not facts.check(kind, c["gold"], facts.render(kind, o["gold"]), **extra)]  # fmt: skip
        canary = rng.choice(wrong) if wrong else None
        question = QUESTIONS[c["template"]].format(id=c["subject"])
        rows.append({
            "question_id": hashlib.sha256(question.encode()).hexdigest()[:12],
            "split": split_of(c["entity"]),
            "template": c["template"],
            "entity": c["entity"],
            "question": question,
            "kind": kind,
            "gold": json.dumps(c["gold"]),
            "extra": json.dumps(extra),
            "canary": json.dumps(canary),
            "live": json.dumps(c["live"]),
            "snapshot_date": snapshot_date,
        })  # fmt: skip
    return rows


def validate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """Keep rows whose verifier passes the oracle, rejects the no-op and the canary, and is deterministic."""
    kept, dropped = [], Counter()
    for row in rows:
        kind, gold, extra = row["kind"], json.loads(row["gold"]), json.loads(row["extra"])
        canary = json.loads(row["canary"])
        oracle = facts.render(kind, gold)
        checks = {
            "oracle": facts.check(kind, gold, oracle, **extra) and facts.check(kind, gold, oracle, **extra),
            "no_op": not facts.check(kind, gold, "I could not find this on the Hub.", **extra),
            "canary": canary is not None and not facts.check(kind, gold, facts.render(kind, canary), **extra),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            dropped.update(failed)
        else:
            kept.append(row)
    return kept, dropped


CARD = """---
license: apache-2.0
language: [en]
task_categories: [question-answering]
tags: [agent-distill, agents, hf-mcp, hf-cli]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# Hugging Face Hub facts

Questions about the Hugging Face Hub with one exact answer each, for agents that look things up on the Hub (the
`hf-hub-qa-mcp` task of agent-distill, Hub MCP server; shareable with its `hf` CLI twin). Gold values were read from the
Hub API on **{date}** and are frozen (`snapshot_date`); `live` says how to re-read a value, so a scorer can tell a
row the Hub has since changed from a wrong answer. Splits are disjoint by entity (repo or paper).

Columns: `question`, `kind` (how `gold` is checked: category, repo, number, title, first_author, arxiv), `gold` and
`extra` (JSON), `canary` (a known-wrong value the checker rejects), `template`, `entity`, `split`.

| split | rows |
|---|---|
{table}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Folder for data/<split>.jsonl and README.md")
    parser.add_argument("--push", help="Dataset repo to create (private), scan, fill and then make public")
    parser.add_argument("--collection", default="", help="Collection the dataset joins")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    date = time.strftime("%Y-%m-%d", time.gmtime())
    hub = api()
    rows, dropped = validate(finalize(candidates(hub), date, args.seed))
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    random.Random(args.seed).shuffle(rows)  # stages take the first N rows: any prefix mixes templates
    counts = Counter(r["split"] for r in rows)
    for split in ("train", "validation", "test"):
        with (out / "data" / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                if row["split"] == split:
                    handle.write(json.dumps(row) + "\n")
    table = "\n".join(f"| {s} | {counts[s]} |" for s in ("train", "validation", "test"))
    (out / "README.md").write_text(CARD.replace("{date}", date).replace("{table}", table))
    print(f"{len(rows)} rows {dict(counts)}; templates {dict(Counter(r['template'] for r in rows))}; "
          f"dropped by validation {dict(dropped)}")  # fmt: skip
    if args.push:
        from agent_distill.credentials import scan_files

        hits = scan_files(sorted(out.rglob("*")))
        if hits:
            raise SystemExit(f"credential scan failed: {hits}")
        created = not hub.repo_exists(args.push, repo_type="dataset")
        hub.create_repo(args.push, repo_type="dataset", private=True, exist_ok=True)
        hub.upload_folder(repo_id=args.push, repo_type="dataset", folder_path=str(out),
                          commit_message=f"{len(rows)} Hub fact questions, snapshot {date}")  # fmt: skip
        hub.update_repo_settings(args.push, repo_type="dataset", private=False)  # scanned above: public
        if created and args.collection:
            hub.add_collection_item(args.collection, item_id=args.push, item_type="dataset", exists_ok=True,
                                    note=f"Hub fact questions (hf-hub-qa-mcp, hf-hub-qa), snapshot {date}")  # fmt: skip
        print(f"pushed to {args.push}")


if __name__ == "__main__":
    main()
