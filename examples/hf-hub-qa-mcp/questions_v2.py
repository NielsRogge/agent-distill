"""Build v2 of the Hub fact questions (`agent-distillation-hf/hf-hub-facts-v2`): harder questions with one exact, frozen
answer each.

v1 (questions.py) asked single-hop lookups, which base Qwen3.5-4B already solved 96% of the time, leaving SFT no room.
Every v2 question needs at least two lookups, or listing a repo's files:

| template | needs | kind |
|---|---|---|
| base_license, base_pipeline | the model's card → its base model → that model's license / pipeline tag | category |
| base_params | the model's card → its base model → its safetensors metadata | number (±1%) |
| paper_title, paper_first_author | the model's card → its one linked arXiv paper → the paper | title, first_author |
| params_sum | three models' safetensors metadata, added up | number (±1%) |
| gguf_count, safetensors_count | the repo's full file listing (every folder) | count |

Two-hop questions are only asked when the hop matters: the base model's value differs from the model's own (license,
pipeline tag, parameter count), or the model has no value of its own. Count questions take repos with 2-40 matching
files. A triple for `params_sum` is drawn from models of one split, and its split follows its first model, so test
triples never reuse train models.

The design is v1's: gold values are read from the Hub API at build time and frozen (`snapshot_date`), `live` says how
to re-read them (agent_distill.hub_facts), every row passes oracle / no-op / canary validation (questions.validate), and
splits are disjoint by entity.

    uv run python examples/hf-hub-qa-mcp/questions_v2.py --out /tmp/facts-v2           # build and validate, no upload
    uv run python examples/hf-hub-qa-mcp/questions_v2.py --out /tmp/facts-v2 \\
        --push agent-distillation-hf/hf-hub-facts-v2 \\
        --collection agent-distillation-hf/hugging-face-hub-qa-6b0a9ec0b0d291cc23d3e865
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

from agent_distill.hub_facts import EXPAND, model_fields, text

try:  # imported from the task package (tests)
    from .questions import PIPELINE_TAGS, api, finalize, known, split_of, validate, write_and_push
except ImportError:  # run as a script
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from questions import PIPELINE_TAGS, api, finalize, known, split_of, validate, write_and_push  # type: ignore

MODELS_PER_TAG = 60
GGUF_REPOS = 300
COUNT_SAMPLE = 220  # repos whose files are listed, per count template
COUNT_RANGE = (2, 40)
SUMS = 260
TEMPLATES_PER_MODEL = 2

QUESTIONS = {
    "base_license": "{id} was made from a base model. What is the license of that base model?",
    "base_pipeline": "{id} was made from a base model. What is the pipeline tag (task) of that base model?",
    "base_params": "{id} was made from a base model. How many parameters does that base model have, according to its "
    "safetensors metadata on the Hub?",
    "paper_title": "The Hugging Face model {id} links an arXiv paper. What is the title of that paper?",
    "paper_first_author": "The Hugging Face model {id} links an arXiv paper. Who is the first author of that paper?",
    "params_sum": "What is the total number of parameters of {id} combined, according to their safetensors metadata "
    "on the Hub?",
    "gguf_count": "How many .gguf files does the Hugging Face repo {id} contain, counting every folder?",
    "safetensors_count": "How many .safetensors files does the Hugging Face model {id} contain, counting every folder?",
}
KINDS = {
    "base_license": "category", "base_pipeline": "category", "base_params": "number", "paper_title": "title",
    "paper_first_author": "first_author", "params_sum": "number", "gguf_count": "count",
    "safetensors_count": "count",
}  # fmt: skip
HEDGED = {"base_license": "license", "base_pipeline": "pipeline"}

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

# Hugging Face Hub facts, v2

Harder questions about the Hugging Face Hub, each with one exact answer, for agents that look things up on the Hub
(agent-distill's `hf-hub-qa-mcp` and `hf-hub-qa` tasks). Every question needs at least two lookups or a repo's full
file listing: the base model's license, pipeline tag or size; the title or first author of the paper a model links;
the total size of three models; the number of GGUF or safetensors files in a repo. v1
([agent-distillation-hf/hf-hub-facts](https://huggingface.co/datasets/agent-distillation-hf/hf-hub-facts))
asked single lookups and was too easy.

Gold values were read from the Hub API on **{date}** and are frozen (`snapshot_date`); `live` says how to re-read a
value, so a scorer can tell a row the Hub has since changed from a wrong answer. Splits are disjoint by entity.

Columns: `question`, `kind` (how `gold` is checked: category, number, title, first_author, count), `gold` and `extra`
(JSON), `canary` (a known-wrong value the checker rejects), `template`, `entity`, `split`.

| split | rows |
|---|---|
{table}
"""


def differs(a: Any, b: Any) -> bool:
    """`a` is set and is not `b` (numbers: more than 2% apart)."""
    if a in (None, "", []):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
        return abs(a - b) > 0.02 * abs(b)
    return a != b


def file_count(hub: Any, repo: str, suffix: str) -> int | None:
    try:
        return sum(1 for name in hub.list_repo_files(repo) if name.lower().endswith(suffix))
    except Exception:  # gated, moved or rate limited: skip the repo
        return None


def candidates(hub: Any, seed: int = 0, log: Any = print) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []

    def add(template: str, entity: str, subject: str, gold: Any, live: dict[str, Any], split_entity: str = "",
            **extra: Any) -> None:  # fmt: skip
        out.append({"template": template, "entity": entity, "subject": subject, "gold": gold, "live": live,
                    "split_entity": split_entity, "extra": extra})  # fmt: skip

    models: dict[str, dict[str, Any]] = {}
    for tag in PIPELINE_TAGS:
        for info in hub.list_models(pipeline_tag=tag, sort="downloads", limit=MODELS_PER_TAG, expand=EXPAND):
            if not info.private:
                models.setdefault(info.id, model_fields(info))
        log(f"models: {tag} ({len(models)} total)")
    gguf: dict[str, dict[str, Any]] = {}
    for info in hub.list_models(filter="gguf", sort="downloads", limit=GGUF_REPOS, expand=EXPAND):
        if not info.private:
            gguf[info.id] = model_fields(info)
            models.setdefault(info.id, gguf[info.id])
    log(f"{len(models)} models, {len(gguf)} GGUF repos")

    bases: dict[str, dict[str, Any] | None] = {}
    for ident, f in models.items():
        entity = f"model:{ident}"
        base_id = f["base"]
        if base_id and base_id != ident:
            if base_id not in bases:
                try:
                    bases[base_id] = model_fields(hub.model_info(base_id, expand=EXPAND))
                except Exception:  # a private, gated or missing base model: no two-hop question
                    bases[base_id] = None
            base = bases[base_id]
            if base:
                if known(base["license"]) and differs(base["license"], f["license"]):
                    add(
                        "base_license",
                        entity,
                        ident,
                        base["license"],
                        {"type": "base", "id": ident, "field": "license"},
                    )
                if base["pipeline"] and differs(base["pipeline"], f["pipeline"]):
                    add("base_pipeline", entity, ident, base["pipeline"],
                        {"type": "base", "id": ident, "field": "pipeline"})  # fmt: skip
                if base["params"] and differs(base["params"], f["params"] or 0):
                    add("base_params", entity, ident, int(base["params"]),
                        {"type": "base", "id": ident, "field": "params"})  # fmt: skip
        if len(f["arxiv"]) == 1:
            try:
                paper = hub.paper_info(f["arxiv"][0])
            except Exception:  # not on HF Papers
                paper = None
            if paper is not None:
                authors = [text(a.name) for a in paper.authors or [] if text(a.name)]
                if text(paper.title):
                    add("paper_title", entity, ident, text(paper.title),
                        {"type": "paper_of", "id": ident, "field": "title"})  # fmt: skip
                if len(authors) >= 2:  # with one author, "first" is trivial
                    add("paper_first_author", entity, ident, authors[0],
                        {"type": "paper_of", "id": ident, "field": "first_author"}, authors=authors)  # fmt: skip
    log(f"two-hop and paper candidates: {len(out)}")

    for suffix, template, pool in ((".gguf", "gguf_count", sorted(gguf)),
                                   (".safetensors", "safetensors_count",
                                    sorted(i for i, f in models.items() if (f["params"] or 0) > 2e9))):  # fmt: skip
        for ident in rng.sample(pool, min(COUNT_SAMPLE, len(pool))):
            n = file_count(hub, ident, suffix)
            if n is not None and COUNT_RANGE[0] <= n <= COUNT_RANGE[1]:
                add(template, f"model:{ident}", ident, n, {"type": "files", "id": ident, "field": suffix})
        log(f"{template}: {sum(1 for c in out if c['template'] == template)}")

    by_split: dict[str, list[str]] = {}
    for ident, f in models.items():
        if f["params"]:
            by_split.setdefault(split_of(f"model:{ident}"), []).append(ident)
    for _split, pool in sorted(by_split.items()):
        share = round(SUMS * len(pool) / max(1, sum(len(p) for p in by_split.values())))
        for _ in range(share):
            trio = sorted(rng.sample(sorted(pool), 3))
            total = sum(int(models[i]["params"]) for i in trio)
            subject = f"{trio[0]}, {trio[1]} and {trio[2]}"
            add("params_sum", "sum:" + "|".join(trio), subject, total, {"type": "sum", "ids": trio, "field": "params"},
                split_entity=f"model:{trio[0]}")  # fmt: skip
    log(f"params_sum: {sum(1 for c in out if c['template'] == 'params_sum')}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Folder for data/<split>.jsonl and README.md")
    parser.add_argument("--push", help="Dataset repo to create (private), scan, fill and then make public")
    parser.add_argument("--collection", default="", help="Collection the dataset joins")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    date = time.strftime("%Y-%m-%d", time.gmtime())
    hub = api()
    cands = candidates(hub, args.seed)
    rows, dropped = validate(finalize(cands, date, args.seed, questions=QUESTIONS, kinds=KINDS, hedged=HEDGED,
                                      per_model=TEMPLATES_PER_MODEL))  # fmt: skip
    write_and_push(rows, dropped, Path(args.out), date, card=CARD, push=args.push, collection=args.collection,
                   note=f"Hub fact questions v2 (multi-hop, counts; hf-hub-qa-mcp, hf-hub-qa), snapshot {date}",
                   hub=hub)  # fmt: skip


if __name__ == "__main__":
    main()
