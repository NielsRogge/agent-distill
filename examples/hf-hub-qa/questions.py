"""Seed questions of the hf-hub-qa task, expanded from the user's examples (2026-11-25):

    list the top trending models on Hugging Face / what are some good models for object detection /
    which decision models does the Transformers library support / what are some good papers on knowledge distillation

Each question has a category and, where the Hub can answer it exactly, a `reference`: a canonical command of the
pinned `hf` CLI that the stage runs when it scores the episode (so the reference is as fresh as the agent's view):

- {"ids": [argv...], "key": "id", "k": 5}: the answer should name the top-k ids of that listing (a match anywhere
  in its top 2k counts, since trending lists move while an episode runs).
- {"fact": [argv...], "path": "card_data.license"}: the answer should state that value.

Only questions that ask for a ranking (trending, most downloaded or liked, a leaderboard, most recent) or a fact have
a reference; "good models for X" questions have many right answers and are scored by the judge (and the grounding
check) only.

    uv run --extra hub python examples/hf-hub-qa/questions.py --push agent-distillation-hf/hf-hub-questions \\
        --collection agent-distillation-hf/hugging-face-hub-qa-6b0a9ec0b0d291cc23d3e865   # (re)publish the dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any


def ids(*argv: str, k: int = 5, key: str = "id") -> dict[str, Any]:
    return {"ids": list(argv), "key": key, "k": k}


def fact(*argv: str, path: str) -> dict[str, Any]:
    return {"fact": list(argv), "path": path}


TOP = ("--limit", "10")
QUESTIONS: list[tuple[str, str, dict[str, Any] | None]] = [
    # Trending and recent
    ("trending", "List the top trending models on Hugging Face right now.",
     ids("models", "list", "--sort", "trending_score", *TOP)),
    ("trending", "What are the top 3 trending text-to-image models on the Hub at the moment?",
     ids("models", "list", "--pipeline-tag", "text-to-image", "--sort", "trending_score", "--limit", "6", k=3)),
    ("trending", "Which datasets are trending on Hugging Face right now?",
     ids("datasets", "list", "--sort", "trending_score", *TOP)),
    ("trending", "Show me the most popular trending Spaces today.",
     ids("spaces", "list", "--sort", "trending_score", *TOP)),
    ("trending", "What are today's trending papers on Hugging Face?",
     ids("papers", "list", "--sort", "trending", *TOP)),
    ("trending", "Which Qwen models are trending the most right now?",
     ids("models", "list", "--author", "Qwen", "--sort", "trending_score", "--limit", "6", k=3)),
    ("trending", "What are the trending automatic speech recognition models this week?",
     ids("models", "list", "--pipeline-tag", "automatic-speech-recognition", "--sort", "trending_score", "--limit", "6",
         k=3)),
    # Model discovery
    ("discovery", "What are some good models for object detection?", None),
    ("discovery", "I need a small instruction-tuned language model (under 3B parameters) that I can run on a laptop. "
     "What do you recommend?", None),
    ("discovery", "Which speech recognition models work well for Dutch?", None),
    ("discovery", "What are the most downloaded sentence embedding models for multilingual retrieval?", None),
    ("discovery", "Are there GGUF versions of Qwen/Qwen3.5-9B that I can run with llama.cpp?", None),
    ("discovery", "Recommend some image segmentation models with a permissive license.", None),
    ("discovery", "What models can I use for time series forecasting?", None),
    ("discovery", "What are good open text-to-speech models?", None),
    # Libraries
    ("libraries", "Which decision models does the Transformers library support?",  # the Hub's decision-model tag
     ids("models", "list", "--filter", "transformers", "--filter", "decision-model", "--sort", "trending_score",
         "--limit", "6", k=3)),
    ("libraries", "What are the most downloaded models for the diffusers library?",
     ids("models", "list", "--filter", "diffusers", "--sort", "downloads", *TOP)),
    ("libraries", "Which sentence-transformers models are the most popular?",
     ids("models", "list", "--filter", "sentence-transformers", "--sort", "downloads", *TOP)),
    ("libraries", "What are the most liked timm models on the Hub?",
     ids("models", "list", "--filter", "timm", "--sort", "likes", *TOP)),
    ("libraries", "Which models on the Hub can I run with MLX on a Mac? Give me a few popular ones.", None),
    ("libraries", "Are there popular LoRA adapters for FLUX on the Hub?", None),
    # One model
    ("model-details", "What is the license of Qwen/Qwen3.5-9B?",
     fact("models", "info", "Qwen/Qwen3.5-9B", path="card_data.license")),
    ("model-details", "Which library is meta-llama/Llama-3.1-8B-Instruct meant to be used with?",
     fact("models", "info", "meta-llama/Llama-3.1-8B-Instruct", path="library_name")),
    ("model-details", "What task (pipeline) is openai/whisper-large-v3 for?",
     fact("models", "info", "openai/whisper-large-v3", path="pipeline_tag")),
    ("model-details", "Is google/gemma-3-27b-it a gated model? What do I need to do to download it?", None),
    ("model-details", "Which files are in the openai-community/gpt2 repository, and which ones hold the weights?",
     None),
    ("model-details", "What base model was unsloth/Qwen3.5-9B-GGUF made from?", None),
    ("model-details", "How many parameters does Qwen/Qwen3.5-9B have according to its safetensors metadata?", None),
    # Datasets
    ("datasets", "What are some good datasets for instruction tuning a language model?", None),
    ("datasets", "Find me some Dutch text datasets on the Hub.", None),
    ("datasets", "What splits and columns does the rajpurkar/squad dataset have?", None),
    ("datasets", "Which models top the SWE-bench Verified leaderboard on the Hub?",
     ids("datasets", "leaderboard", "SWE-bench/SWE-bench_Verified", "--limit", "6", key="model_id", k=3)),
    ("datasets", "What are the most downloaded datasets for code generation?", None),
    ("datasets", "Which object detection datasets are popular on the Hub?", None),
    ("datasets", "What is the license of the HuggingFaceFW/fineweb dataset?",
     fact("datasets", "info", "HuggingFaceFW/fineweb", path="card_data.license")),
    # Papers
    ("papers", "What are some good papers on knowledge distillation?", None),
    ("papers", "Find recent papers on speculative decoding.", None),
    ("papers", "What are some papers about GRPO or reinforcement learning for reasoning in LLMs?", None),
    ("papers", "What is the paper 2205.02340 about, and who wrote it?", None),
    ("papers", "Which papers are trending this week on Hugging Face, and what are they about?", None),
    ("papers", "Are there papers on Hugging Face about distilling large models into small agents?", None),
    ("papers", "Find papers on vision-language-action models for robotics.", None),
    # Spaces
    ("spaces", "Is there a Space where I can try background removal on an image?", None),
    ("spaces", "What are the most liked Spaces on Hugging Face?",
     ids("spaces", "list", "--sort", "likes", *TOP)),
    ("spaces", "Find a Space for text-to-speech that I can try in the browser.", None),
    ("spaces", "Which Spaces are leaderboards for open LLMs?", None),
    ("spaces", "Are there Spaces that run image generation on ZeroGPU?", None),
    # Organizations and collections
    ("orgs", "What collections has NVIDIA published on Hugging Face?", None),
    ("orgs", "Which models has the Qwen organization released most recently?",
     ids("models", "list", "--author", "Qwen", "--sort", "created_at", "--limit", "6", k=3)),
    ("orgs", "What datasets has HuggingFaceFW published?", None),
    ("orgs", "Which models from mistralai are the most downloaded?",
     ids("models", "list", "--author", "mistralai", "--sort", "downloads", "--limit", "6", k=3)),
]  # fmt: skip


def rows() -> list[dict[str, Any]]:
    out = []
    for category, question, reference in QUESTIONS:
        qid = hashlib.sha256(question.encode()).hexdigest()[:12]
        out.append({"question_id": qid, "category": category, "question": question,
                    "reference": json.dumps(reference) if reference else ""})  # fmt: skip
    return out


CARD = """---
license: apache-2.0
language: [en]
task_categories: [question-answering]
tags: [agent-distill, agents, hf-cli]
configs:
- config_name: default
  data_files:
  - split: seed
    path: data/seed.jsonl
---

# Hugging Face Hub questions

Questions about the Hugging Face Hub for an agent that answers them with the `hf` CLI (agent-distill's `hf-hub-qa`
task). The seed split was expanded from a handful of example questions. `reference`, when set, is a canonical `hf`
command (JSON) whose live output the task's rubrics check the answer against: the top-k ids of a ranked listing, or
one fact.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", help="Publish the seed questions to this dataset repo (public; split `seed`)")
    parser.add_argument("--collection", default="", help="Collection the dataset joins when it is created")
    args = parser.parse_args()
    data = rows()
    if not args.push:
        print(json.dumps(data, indent=2))
        return
    import io

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    created = not api.repo_exists(args.push, repo_type="dataset")
    api.create_repo(args.push, repo_type="dataset", private=False, exist_ok=True)
    lines = "".join(json.dumps(r) + "\n" for r in data).encode()
    api.create_commit(args.push, repo_type="dataset", commit_message=f"{len(data)} seed questions",
                      operations=[CommitOperationAdd("data/seed.jsonl", io.BytesIO(lines)),
                                  CommitOperationAdd("README.md", io.BytesIO(CARD.encode()))])  # fmt: skip
    if created and args.collection:
        api.add_collection_item(args.collection, item_id=args.push, item_type="dataset", exists_ok=True,
                                note="hf-hub-qa: questions about the Hub for an hf-CLI agent (seed split)")  # fmt: skip
    print(f"pushed {len(data)} questions to {args.push}")


if __name__ == "__main__":
    main()
