"""Open-ended training questions for `hf-hub-qa` (Option #3: mixed factoid + open-ended distillation).

Builds 240 diverse open-ended Hugging Face Hub questions across 10 categories (trending, discovery, libraries,
model-details, datasets, papers, spaces, orgs, comparison, workflow), strictly disjoint from the 51 evaluation
questions in `questions.py` (`agent-distillation-hf/hf-hub-questions`).

    uv run --extra hub python examples/hf-hub-qa/open_ended_questions.py --check
    uv run --extra hub python examples/hf-hub-qa/open_ended_questions.py \\
        --push agent-distillation-hf/hf-hub-questions-train \\
        --collection agent-distillation-hf/hugging-face-hub-qa-6b0a9ec0b0d291cc23d3e865
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .questions import QUESTIONS as EVAL_QUESTIONS
    from .questions import TOP, fact, ids
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from questions import QUESTIONS as EVAL_QUESTIONS  # type: ignore[import-not-found]
    from questions import TOP, fact, ids  # type: ignore[import-not-found]

TRAIN_QUESTIONS: list[tuple[str, str, dict[str, Any] | None]] = [
    # ------------------------------------------------------------------
    # 1. Trending & recent (24) — different tasks/orgs/filters from seed
    # ------------------------------------------------------------------
    (
        "trending",
        "What are the top 3 trending text-generation models on Hugging Face right now?",
        ids("models", "list", "--pipeline-tag", "text-generation", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which image-text-to-text (vision-language) models are trending on the Hub right now?",
        ids("models", "list", "--pipeline-tag", "image-text-to-text", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Show the top 3 trending text-to-video models on Hugging Face today.",
        ids("models", "list", "--pipeline-tag", "text-to-video", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "What are the top trending feature-extraction (embedding) models at the moment?",
        ids("models", "list", "--pipeline-tag", "feature-extraction", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which image-to-image models are currently trending on Hugging Face?",
        ids("models", "list", "--pipeline-tag", "image-to-image", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which 3 text-to-speech models have the highest trending score right now?",
        ids("models", "list", "--pipeline-tag", "text-to-speech", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "What are the top trending image-classification models on the Hub right now?",
        ids(
            "models",
            "list",
            "--pipeline-tag",
            "image-classification",
            "--sort",
            "trending_score",
            "--limit",
            "6",
            k=3,
        ),
    ),
    (
        "trending",
        "Which zero-shot-image-classification models are trending on Hugging Face?",
        ids(
            "models",
            "list",
            "--pipeline-tag",
            "zero-shot-image-classification",
            "--sort",
            "trending_score",
            "--limit",
            "6",
            k=3,
        ),
    ),
    (
        "trending",
        "What are the top 3 trending depth-estimation models on the Hub right now?",
        ids("models", "list", "--pipeline-tag", "depth-estimation", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which text-classification models are trending on Hugging Face right now?",
        ids(
            "models",
            "list",
            "--pipeline-tag",
            "text-classification",
            "--sort",
            "trending_score",
            "--limit",
            "6",
            k=3,
        ),
    ),
    (
        "trending",
        "What are the top 3 trending image-to-video models on the Hub right now?",
        ids("models", "list", "--pipeline-tag", "image-to-video", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which google models are trending the most on Hugging Face right now?",
        ids("models", "list", "--author", "google", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which meta-llama models are trending the most on Hugging Face right now?",
        ids("models", "list", "--author", "meta-llama", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which deepseek-ai models are currently trending on Hugging Face?",
        ids("models", "list", "--author", "deepseek-ai", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which microsoft models are trending the most on the Hub today?",
        ids("models", "list", "--author", "microsoft", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "What are the top trending models published by allenai right now?",
        ids("models", "list", "--author", "allenai", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "Which HuggingFaceTB models are trending on the Hub right now?",  # pragma: allowlist secret
        ids(
            "models",
            "list",
            "--author",
            "HuggingFaceTB",  # pragma: allowlist secret
            "--sort",
            "trending_score",
            "--limit",
            "6",
            k=3,
        ),
    ),
    (
        "trending",
        "Which unsloth models are trending the most right now?",
        ids("models", "list", "--author", "unsloth", "--sort", "trending_score", "--limit", "6", k=3),
    ),
    (
        "trending",
        "What are the most recently created models from deepseek-ai?",
        ids("models", "list", "--author", "deepseek-ai", "--sort", "created_at", "--limit", "6", k=3),
    ),
    (
        "trending",
        "What are the most recently uploaded datasets by HuggingFaceH4?",  # pragma: allowlist secret
        ids(
            "datasets",
            "list",
            "--author",
            "HuggingFaceH4",  # pragma: allowlist secret
            "--sort",
            "created_at",
            "--limit",
            "6",
            k=3,
        ),
    ),
    (
        "trending",
        "Which datasets are the most downloaded overall on Hugging Face?",
        ids("datasets", "list", "--sort", "downloads", *TOP),
    ),
    (
        "trending",
        "Which models have the highest all-time likes on Hugging Face?",
        ids("models", "list", "--sort", "likes", *TOP),
    ),
    (
        "trending",
        "Which datasets have the most likes on Hugging Face?",
        ids("datasets", "list", "--sort", "likes", *TOP),
    ),
    (
        "trending",
        "Which trending models on the Hub carry an Apache-2.0 license?",
        None,
    ),
    # ------------------------------------------------------------------
    # 2. Model discovery (34) — diverse tasks, languages, constraints
    # ------------------------------------------------------------------
    ("discovery", "What are some popular open models for code generation and coding assistance?", None),
    ("discovery", "Recommend a few strong open vision-language models for reading charts and documents.", None),
    ("discovery", "What are good models on the Hub for optical character recognition (OCR)?", None),
    ("discovery", "I need a cross-encoder reranker model for RAG search pipelines. What do you recommend?", None),
    ("discovery", "Which open models work well for math reasoning?", None),
    ("discovery", "What models on the Hub can do monocular depth estimation from RGB images?", None),
    ("discovery", "Recommend popular open models for text-to-music or audio generation.", None),
    ("discovery", "What are some strong open models for text-to-video generation?", None),
    ("discovery", "Which models on Hugging Face are popular for named entity recognition (NER)?", None),
    ("discovery", "Recommend good open models for abstractive text summarization.", None),
    ("discovery", "What models can I use for English-to-French and French-to-English machine translation?", None),
    ("discovery", "Which open models are popular for German language generation or chat?", None),
    ("discovery", "Find popular open LLMs that are specifically trained or tuned for Japanese.", None),
    ("discovery", "What are good open models on the Hub for Arabic NLP or chat?", None),
    ("discovery", "Which open models work well for Spanish speech recognition?", None),
    ("discovery", "Recommend some open models for speaker diarization or audio classification.", None),
    ("discovery", "What models on the Hub are good for zero-shot text classification?", None),
    ("discovery", "Which models can I use for zero-shot image classification (CLIP-style)?", None),
    ("discovery", "Recommend open models for image super-resolution or upscaling.", None),
    ("discovery", "What open models on the Hub are built for table question answering?", None),
    ("discovery", "Find popular biomedical or clinical language models on Hugging Face.", None),
    ("discovery", "Which protein or biology foundation models are popular on Hugging Face?", None),
    ("discovery", "Recommend small vision backbones suitable for edge image classification.", None),
    ("discovery", "What are some strong 7B to 9B open instruction-tuned LLMs with an Apache-2.0 or MIT license?", None),
    ("discovery", "Find popular open models specifically designed for function calling and tool use.", None),
    ("discovery", "Which open embedding models support long documents (8K tokens or more)?", None),
    ("discovery", "Recommend open models for document visual question answering.", None),
    ("discovery", "What models on Hugging Face are popular for sentiment analysis?", None),
    ("discovery", "Which open models can generate 3D assets or meshes from images or text?", None),
    ("discovery", "Are there quantized GGUF versions of Qwen/Qwen2.5-Coder-7B-Instruct on the Hub?", None),
    ("discovery", "Are there FP8 or AWQ quantized builds of meta-llama/Llama-3.3-70B-Instruct on the Hub?", None),
    ("discovery", "What are some popular open models for image editing guided by text instructions?", None),
    ("discovery", "Which open models on the Hub are good for toxicity detection or content moderation?", None),
    ("discovery", "Recommend a few open models for voice activity detection or sound event classification.", None),
    # ------------------------------------------------------------------
    # 3. Libraries & frameworks (24)
    # ------------------------------------------------------------------
    (
        "libraries",
        "What are the most downloaded models for the peft library on the Hub?",
        ids("models", "list", "--filter", "peft", "--sort", "downloads", *TOP),
    ),
    (
        "libraries",
        "What are the most downloaded ONNX models on Hugging Face?",
        ids("models", "list", "--filter", "onnx", "--sort", "downloads", *TOP),
    ),
    (
        "libraries",
        "Which transformers models for fill-mask are the most downloaded?",
        ids("models", "list", "--filter", "transformers", "--pipeline-tag", "fill-mask", "--sort", "downloads", *TOP),
    ),
    (
        "libraries",
        "Which transformers models for token-classification are the most downloaded?",
        ids(
            "models",
            "list",
            "--filter",
            "transformers",
            "--pipeline-tag",
            "token-classification",
            "--sort",
            "downloads",
            *TOP,
        ),
    ),
    (
        "libraries",
        "What are the most liked GGUF models on Hugging Face?",
        ids("models", "list", "--filter", "gguf", "--sort", "likes", *TOP),
    ),
    (
        "libraries",
        "Which safetensors models are the most liked on the Hub?",
        ids("models", "list", "--filter", "safetensors", "--sort", "likes", *TOP),
    ),
    (
        "libraries",
        "What are the most downloaded spaCy models on Hugging Face?",
        ids("models", "list", "--filter", "spacy", "--sort", "downloads", *TOP),
    ),
    (
        "libraries",
        "Which SpeechBrain models are the most downloaded on the Hub?",
        ids("models", "list", "--filter", "speechbrain", "--sort", "downloads", *TOP),
    ),
    (
        "libraries",
        "What are the most liked Transformers.js-compatible models on Hugging Face?",
        ids("models", "list", "--filter", "transformers.js", "--sort", "likes", *TOP),
    ),
    (
        "libraries",
        "Which open_clip models have the most downloads on Hugging Face?",
        ids("models", "list", "--filter", "open_clip", "--sort", "downloads", *TOP),
    ),
    (
        "libraries",
        "What are the most downloaded SetFit models on Hugging Face?",
        ids("models", "list", "--filter", "setfit", "--sort", "downloads", *TOP),
    ),
    (
        "libraries",
        "Which Keras models are the most downloaded on Hugging Face?",
        ids("models", "list", "--filter", "keras", "--sort", "downloads", *TOP),
    ),
    ("libraries", "Which popular models on the Hub are tagged for vLLM serving?", None),
    ("libraries", "Find some popular Flair NLP models hosted on Hugging Face.", None),
    ("libraries", "Are there popular ControlNet adapters for SDXL in diffusers on the Hub?", None),
    ("libraries", "What popular LoRA or PEFT adapters exist for Qwen2.5 models on Hugging Face?", None),
    ("libraries", "Which ColPali or late-interaction document retrieval models are hosted on the Hub?", None),
    ("libraries", "Find popular Ultralytics vision models hosted on Hugging Face.", None),
    ("libraries", "Which JAX/Flax models are popular on Hugging Face?", None),
    ("libraries", "What models on the Hub work with pyannote-audio?", None),
    ("libraries", "Which popular models on Hugging Face are packaged for llama.cpp?", None),
    ("libraries", "Are there popular ESPnet speech models hosted on Hugging Face?", None),
    ("libraries", "Which fastText language identification or embedding models are on the Hub?", None),
    ("libraries", "Find popular scikit-learn (sklearn) models on Hugging Face.", None),
    # ------------------------------------------------------------------
    # 4. Model details (24) — completely disjoint repos from seed & facts-v2 eval
    # ------------------------------------------------------------------
    (
        "model-details",
        "What is the license of Qwen/Qwen2.5-Coder-7B-Instruct?",
        fact("models", "info", "Qwen/Qwen2.5-Coder-7B-Instruct", path="card_data.license"),
    ),
    (
        "model-details",
        "Which library is BAAI/bge-m3 meant to be used with?",
        fact("models", "info", "BAAI/bge-m3", path="library_name"),
    ),
    (
        "model-details",
        "What pipeline task is sentence-transformers/all-MiniLM-L6-v2 configured for?",
        fact("models", "info", "sentence-transformers/all-MiniLM-L6-v2", path="pipeline_tag"),
    ),
    (
        "model-details",
        "What is the license of deepseek-ai/DeepSeek-R1-Distill-Qwen-7B?",
        fact("models", "info", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", path="card_data.license"),
    ),
    (
        "model-details",
        "What library is stabilityai/stable-diffusion-xl-base-1.0 designed for?",
        fact("models", "info", "stabilityai/stable-diffusion-xl-base-1.0", path="library_name"),
    ),
    (
        "model-details",
        "What task is facebook/bart-large-cnn built for?",
        fact("models", "info", "facebook/bart-large-cnn", path="pipeline_tag"),
    ),
    (
        "model-details",
        "What is the license of HuggingFaceTB/SmolLM2-1.7B-Instruct?",  # pragma: allowlist secret
        fact(
            "models",
            "info",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # pragma: allowlist secret
            path="card_data.license",
        ),
    ),
    (
        "model-details",
        "What pipeline tag does openai/clip-vit-large-patch14 have?",
        fact("models", "info", "openai/clip-vit-large-patch14", path="pipeline_tag"),
    ),
    ("model-details", "Is black-forest-labs/FLUX.1-dev a gated model, and what is its license?", None),
    ("model-details", "Is meta-llama/Llama-3.2-1B-Instruct gated, and how many parameters does it have?", None),
    ("model-details", "Is pyannote/speaker-diarization-3.1 gated? What library does it use?", None),
    ("model-details", "Which weight files are in google-bert/bert-base-uncased, and does it ship safetensors?", None),
    ("model-details", "How many parameters does microsoft/Phi-3.5-mini-instruct have, and what is its license?", None),
    ("model-details", "What architecture and parameter count does FacebookAI/roberta-large have?", None),
    ("model-details", "Does black-forest-labs/FLUX.1-schnell require gating, and what license does it use?", None),
    ("model-details", "What is the base model of deepseek-ai/DeepSeek-R1-Distill-Qwen-7B?", None),
    (
        "model-details",
        "What files are inside sentence-transformers/all-MiniLM-L6-v2, including ONNX or OpenVINO exports?",
        None,
    ),
    ("model-details", "What languages and license does BAAI/bge-reranker-v2-m3 list on its model card?", None),
    ("model-details", "What task, library, and parameter size does facebook/dinov2-large have?", None),
    ("model-details", "What is the license and parameter count of allenai/OLMo-2-1124-7B-Instruct?", None),
    ("model-details", "What files are stored in Helsinki-NLP/opus-mt-en-fr and which library loads it?", None),
    ("model-details", "What is the license, gating status, and task of google/siglip-so400m-patch14-384?", None),
    ("model-details", "What parameter count and license does ibm-granite/granite-3.1-8b-instruct have?", None),
    (
        "model-details",
        "How many safetensors shards does Qwen/Qwen2.5-VL-7B-Instruct have, and what is its total parameter count?",
        None,
    ),
    # ------------------------------------------------------------------
    # 5. Datasets (28)
    # ------------------------------------------------------------------
    (
        "datasets",
        "What is the license of the HuggingFaceH4/ultrachat_200k dataset?",  # pragma: allowlist secret
        fact(
            "datasets",
            "info",
            "HuggingFaceH4/ultrachat_200k",  # pragma: allowlist secret
            path="card_data.license",
        ),
    ),
    (
        "datasets",
        "What is the license of open-r1/OpenR1-Math-220k?",
        fact("datasets", "info", "open-r1/OpenR1-Math-220k", path="card_data.license"),
    ),
    (
        "datasets",
        "What are the most liked datasets from the HuggingFaceFW organization?",  # pragma: allowlist secret
        ids(
            "datasets",
            "list",
            "--author",
            "HuggingFaceFW",  # pragma: allowlist secret
            "--sort",
            "likes",
            "--limit",
            "6",
            k=3,
        ),
    ),
    ("datasets", "What are some good datasets on the Hub for mathematical reasoning?", None),
    ("datasets", "Recommend popular datasets for RLHF or DPO preference tuning.", None),
    ("datasets", "Find good multimodal / visual question answering datasets on Hugging Face.", None),
    ("datasets", "What datasets on the Hub are useful for training text-to-speech or speech synthesis models?", None),
    ("datasets", "Find popular parallel corpora on Hugging Face for machine translation.", None),
    ("datasets", "Which datasets on the Hub are widely used for named entity recognition?", None),
    ("datasets", "What are popular datasets for news or document summarization on Hugging Face?", None),
    ("datasets", "Find datasets on the Hub for function calling or tool-use agent training.", None),
    ("datasets", "Which biomedical or medical QA datasets are popular on Hugging Face?", None),
    ("datasets", "Find some French text datasets hosted on Hugging Face.", None),
    ("datasets", "Find some German text or instruction datasets on the Hub.", None),
    ("datasets", "What popular Japanese text or chat datasets exist on Hugging Face?", None),
    ("datasets", "Find Spanish NLP or speech datasets on Hugging Face.", None),
    ("datasets", "What popular image captioning datasets are available on the Hub?", None),
    ("datasets", "What configs, splits, and columns does the stanfordnlp/imdb dataset have?", None),
    ("datasets", "Inspect the openai/gsm8k dataset: what configs, splits, and fields does it contain?", None),
    ("datasets", "What splits and approximate size does Salesforce/wikitext have?", None),
    ("datasets", "Is lmsys/lmsys-chat-1m gated, and what is its license?", None),
    ("datasets", "What files and parquet splits are in tatsu-lab/alpaca?", None),
    ("datasets", "Which datasets on Hugging Face are popular for tabular machine learning?", None),
    ("datasets", "Find popular OCR or document layout datasets on the Hub.", None),
    ("datasets", "What datasets are popular on Hugging Face for embedding or semantic retrieval training?", None),
    ("datasets", "Find popular synthetic reasoning trace datasets on Hugging Face.", None),
    ("datasets", "What datasets on the Hub focus on AI safety, red-teaming, or toxicity evaluation?", None),
    ("datasets", "Which video understanding or video captioning datasets are popular on Hugging Face?", None),
    # ------------------------------------------------------------------
    # 6. Papers (26) — disjoint topics and arXiv IDs from seed
    # ------------------------------------------------------------------
    ("papers", "What are some good papers on Hugging Face about LoRA or parameter-efficient fine-tuning?", None),
    ("papers", "Find papers on Hugging Face about FlashAttention or efficient attention kernels.", None),
    ("papers", "What papers on Hugging Face discuss Mixture-of-Experts (MoE) architectures?", None),
    ("papers", "Find papers on Hugging Face about LLM quantization methods like AWQ, GPTQ, or FP4.", None),
    ("papers", "What are some good papers on Hugging Face about retrieval-augmented generation (RAG)?", None),
    ("papers", "Find papers on Hugging Face about extending LLM context length (long-context modeling).", None),
    ("papers", "What papers on Hugging Face cover Diffusion Transformers (DiT) or flow matching?", None),
    ("papers", "Find papers on Hugging Face about Direct Preference Optimization (DPO) and alignment.", None),
    ("papers", "What papers on Hugging Face explore synthetic data generation for training LLMs?", None),
    ("papers", "Find papers on Hugging Face about state space models such as Mamba.", None),
    ("papers", "What papers on Hugging Face discuss test-time compute scaling or reasoning search?", None),
    ("papers", "Find papers on Hugging Face about mechanistic interpretability or sparse autoencoders.", None),
    ("papers", "What papers on Hugging Face cover multimodal vision-language models?", None),
    ("papers", "Find papers on Hugging Face about model merging or model soups.", None),
    ("papers", "What papers on Hugging Face focus on tool use and web or coding agents?", None),
    ("papers", "Find papers on Hugging Face about video generation models.", None),
    ("papers", "What papers on Hugging Face cover world models for video or simulation?", None),
    ("papers", "Find papers on Hugging Face about neural pruning or structured compression.", None),
    ("papers", "What papers on Hugging Face study GUI or computer-use agents?", None),
    ("papers", "Find papers on Hugging Face about speech-to-speech or audio language models.", None),
    ("papers", "What is the paper 2305.14314 about, and who are its authors?", None),
    ("papers", "Summarize what arXiv paper 2312.00752 is about and list its main authors.", None),
    ("papers", "What is arXiv paper 2205.14135 about, and what models or repos on the Hub link to it?", None),
    ("papers", "What is paper 2307.09288 about, and who wrote it?", None),
    ("papers", "What is arXiv paper 2302.13971 about, and who are its lead authors?", None),
    ("papers", "Find papers on Hugging Face about benchmark contamination or LLM evaluation.", None),
    # ------------------------------------------------------------------
    # 7. Spaces (22) — disjoint from background removal, most liked, TTS, LLM leaderboard, ZeroGPU image gen
    # ------------------------------------------------------------------
    ("spaces", "Find a Hugging Face Space where I can transcribe audio with Whisper or another ASR model.", None),
    ("spaces", "Are there Spaces on Hugging Face for generating videos from text or images?", None),
    ("spaces", "Find Spaces on the Hub for OCR or parsing PDFs into Markdown.", None),
    ("spaces", "Are there Spaces on Hugging Face for generating 3D models from a single image?", None),
    ("spaces", "Find a Space on Hugging Face where I can try voice cloning.", None),
    ("spaces", "Which Spaces on Hugging Face let you generate music or sound effects?", None),
    ("spaces", "Find Spaces on the Hub for image upscaling or restoration.", None),
    ("spaces", "Are there Spaces on Hugging Face for monocular depth estimation?", None),
    ("spaces", "Find a Space on Hugging Face for translating text between languages.", None),
    ("spaces", "Which Spaces on the Hub let users chat with open vision-language models?", None),
    ("spaces", "Are there Spaces on Hugging Face that showcase coding assistants or live code generation?", None),
    ("spaces", "Find Spaces on the Hub that use ZeroGPU for video or 3D generation.", None),
    ("spaces", "Find popular Spaces hosted by the HuggingFaceH4 organization.", None),  # pragma: allowlist secret
    ("spaces", "Which Spaces has the stabilityai organization published on Hugging Face?", None),
    ("spaces", "What Spaces has black-forest-labs published on Hugging Face?", None),
    ("spaces", "Find Spaces on Hugging Face built with the Docker SDK that are popular.", None),
    ("spaces", "Are there Spaces on the Hub for evaluating or comparing embedding models (like MTEB)?", None),
    ("spaces", "Find Spaces on Hugging Face for face restoration or portrait editing.", None),
    ("spaces", "Which Spaces on the Hub demonstrate document question answering or visual retrieval?", None),
    ("spaces", "Are there Spaces on Hugging Face for trying out object segmentation interactively?", None),
    ("spaces", "Find Spaces on the Hub dedicated to speech-to-speech or real-time voice chat.", None),
    (
        "spaces",
        "Which Spaces on Hugging Face were created most recently and already have traction?",
        None,
    ),
    # ------------------------------------------------------------------
    # 8. Organizations & collections (24)
    # ------------------------------------------------------------------
    (
        "orgs",
        "Which models from meta-llama are the most downloaded on Hugging Face?",
        ids("models", "list", "--author", "meta-llama", "--sort", "downloads", "--limit", "6", k=3),
    ),
    (
        "orgs",
        "Which models from google are the most downloaded on the Hub?",
        ids("models", "list", "--author", "google", "--sort", "downloads", "--limit", "6", k=3),
    ),
    (
        "orgs",
        "Which models from BAAI are the most downloaded on Hugging Face?",
        ids("models", "list", "--author", "BAAI", "--sort", "downloads", "--limit", "6", k=3),
    ),
    (
        "orgs",
        "Which models from sentence-transformers are the most downloaded?",
        ids("models", "list", "--author", "sentence-transformers", "--sort", "downloads", "--limit", "6", k=3),
    ),
    (
        "orgs",
        "Which models from openai are the most downloaded on Hugging Face?",
        ids("models", "list", "--author", "openai", "--sort", "downloads", "--limit", "6", k=3),
    ),
    (
        "orgs",
        "Which models from deepseek-ai have the most likes on Hugging Face?",
        ids("models", "list", "--author", "deepseek-ai", "--sort", "likes", "--limit", "6", k=3),
    ),
    (
        "orgs",
        "Which models from microsoft have the most likes on Hugging Face?",
        ids("models", "list", "--author", "microsoft", "--sort", "likes", "--limit", "6", k=3),
    ),
    (
        "orgs",
        "Which models from stabilityai are the most liked on the Hub?",
        ids("models", "list", "--author", "stabilityai", "--sort", "likes", "--limit", "6", k=3),
    ),
    (
        "orgs",
        "Which models from black-forest-labs are the most liked on the Hub?",
        ids("models", "list", "--author", "black-forest-labs", "--sort", "likes", "--limit", "6", k=3),
    ),
    ("orgs", "What collections has HuggingFaceTB published on Hugging Face?", None),  # pragma: allowlist secret
    ("orgs", "What collections has BAAI published on Hugging Face?", None),
    ("orgs", "What collections has allenai published on Hugging Face?", None),
    ("orgs", "What collections has google published on Hugging Face?", None),
    ("orgs", "What collections has meta-llama published on Hugging Face?", None),
    ("orgs", "What collections has deepseek-ai published on Hugging Face?", None),
    ("orgs", "What datasets has allenai published on Hugging Face?", None),
    ("orgs", "What datasets has facebook published on Hugging Face?", None),
    ("orgs", "What datasets has HuggingFaceTB published on Hugging Face?", None),  # pragma: allowlist secret
    ("orgs", "What models has apple published on Hugging Face?", None),
    ("orgs", "What models has LiquidAI published on Hugging Face?", None),
    ("orgs", "What models has CohereLabs published on Hugging Face?", None),
    ("orgs", "What models has EleutherAI published on Hugging Face?", None),
    ("orgs", "What models has ibm-granite published on Hugging Face?", None),
    ("orgs", "What models has kyutai published on Hugging Face?", None),
    # ------------------------------------------------------------------
    # 9. Comparison & synthesis (22) — teach concise multi-lookup synthesis
    # ------------------------------------------------------------------
    (
        "comparison",
        "Compare black-forest-labs/FLUX.1-dev and black-forest-labs/FLUX.1-schnell "
        "in terms of gating, license, and popularity.",
        None,
    ),
    (
        "comparison",
        "Compare BAAI/bge-m3 and intfloat/multilingual-e5-large: which is more downloaded, "
        "what libraries do they use, and what are their licenses?",
        None,
    ),
    (
        "comparison",
        "Compare Qwen/Qwen2.5-7B-Instruct and mistralai/Mistral-7B-Instruct-v0.3 "
        "in terms of license, gating, parameter count, and downloads.",
        None,
    ),
    (
        "comparison",
        "Compare HuggingFaceTB/SmolLM2-1.7B-Instruct and meta-llama/Llama-3.2-1B-Instruct "  # pragma: allowlist secret
        "for on-device deployment (size, gating, license, downloads).",
        None,
    ),
    (
        "comparison",
        "Compare Qwen/Qwen2.5-7B-Instruct and Qwen/Qwen2.5-14B-Instruct in parameter count, license, and downloads.",
        None,
    ),
    (
        "comparison",
        "Compare microsoft/deberta-v3-large and FacebookAI/roberta-large "
        "by downloads, likes, license, and pipeline tag.",
        None,
    ),
    (
        "comparison",
        "Compare google/flan-t5-base and facebook/bart-large-cnn: "
        "what tasks, licenses, and parameter counts do they have?",
        None,
    ),
    (
        "comparison",
        "Compare jinaai/jina-embeddings-v3 and sentence-transformers/all-MiniLM-L6-v2 "
        "on downloads, likes, license, and library.",
        None,
    ),
    (
        "comparison",
        "Compare google/gemma-2-9b-it and allenai/OLMo-2-1124-7B-Instruct in terms of gating, license, and popularity.",
        None,
    ),
    (
        "comparison",
        "Compare microsoft/Florence-2-large and Qwen/Qwen2.5-VL-7B-Instruct "
        "on task, license, downloads, and parameter size.",
        None,
    ),
    (
        "comparison",
        "Compare deepseek-ai/DeepSeek-V3 and meta-llama/Llama-3.3-70B-Instruct in terms of gating, license, and likes.",
        None,
    ),
    (
        "comparison",
        "Compare HuggingFaceH4/ultrachat_200k and open-r1/OpenR1-Math-220k: "  # pragma: allowlist secret
        "what are their domains, licenses, and popularity?",
        None,
    ),
    (
        "comparison",
        "Compare the top trending text-to-image models vs. the most all-time liked text-to-image models on the Hub.",
        None,
    ),
    (
        "comparison",
        "Compare the most downloaded diffusers models vs. the most downloaded transformers text-generation models.",
        None,
    ),
    (
        "comparison",
        "I want an open 7B-8B model for coding with an Apache-2.0 license. "
        "Compare two or three top options from the Hub.",
        None,
    ),
    (
        "comparison",
        "I need a permissive-license reranker for English and multilingual search. "
        "Compare two or three options on the Hub.",
        None,
    ),
    (
        "comparison",
        "Compare two popular open vision backbones (DINOv2 vs. SigLIP) hosted on Hugging Face "
        "by size, task, and license.",
        None,
    ),
    (
        "comparison",
        "Compare unsloth and bartowski as GGUF quantization providers on Hugging Face: "
        "what are their top trending or most downloaded repos?",
        None,
    ),
    (
        "comparison",
        "Compare the top 3 most downloaded datasets for math vs. "
        "the top 3 most downloaded datasets for summarization on the Hub.",
        None,
    ),
    (
        "comparison",
        "Compare two popular speech diarization or ASR pipelines on the Hub "
        "in terms of gating, license, and downloads.",
        None,
    ),
    (
        "comparison",
        "Which is more popular on the Hub right now by downloads and likes: "
        "Qwen/Qwen2.5-Coder-7B-Instruct or deepseek-ai/DeepSeek-R1-Distill-Qwen-7B?",
        None,
    ),
    (
        "comparison",
        "Compare google-bert/bert-base-uncased and FacebookAI/roberta-large "
        "in terms of downloads, likes, and parameter count.",
        None,
    ),
    # ------------------------------------------------------------------
    # 10. Multi-criteria workflow / Hub navigation (12)
    # ------------------------------------------------------------------
    (
        "workflow",
        "Find 3 popular Apache-2.0 licensed text-generation models on the Hub that ship safetensors weights.",
        None,
    ),
    (
        "workflow",
        "Find popular MIT-licensed vision-language models on Hugging Face "
        "and note their parameter counts if available.",
        None,
    ),
    (
        "workflow",
        "Check if Qwen/Qwen2.5-7B-Instruct has popular community GGUF quantizations and name a few of them.",
        None,
    ),
    (
        "workflow",
        "Find popular collections on Hugging Face curated for open reasoning models.",
        None,
    ),
    (
        "workflow",
        "Find popular collections on Hugging Face focused on multimodal or vision-language models.",
        None,
    ),
    (
        "workflow",
        "What are the most popular Hugging Face datasets tagged for code or programming "
        "that carry a permissive license?",
        None,
    ),
    (
        "workflow",
        "Find 3 popular text-to-image models on the Hub that are NOT gated and explain their licenses.",
        None,
    ),
    (
        "workflow",
        "Which organizations publish the most popular open medical or biology models on Hugging Face? "
        "Give specific repo examples.",
        None,
    ),
    (
        "workflow",
        "Find popular open models for document OCR that work with transformers, "
        "and check whether they require trust_remote_code or custom code.",
        None,
    ),
    (
        "workflow",
        "Find 3 popular audio-classification models on the Hub and summarize what kind of audio each one classifies.",
        None,
    ),
    (
        "workflow",
        "Find popular open-source text-embedding models on the Hub that have ONNX weights available.",
        None,
    ),
    (
        "workflow",
        "Find a popular open-source speech-to-text model for multilingual transcription (besides Whisper) "
        "and summarize its key details.",
        None,
    ),
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9._/-]*", text.lower())) - {
        "the",
        "a",
        "an",
        "on",
        "of",
        "for",
        "in",
        "to",
        "and",
        "or",
        "is",
        "are",
        "what",
        "which",
        "with",
        "that",
        "can",
        "i",
        "do",
        "you",
        "me",
        "some",
        "most",
        "hugging",
        "face",
        "hub",
    }


def _repo_and_arxiv_ids(text: str) -> set[str]:
    repos = {m.lower() for m in re.findall(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\b", text)}
    arxiv = set(re.findall(r"\b\d{4}\.\d{4,5}\b", text))
    return repos | arxiv


def check_disjoint() -> None:
    """Verify training questions have unique IDs and zero overlap with the 51 `questions.py` eval items."""
    eval_texts = {q for _, q, _ in EVAL_QUESTIONS}
    eval_qids = {hashlib.sha256(q.encode()).hexdigest()[:12] for q in eval_texts}
    eval_ids = set().union(*(_repo_and_arxiv_ids(q) for q in eval_texts))
    eval_tok = [(q, _tokens(q)) for q in eval_texts]
    eval_refs = {json.dumps(ref, sort_keys=True) for _, _, ref in EVAL_QUESTIONS if ref}

    seen_qids: set[str] = set()
    for _, q, ref in TRAIN_QUESTIONS:
        qid = hashlib.sha256(q.encode()).hexdigest()[:12]
        assert qid not in seen_qids, f"duplicate train question: {q}"
        seen_qids.add(qid)
        assert qid not in eval_qids and q not in eval_texts, f"exact overlap with eval: {q}"
        if ref:
            assert json.dumps(ref, sort_keys=True) not in eval_refs, f"reference overlap with eval: {q}"
        shared_ids = _repo_and_arxiv_ids(q) & eval_ids
        assert not shared_ids, f"train question shares eval repo/arxiv id {shared_ids}: {q}"
        t = _tokens(q)
        for eq, et in eval_tok:
            union = len(t | et)
            jacc = len(t & et) / union if union else 0.0
            assert jacc < 0.72, f"too similar (Jaccard={jacc:.2f}) to eval {eq!r}: {q!r}"


def rows() -> list[dict[str, Any]]:
    check_disjoint()
    out = []
    for category, question, reference in TRAIN_QUESTIONS:
        qid = hashlib.sha256(question.encode()).hexdigest()[:12]
        out.append(
            {
                "question_id": qid,
                "category": category,
                "question": question,
                "reference": json.dumps(reference) if reference else "",
            }
        )
    return out


CARD = """---
license: apache-2.0
language: [en]
task_categories: [question-answering]
tags: [agent-distill, agents, hf-cli]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Hugging Face Hub open-ended training questions

240 open-ended training questions about the Hugging Face Hub for agent distillation
(`examples/hf-hub-qa/open_ended.py`), strictly disjoint from the 51 evaluation questions in
`agent-distillation-hf/hf-hub-questions` (`questions.py`). `reference`, when set, is a canonical `hf` CLI command
(JSON) checked live by `judged_rubrics.py`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Validate count and disjointness from eval questions")
    parser.add_argument("--push", help="Publish the training questions to this dataset repo (public; split `train`)")
    parser.add_argument("--collection", default="", help="Collection the dataset joins when it is created")
    args = parser.parse_args()
    data = rows()
    if args.check and not args.push:
        from collections import Counter

        cats = Counter(r["category"] for r in data)
        refs = sum(1 for r in data if r["reference"])
        print(f"OK: {len(data)} disjoint training questions ({refs} with live references); categories={dict(cats)}")
        return
    if not args.push:
        print(json.dumps(data, indent=2))
        return
    import io

    from huggingface_hub import CommitOperationAdd, HfApi  # pragma: allowlist secret

    api = HfApi()
    created = not api.repo_exists(args.push, repo_type="dataset")
    api.create_repo(args.push, repo_type="dataset", private=False, exist_ok=True)
    lines = "".join(json.dumps(r) + "\n" for r in data).encode()
    api.create_commit(
        args.push,
        repo_type="dataset",
        commit_message=f"{len(data)} open-ended training questions (disjoint from agent-distillation-hf/hf-hub-questions)",  # noqa: E501
        operations=[
            CommitOperationAdd("data/train.jsonl", io.BytesIO(lines)),
            CommitOperationAdd("README.md", io.BytesIO(CARD.encode())),
        ],
    )
    if created and args.collection:
        api.add_collection_item(
            args.collection,
            item_id=args.push,
            item_type="dataset",
            exists_ok=True,
            note="hf-hub-qa: 240 open-ended training questions for an hf-CLI agent (disjoint from seed eval)",
        )
    print(f"pushed {len(data)} questions to {args.push}")


if __name__ == "__main__":
    main()
