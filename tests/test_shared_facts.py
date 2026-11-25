"""Shared fact checking (rubrics.facts), grounding (rubrics.grounding) and live Hub facts (hub_facts); no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_distill import hub_facts
from agent_distill.rubrics import facts, grounding

LICENSES = ["apache-2.0", "mit", "cc-by-4.0", "llama3.1", "llama3", "openrail++", "other"]


@pytest.mark.parametrize(("gold", "answer", "ok"), [
    ("apache-2.0", "It is licensed under **Apache 2.0**.", True),
    ("apache-2.0", "The license is `apache-2.0`.", True),
    ("apache-2.0", "Apache License, Version 2.0", True),
    ("apache-2.0", "Apache 2.0 or MIT, I am not sure.", False),  # hedged
    ("mit", "MIT License.", True),
    ("mit", "You can submit a request.", False),
    ("llama3.1", "the Llama 3.1 Community License", True),
    ("cc-by-4.0", "CC BY 4.0", True),
    ("text-generation", "Its pipeline tag is text2text-generation.", False),
    ("text2text-generation", "Its pipeline tag is text2text-generation.", True),
])  # fmt: skip
def test_category(gold, answer, ok):
    hedge = [*LICENSES, "text-generation", "text2text-generation"]
    assert facts.check("category", gold, answer, hedge=hedge) is ok


def test_repo_number_title_author_arxiv():
    assert facts.check("repo", "Qwen/Qwen3.5-4B-Base", "Its base model is `Qwen/Qwen3.5-4B-Base`.")
    assert not facts.check("repo", "Qwen/Qwen3.5-4B", "Its base model is Qwen/Qwen3.5-4B-Base.")
    assert facts.check("number", 4659865088, "about 4.66B parameters")
    assert facts.check("number", 4659865088, "4,659,865,088 parameters (BF16)")
    assert not facts.check("number", 4659865088, "4B parameters")
    assert facts.check("number", 494032768, "494M")
    title = "Knowledge Distillation of Russian Language Models with Reduction of Vocabulary"
    assert facts.check("title", title, f'The paper is "{title.lower()}".')
    assert not facts.check("title", title, "It is about distilling Russian models.")
    authors = ["Alina Kolesnikova", "Yuri Kuratov", "Vasily Konovalov"]
    assert facts.check("first_author", "Alina Kolesnikova", "The first author is Alina Kolesnikova.", authors=authors)
    assert not facts.check("first_author", "Alina Kolesnikova", "Yuri Kuratov and Alina Kolesnikova", authors=authors)
    assert facts.check("first_author", "José Núñez", "jose nunez wrote it", authors=["José Núñez"])
    assert facts.check("arxiv", ["2205.02340"], "It links arXiv 2205.02340v2.")
    assert not facts.check("arxiv", ["2205.02340"], "Either 2205.02340 or 2301.00001.")
    assert not facts.check("category", "mit", None)
    with pytest.raises(ValueError):
        facts.check("colour", "red", "red")


@pytest.mark.parametrize(("kind", "gold"), [
    ("category", "apache-2.0"), ("repo", "Qwen/Qwen3.5-4B-Base"), ("number", 4659865088),
    ("title", "Knowledge Distillation of Russian Language Models"), ("first_author", "Alina Kolesnikova"),
    ("arxiv", ["2205.02340"]),
])  # fmt: skip
def test_oracle_passes_and_no_op_fails(kind, gold):
    assert facts.check(kind, gold, facts.render(kind, gold))
    assert not facts.check(kind, gold, "I could not find this on the Hub.")


def test_cited_ids(monkeypatch):
    monkeypatch.setattr(grounding, "exists_on_hub", lambda repo_id: repo_id == "BAAI/bge-m3")
    text = ("Try `Qwen/Qwen3.5-9B` or [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), see paper "
            "2205.02340. Use PEFT/LoRA in F16/BF16, or BAAI/bge-m3 for retrieval, and/or more.")  # fmt: skip
    assert grounding.cited_ids(text) == {"Qwen/Qwen3.5-9B", "HuggingFaceFW/fineweb", "2205.02340", "BAAI/bge-m3"}
    assert grounding.cited_ids("An invented `Made/Up-Model-7B`.") == {"Made/Up-Model-7B"}
    org_pages = (
        "See [`Qwen`](https://huggingface.co/Qwen/models) and [its datasets](https://huggingface.co/Qwen/datasets)."
    )
    assert grounding.cited_ids(org_pages) == set()
    shorthand = "Try `NeoQuasar/Kronos-mini/small`, `Kronos-Tokenizer-base/2k` or `mmproj-Qwen3.5-9B-bf16/f16.gguf`."
    assert grounding.cited_ids(shorthand) == set()
    assert grounding.cited_ids("BAAI/bge-m3 in prose", check_hub=False) == set()


def test_grounded_score():
    tools = 'hub_repo_details {"repo_ids": ["acme/Model-1"]}\nlicense: apache-2.0, see 2205.02340'
    assert grounding.grounded_score("No ids here.", tools) is None
    assert grounding.grounded_score("`acme/Model-1`, paper 2205.02340", tools) == (1.0, {"ids": 2.0, "unseen": 0.0})
    assert grounding.grounded_score("`acme/Model-1` or `acme/Other-2`", tools) == (0.5, {"ids": 2.0, "unseen": 1.0})


class FakeHub:
    def model_info(self, ident, expand=None, token=None):
        assert token is False  # anonymous, as the agent sees the Hub
        cards = {"org/ft": {"license": "mit", "base_model": ["org/base"]}, "org/base": {"license": "apache-2.0"}}
        tags = ["arxiv:2205.02340", "license:mit"]
        return SimpleNamespace(card_data=cards[ident], pipeline_tag="text-generation", library_name="transformers",
                               safetensors=SimpleNamespace(total=7), tags=tags)  # fmt: skip

    def dataset_info(self, ident, expand=None, token=None):
        return SimpleNamespace(card_data={"license": "cc-by-4.0"})

    def space_info(self, ident, expand=None, token=None):
        return SimpleNamespace(sdk="gradio")

    def paper_info(self, ident):
        return SimpleNamespace(title="A  Title\n Split", authors=[SimpleNamespace(name="Ada Lovelace")])


@pytest.mark.parametrize(("spec", "value"), [
    ({"type": "model", "id": "org/ft", "field": "license"}, "mit"),
    ({"type": "model", "id": "org/ft", "field": "base"}, "org/base"),
    ({"type": "model", "id": "org/ft", "field": "params"}, 7),
    ({"type": "model", "id": "org/ft", "field": "arxiv"}, ["2205.02340"]),
    ({"type": "base", "id": "org/ft", "field": "license"}, "apache-2.0"),
    ({"type": "dataset", "id": "d/x", "field": "license"}, "cc-by-4.0"),
    ({"type": "space", "id": "s/x", "field": "sdk"}, "gradio"),
    ({"type": "paper", "id": "2205.02340", "field": "title"}, "A Title Split"),
    ({"type": "paper", "id": "2205.02340", "field": "first_author"}, "Ada Lovelace"),
])  # fmt: skip
def test_live_value(spec, value):
    assert hub_facts.live_value(spec, hub=FakeHub()) == value


def test_single_and_card():
    assert hub_facts.single(["a/b"]) == "a/b" and hub_facts.single(["a/b", "c/d"]) is None
    assert hub_facts.single("a/b") == "a/b" and hub_facts.single("") is None
    assert hub_facts.card(SimpleNamespace(card_data=SimpleNamespace(to_dict=lambda: {"license": "mit"}))) == {
        "license": "mit"
    }
