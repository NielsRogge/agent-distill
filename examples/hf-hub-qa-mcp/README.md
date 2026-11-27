# hf-hub-qa-mcp: Hugging Face Hub Q&A via the HF MCP server

Distill **GLM-5.3** into **Qwen3.5-4B** inside `pi@1.6.0` using only three read-only tools from the anonymous [Hugging Face MCP server](https://huggingface.co/docs/hub/en/agents-mcp) (`hub_repo_search`, `hub_repo_details`, `hf_fs`). <!-- pragma: allowlist secret -->

This is the MCP twin of [`hf-hub-qa`](../hf-hub-qa/README.md), which answers the same questions through the `hf` CLI. Everything except the tool surface is shared so the two interfaces can be compared directly.

## Headline results

Evaluated on all **377 test questions** of `nielsr/hf-hub-facts-v2` in `pi@1.6.0` against HF MCP server `0.4.39`:

| Model | Thinking | Passed | MCP calls / ep | Failed MCP calls |
| --- | :---: | ---: | ---: | ---: |
| Qwen3.5-4B (base) | off | 272/377 (72.1%) | 6.13 | 45.0% |
| + Action-only SFT (`run.toml`, 599 traces) | off | 307/377 (81.4%) | 3.38 | 11.9% |
| + SFT + GRPO (`run.toml` phase 2, 100 steps) | off | **362/377 (96.0%)** | **1.75** | **1.2%** |
| Qwen3.5-4B (base) | on | 336/377 (89.1%) | 6.47 | — |
| **+ Cross-format SFT, recipe (a) (`run-think.toml`)** | **on** | **370/377 (98.1%)** | **2.23** | — |
| + Cross-format SFT, recipe (b) (+ CoT-proxies) | on | 365/377 (96.8%) | 3.32 | — |
| + Full-CoT SFT, variant (c) (teacher's thought supervised) | on | 333/377 (88.3%) | 2.52 | — |
| GLM-5.3 (teacher, first record) | on | 363/377 (96.3%) | 3.24 | 6.6% |
| GLM-5.3 (teacher, final record after reruns) | on | 372/377 (98.7%) | 3.15 | 6.6% |

### Key takeaways

- **Recommended route — cross-format SFT (`run-think.toml`):** trains `qwen3.5-4b-think` on the 599 accepted GLM-5.3 traces using [CoT-Proxy](https://arxiv.org/abs/2611.33148) recipe (a) (empty unsupervised think block, stripped prose around tool calls, out-of-domain KL replay on 1,000 OpenR1-Math/OpenAssistant generations; 8 passes at `lr=3e-5`, LoRA `r=16`; held-out drift `1.6e-3` nats/token).
  - Reaches **98.1% (370/377)** for ~$13 in GPU spend (~$8 SFT, ~$5 eval; ~$1 shared replay pool) without needing a difficulty probe or GRPO.
  - Paired exact McNemar: beats non-thinking SFT (66 fixed / 3 broken, `p = 1.9e-16`) and thinking base (39 / 5, `p = 1.4e-7`), and ties both SFT + GRPO (15 / 7, `p = 0.13`) and the teacher's first record (13 / 6, `p = 0.17`).
  - Adding CoT-proxy summaries (recipe (b), 2-sentence turn summaries by Qwen3-4B-Instruct-2507) did not help on these short 2–4 call episodes (5 fixed / 10 broken vs. recipe (a), `p = 0.30`).
- **Supervising the teacher's thought hurts — variant (c):** recipe (a) with KL replay and the same hyperparameters, but only `<think>\n` is prompt and the teacher's thought, `</think>` and the action are trained (`--think-target native --keep-reasoning`; held-out drift `2.7e-3` nats/token).
  - Scores **333/377 (88.3%)**: 5 fixed / 42 broken vs. recipe (a) (`p = 2.5e-8`), level with the thinking base (38 / 41, `p = 0.82`), below the teacher's first record (6 / 36, `p = 2.8e-6`).
  - All 35 failures are the 300 s episode timeout every run has, each cut off inside a think block; 22 of them after 500+ words of one thought re-checking tool output ("Hmm, but wait… let me reconsider"). Without them it passes 333/342.
  - As in the paper's Table 8, (c) takes on the teacher's phrasing while (a) keeps the base's own:

    | Think blocks | Words/turn (median / mean) | Most common openings |
    | --- | ---: | --- |
    | GLM-5.3 teacher (training traces) | 62 / 408 | "The user wants" 16%, "The base model" 9%, "The model card" 8% |
    | Qwen3.5-4B base (thinking on) | 53 / 102 | "I keep getting" 15%, "I can see" 14%, "The user is" 12% |
    | Recipe (a) | 58 / 62 | "The user is" 28%, "I found the" 26%, "I can see" 4% |
    | Variant (c) | 36 / 119 | "The base model" 19%, "The user wants" 18%, "The user asks" 12% |

  - Cost: ~$10 SFT (1 h 53 min on one H200) and ~$10 eval (1 h 45 min, timeouts included).
- **Non-thinking SFT + GRPO (`run.toml`, phase 2):** a difficulty probe over all 1,162 train rows (4 attempts each at `T=0.8`) selected 350 rows solved 1–3/4 times. Running 100 steps of TRL async GRPO in Pi (2 prompts/step, 8 generations/prompt, 16 concurrent rollouts; $74.57 total for probe + GRPO + eval) lifted accuracy from **81.4% to 96.0%** (64 fixed / 9 broken, `p = 2.4e-11`).
  - GRPO fixed the exact templates SFT could not (recursive file counts: `gguf_count` 17 → 34/37, `safetensors_count` 12 → 20/20) by learning single precise `hf_fs find --recursive` calls, cutting tool calls to **1.75/episode** and failed calls to **1.2%**.
- **Teacher counting rule:** the traces stage reruns episodes cut short by timeouts or provider errors (`--rerun-cut-short`), whereas student evaluations never rerun. Across 377 test questions, 11 teacher episodes were rerun (9 passed), yielding **98.7%** by final record and **96.3%** under the students' first-record rule.

### Per-template breakdown (377 test questions)

| Template | n | Base (off) | SFT (off) | SFT + GRPO (off) | Base (on) | Recipe (a) (on) | Teacher (first) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_params` | 81 | 57 | 63 | 79 | 58 | **81** | 80 |
| `paper_title` | 69 | 47 | 61 | 67 | 66 | **69** | 69 |
| `paper_first_author` | 67 | 56 | 66 | 66 | 63 | 66 | 67 |
| `params_sum` | 52 | 45 | 40 | 49 | 52 | **52** | 41 |
| `gguf_count` | 37 | 15 | 17 | **34** | 30 | 33 | 37 |
| `base_pipeline` | 36 | 25 | 34 | 33 | 35 | **36** | 36 |
| `safetensors_count` | 20 | 13 | 12 | **20** | 18 | 19 | 20 |
| `base_license` | 15 | 14 | 14 | 14 | 14 | 14 | 13 |

## CLI vs. MCP comparison

Apples-to-apples comparison on the first **150 test questions** of `nielsr/hf-hub-facts-v2` (GLM-5.3 → Qwen3.5-4B, thinking off, `pi@1.6.0`, shared verifier):

| Model (thinking off) | `hf` CLI (`bash` + skill) | HF MCP server (3 tools) |
| --- | ---: | ---: |
| Qwen3.5-4B (base) | 71/150 (47.3%) | 110/150 (73.3%) |
| Qwen3.5-4B + SFT | 117/150 (78.0%) | 120/150 (80.0%) |
| GLM-5.3 (teacher) | 148/150 (98.7%) | 148/150 (98.7%) |
| SFT vs. base (fixed / broken) | 62 / 16 | 25 / 15 |

- **Base:** a constrained 3-tool MCP interface is much easier for an untrained 4B model than an open shell (73.3% vs. 47.3%), because it cannot wander off into ad-hoc scrapers.
- **After SFT:** the gap disappears (~80% on both). With thinking enabled and cross-format SFT (recipe (a)), both surfaces match the teacher (98.1% MCP on 377 rows; 96.7% CLI on 150 rows).

### Open-ended judged evaluation and mixed SFT (`judged.py`, `open_ended.py`)

To test whether MCP's typed JSON tools avoid the open-ended confabulation drop seen in CLI mixed SFT, we mirrored `hf-hub-qa`'s judged (`nielsr/hf-hub-questions`, 51 rows) and open-ended training (`nielsr/hf-hub-questions-train`, 240 disjoint rows) tasks with identical grounding, live `hf` reference checks (stage-only), and Gemini 4 Hydrogen judge (`judged_rubrics.py`). GLM-5.3 open-ended MCP traces accepted **232/240 (96.7%)** across 264 episodes (**226** with `<= 12` calls, median 2 calls, $7.87 teacher spend). Training `qwen3.5-4b-think` (recipe (a) + KL replay) on `599 factoid + 226 MCP open-ended = 825 traces` (`run-think-mixed.toml`):

| Model (`pi@1.6.0`, thinking on) | Full `test` (377) | `fact_correct` (377) | Judged (51) | Judged grounded | Judged live `reference` | Judged `answer_quality` | Judged med. calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MCP Base `think` | 336/377 (89.1%) | — | 35/51 (68.6%) | 47/48 (0.999) | 14/18 (0.785) | 0.775 | 2.0 |
| **MCP Factoid Recipe (a)-599** (`run-think.toml`) | **370/377 (98.1%)** | **371/377 (98.4%)** | **36/51 (70.6%)** | **48/49 (0.998)** | 13/17 (0.831) | 0.769 | **2.0** |
| **MCP Mixed Recipe (a)** (`run-think-mixed.toml`) | 364/377 (96.6%) | 365/376 (97.1%) | **36/51 (70.6%)** | 41/45 (0.990) | **16/18 (0.922)** | **0.775** | **2.0** |
| CLI Base `think` (twin) | — | — | 31/51 (60.8%) | 43/44 (0.999) | 13/17 (0.765) | 0.725 | 4.0 |
| CLI Unrewritten Mixed SFT (twin) | 365/377 (96.8%) | 370/376 (98.4%) | 29/51 (56.9%) | 39/45 (0.988) | 18/18 (1.000) | 0.731 | 6.0 |
| CLI Grounded Mixed SFT (twin, Follow-up 2) | 366/377 (97.1%) | 372/376 (98.9%) | **36/51 (70.6%)** | 40/42 (0.998) | 18/19 (0.965) | **0.806** | 7.0 |

- **MCP avoids CLI's mixed-SFT confabulation collapse without trace rewriting:** MCP Base (`35/51 = 68.6%`) already starts +7.8pp above CLI Base (`31/51`, paired `+14/-10`, `p = 0.54`), and both MCP Factoid Recipe (a)-599 and MCP Mixed SFT hold **36/51 (70.6%)** (`+11/-11`, `p = 1.0`). Adding open-ended traces boosts live `reference` retrieval from `13/17 (0.831)` to **16/18 (0.922)** while keeping `364/377 (96.6%)` factual (`0` limit hits, `2.34` MCP calls/ep, `0.11` errors/ep; vs MCP Recipe (a)-599 `+3/-9`, `p = 0.15`).
- **CLI catches up completely once its open-ended teacher traces are strictly tool-grounded:** Follow-up 2 on CLI matches MCP Mixed on both `judged.py` (`36/51` vs `36/51`, `+9/-9`, `p = 1.0`) and `hf-hub-facts-v2` (`366/377` vs `364/377`, `+11/-9`, `p = 0.82`), with MCP still requiring ~3× fewer tool calls (median 2 vs 7 on `judged.py`, 2 vs 4 on facts).

## Task, MCP setup, and rubrics

- **Pinned, read-only MCP tools:** connects anonymously (no `Authorization` header, no HF token) to `https://huggingface.co/mcp` (server `0.4.39`) via [Pi's built-in MCP support](https://docs.pi.dev/latest/mcp) (`--no-extensions -e builtin:mcp`). <!-- pragma: allowlist secret -->
  - Only `hub_repo_search`, `hub_repo_details`, and `hf_fs` are exposed (`exposure: "hidden"` by default; no `bash`, `read`, `write`, `edit`, `hf_whoami`, or codemode).
  - Full tool schemas and sha256 are frozen in [`mcp-tools.json`](mcp-tools.json). Every episode stage verifies `tools/list` against this snapshot and refuses to start on drift unless `--allow-mcp-drift` is passed.
  - Agents run as unprivileged UIDs with no Hub credentials.
- **Dataset (`questions_v2.py`):** [`nielsr/hf-hub-facts-v2`](https://huggingface.co/datasets/nielsr/hf-hub-facts-v2) (1,162 train / 149 validation / 377 test, frozen 2026-11-25). Splits are entity-disjoint and every row passes oracle, no-op, and canary checks. Single-hop v1 (`nielsr/hf-hub-facts`) was retired after base Qwen3.5-4B scored 96.1% on validation. <!-- pragma: allowlist secret -->
- **Rubrics (`rubrics.py`):**
  - **Gates (0 on failure):** `answered` (natural-language `TextAnswer`), `allowed_tools` (no calls outside the 3-tool allow-list), and `used_hub` (≥ 1 successful MCP call).
  - **Scores:** `fact_correct` (shared `agent_distill.rubrics.facts` checker; drifted live Hub values are marked unscorable, not wrong) and `grounded` (every named Hub ID appears in MCP tool outputs), plus a `+0.1` tool-efficiency bonus on passing episodes.

## Running it

```bash
# Recommended: thinking student with cross-format SFT (recipe a)
uv run distill run examples/hf-hub-qa-mcp/run-think.toml --dry-run

# Phase 1: non-thinking SFT pipeline (traces -> sft -> merge -> eval)
uv run distill run examples/hf-hub-qa-mcp/run.toml --dry-run

# Phase 2: difficulty probe + GRPO on top of Phase 1 SFT
uv run distill run examples/hf-hub-qa-mcp/run.toml --stages probe,grpo,grpo-merge --dry-run

# Open-ended judged baselines, open-ended teacher traces, and mixed factual + open-ended SFT
uv run distill run examples/hf-hub-qa-mcp/run-think-judged-baselines.toml --dry-run
uv run distill run examples/hf-hub-qa-mcp/run-think-open-ended-traces.toml --dry-run
uv run distill run examples/hf-hub-qa-mcp/run-think-mixed.toml --dry-run
uv run distill run examples/hf-hub-qa-mcp/run-think-mixed-judged.toml --dry-run

# End-to-end smoke check (8 traces, 2 SFT steps, 5-row eval)
uv run distill smoke examples/hf-hub-qa-mcp/run.toml \
  --set 'stages=["traces","sft","merge","eval"]' \
  --set 'eval.variants=["base","sft","teacher"]' --dry-run
```

## Caveats and published artifacts

- **Phase 1 SFT tool-list bug (fixed in #19):** Phase 1 non-thinking SFT (`run.toml`) rendered training prompts with a phantom `submit` tool instead of the MCP tool schemas. Published evaluation numbers (81.4%) were measured in the real harness with real MCP tools, so the SFT gain is a lower bound; GRPO rollouts and `run-think.toml` (recipe (a)/(b)) all used the real MCP tool definitions.
- **Reasoning-leak guard:** serving Qwen3.5 in thinking mode without a reasoning parser leaked `</think>` into 148/150 early base answers. `agent_distill.reasoning` now flags leakage automatically, and `qwen3.5-4b-think` enables vLLM's `qwen3` reasoning parser while `qwen3.5-4b-nothink` disables thinking.
- **Artifacts** (all in the [agent-distillation collection](https://huggingface.co/collections/nielsr/agent-distillation-6b0445568a39b1d979c52e31)): <!-- pragma: allowlist secret -->
  - Dataset: [`nielsr/hf-hub-facts-v2`](https://huggingface.co/datasets/nielsr/hf-hub-facts-v2) <!-- pragma: allowlist secret -->
  - Shared replay pool: [`nielsr/qwen35-4b-think-replay-openr1-oasst`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-replay-openr1-oasst) <!-- pragma: allowlist secret -->
  - Non-thinking GRPO: [`…-grpo-lora`](https://huggingface.co/nielsr/qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160-grpo-lora), [`…-grpo-merged`](https://huggingface.co/nielsr/qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160-grpo-merged) <!-- pragma: allowlist secret -->
  - Cross-format SFT (recommended): [`…-think-cfx-a-…-lora`](https://huggingface.co/nielsr/qwen35-4b-think-cfx-a-hf-hub-qa-mcp-v2-glm-53-pi-160-lora), [`…-think-cfx-a-…-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-cfx-a-hf-hub-qa-mcp-v2-glm-53-pi-160-eval) <!-- pragma: allowlist secret -->
  - Cross-format SFT + CoT-proxies: [`…-think-cfx-b-…-lora`](https://huggingface.co/nielsr/qwen35-4b-think-cfx-b-hf-hub-qa-mcp-v2-glm-53-pi-160-lora), [`…-think-cfx-b-…-summaries`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-cfx-b-hf-hub-qa-mcp-v2-glm-53-pi-160-summaries), [`…-think-cfx-b-…-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-cfx-b-hf-hub-qa-mcp-v2-glm-53-pi-160-eval) <!-- pragma: allowlist secret -->
  - Full-CoT SFT, variant (c): [`…-think-cfx-c-…-lora`](https://huggingface.co/nielsr/qwen35-4b-think-cfx-c-hf-hub-qa-mcp-v2-glm-53-pi-160-lora), [`…-think-cfx-c-…-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-cfx-c-hf-hub-qa-mcp-v2-glm-53-pi-160-eval) <!-- pragma: allowlist secret -->
  - Thinking base eval: [`…-think-…-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-hf-hub-qa-mcp-v2-glm-53-pi-160-eval) <!-- pragma: allowlist secret -->
  - Open-ended judged & mixed SFT (Follow-up 4): baselines [`…-mcp-judged-…-eval`](https://huggingface.co/datasets/nielsr/cursor-qwen35-4b-think-cfx-a-hf-hub-qa-mcp-judged-pi-160-eval), open-ended traces [`…-mcp-open-ended-…-traces`](https://huggingface.co/datasets/nielsr/cursor-qwen35-4b-think-hf-hub-qa-mcp-open-ended-glm-53-pi-160-traces), mixed adapter [`…-mixed-…-mcp-v2-…-lora`](https://huggingface.co/nielsr/cursor-qwen35-4b-think-cfx-a-mixed-hf-hub-qa-mcp-v2-pi-160-lora), factual eval [`…-mixed-…-mcp-v2-…-eval`](https://huggingface.co/datasets/nielsr/cursor-qwen35-4b-think-cfx-a-mixed-hf-hub-qa-mcp-v2-pi-160-eval), judged eval [`…-mixed-…-mcp-judged-…-eval`](https://huggingface.co/datasets/nielsr/cursor-qwen35-4b-think-cfx-a-mixed-hf-hub-qa-mcp-judged-pi-160-eval) <!-- pragma: allowlist secret -->
