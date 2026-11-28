# hf-hub-qa: Hugging Face Hub Q&A with the `hf` CLI

Distill **GLM-5.3** into **Qwen3.5-4B** inside `pi@1.6.0` to answer multi-hop questions about the Hugging Face Hub using the pinned `hf` CLI (`huggingface_hub==2.4.0`, anonymous/read-only) and its `hf-cli` Agent Skill. <!-- pragma: allowlist secret -->

This is the CLI twin of [`hf-hub-qa-mcp`](../hf-hub-qa-mcp/README.md), which answers the same questions through the HF MCP server. Both share the same teacher, student, dataset, and verifier so only the tool surface differs.

## Headline results

Evaluated on `test[:150]` of `agent-distillation-hf/hf-hub-facts-v2` (first record per row; the teacher had no reruns, so first and final records are identical):

| Model | Thinking | Passed | Tool calls / ep | `hf` errors / ep | Read skill | Hit call limit |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B (base) | off | 71/150 (47.3%) | 8.62 | 0.36 | 2.0% | 40 |
| + Phase 1 SFT (`run.toml`, 600 traces) | off | 117/150 (78.0%) | 7.19 | 0.95 | 93.3% | 19 |
| Qwen3.5-4B (base) | on | 130/150 (86.7%) | 6.09 | 1.53 | 0.0% | 12 |
| **+ Cross-format SFT, recipe (a) (`run-think.toml`)** | **on** | **145/150 (96.7%)** | **4.45** | **0.23** | **92.0%** | **0** |
| + Curated Full-CoT SFT (`run-think-fullcot.toml`, 1,025 traces) | on | 148/150 (98.7%) | 4.10 | 0.04 | 93.3% | 0 |
| GLM-5.3 (teacher) | on | 148/150 (98.7%) | 4.58 | 0.08 | 93.3% | 0 |

Full-CoT SFT on the remaining splits: **369/377 (97.9%)** on full `test` (`98.9%` `fact_correct`) and **145/149 (97.3%)** on `validation`.

### Key takeaways

- **Recommended route — cross-format SFT (`run-think.toml`):** trains `qwen3.5-4b-think` on the 600 Phase 1 traces using CoT-Proxy recipe (a) (empty unsupervised think block, stripped prose around tool calls, out-of-domain KL replay; 8 passes at `lr=3e-5`, LoRA `r=16`; held-out drift `2.1e-3` nats/token).
  - Beats Phase 1 non-thinking SFT (31 fixed / 3 broken, exact McNemar `p = 7.7e-7`) and thinking base (18 / 3, `p = 0.0015`), and is statistically level with the GLM-5.3 teacher (1 ahead / 4 behind, `p = 0.38`).
  - Costs ~$13 in GPU spend (~$7 SFT, ~$6 eval) when reusing traces and the shared replay pool.
- **Full-CoT SFT (`run-think-fullcot.toml`) and its generalization trade-off:** trained on 1,025 traces (all 1,162 train rows filtered to `hf_errors=0` and `tool_calls<=12`) with full reasoning supervision (`keep_reasoning=true`, 3 epochs, cosine `lr=1e-4`, LoRA `r=32`, no KL replay).
  - Scores 148/150 on `test[:150]` (3 ahead / 0 behind recipe (a), `p = 0.25`, not significant; both get 148/150 on `fact_correct`, with the 3 differences being grounding misses on recipe (a)). Because it changes data volume, zero-error curation, target format, and schedule at once, the difference cannot be attributed to full-CoT targets alone.
  - On the 51 open-ended judged questions (`run-think-fullcot-judged.toml`), it drops to **26/51 (51.0%)** vs. GLM-5.3's **43/51 (84.3%)**: 3 epochs on terse factoid lookups without KL replay over-specializes the 4B model, causing 11/25 failures to hit the 20-call limit on broad prompts and 13/25 to miss the quality bar. Recipe (a), with KL replay, scores **33/51 (64.7%)** there (12 ahead / 5 behind full-CoT, `p = 0.14`): the direction #21 predicted, but not significant at 51 questions, and the thinking base has not been run on them.
- **Why non-thinking base struggles with a shell:** with thinking off, base Qwen3.5-4B ignores the `hf-cli` skill (2% of episodes), writes `curl`/Python scrapers (130/150 episodes), and runs out of calls (40/150). Action-only SFT lifts it by +30.7 points (62 fixed / 16 broken, `p = 1.5e-7`) to match the MCP SFT student (78.0% vs. 80.0%), though recursive file counts (`gguf_count` 6/15, `safetensors_count` 5/15) remain hard without thinking or GRPO.

### Per-template breakdown (`test[:150]`)

| Template | n | Base (off) | SFT (off) | Base (on) | Recipe (a) (on) | Teacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_params` | 30 | 10 | 27 | 18 | **30** | 30 |
| `params_sum` | 26 | 11 | 22 | 26 | **26** | 26 |
| `paper_first_author` | 26 | 16 | 23 | 23 | 25 | 26 |
| `paper_title` | 24 | 15 | 21 | 24 | 23 | 24 |
| `gguf_count` | 15 | 6 | 6 | 12 | 13 | 15 |
| `safetensors_count` | 12 | 4 | 5 | 12 | **12** | 12 |
| `base_pipeline` | 12 | 7 | 8 | 11 | **12** | 11 |
| `base_license` | 5 | 2 | 5 | 4 | **4** | 4 |

### Data or recipe? Recipe (a) on #21's curated traces (`run-think-curated.toml`)

Full-CoT SFT (#21) changed both the data (1,025 curated traces instead of 600) and the recipe (full-CoT targets, 3 epochs at `lr=1e-4`, LoRA `r=32`, no replay). This ablation keeps recipe (a) at its profile defaults (8 passes, `lr=3e-5`, LoRA `r=16`, KL replay) and trains it on exactly #21's curated set: both trace repos, `hf_errors=0`, `tool_calls<=12`, 1,161 accepted → 1,025 kept. SFT took 2 h 27 min on one H200, with held-out drift `2.2e-3` nats/token. Full `test`, first record per row:

| Model | Traces | `test[:150]` | Full `test` (377) | `fact_correct` (377) | Tool calls / ep | `hf` errors / ep | Hit call limit | Judged (51) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recipe (a) (`run-think.toml`) | 600 | 145/150 | not run | | 4.45 ¹ | 0.23 ¹ | 0 ¹ | 33/51 |
| **Recipe (a), curated (`run-think-curated.toml`)** | 1,025 | 144/150 | 364/377 (96.6%) | 371 | 4.69 | 0.19 | 3 | 26/51 |
| Full-CoT SFT (`run-think-fullcot.toml`) | 1,025 | 148/150 | 369/377 (97.9%) | 373 | 4.29 | 0.06 | 2 | 26/51 |
| GLM-5.3 (teacher) | | 148/150 | 375/377 (99.5%) | | 4.27 | 0.10 | 0 | 41/51 ² |

¹ On `test[:150]`. ² The pilot's published records (`agent-distillation-hf/hf-hub-qa-pilot-glm-53-pi-160-eval`).

Paired comparisons, exact McNemar (curated recipe (a) only / other only):

- **vs recipe (a) on 600 traces, `test[:150]`:** 4 / 5, `p = 1.0`. The 425 extra curated traces changed nothing.
- **vs full-CoT (#21), 377 rows:** 4 / 9, `p = 0.27`. Same traces, different recipe: 5 rows behind, not significant.
- **vs the teacher, 377 rows:** 2 / 13, `p = 0.0074`. Significantly behind. #21 was not significantly behind the teacher (2 / 8, `p = 0.11`).
- **Judged (51):**
  - vs #21: 11 / 11, `p = 1.0`.
  - vs recipe (a)-600: 5 / 12, `p = 0.14`.
  - vs the teacher's published records: 4 / 19, `p = 0.0026`.

What this suggests:

- **The data alone does not explain #21's score.** Recipe (a) on #21's curated traces scores the same as on 600 traces. It trails #21 by 5 rows of 377, a gap that is not significant.
- **#21's edge, if real, comes from its recipe.** The clearest difference is that #21 makes fewer `hf` errors per episode (0.06 against 0.19). Both students trained on the same zero-error traces, so the recipe's full-CoT targets or its longer schedule are the likely cause, not the data.
- **The curated student's 13 failures on `test`:**
  - 7 answered correctly but named a Hub id the agent never looked up;
  - 3 were wrong;
  - 3 hit the 15-call limit.
- **On the judged questions, more curated data did not help recipe (a).** It falls from 33/51 to 26/51, which is #21's score (`p = 0.14`, so not significant). It hits the 20-call limit less often: 7 times, against 11 for recipe (a)-600 and 12 for #21. Its failures are answers that miss the quality bar. On these open-ended questions, the curated set's short, zero-error factoid traces may matter more than the recipe.

Spend: about $22.6 of GPU (SFT $12.2, eval of 377 rows $7.9, judged eval $2.4) and under $1 of judge. No new teacher traces. Adapter: [`…-cfx-a-curated-…-lora`](https://huggingface.co/agent-distillation-hf/qwen35-4b-think-cfx-a-curated-hf-hub-qa-v2-glm-53-pi-160-lora). Evaluations: [`…-cfx-a-curated-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/qwen35-4b-think-cfx-a-curated-hf-hub-qa-v2-glm-53-pi-160-eval) and [`…-cfx-a-curated-…-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/qwen35-4b-think-cfx-a-curated-hf-hub-qa-judged-glm-53-pi-160-eval). <!-- pragma: allowlist secret -->

### Mixed factual + open-ended SFT (`run-think-mixed.toml`)

Factoid-only SFT (whether 600 traces, 1,025 curated traces, or full-CoT) leaves open-ended judged performance at 26–33/51 (51.0–64.7%). To test whether open-ended training traces lift `judged.py` without hurting `hf-hub-facts-v2`, we built 240 open-ended training questions (`open_ended_questions.py`, published at `agent-distillation-hf/hf-hub-questions-train`, strictly disjoint from both `agent-distillation-hf/hf-hub-questions` and `agent-distillation-hf/hf-hub-facts-v2` test/validation entities), collected accepted GLM-5.3 traces with `open_ended.py` (**238/240 = 99.2%** accepted across 267 episodes, $14.19 teacher spend), and trained `qwen3.5-4b-think` (recipe (a) + KL replay, 8 passes, 378 updates, held-out drift `3.18e-3` nats/token) on the mixed pool filtered to `<= 12` tool calls: **802 traces** (597 factoid + 205 open-ended, 74.4% / 25.6%).

| Model | Traces | `test[:150]` | Full `test` (377) | `fact_correct` (377) | Tool calls / ep | `hf` errors / ep | Hit call limit | Judged (51) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recipe (a) (`run-think.toml`) | 600 fact | 145/150 (96.7%) | not run | | 4.45 ¹ | 0.23 ¹ | 0 ¹ | 33/51 (64.7%) |
| Recipe (a), curated (`run-think-curated.toml`) | 1,025 fact | 144/150 (96.0%) | 364/377 (96.6%) | 371 | 4.69 | 0.19 | 3 | 26/51 (51.0%) |
| **Recipe (a), mixed (`run-think-mixed.toml`)** | **597 fact + 205 open** | **145/150 (96.7%)** | **365/377 (96.8%)** | **370** | **4.64** | **0.34** | **4** | **29/51 (56.9%)** |
| Full-CoT SFT (`run-think-fullcot.toml`) | 1,025 fact | 148/150 (98.7%) | 369/377 (97.9%) | 373 | 4.29 | 0.06 | 2 | 26/51 (51.0%) |
| GLM-5.3 (teacher) | | 148/150 (98.7%) | 375/377 (99.5%) | | 4.27 | 0.10 | 0 | 41/51 (80.4%) ² |

¹ On `test[:150]`. ² The pilot's published records (`agent-distillation-hf/hf-hub-qa-pilot-glm-53-pi-160-eval`).

Paired comparisons, exact two-sided McNemar (mixed only / other only):

- **Factual `test[:150]` vs recipe (a)-600:** 4 / 4, `p = 1.0`. Adding 25.6% open-ended traces preserved factual accuracy completely (and reached the highest full-`test` score among recipe (a) runs at **365/377 = 96.8%**, vs curated 11 / 10 `p = 1.0` and full-CoT 5 / 9 `p = 0.42`; its 12 failures are 5 grounding misses on correct facts, 3 wrong facts, and 4 call-limit hits).
- **Judged (51):** 29/51 (56.9%) — vs recipe (a)-600: 5 / 9 (`p = 0.42`); vs curated: 11 / 8 (`p = 0.65`); vs full-CoT: 12 / 9 (`p = 0.66`); vs teacher: 4 / 16 (`p = 0.012`).
- **What mixed SFT improved vs. why judged still plateaus around ~57–65%:**
  - **CLI retrieval and efficiency improved cleanly:** on `judged.py`, live `reference` recall reached **18/18 (100.0%)** (vs 16/18 on recipe (a)-600), `read_skill` reached **51/51 (100%)**, median tool calls dropped from **8 → 6** (closest of any student to GLM-5.3's 5), and 20-call limit hits fell from 9 (recipe (a)-600) and 11 (full-CoT) to **6**.
  - **The remaining bottleneck is 4B parametric confabulation from teacher style mimicry:** GLM-5.3's open-ended traces enrich tool listings with encyclopedic descriptions drawn from frontier world knowledge (e.g., token counts, lab affiliations, architecture blurbs) without reading every model card. Distilled on those traces, the 4B student faithfully imitates the rich table/summary format (`relevance` 0.78–1.0, `clarity` 0.78–1.0) but fills in unverified details from shaky 4B memory, tanking the judge's `correctness` score. Of its 22 judged failures: **12** missed `answer_quality` specifically on `correctness` (0.11–0.56) despite high relevance/clarity, **4** passed both the quality bar (`≥ 7/10`) and live `reference` (`1.0`) but failed only the strict `grounded == 1.0` gate by 1–2 extra IDs (`grounded = 0.75–0.97`), and **6** hit the 20-call limit. Breaking this ceiling on a 4B student likely requires on-policy GRPO directly on `open_ended.py` (penalizing confabulation and limit hits) or rewriting teacher open-ended answers to contain only facts present in tool outputs.

Spend: $14.19 teacher traces + ~$2 judge + ~$20 GPU (SFT ~$10.5, 377-row eval ~$7.2, judged eval ~$2.2). Datasets: [`agent-distillation-hf/hf-hub-questions-train`](https://huggingface.co/datasets/agent-distillation-hf/hf-hub-questions-train), [`…-open-ended-…-traces`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-hf-hub-qa-open-ended-glm-53-pi-160-traces). Adapter: [`…-cfx-a-mixed-…-lora`](https://huggingface.co/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-hf-hub-qa-v2-glm-53-pi-160-lora). Evaluations: [`…-cfx-a-mixed-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-hf-hub-qa-v2-glm-53-pi-160-eval) and [`…-cfx-a-mixed-…-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-hf-hub-qa-judged-glm-53-pi-160-eval). <!-- pragma: allowlist secret -->

### Closing the open-ended gap: strict-grounding ablations and open-ended GRPO

Following the confabulation diagnosis above, three targeted CLI follow-ups tested (1) inference-time grounding instructions alone, (2) rewriting the 205 open-ended teacher answers to be strictly tool-grounded before SFT, and (3) on-policy GRPO directly on `open_ended.py` starting from the mixed SFT checkpoint:

| Model | Full `test` (377) | `fact_correct` (377) | Facts limit hits | Judged (51) | Judged grounded | Judged live `reference` | Judged `answer_quality` | Judged med. calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLI Base `think` | — | — | — | 31/51 (60.8%) | 43/44 (0.999) | 13/17 (0.765) | 0.725 | 4.0 |
| Factoid Recipe (a)-600 (`run-think.toml`) | — | — | — | 33/51 (64.7%) | 42/43 (0.998) | 16/18 (0.907) | 0.773 | 8.0 |
| Mixed SFT (`run-think-mixed.toml`) | 365/377 (96.8%) | 370/376 (98.4%) | 4 | 29/51 (56.9%) | 39/45 (0.988) | 18/18 (1.000) | 0.731 | 6.0 |
| **F1: Mixed SFT + strict-grounding prompt** (`judged_grounded.py`) | *(same adapter)* | *(same)* | *(same)* | 32/51 (62.7%) | 40/47 (0.988) | 18/18 (1.000) | 0.785 | 5.0 |
| **F2: Grounded Mixed SFT** (`run-think-mixed-grounded.toml`) | **366/377 (97.1%)** | **372/376 (98.9%)** | **0** | **36/51 (70.6%)** | **40/42 (0.998)** | **18/19 (0.965)** | **0.806** | 7.0 |
| **F3: Open-ended GRPO** (`run-think-mixed-grpo.toml`, 40 steps) | 317/377 (84.1%) | 318/376 (84.6%) | 40 | 34/51 (66.7%) | 47/48 (0.997) | 13/18 (0.752) | 0.741 | **3.0** |
| MCP Mixed SFT (twin, `hf-hub-qa-mcp`) | 364/377 (96.6%) | 365/376 (97.1%) | 0 | **36/51 (70.6%)** | 41/45 (0.990) | 16/18 (0.922) | 0.775 | 2.0 |

Key takeaways (paired exact two-sided McNemar):

- **F1 — Prompting recovers half the drop, but not the baked-in habit (`32/51 = 62.7%`):** adding a strict-grounding paragraph (`system_prompt_grounded.txt`) to the exact PR #30 mixed adapter lifted judged pass rate by **+5.9pp** (`+11/-8`, `p = 0.65`), `answer_quality` `0.731 → 0.785`, `correctness` `0.652 → 0.721`, cut median calls `6 → 5`, and dropped `hf_errors` `0.51 → 0.27` while holding `reference = 18/18 (100%)`. Seven `grounded < 1.0` misses remained (`40/47`), confirming part of the confabulation habit sits in the SFT weights.
- **F2 — Strictly tool-grounded trace rewriting is the clear CLI winner (`366/377 = 97.1%` factual AND `36/51 = 70.6%` judged):** `rewrite_grounded_traces.py` rewrote only the final assistant answer of the 205 open-ended traces via GLM-5.3 so every claim is directly supported by the episode's own tool output, then re-scored with `open_ended.py` rubrics (`correctness >= 8/10`), keeping **194/205 (94.6%)**. Retraining Recipe (a) SFT on `597 factoid + 194 grounded open-ended = 791 traces`:
  - **Lifts `judged.py` by +13.7pp over unrewritten mixed SFT (`29/51 → 36/51`, `+13/-6`, `p = 0.17`)**, beating CLI Base `think` (`31/51`, `+14/-9`, `p = 0.40`) and Factoid Recipe (a)-600 (`33/51`, `+10/-7`, `p = 0.63`), and **tying the best MCP student (`36/51`, `+9/-9`, `p = 1.0`)**.
  - Achieves the highest `answer_quality` (**0.806**) and `correctness` (**0.762**) of any student (CLI or MCP), cuts ungrounded IDs to **2 out of 528 (`grounded = 0.998`)**, keeps `reference = 18/19 (0.965)` and `hf_errors = 0.47`.
  - **Preserves peak factual accuracy on `hf-hub-facts-v2`:** **366/377 (97.1%)**, `fact_correct = 372/376 (98.9%)`, **0 tool-call limit hits** (paired vs unrewritten mixed `+10/-9`, `p = 1.0`; vs MCP mixed `+11/-9`, `p = 0.82`).
- **F3 — Pure open-ended GRPO fixes grounding & judged (`34/51 = 66.7%`) but causes factual task-forgetting (`317/377 = 84.1%`):** probing `open_ended.py` (`k = 4`) qualified **161/240 (67.1%)** partial-pass rows; 40 steps of TRL async GRPO (`num_generations=8`, 2× H200, 0 zero-std groups) lifted `judged.py` by **+9.8pp** (`29/51 → 34/51`, `+13/-8`, `p = 0.38`), drove `grounded` to **47/48 (0.997, 1 unseen ID of 405)**, and halved median tool calls (`6 → 3`, limit hits `6 → 2`). However, without factual prompts in the GRPO pool (and with `beta = 0`), the policy stopped reading `SKILL.md` (`read_skill: 1.0 → 0.0`) and over-explored on factoid lookups (`40` limit-15 hits, `hf_errors: 0.34 → 1.25`), dropping `hf-hub-facts-v2` by **-12.7pp** (`+8/-56`, `p < 1e-4`). Any future RL stage should mix both factual and open-ended prompts or add a non-zero KL anchor.

Published follow-up artifacts (all public in the [agent-distillation collection](https://huggingface.co/collections/agent-distillation-hf/agent-distillation-6b09e6966000e7910fd46633)): <!-- pragma: allowlist secret -->
- **F1:** [`…-mixed-grounded-prompt-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-grounded-prompt-judged-pi-160-eval) <!-- pragma: allowlist secret -->
- **F2 (recommended CLI all-rounder):** rewritten traces [`…-open-ended-grounded-…-traces`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-hf-hub-qa-open-ended-grounded-glm-53-pi-160-traces), adapter [`…-mixed-grounded-…-lora`](https://huggingface.co/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-grounded-hf-hub-qa-v2-pi-160-lora), factual eval [`…-mixed-grounded-…-v2-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-grounded-hf-hub-qa-v2-pi-160-eval), judged eval [`…-mixed-grounded-…-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-grounded-hf-hub-qa-judged-pi-160-eval) <!-- pragma: allowlist secret -->
- **F3:** merged SFT [`…-grpo-…-sft-merged`](https://huggingface.co/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-open-ended-grpo-pi-160-sft-merged), GRPO adapter [`…-grpo-lora`](https://huggingface.co/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-open-ended-grpo-pi-160-grpo-lora), GRPO merged [`…-grpo-merged`](https://huggingface.co/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-open-ended-grpo-pi-160-grpo-merged), probe [`…-grpo-pi-160-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-open-ended-grpo-pi-160-eval), factual eval [`…-grpo-facts-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-open-ended-grpo-facts-pi-160-eval), judged eval [`…-grpo-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-open-ended-grpo-judged-pi-160-eval) <!-- pragma: allowlist secret -->

### Non-thinking SFT → GRPO on full `test` (`run-grpo.toml`)

Phase 2 reran non-thinking SFT on the 600 Phase 1 traces after #19 (tools exactly `bash` and `read`). A difficulty probe then ran the SFT model 4 times on 700 sampled train rows, and GRPO trained from the SFT merge on the 157 rows it solved on 1–3 of the 4 attempts. That is 78 steps, about 38 minutes on 2× H200. The evaluation covers all 377 `test` rows, first record per row, with the MCP twin's numbers in brackets (its teacher counted on first records as well):

| Model | Thinking | Passed | Tool calls / ep | `hf` errors / ep | Read skill | Hit call limit |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B (base) | off | 206/377 (54.6%) [72.1%] | 8.9 | 0.50 | 2% | 91 |
| + SFT (post-#19) | off | 321/377 (85.1%) [81.4%] | 7.0 | 0.93 | 93% | 30 |
| **+ GRPO** | off | **331/377 (87.8%)** [96.0%] | **4.9** | **0.22** | 70% | 18 |
| GLM-5.3 (teacher) | on | 375/377 (99.5%) [96.3%] | 4.3 | 0.10 | 92% | 0 |

- **Paired comparisons:** SFT against base fixed 141 rows and broke 26 (sign test `p ≈ 2.6e-20`). GRPO against SFT fixed 46 and broke 36 (`p ≈ 0.32`, not significant).
- **GRPO bought mainly efficiency:** 30% fewer tool calls, a quarter of the rejected `hf` calls, and 18 call-limit hits instead of 30. It gained only +2.7 points of accuracy, against +14.6 for the MCP twin. The likely cause is the small pool: 157 qualified rows (22% of those sampled), against 350 for the MCP twin.
- **The thinking students do better on the CLI:** Full-CoT SFT scores 97.9% on the same 377 rows, and recipe (a) 96.7% on `test[:150]`.
- **Per template (base / SFT / GRPO / teacher, of n):**
  - `base_params` 30 / 73 / 73 / 80 of 81
  - `paper_title` 52 / 63 / 68 / 69 of 69
  - `paper_first_author` 42 / 58 / 63 / 67 of 67
  - `params_sum` 24 / 50 / 44 / 52 of 52
  - `gguf_count` 11 / 23 / 23 / 37 of 37
  - `base_pipeline` 29 / 32 / 33 / 35 of 36
  - `safetensors_count` 9 / 10 / 12 / 20 of 20
  - `base_license` 9 / 12 / 15 / 15 of 15
- **File counts remain the gap:** 14 of GRPO's 18 call-limit failures are count questions. `params_sum` regressed by 6 rows; 6 of GRPO's 8 failures there added the sizes with a script, and the cause has not been diagnosed.
- **GRPO's 46 failures:**
  - 18 hit the call limit;
  - 14 were wrong;
  - 13 were correct but named a Hub id the agent never looked up;
  - 1 never used the Hub.

## Task and rubrics

- **Agent setup:** Pi (`pi@1.6.0`) with `bash` and `read`, the pinned `hf` CLI (`huggingface_hub==2.4.0`, logged out / read-only), `curl`/`jq`/`wget`, and the `hf-cli` Agent Skill (`skills/hf-cli/SKILL.md`, loaded via `--skill`). Each episode runs under an unprivileged UID (`agent_distill.harnesses.isolation`). The agent's final natural-language message is the answer (`TextAnswer`). <!-- pragma: allowlist secret -->
- **Main dataset (`task.py`):** `agent-distillation-hf/hf-hub-facts-v2` (1,162 train / 149 validation / 377 test, frozen 2026-11-25, entity-disjoint splits), shared with `hf-hub-qa-mcp`. Every question requires multi-hop lookups or recursive file counting (base model's license/pipeline/size, linked paper title/first author, parameter sum of three models, or repo GGUF/safetensors count). Single-hop v1 (`agent-distillation-hf/hf-hub-facts`) was retired after base Qwen3.5-4B hit 93% CLI / 96% MCP on validation.
- **Deterministic rubrics (`rubrics.py`):**
  - **Gates (0 on failure):** `answered`, `read_only` (no write/login commands), and `used_hub` (at least one successful `hf` call).
  - **Scores:** `fact_correct` (shared `agent_distill.rubrics.facts` checker; drifted live values are marked unscorable) and `grounded` (every named Hub ID appears in tool output), plus a `+0.1` tool-efficiency bonus on passing episodes.
- **Secondary task (`judged.py`):** 51 open-ended questions in `agent-distillation-hf/hf-hub-questions` (`questions.py`), using the same gates and grounding plus live `hf` command references (18 factual/ranked questions) and a Gemini 4 Hydrogen quality judge (`≥ 7/10` to pass).

### Open-ended judged evaluation (51 questions)

| Model | Passed | Median tool calls | Spend |
| --- | ---: | ---: | ---: |
| Kimi-K3.1-Pro (teacher pilot) | 49/51 (96.1%) | 10 | $12.31 |
| GLM-5.3 (teacher pilot) | 43/51 (84.3%) | 5 | $2.72 |
| GLM-5.3 (same-day re-run, `run-teacher-judged.toml`) | 43/51 (84.3%) | 5 | $2.70 + <$1 judge |
| **Grounded Mixed Recipe (a) (`run-think-mixed-grounded-judged.toml`)** | **36/51 (70.6%)** | **7** | ~$2.5 GPU + <$1 judge |
| Open-ended GRPO (`run-think-mixed-grpo-eval-judged.toml`) | 34/51 (66.7%) | 3 | ~$2.5 GPU + <$1 judge |
| Recipe (a) (`run-think.toml` adapter, same settings) | 33/51 (64.7%) | 8 | ~$3 GPU + <$1 judge |
| Mixed Recipe (a) + strict-grounding prompt (`run-think-mixed-judged-grounded.toml`) | 32/51 (62.7%) | 5 | ~$2.5 GPU + <$1 judge |
| Qwen3.5-4B Base (`think`) | 31/51 (60.8%) | 4 | ~$2.5 GPU + <$1 judge |
| Recipe (a), mixed GLM-5.3 (`run-think-mixed-judged.toml`) | 29/51 (56.9%) | 6 | ~$2.2 GPU + <$1 judge |
| **Recipe (a), mixed Inkling-2-Mini (`run-think-mixed-inkling-judged.toml`)** | **28/51 (54.9%)** | **8** | **$2.52 GPU + <$1 judge** |
| Full-CoT SFT (`run-think-fullcot-judged.toml`) | 26/51 (51.0%) | 8 | <$1 judge |
| Inkling-2-Mini base, 8,192 output tokens (`run-inkling-judged.toml`) | **49/51 (96.1%)** | 5 | ~$1.6 GPU + <$1 judge |
| LFM3-1.7B base, thinking (`run-lfm3-judged.toml`) | 20/51 (39.2%) | 4 | $0.87 GPU + judge |
| **LFM3-1.7B recipe (a), mixed Inkling-2-Mini (`run-lfm3-think-mixed-inkling-judged.toml`)** | **20/51 (39.2%)** | **3** | **$0.38 GPU + <$1 judge** |
| LFM3-1.7B recipe (a) (`run-lfm3-think-judged.toml`, the CLI adapter) | 12/51 (23.5%) | 4 | $0.40 GPU + judge |

The teacher pilot's published records (`agent-distillation-hf/hf-hub-qa-pilot-glm-53-pi-160-eval`, one record per question) score 41/51, not the 43/51 reported for the pilot; paired against them, recipe (a) is 4 ahead / 12 behind (`p = 0.08`). Recipe (a) hit the 20-call limit 9 times, full-CoT 11 times. Evaluation: [`…-think-cfx-a-…-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/qwen35-4b-think-cfx-a-hf-hub-qa-judged-glm-53-pi-160-eval). <!-- pragma: allowlist secret -->

**Base Inkling-2-Mini** (`run-inkling-judged.toml`: the settings of `run-think-fullcot-judged.toml`, with the
`inkling-2-mini-8k` profile from [`../inkling-2-mini`](../inkling-2-mini/students/inkling-2-mini-8k.toml)) passes
**49/51** without any training. Paired on the same 51 questions (first record per row, exact McNemar;
`analyze_inkling_judged.py`):

| Inkling-2-Mini vs. | Other | Only Inkling / only other | p |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B base, thinking | 31/51 | 19 / 1 | 4e-05 |
| Recipe (a) | 33/51 | 16 / 0 | 3.1e-05 |
| Recipe (a), curated | 26/51 | 23 / 0 | 2.4e-07 |
| Recipe (a), mixed | 29/51 | 20 / 0 | 1.9e-06 |
| Full-CoT SFT | 26/51 | 24 / 1 | 1.5e-06 |
| GLM-5.3 teacher (published pilot records) | 41/51 | 9 / 1 | 0.021 |
| GLM-5.3 teacher, same-day re-run (`run-teacher-judged.toml`) | 43/51 | 7 / 1 | 0.07 |

- **No timeouts or truncations, and one call limit.** Inkling uses a median of 5 tool calls (the teacher's number). It
  reaches the 20-call limit once, and that answer still passes; the Qwen students fail on the limit 5–11 times each. It
  reads the skill in 50/51 episodes and never leaks reasoning.
- **Two failures, both on hard questions.**
  - "Which object detection datasets are popular on the Hub?": the judge scored correctness 3/10.
  - "What collections has NVIDIA published on Hugging Face?": the answer cleared the quality bar (7/10), but 2 of its 15
    Hub IDs were never seen in a tool result (`grounded` = 0.87).
  - All 5 Qwen students and the teacher fail the first question. Three students fail the second; the teacher passes it.
- **Inkling doesn't fail the questions that are hard for the Qwen students.** By how many of the 5 students fail a
  question, Inkling passes 10/10 at zero failures, 9/9 at one, 11/11 at two, 8/9 at three, 8/8 at four and 3/4 at five.
  It also passes 9 of the teacher's 10 failures. The students' misses are mostly judge correctness (7–15 per student),
  call limits (5–11) and the grounded gate, which Inkling avoids.
- **Hub drift vs. teacher variance.** The references and the judge see the live Hub, and the baselines ran days to
  weeks earlier. To separate the two effects, `run-teacher-judged.toml` re-ran GLM-5.3 on the same day as Inkling, with
  the pilot's settings ($2.70 of teacher spend).
  - The re-run scores **43/51**, against 41/51 for the pilot records (8 / 6 discordant, p = 0.79). The total is stable,
    but 14 of the 51 rows flip between the two runs.
  - Most flips come from the strict `grounded` gate, which fails an answer if any Hub ID was never seen in a tool
    result. The pilot failed it on 6 rows; the re-run passes 5 of those and fails it on 4 others.
  - Only one reference score changed, on "today's trending papers": the re-run listed today's daily papers instead of
    `--sort trending`. That is the teacher answering a different reading of the question, not a Hub change.
  - The re-run passes 8 of Inkling's 9 wins over the pilot records. Those wins came from teacher run-to-run variance,
    not from Hub changes that favoured Inkling.
  - Against the same-day teacher, Inkling is 7 ahead and 1 behind (49 vs. 43, p = 0.07): ahead, but not significantly
    at 51 questions. Its gap over every Qwen student stays significant.

Evaluations: [`inkling-2-mini-8k-…-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/inkling-2-mini-8k-hf-hub-qa-judged-glm-53-pi-160-eval) and the teacher re-run [`hf-hub-qa-judged-rerun-glm-53-pi-160-eval`](https://huggingface.co/datasets/agent-distillation-hf/hf-hub-qa-judged-rerun-glm-53-pi-160-eval). <!-- pragma: allowlist secret -->

**LFM3-1.7B** (`run-lfm3-judged.toml` for the base with thinking on; `run-lfm3-think-judged.toml` for the recipe-(a) CLI
adapter of [`run-lfm3-think.toml`](run-lfm3-think.toml)) uses the same settings: 600 s, 20 tool calls and the Gemini
judge. The question is whether LFM3's factual score (85–88% base, 97.1% after recipe (a); see the
[MCP README](../hf-hub-qa-mcp/README.md)) reflects its general Hub ability. Paired on the 51 questions:

| vs. | LFM3 base (20/51): only LFM3 / only other, p | LFM3 recipe (a) (12/51): only (a) / only other, p |
| --- | --- | --- |
| LFM3-1.7B base | - | 3 / 11, p = 0.057 |
| Qwen3.5-4B base, thinking (31) | 7 / 18, p = 0.043 | 3 / 22, p = 0.00016 |
| Qwen3.5-4B recipe (a) (33) | 3 / 16, p = 0.0044 | 1 / 22, p = 5.7e-06 |
| Inkling-2-Mini 8k (49) | 0 / 29, p = 3.7e-09 | 0 / 37, p = 1.5e-11 |
| GLM-5.3 teacher, same-day re-run (43) | 3 / 26, p = 1.5e-05 | 2 / 33, p = 3.7e-08 |

- **The factual score overstates LFM3's general Hub ability.** On the factual CLI questions the base ties the thinking
  Qwen base (132 vs. 130 of 150, p = 0.86). On open-ended questions it is significantly below it (20 vs. 31,
  p = 0.043) and far below Inkling and the teacher. After recipe (a) the gap widens: +36 factual rows, but 12/51 here,
  8 fewer than the base (not significant at 51 questions, p = 0.057). Qwen recipe (a) gained on both kinds of question.
- **Its failures are answer quality, not budget:**
  - Base: 12 answers miss on the judge's correctness, 9 fail the grounded gate and 5 miss the references. There is
    only 1 call-limit failure and 1 output-limit failure.
  - Recipe (a): 17 correctness misses, 14 grounded and 7 reference misses, 1 call limit.
  - The adapter does not get terser. It writes a median of 232 words (base 274), makes 4 calls and reads the skill
    more often (42 vs. 21 episodes). But it answers the wrong reading of the question more often: it sorts "trending"
    by downloads, takes "decision models" for a Hub tag, and lists IDs it never saw. Factoid-lookup SFT seems to push
    it toward listing and enumerating.
- **It fails easy questions too.** By how many of the 5 Qwen students fail a question, the base passes 7/10 at zero
  failures, 6/9 at one and 3/11 at two; the adapter passes 5/10, 4/9 and 1/11. Discovery questions are 0/8 for both.
- **For training LFM3 beyond factoid lookups,** the mixed factual + open-ended traces of `run-think-mixed.toml` (above)
  are the obvious next run. They kept Qwen's factual accuracy, though they did not lift its judged score (29/51).

Evaluations: [`lfm3-17b-…-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/lfm3-17b-hf-hub-qa-judged-glm-53-pi-160-eval) and [`lfm3-17b-think-sft-…-judged-…-eval`](https://huggingface.co/datasets/agent-distillation-hf/lfm3-17b-think-sft-hf-hub-qa-judged-glm-53-pi-160-eval). <!-- pragma: allowlist secret -->

### Inkling-2-Mini as open-ended teacher (`run-inkling-open-ended-traces.toml`, `run-think-mixed-inkling.toml`, `run-lfm3-think-mixed-inkling.toml`)

Untrained Inkling-2-Mini (8k profile) passes **49/51** on `judged.py` and avoids the call-limit, correctness, and unseen-ID misses that trip up Qwen3.5-4B and LFM3-1.7B. Because recipe (a) leaves the think block empty and strips prose around tool calls, a cross-family teacher's actions and final answers can drop straight into the SFT mix. To test whether Inkling-2-Mini's open-ended traces make better training data than GLM-5.3's (#30), we added self-served vLLM teacher support to `agent_distill.stages.traces` (`examples/inkling-2-mini/teachers/inkling-2-mini-8k.toml`), generated open-ended traces on the 240 disjoint training questions of `open_ended.py` (`timeout=600`, `max_tool_calls=20`, Gemini 4 Hydrogen judge, `pi@1.6.0`), curated to `<= 12` tool calls, and trained both **Qwen3.5-4B** (`run-think-mixed-inkling.toml`) and **LFM3-1.7B** (`run-lfm3-think-mixed-inkling.toml`) with recipe (a) on the exact same 597 factual GLM-5.3 traces plus Inkling's curated open-ended traces.

#### 1. Trace generation & curation (Inkling-2-Mini vs. GLM-5.3)

| Trace teacher (`open_ended.py`, 240 rows) | Accepted | Attempt-1 pass | Total episodes | Curated (`<= 12` calls) | Curated calls mean (med) | Curated answer words mean (med) | Median novel-token ratio ¹ | Curated `read_skill` | Curated `hf_errors` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Inkling-2-Mini (8k, self-served)** | **231/240 (96.2%)** | **215/240 (89.6%)** | **265** | **203/231 (87.9%)** | **5.64 (5)** | **271.2 (237)** | **0.519** | **196/203 (96.6%)** | **0.06** |
| GLM-5.3 (#30, API) | 238/240 (99.2%) | 213/240 (88.8%) | 267 | 205/238 (86.1%) | 5.12 (4) | 268.2 (260) | 0.556 | 198/205 (96.6%) | 0.03 |
| GLM-5.3 grounded rewrite (F2, #33) | — | — | — | 194/205 (94.6%) | 5.00 (4) | 216.8 (191.5) | 0.356 | 188/194 (96.9%) | 0.03 |

¹ Fraction of `>= 4`-character alphanumeric tokens in the final answer that do not appear anywhere in the episode's tool outputs (proxy for parametric elaboration beyond tool results).

- **Acceptance & failure modes:** Inkling accepted **231/240 (96.2%)** with a slightly higher attempt-1 pass rate than GLM-5.3 (`89.6%` vs `88.8%`), 0 reasoning leaks, and 0 cut-short episodes; 9 questions failed both attempts (vs. 2 for GLM-5.3). Across Inkling's 34 failed episodes (any reason; episodes can trigger more than one), **23** failed the strict `grounded` gate, **6** missed `answer_quality` on `correctness`, **6** hit the 20-call limit, and **1** missed `clarity` — matching GLM-5.3's 29 failed episodes (`grounded` 22, `correctness` 4, `call limit` 3, `read_only` 1).
- **Training mix:** `<= 12` tool calls keeps **203** Inkling open-ended traces, giving an **800-trace** mixed pool (597 factual GLM-5.3 + 203 Inkling open-ended, 74.6% / 25.4%) vs. #30's 802 traces (597 + 205). Both students trained for 377 updates (8 passes, `lr=3e-5`, LoRA `r=16`, KL replay): Qwen3.5-4B held-out drift **`3.17e-3` nats/token** (vs. #30's `3.18e-3`); LFM3-1.7B held-out drift **`3.03e-3` nats/token** (run on `l40sx1` for SFT, `l4x1` for eval).

#### 2. Benchmark results (first record per question)

| Model | Open-ended (51) | Factual `test[:150]` | Factual full `test` (377) | Factual `fact_correct` | Open-ended calls mean (med) | Open-ended `read_skill` | Factual calls mean (med) | Factual `hf_errors` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling-2-Mini base (8k) | **49/51 (96.1%)** | — | 363/377 (96.3%) | 363/376 (96.5%) | 6.73 (5) | 50/51 (98.0%) | 5.81 (5) | 0.24 |
| GLM-5.3 teacher (same-day rerun / test) | 43/51 (84.3%) | 148/150 (98.7%) | 375/377 (99.5%) | 377/377 (100.0%) | 5.86 (5) | 49/51 (96.1%) | 4.27 (4) | 0.10 |
| Qwen3.5-4B base, thinking | 31/51 (60.8%) | 130/150 (86.7%) | — | — | 6.86 (4) | 0/51 (0.0%) | 6.09 ¹ | 1.53 ¹ |
| Qwen3.5-4B recipe (a) (600 factual) | 33/51 (64.7%) | 145/150 (96.7%) | — | — | 10.39 (8) | 50/51 (98.0%) | 4.45 ¹ | 0.23 ¹ |
| Qwen3.5-4B mixed GLM-5.3 (#30) | 29/51 (56.9%) | 145/150 (96.7%) | 365/377 (96.8%) | 370/376 (98.4%) | 8.94 (6) | 51/51 (100.0%) | 4.64 (4) | 0.34 |
| Qwen3.5-4B mixed grounded GLM-5.3 (F2, #33) | 36/51 (70.6%) | — | 366/377 (97.1%) | 372/376 (98.9%) | 9.82 (7) | 51/51 (100.0%) | 4.49 (4) | 0.33 |
| **Qwen3.5-4B mixed Inkling-2-Mini** | **28/51 (54.9%)** | **144/150 (96.0%)** | **364/377 (96.6%)** | **367/376 (97.6%)** | **9.84 (8)** | **51/51 (100.0%)** | **5.26 (5)** | **0.58** |
| LFM3-1.7B base, thinking | 20/51 (39.2%) | 132/150 (88.0%) | 330/377 (87.5%) | 331/376 (88.0%) | 6.12 (4) | 21/51 (41.2%) | 6.14 (5) | 2.19 |
| LFM3-1.7B recipe (a) (600 factual, #36) | 12/51 (23.5%) | — | 366/377 (97.1%) | 367/376 (97.6%) | 6.12 (4) | 42/51 (82.4%) | 4.06 (3) | 0.38 |
| **LFM3-1.7B mixed Inkling-2-Mini** | **20/51 (39.2%)** | **138/150 (92.0%)** | **352/377 (93.4%)** | **359/377 (95.2%)** | **5.98 (3)** | **48/51 (94.1%)** | **4.25 (3)** | **0.23** |

¹ On `test[:150]`.

#### 3. Exact paired McNemar tests (student only / other only, two-sided `p`)

**Qwen3.5-4B mixed Inkling-2-Mini** (`28/51` open-ended, `364/377` factual, `144/150` on `test[:150]`):

| Compared against | Open-ended (51): only student / only other, `p` | Factual: only student / only other, `p` (split) |
| --- | --- | --- |
| Qwen3.5-4B base, thinking | 9 / 12, `p = 0.66` (28 vs. 31) | **18 / 4, `p = 0.0043`** (144 vs. 130, `test[:150]`) |
| Qwen3.5-4B recipe (a) | 7 / 12, `p = 0.36` (28 vs. 33) | 3 / 4, `p = 1.0` (144 vs. 145, `test[:150]`) |
| Qwen3.5-4B mixed GLM-5.3 (#30) | 8 / 9, `p = 1.0` (28 vs. 29) | 9 / 10, `p = 1.0` (364 vs. 365, full 377) |
| Qwen3.5-4B mixed grounded (F2, #33) | 3 / 11, `p = 0.057` (28 vs. 36) | 8 / 10, `p = 0.82` (364 vs. 366, full 377) |
| Inkling-2-Mini base (8k) | **2 / 23, `p = 1.9e-5`** (28 vs. 49) | 11 / 10, `p = 1.0` (364 vs. 363, full 377) |
| GLM-5.3 teacher (same-day rerun / test) | **5 / 20, `p = 0.0041`** (28 vs. 43) | **2 / 13, `p = 0.0074`** (364 vs. 375, full 377) |

**LFM3-1.7B mixed Inkling-2-Mini** (`20/51` open-ended, `352/377` factual):

| Compared against | Open-ended (51): only student / only other, `p` | Factual (full 377): only student / only other, `p` |
| --- | --- | --- |
| LFM3-1.7B base, thinking | 8 / 8, `p = 1.0` (20 vs. 20) | **42 / 20, `p = 0.0072`** (352 vs. 330) |
| LFM3-1.7B recipe (a) (#36) | **10 / 2, `p = 0.039`** (20 vs. 12) | **6 / 20, `p = 0.0094`** (352 vs. 366) |
| Inkling-2-Mini base (8k) | **0 / 29, `p = 3.7e-9`** (20 vs. 49) | 12 / 23, `p = 0.090` (352 vs. 363) |
| GLM-5.3 teacher (same-day rerun / test) | **2 / 25, `p = 5.7e-6`** (20 vs. 43) | **1 / 24, `p = 1.6e-6`** (352 vs. 375) |

#### 4. Open-ended (51) failure breakdown (any reason; episodes can fail multiple bars)

| Model | Failed | Judge correctness | Grounded gate (`unseen/ids`) | Reference miss (`pass/n`) | Call limit (20) | Timeout (600 s) | Output limit | Calls mean (med) | Read skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3.5-4B mixed Inkling** | **23** | **14** | **3** (`4/484`) | **0** (`18/18`) | **8** | **0** | **0** | **9.84 (8)** | **51/51 (100%)** |
| Qwen3.5-4B mixed GLM-5.3 (#30) | 22 | 12 | 6 (`8/571`) | 0 (`18/18`) | 6 | 0 | 0 | 8.94 (6) | 51/51 (100%) |
| Qwen3.5-4B mixed grounded (F2) | 15 | 7 | 2 (`2/528`) | 1 (`18/19`) | 8 | 0 | 0 | 9.82 (7) | 51/51 (100%) |
| Qwen3.5-4B recipe (a) | 18 | 8 | 1 (`1/433`) | 2 (`16/18`) | 8 | 0 | 0 | 10.39 (8) | 50/51 (98%) |
| Qwen3.5-4B base, thinking | 20 | 12 ¹ | 1 (`1/374`) | 4 (`13/17`) | 5 | 0 | 0 | 6.86 (4) | 0/51 (0%) |
| **LFM3-1.7B mixed Inkling** | **31** | **23** | **12** (`33/426`) | **2** (`16/18`) | **2** | **0** | **5** | **5.98 (3)** | **48/51 (94%)** |
| LFM3-1.7B recipe (a) (#36) | 39 | 35 ² | 14 (`22/442`) | 7 (`11/18`) | 1 | 0 | 0 | 6.12 (4) | 42/51 (82%) |
| LFM3-1.7B base, thinking | 31 | 21 ³ | 9 (`19/414`) | 6 (`13/19`) | 1 | 0 | 1 | 6.12 (4) | 21/51 (41%) |
| Inkling-2-Mini base (8k) | 2 | 1 | 1 (`2/576`) | 0 (`19/19`) | 0 | 0 | 0 | 6.73 (5) | 50/51 (98%) |
| GLM-5.3 teacher rerun | 8 | 3 | 5 (`5/653`) | 1 (`18/19`) | 0 | 0 | 0 | 5.86 (5) | 49/51 (96%) |

¹ Plus 1 `relevance` quality miss. ² Plus 1 `relevance` quality miss. ³ Plus 2 `completeness` and 2 `relevance` quality misses. All runs had 0 reasoning leaks.

#### 5. Interpretation

- **The hypothesis is rejected for Qwen3.5-4B — swapping GLM-5.3 for Inkling-2-Mini did not close the open-ended gap.** On the direct apples-to-apples comparison (same 597 factual traces, same recipe (a), same replay pool, ~203 vs. 205 open-ended traces), Qwen3.5-4B mixed-Inkling scores **28/51 (54.9%)** vs. GLM-mixed's **29/51 (56.9%)** (`8/9`, `p = 1.0`), and still trails Inkling base (`49/51`, `p = 1.9e-5`) and the GLM-5.3 rerun (`43/51`, `p = 0.0041`) significantly. Its factual accuracy held completely: **364/377 (96.6%)** vs. GLM-mixed's `365/377` (`9/10`, `p = 1.0`), tying Inkling base (`363/377`, `p = 1.0`) and beating Qwen thinking base on `test[:150]` (`144` vs. `130`, `p = 0.0043`).
- **Why Inkling's traces behaved like GLM-5.3's despite Inkling scoring 49/51 itself:** inspecting the curated trace pools reveals that Inkling-2-Mini's 203 open-ended traces have virtually the same surface elaboration as GLM-5.3's 205 traces — **271 words mean / 237 median** vs. **268 / 260**, and a median novel-token ratio (answer tokens absent from tool outputs) of **0.519** vs. **0.556**. Contrast F2's strictly tool-grounded rewrite (#33), which cut answer length to **217 / 191.5** words and novel-token ratio to **0.356**, lifting Qwen3.5-4B to **36/51** (`3/11`, `p = 0.057` vs. mixed-Inkling). Untrained Inkling-2-Mini passes 49/51 because *its own* parametric background knowledge is accurate, not because its final answers stick strictly to tool outputs. When a 4B student imitates that style under recipe (a), it still suffers **14** judge-correctness misses (vs. 12 on GLM-mixed and 7 on F2) and **8** call-limit hits (mirroring Inkling's slightly higher mean tool calls, `5.64` vs. `5.12`), even though unseen-ID grounding tightened (`3` vs. `6` failures, `4/484` vs. `8/571` unseen IDs) and live `reference` recall stayed `18/18 (100%)`.
- **For LFM3-1.7B, the mixed pool significantly reversed the open-ended collapse of factoid-only SFT, but introduced a significant factual trade-off:**
  - **Open-ended:** **20/51 (39.2%)** beats factoid-only LFM3 recipe (a) (`12/51`) by **+8 questions (`10/2`, `p = 0.039`, statistically significant even at `n = 51`)**, fully restoring base LFM3's score (`20/51`, `8/8`, `p = 1.0`), lifting `read_skill` to `94.1%` and `reference` recall to `16/18`, and cutting correctness misses from `35 → 23` and reference misses from `7 → 2`. However, 5 episodes hit the output token limit (vs. 0 on recipe (a) and 1 on base), and 12 still failed the grounded gate (`33/426` unseen IDs).
  - **Factual:** **352/377 (93.4%)** remains significantly above LFM3 base (`330/377`, `42/20`, `p = 0.0072`) with `hf_errors` cut from `2.19 → 0.23`, **but drops significantly below factoid-only LFM3 recipe (a)** (`366/377`, `6/20`, `p = 0.0094`) due to more call-limit hits (`1 → 8`) and grounded misses (`1 → 7`), notably on `gguf_count` (`36 → 32`) and `base_params` (`81 → 78`).
  - **Attribution:** because LFM3 has no GLM-mixed control, LFM3's +8 open-ended recovery is attributable to **mixing open-ended traces into the 1.7B training pool**, not to Inkling specifically as a teacher (given that Qwen's controlled Inkling vs. GLM-5.3 comparison was a dead heat, `p = 1.0`).
- **Statistical power at `n = 51`:** at 51 questions, only large gaps (`>= 8–10` net discordant pairs) reach `p < 0.05`. The significant findings here are: (1) both new students remain far behind Inkling base (`p <= 1.9e-5`) and the GLM-5.3 rerun (`p <= 0.0041`); (2) LFM3 mixed-Inkling significantly recovers open-ended performance over LFM3 recipe (a) (`10/2`, `p = 0.039`); and (3) on factual `test` (`n = 377`), LFM3 mixed-Inkling beats base (`p = 0.0072`) while trailing factoid-only recipe (a) (`p = 0.0094`). Differences among Qwen base (`31`), recipe (a) (`33`), mixed-GLM (`29`), and mixed-Inkling (`28`) are all within noise (`p >= 0.36`); even F2's +8 lead over mixed-Inkling (`36` vs. `28`, `3/11`, `p = 0.057`) is borderline at `n = 51`.

#### 6. Spend & published artifacts

Total spend: **$35.07 GPU** (exact job runtime × HF Jobs rates) + **~$2.70 Gemini 4 Hydrogen judge** (~356 judged episodes across smokes, traces, and evals) ≈ **$37.80 total** (within the $60 cap; **$0** GLM-5.3 spend):

- **Inkling open-ended traces (`rtx-pro-6000`, $2.75/h):** smoke 7.7 min ($0.35) + shard 0 61.5 min ($2.82) + shard 1 53.4 min ($2.45) = **$5.62 GPU** (+ ~$1.95 judge).
- **Qwen3.5-4B mixed-Inkling (`h200`, $5.00/h):** SFT smoke 7.4 min ($0.62) + full SFT 133.3 min ($11.10) + eval smoke 5.0 min ($0.41) + 377-factual eval 92.5 min ($7.71) + 51-judged eval 30.2 min ($2.52) = **$22.37 GPU** (+ ~$0.35 judge).
- **LFM3-1.7B mixed-Inkling (`l40sx1` $1.80/h SFT, `l4x1` $0.80/h eval):** SFT smoke 3.0 min ($0.09) + full SFT 198.0 min ($5.94) + eval smoke 5.6 min ($0.07) + 377-factual eval 44.4 min ($0.59) + 51-judged eval 28.9 min ($0.39) = **$7.08 GPU** (+ ~$0.38 judge).

Published artifacts (all public, scanned with `distill publish`, sessions verified in Data Studio):
- **Hub QA · Inkling-2-Mini:** open-ended traces [`cursor-inkling-2-mini-8k-hf-hub-qa-open-ended-pi-160-traces`](https://huggingface.co/datasets/agent-distillation-hf/cursor-inkling-2-mini-8k-hf-hub-qa-open-ended-pi-160-traces) <!-- pragma: allowlist secret -->
- **Hub QA · Qwen3.5:** adapter [`cursor-qwen35-4b-think-cfx-a-mixed-inkling-hf-hub-qa-v2-pi-160-lora`](https://huggingface.co/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-inkling-hf-hub-qa-v2-pi-160-lora), factual eval [`cursor-qwen35-4b-think-cfx-a-mixed-inkling-hf-hub-qa-v2-pi-160-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-inkling-hf-hub-qa-v2-pi-160-eval), judged eval [`cursor-qwen35-4b-think-cfx-a-mixed-inkling-hf-hub-qa-judged-pi-160-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-qwen35-4b-think-cfx-a-mixed-inkling-hf-hub-qa-judged-pi-160-eval) <!-- pragma: allowlist secret -->
- **Hub QA · LFM3:** adapter [`cursor-lfm3-17b-think-cfx-a-mixed-inkling-hf-hub-qa-v2-pi-160-lora`](https://huggingface.co/agent-distillation-hf/cursor-lfm3-17b-think-cfx-a-mixed-inkling-hf-hub-qa-v2-pi-160-lora), factual eval [`cursor-lfm3-17b-think-cfx-a-mixed-inkling-hf-hub-qa-v2-pi-160-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-lfm3-17b-think-cfx-a-mixed-inkling-hf-hub-qa-v2-pi-160-eval), judged eval [`cursor-lfm3-17b-think-cfx-a-mixed-inkling-hf-hub-qa-judged-pi-160-eval`](https://huggingface.co/datasets/agent-distillation-hf/cursor-lfm3-17b-think-cfx-a-mixed-inkling-hf-hub-qa-judged-pi-160-eval) <!-- pragma: allowlist secret -->

## Running it

```bash
# Recommended: thinking student with cross-format SFT (recipe a)
uv run distill run examples/hf-hub-qa/run-think.toml --dry-run

# Phase 1: non-thinking SFT baseline (traces -> sft -> merge -> eval)
uv run distill run examples/hf-hub-qa/run.toml --dry-run

# Non-thinking SFT rerun -> probe -> GRPO -> eval on all 377 test rows
uv run distill run examples/hf-hub-qa/run-grpo.toml --dry-run

# Base Inkling-2-Mini, thinking on, all 377 test rows (see the MCP README)
uv run distill run examples/hf-hub-qa/run-inkling.toml --dry-run

# Base LFM3-1.7B, thinking on / off, all 377 test rows, one L4 (see the MCP README's LFM3-1.7B section)
uv run distill run examples/hf-hub-qa/run-lfm3.toml --dry-run
uv run distill run examples/hf-hub-qa/run-lfm3-nothink.toml --dry-run
# LFM3-1.7B recipe (a) (SFT on one L4, adapter eval on 377 rows), and both LFM3 models on the 51 judged questions
uv run distill run examples/hf-hub-qa/run-lfm3-think.toml --dry-run
uv run distill run examples/hf-hub-qa/run-lfm3-judged.toml --dry-run      # and run-lfm3-think-judged.toml
uv run --extra hub python examples/hf-hub-qa/analyze_inkling_judged.py --model lfm3-1.7b   # or lfm3-1.7b-recipe-a

# Curated full-CoT SFT experiment (1,025 zero-error traces)
uv run distill run examples/hf-hub-qa/run-think-fullcot.toml --dry-run

# Data-vs-recipe ablation: recipe (a) on the same 1,025 curated traces, then its judged evaluation
uv run distill run examples/hf-hub-qa/run-think-curated.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-curated-judged.toml --dry-run

# Mixed factual + open-ended SFT: disjoint open-ended traces, mixed recipe (a) SFT + 377-row eval, and judged eval
uv run --extra hub python examples/hf-hub-qa/open_ended_questions.py --check
uv run distill run examples/hf-hub-qa/run-think-open-ended-traces.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-mixed.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-mixed-judged.toml --dry-run

# Mixed factual + open-ended SFT with Inkling-2-Mini as self-served open-ended teacher
uv run distill run examples/hf-hub-qa/run-inkling-open-ended-traces.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-mixed-inkling.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-mixed-inkling-judged.toml --dry-run
uv run distill run examples/hf-hub-qa/run-lfm3-think-mixed-inkling.toml --dry-run
uv run distill run examples/hf-hub-qa/run-lfm3-think-mixed-inkling-judged.toml --dry-run
uv run --extra hub python examples/hf-hub-qa/analyze_inkling_judged.py --model qwen3.5-4b-mixed-inkling
uv run --extra hub python examples/hf-hub-qa/analyze_inkling_judged.py --model lfm3-1.7b-mixed-inkling

# Follow-up 1: prompt-only strict-grounding ablation on the mixed SFT adapter
uv run distill run examples/hf-hub-qa/run-think-mixed-judged-grounded.toml --dry-run

# Follow-up 2 (recommended all-rounder): strictly tool-grounded trace rewrite -> mixed recipe (a) SFT + evals
uv run --extra hub python examples/hf-hub-qa/rewrite_grounded_traces.py --dry-run \
  --source-repo agent-distillation-hf/cursor-qwen35-4b-think-hf-hub-qa-open-ended-glm-53-pi-160-traces \
  --output-repo agent-distillation-hf/cursor-qwen35-4b-think-hf-hub-qa-open-ended-grounded-glm-53-pi-160-traces
uv run distill run examples/hf-hub-qa/run-think-mixed-grounded.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-mixed-grounded-judged.toml --dry-run

# Follow-up 3: on-policy GRPO on open_ended.py + factual & judged evals
uv run distill run examples/hf-hub-qa/run-think-mixed-grpo.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-mixed-grpo-eval-facts.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-mixed-grpo-eval-judged.toml --dry-run

# Secondary judged evaluation (teacher pilot / student eval)
uv run distill run examples/hf-hub-qa/run-judged-pilot.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-fullcot-judged.toml --dry-run

# Export the 51 judged questions as JSON
uv run --extra hub python examples/hf-hub-qa/questions.py
```

`run-think-fullcot-grpo.toml` is kept as a config-only template (not run, since full-CoT SFT already reached 97.9% on full test).

## Caveats and notes

- **Phase 1 SFT tool-list mismatch:** Phase 1 non-thinking SFT (`run.toml`) ran before PR #19 and included a phantom `submit` tool in its SFT prompt (evaluation offered only `bash` and `read`, matching traces). Both `run-think.toml` and `run-think-fullcot.toml` use the fixed `[bash, read]` tool list.
- **Shell tool parity across job images:** Student TRL images initially lacked `curl`, `jq`, and `wget` (causing 139 `curl: command not found` errors and depressing non-thinking validation from 61% to 43%). `Task.commands` and `JOB_SETUP` now install and verify them before every job.
- **Published runs:** Phase 1 (`qwen35-4b-hf-hub-qa-v2-glm-53-pi-160`), Phase 2 GRPO (`qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-grpo`), recommended recipe (a) (`qwen35-4b-think-cfx-a-hf-hub-qa-v2-glm-53-pi-160`), and curated full-CoT (`cursor-qwen35-4b-think-hf-hub-qa-v2-glm-53-pi-160`) are published in the [agent-distillation collection](https://huggingface.co/collections/agent-distillation-hf/agent-distillation-6b09e6966000e7910fd46633). <!-- pragma: allowlist secret -->
- **Approximate spend:** Phase 1 traces ~$21 + teacher eval $4.49 + GPU ~$12; recipe (a) SFT + eval ~$13 (reusing traces and replay pool); full-CoT additional 562 traces $17.02 + GPU ~$16; Phase 2 GRPO (`run-grpo.toml`): GPU ~$43 (SFT $6.75, probe $11, GRPO pilot $1.90, GRPO $6.30, merges ~$0.60, evaluations ~$16) + teacher eval $11.80.
