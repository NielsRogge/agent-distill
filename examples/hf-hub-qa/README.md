# hf-hub-qa: Hugging Face Hub Q&A with the `hf` CLI

Distill **GLM-5.3** into **Qwen3.5-4B** inside `pi@1.6.0` to answer multi-hop questions about the Hugging Face Hub using the pinned `hf` CLI (`huggingface_hub==2.4.0`, anonymous/read-only) and its `hf-cli` Agent Skill. <!-- pragma: allowlist secret -->

This is the CLI twin of [`hf-hub-qa-mcp`](../hf-hub-qa-mcp/README.md), which answers the same questions through the HF MCP server. Both share the same teacher, student, dataset, and verifier so only the tool surface differs.

## Headline results

Evaluated on `test[:150]` of `nielsr/hf-hub-facts-v2` (first record per row; the teacher had no reruns, so first and final records are identical):

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

¹ On `test[:150]`. ² The pilot's published records (`nielsr/hf-hub-qa-pilot-glm-53-pi-160-eval`).

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

Spend: about $22.6 of GPU (SFT $12.2, eval of 377 rows $7.9, judged eval $2.4) and under $1 of judge. No new teacher traces. Adapter: [`…-cfx-a-curated-…-lora`](https://huggingface.co/nielsr/qwen35-4b-think-cfx-a-curated-hf-hub-qa-v2-glm-53-pi-160-lora). Evaluations: [`…-cfx-a-curated-…-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-cfx-a-curated-hf-hub-qa-v2-glm-53-pi-160-eval) and [`…-cfx-a-curated-…-judged-…-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-cfx-a-curated-hf-hub-qa-judged-glm-53-pi-160-eval). <!-- pragma: allowlist secret -->

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
- **Main dataset (`task.py`):** `nielsr/hf-hub-facts-v2` (1,162 train / 149 validation / 377 test, frozen 2026-11-25, entity-disjoint splits), shared with `hf-hub-qa-mcp`. Every question requires multi-hop lookups or recursive file counting (base model's license/pipeline/size, linked paper title/first author, parameter sum of three models, or repo GGUF/safetensors count). Single-hop v1 (`nielsr/hf-hub-facts`) was retired after base Qwen3.5-4B hit 93% CLI / 96% MCP on validation.
- **Deterministic rubrics (`rubrics.py`):**
  - **Gates (0 on failure):** `answered`, `read_only` (no write/login commands), and `used_hub` (at least one successful `hf` call).
  - **Scores:** `fact_correct` (shared `agent_distill.rubrics.facts` checker; drifted live values are marked unscorable) and `grounded` (every named Hub ID appears in tool output), plus a `+0.1` tool-efficiency bonus on passing episodes.
- **Secondary task (`judged.py`):** 51 open-ended questions in `nielsr/hf-hub-questions` (`questions.py`), using the same gates and grounding plus live `hf` command references (18 factual/ranked questions) and a Gemini 4 Hydrogen quality judge (`≥ 7/10` to pass).

### Open-ended judged evaluation (51 questions)

| Model | Passed | Median tool calls | Spend |
| --- | ---: | ---: | ---: |
| Kimi-K3.1-Pro (teacher pilot) | 49/51 (96.1%) | 10 | $12.31 |
| GLM-5.3 (teacher pilot) | 43/51 (84.3%) | 5 | $2.72 |
| Recipe (a) (`run-think.toml` adapter, same settings) | 33/51 (64.7%) | 8 | ~$3 GPU + <$1 judge |
| Full-CoT SFT (`run-think-fullcot-judged.toml`) | 26/51 (51.0%) | 8 | <$1 judge |

The teacher pilot's published records (`nielsr/hf-hub-qa-pilot-glm-53-pi-160-eval`, one record per question) score 41/51, not the 43/51 reported for the pilot; paired against them, recipe (a) is 4 ahead / 12 behind (`p = 0.08`). Recipe (a) hit the 20-call limit 9 times, full-CoT 11 times. Evaluation: [`…-think-cfx-a-…-judged-…-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-think-cfx-a-hf-hub-qa-judged-glm-53-pi-160-eval). <!-- pragma: allowlist secret -->

## Running it

```bash
# Recommended: thinking student with cross-format SFT (recipe a)
uv run distill run examples/hf-hub-qa/run-think.toml --dry-run

# Phase 1: non-thinking SFT baseline (traces -> sft -> merge -> eval)
uv run distill run examples/hf-hub-qa/run.toml --dry-run

# Non-thinking SFT rerun -> probe -> GRPO -> eval on all 377 test rows
uv run distill run examples/hf-hub-qa/run-grpo.toml --dry-run

# Curated full-CoT SFT experiment (1,025 zero-error traces)
uv run distill run examples/hf-hub-qa/run-think-fullcot.toml --dry-run

# Data-vs-recipe ablation: recipe (a) on the same 1,025 curated traces, then its judged evaluation
uv run distill run examples/hf-hub-qa/run-think-curated.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think-curated-judged.toml --dry-run

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
- **Published runs:** Phase 1 (`qwen35-4b-hf-hub-qa-v2-glm-53-pi-160`), Phase 2 GRPO (`qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-grpo`), recommended recipe (a) (`qwen35-4b-think-cfx-a-hf-hub-qa-v2-glm-53-pi-160`), and curated full-CoT (`cursor-qwen35-4b-think-hf-hub-qa-v2-glm-53-pi-160`) are published in the [agent-distillation collection](https://huggingface.co/collections/nielsr/agent-distillation-6b0445568a39b1d979c52e31). <!-- pragma: allowlist secret -->
- **Approximate spend:** Phase 1 traces ~$21 + teacher eval $4.49 + GPU ~$12; recipe (a) SFT + eval ~$13 (reusing traces and replay pool); full-CoT additional 562 traces $17.02 + GPU ~$16; Phase 2 GRPO (`run-grpo.toml`): GPU ~$43 (SFT $6.75, probe $11, GRPO pilot $1.90, GRPO $6.30, merges ~$0.60, evaluations ~$16) + teacher eval $11.80.
