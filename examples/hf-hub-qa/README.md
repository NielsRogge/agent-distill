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
  - On the 51 open-ended judged questions (`run-think-fullcot-judged.toml`), it drops to **26/51 (51.0%)** vs. GLM-5.3's **43/51 (84.3%)**: 3 epochs on terse factoid lookups without KL replay over-specializes the 4B model, causing 11/25 failures to hit the 20-call limit on broad prompts and 13/25 to miss the quality bar. (Recipe (a) has not been evaluated on the judged split yet.)
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
| Full-CoT SFT (`run-think-fullcot-judged.toml`) | 26/51 (51.0%) | — | <$1 judge |

## Running it

```bash
# Recommended: thinking student with cross-format SFT (recipe a)
uv run distill run examples/hf-hub-qa/run-think.toml --dry-run

# Phase 1: non-thinking SFT baseline (traces -> sft -> merge -> eval)
uv run distill run examples/hf-hub-qa/run.toml --dry-run

# Curated full-CoT SFT experiment (1,025 zero-error traces)
uv run distill run examples/hf-hub-qa/run-think-fullcot.toml --dry-run

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
- **Published runs:** Phase 1 (`qwen35-4b-hf-hub-qa-v2-glm-53-pi-160`), recommended recipe (a) (`qwen35-4b-think-cfx-a-hf-hub-qa-v2-glm-53-pi-160`), and curated full-CoT (`cursor-qwen35-4b-think-hf-hub-qa-v2-glm-53-pi-160`) are published in the [agent-distillation collection](https://huggingface.co/collections/nielsr/agent-distillation-6b0445568a39b1d979c52e31). <!-- pragma: allowlist secret -->
- **Approximate spend:** Phase 1 traces ~$21 + teacher eval $4.49 + GPU ~$12; recipe (a) SFT + eval ~$13 (reusing traces and replay pool); full-CoT additional 562 traces $17.02 + GPU ~$16.
