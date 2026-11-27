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

## Inkling-2-Mini (base, thinking on)

`run-inkling.toml` here and in [`hf-hub-qa`](../hf-hub-qa/run-inkling.toml) evaluate base
[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B MoE / 3.4B active, thinking on: its
native mode) on all 377 test questions with `run-think.toml`'s settings (300 s, 15 tool calls, 4 concurrent episodes).
The `inkling-2-mini` profile serves only on RTX PRO 6000 with the `vllm/vllm-openai:inkling-2-mini` image (inkling tool
and reasoning parsers); its hook modules are copied from url-extraction into [`../inkling-2-mini`](../inkling-2-mini).

Results (Nov 2026; first record per row, exact McNemar tests on the paired rows; eval datasets
[MCP](https://huggingface.co/datasets/nielsr/inkling-2-mini-hf-hub-qa-mcp-v2-glm-53-pi-160-eval) and
[CLI](https://huggingface.co/datasets/nielsr/inkling-2-mini-hf-hub-qa-v2-glm-53-pi-160-eval)). Each job took 76 min
(~$3.50); with the 5-question smoke runs, Phase 1 cost ~$8 of GPU.

| Inkling-2-Mini base vs. | MCP: Inkling / other (only Inkling, only other, p) | CLI: Inkling / other (only Inkling, only other, p) |
|---|---|---|
| Qwen3.5-4B base, thinking | **360** / 336 of 377 (37, 13, p = 0.00094) | **145** / 130 of 150 (19, 4, p = 0.0026) |
| Qwen3.5-4B recipe (a) | 360 / 370 of 377 (6, 16, p = 0.052) | 145 / 145 of 150 (4, 4, p = 1.0) |
| Qwen3.5-4B full-CoT SFT | - | 367 / 369 of 377 (5, 7, p = 0.77) |
| GLM-5.3 teacher | 360 / 363 of 377 (9, 12, p = 0.66) | 367 / **375** of 377 (2, 10, p = 0.039) |

- **Pass rate:** 95.5% on MCP and 97.3% on the CLI (367/377 after the `read_only` fix below; the published records
  say 365). Out of the box it beats the thinking Qwen base on both tasks. It is level with recipe (a), the full-CoT SFT
  and the MCP teacher, and below the CLI teacher.
- **Failures:**
  - Budget exhaustion is most of them: on MCP, 8 replies hit the 4096-token output limit, 5 hit the 15-call limit and
    1 timed out. On the CLI, 5 hit the call limit and 2 timed out.
  - Only 3 answers are wrong facts (2 MCP, 1 CLI), plus 1 CLI wrong fact where the skill was never read.
  - One row per task is unscorable: `AtomicChat/Inkling-2-Mini-Uncensored-GGUF` grew from 7 to 14 `.gguf` files after
    the snapshot, and Inkling answered 14.
  - There are no reasoning leaks or think tags in answers, and no tool-call or reasoning-parser errors. The truncated
    replies are long deliberation, not loops (distinct 8-grams ≈ 1.0).
- **Templates:** the misses sit in the parameter questions. On MCP that is `base_params` 72/81 and `params_sum` 46/52;
  on the CLI, `params_sum` 47/52. Every other template is at 36/37 or better.
- **Cost per episode:**
  - MCP: 4.6 tool calls (teacher 3.2, recipe (a) 2.2); 44 s mean and 18 s median wall time; 2,110 output tokens.
  - CLI: 5.9 calls; 44 s mean and 27 s median wall time.
  - Reasoning: a median of 60 words per turn (MCP) and 39 (CLI), similar to the Qwen thinking runs. Inkling often
    calls a tool with no reasoning at all (186 of its CLI turns).
  - The CLI skill was read in 290/377 episodes (77%).
- **Rubric fix:** the CLI's `read_only` gate counted `hf repos branch list` as a write. That command doesn't exist
  (the `branch` group only has `create` and `delete`) and wrote nothing. The gate now matches only
  `branch create|delete`. A scan of every public hf-hub-qa CLI record found only one other affected record, a GRPO
  probe attempt (`qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-grpo-eval`, `episodes/probe/train-0of4.jsonl`). `distill rescore`
  leaves that file at 648/700 passed. Inkling moves from 365 to 367. No published Qwen or teacher number moves.
- **Caveat:** the profile caps Inkling at 4,096 output tokens per reply, while `qwen3.5-4b-think` gets 8,192. The 8 MCP
  truncations are therefore partly a setting. The 8,192-token re-run below tests this.

### Inkling-2-Mini with 8,192 output tokens

`run-inkling-8k.toml` here and in [`hf-hub-qa`](../hf-hub-qa/run-inkling-8k.toml) repeat the runs above with the
project profile [`inkling-2-mini-8k`](../inkling-2-mini/students/inkling-2-mini-8k.toml). That profile is the built-in
one with `max_output_tokens = 8192`, the Qwen thinking setting. The built-in profile keeps 4,096. Eval datasets:
[MCP](https://huggingface.co/datasets/nielsr/inkling-2-mini-8k-hf-hub-qa-mcp-v2-glm-53-pi-160-eval) and
[CLI](https://huggingface.co/datasets/nielsr/inkling-2-mini-8k-hf-hub-qa-v2-glm-53-pi-160-eval). Each job took about
85 min (~$3.90).

| Inkling-2-Mini 8k vs. | MCP: 8k / other (only 8k, only other, p) | CLI: 8k / other (only 8k, only other, p) |
|---|---|---|
| Inkling-2-Mini 4k (above) | 363 / 360 of 377 (13, 10, p = 0.68) | 363 / 367 of 377 (5, 9, p = 0.42) |
| Qwen3.5-4B base, thinking | **363** / 336 of 377 (35, 8, p = 4.2e-05) | **144** / 130 of 150 (19, 5, p = 0.0066) |
| Qwen3.5-4B recipe (a) | 363 / 370 of 377 (6, 13, p = 0.17) | 144 / 145 of 150 (4, 5, p = 1.0) |
| Qwen3.5-4B full-CoT SFT | - | 363 / 369 of 377 (6, 12, p = 0.24) |
| GLM-5.3 teacher | 363 / 363 of 377 (12, 12, p = 1.0) | 363 / **375** of 377 (2, 14, p = 0.0042) |

- **The longer limit fixes the truncations, but the pass rate doesn't move.** On MCP, 7 of the 8 rows that failed on the
  4,096-token limit now pass (the eighth, `6bbe645c46af`, still runs past 8,192 tokens). Ten other rows that passed
  before now fail: 5 hit the call limit, 3 time out, 1 hits the 8,192-token limit and 1 gives a wrong fact. The net is
  +3 (360 to 363), well within run-to-run noise (13 vs. 10 discordant rows, p = 0.68).
- **Budget failures move from tokens to calls and time.** On MCP, output-limit failures fall from 8 to 2, call-limit
  hits stay at 5 (on different rows) and timeouts rise from 1 to 3. On the CLI, where no reply hit the 4,096-token
  limit, call-limit hits rise from 5 to 7 and timeouts from 2 to 3 (367 to 363, p = 0.42).
- Wrong facts stay rare: 2 on MCP and 3 on the CLI (one with the skill unread). The drifted GGUF row is again
  unscorable on both tasks.
- Cost per episode barely changes: 4.7 MCP calls and 2,236 output tokens (from 2,110); 5.8 CLI calls.

```bash
uv run distill run examples/hf-hub-qa-mcp/run-inkling.toml --dry-run
uv run distill run examples/hf-hub-qa/run-inkling.toml --dry-run
uv run --extra hub python examples/hf-hub-qa-mcp/analyze_inkling.py --task mcp
uv run --extra hub python examples/hf-hub-qa-mcp/analyze_inkling.py --task cli --passed d2303c64450f 330b2ba29472  # rescored rows
uv run distill run examples/hf-hub-qa-mcp/run-inkling-8k.toml --dry-run   # and ../hf-hub-qa/run-inkling-8k.toml
uv run --extra hub python examples/hf-hub-qa-mcp/analyze_inkling.py --task mcp --inkling <8k episodes/base/test.jsonl>
```

## LFM3-1.7B (base, thinking on and off)

`run-lfm3.toml` and `run-lfm3-nothink.toml`, here and in [`hf-hub-qa`](../hf-hub-qa/run-lfm3.toml), evaluate base
[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) out of the box. It is a 1.73B dense conv/attention hybrid, and
no fine-tuning was done. The runs cover all 377 test questions with the Inkling settings (300 s, 15 tool calls,
4 concurrent episodes).

**Serving.** Both profiles use the model card's settings:
- vLLM 0.34.0, which supports LFM3 natively, with `--trust-remote-code` and the revision pinned;
- the `qwen3` reasoning parser and the `qwen3_coder` tool parser;
- temperature 0.1, top_k 50, repetition_penalty 1.1, and 8,192 output tokens per reply;
- one L4 (`l4x1`).

**Profiles.** [`lfm3-1.7b`](../../agent_distill/profiles/students/lfm3-1.7b.toml) runs thinking on, the template's
default and the card's choice for agentic use. [`lfm3-1.7b-nothink`](../../agent_distill/profiles/students/lfm3-1.7b-nothink.toml)
passes `enable_thinking: false`, a mode the card calls inconsistent.

**Data and cost.** Eval datasets:
- thinking on: [MCP](https://huggingface.co/datasets/nielsr/lfm3-17b-hf-hub-qa-mcp-v2-glm-53-pi-160-eval) and
  [CLI](https://huggingface.co/datasets/nielsr/lfm3-17b-hf-hub-qa-v2-glm-53-pi-160-eval);
- thinking off: [MCP](https://huggingface.co/datasets/nielsr/lfm3-17b-nothink-hf-hub-qa-mcp-v2-glm-53-pi-160-eval) and
  [CLI](https://huggingface.co/datasets/nielsr/lfm3-17b-nothink-hf-hub-qa-v2-glm-53-pi-160-eval).

The jobs took 2.0 h (MCP think), 1.2 h (MCP nothink), 1.5 h (CLI think) and 0.9 h (CLI nothink), about $4.50 at
$0.80/h. With the 5-question smoke runs, the total is about $5.50 of GPU.

Numbers use the first record per row, with exact McNemar tests on the paired rows. The Qwen thinking base and recipe (a)
CLI baselines cover only 150 rows.

| LFM3-1.7B vs. | MCP think: LFM3 / other (only LFM3, only other, p) | CLI think: LFM3 / other (only LFM3, only other, p) |
|---|---|---|
| LFM3-1.7B thinking off | **322** / 265 of 377 (91, 34, p = 3.5e-07) | **330** / 296 of 377 (58, 24, p = 0.00022) |
| Qwen3.5-4B base, thinking off | **322** / 272 of 377 (87, 37, p = 8.3e-06) | **330** / 206 of 377 (142, 18, p = 4.3e-25) |
| Qwen3.5-4B base, thinking | 322 / 336 of 377 (30, 44, p = 0.13) | 132 / 130 of 150 (17, 15, p = 0.86) |
| Qwen3.5-4B recipe (a) | 322 / **370** of 377 (4, 52, p = 1.1e-11) | 132 / **145** of 150 (4, 17, p = 0.0072) |
| Inkling-2-Mini 8k | 322 / **363** of 377 (7, 48, p = 1.3e-08) | 330 / **363** of 377 (10, 43, p = 5.6e-06) |
| GLM-5.3 teacher | 322 / **363** of 377 (10, 51, p = 9.6e-08) | 330 / **375** of 377 (2, 47, p = 4.4e-12) |

With thinking off, LFM3 scores 265/377 on MCP and 296/377 on the CLI. Against the non-thinking Qwen base that is
265 vs. 272 on MCP (64, 71, p = 0.61) and 296 vs. 206 on the CLI (130, 40, p = 2.6e-12).

| Template (n) | MCP think | MCP nothink | CLI think | CLI nothink |
|---|---:|---:|---:|---:|
| `base_params` (81) | 58 | 69 | 77 | 68 |
| `paper_title` (69) | 63 | 53 | 68 | 65 |
| `paper_first_author` (67) | 65 | 58 | 63 | 55 |
| `params_sum` (52) | 36 | 43 | 48 | 41 |
| `gguf_count` (37) | 30 | 12 | 18 | 20 |
| `base_pipeline` (36) | 36 | 11 | 32 | 22 |
| `safetensors_count` (20) | 20 | 7 | 10 | 11 |
| `base_license` (15) | 14 | 12 | 14 | 14 |

- **Pass rate, thinking on:** 85.4% on MCP and 87.5% on the CLI. At under half the size, it ties the thinking
  Qwen3.5-4B base on both tasks (p = 0.13 and 0.86). It is 41 to 48 rows below recipe (a), Inkling and the teacher, so a
  distillation recipe has room to gain.
- **Thinking off costs a lot:** −57 rows on MCP and −34 on the CLI. In that mode the model invents tools:
  - On MCP, 111/377 episodes call a tool that doesn't exist (1.4 per episode), such as `find`, `cat` or
    `mcp__hf__hub_search`.
  - On the CLI, 177/377 episodes do (0.68 per episode), mostly calling the skill name `hf-cli` as a tool.
- **Parsing is clean in both modes:**
  - No tool call was left unparsed as raw XML in the text.
  - No answer contains a think tag. With thinking on, every turn carries its reasoning in `reasoning_content`.
  - The 4 "reasoning leaks" flagged with thinking off are narration in the final answer ("The `find` result lists ...").
    They are not template output; 2 of them fail.
  - Pi's "Tool X not found" replies are correct: the unknown tools are the model's own inventions, not a parser drop.
- **Failures, thinking on:**
  - MCP: mostly budget. 31 rows hit the 15-call limit, 17 are wrong facts, 3 hit the output limit, 2 are ungrounded
    and 1 timed out. The misses sit in `base_params` (58/81) and `params_sum` (36/52), where the model pages through
    file listings instead of reading `safetensors` metadata.
  - CLI: mostly not reading the skill. The skill is read in only 38/377 episodes (10%; thinking off, 2/377), against
    77% for Inkling. 29 of the 47 failures are wrong facts in episodes that never read it: the model guesses `hf`
    syntax (`hf model ...`) and hits 2.2 `hf` errors per episode. `gguf_count` (18/37) and `safetensors_count` (10/20)
    suffer most, because they need the right `hf` listing commands.
  - The drifted GGUF row is unscorable on three of the four runs.
- **Cost per episode, thinking on:**
  - MCP: 8.0 tool calls (teacher 3.2, recipe (a) 2.2, Inkling 4.7). Wall time is 73 s mean and 61 s median on one L4,
    with 3,836 output tokens and a median of 133 thinking words per turn.
  - CLI: 6.1 calls; 41 s mean and 24 s median wall time; 1,910 output tokens; 43 thinking words per turn.
  - Thinking off makes episodes faster per call but uses more calls: 11.8 on MCP and 8.6 on the CLI.

### Assessment: is LFM3-1.7B a viable student?

**Yes, with thinking on.** Out of the box, the 1.7B model matches the thinking Qwen3.5-4B base on both tasks. It runs
on a $0.80/h L4 and serves on stock vLLM with stock parsers. No tool call or reasoning-parsing problem showed up in
1,508 episodes.

Its gap to the teacher (41 to 45 rows) is behavioural, and the teacher traces cover it:
- On MCP, it uses too many calls: 2.5× the teacher's.
- On the CLI, it doesn't read the skill and guesses commands.

**Recommended recipe: (a), a thinking student with cross-format SFT.** Don't use the non-thinking Phase 1 recipe. The
card calls thinking-off inconsistent, and in this eval it is 34 to 57 rows worse and invents tools in 30 to 47% of
episodes. Recipe (a) supervises the teacher's concise tool calls after an unsupervised empty think block, which targets
exactly LFM3's two weaknesses: call count and the unread skill. Notes for running it:

- **Think tags:** the pinned chat template renders `<think></think>` on history turns (and `reasoning_content` inside
  it). It is prefix-stable, and with LFM3's tokenizer `stages/sft.py` already puts the supervised span after the empty
  think block (checked with `render_trajectory_examples`). The SFT stage needs no LFM3-specific change. A trained
  student would be served with the `lfm3-1.7b` profile.
- **Replay pool:** sample it from LFM3-1.7B itself with thinking on (the `replay` stage on the `lfm3-1.7b` profile),
  not from Qwen. Its thoughts are about twice as long as Qwen's per turn, 133 vs. ~60 words on MCP.
- **GRPO:** the 31 MCP call-limit failures suggest a later GRPO stage with the same 15-call cap would pay off. Profile
  and LoRA targets for its conv/attention hybrid layers still need to be checked before training. Nothing was trained
  here.

```bash
uv run distill run examples/hf-hub-qa-mcp/run-lfm3.toml --dry-run          # and run-lfm3-nothink.toml
uv run distill run examples/hf-hub-qa/run-lfm3.toml --dry-run              # and run-lfm3-nothink.toml
uv run --extra hub python examples/hf-hub-qa-mcp/analyze_inkling.py --task mcp --model lfm3-1.7b   # or lfm3-1.7b-nothink, --task cli
```

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
