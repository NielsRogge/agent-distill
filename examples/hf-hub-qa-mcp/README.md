# hf-hub-qa-mcp: Hub questions answered through the Hugging Face MCP server

**Status: done. On all 377 v2 test questions, non-thinking Qwen3.5-4B goes from 72.1% (base) to 81.4% (SFT on 599
GLM-5.3 traces) to 96.0% (GRPO, 100 steps), close to the teacher's 98.7%, with fewer MCP calls than the teacher (1.75
against 3.15 per episode). See "Results: GRPO (phase 2)".**

A Pi agent answers questions about the Hugging Face Hub using only the tools of the
[Hugging Face MCP server](https://huggingface.co/docs/hub/en/agents-mcp), through
[Pi's built-in MCP support](https://docs.pi.dev/latest/mcp). This is the MCP twin of
[`hf-hub-qa`](../hf-hub-qa) (branch `claude/hf-cli-task`), where the agent uses the `hf` CLI instead. The two
examples exist to compare *CLI vs MCP* as the way a small student learns to use the Hub, so everything except
the tools is shared wherever possible.

| | hf-hub-qa (CLI) | hf-hub-qa-mcp (this task) |
|---|---|---|
| Harness | `pi@1.6.0` | `pi@1.6.0` |
| Tools | `bash`, `read` + pinned `hf` 2.4.0 + `hf-cli` skill | `mcp__hf__hub_repo_search`, `mcp__hf__hub_repo_details`, `mcp__hf__hf_fs` only |
| Teacher | Kimi-K3.1-Pro (teacher pilot so far) | **GLM-5.3** (user's choice) |
| Student | Qwen3.5-9B (placeholder in the pilot config) | **Qwen/Qwen3.5-4B** (user's choice) |
| Answer | final message in natural language (`TextAnswer`) | same |
| Hub access | anonymous, read-only | anonymous, read-only |

## Tools: pinned and restricted

- **Endpoint:** `https://huggingface.co/mcp`, streamable HTTP, **anonymous** (no `Authorization` header).
  Snapshot taken on 2026-11-25: server `huggingface.co/mcp` version **0.4.39**, four tools, all annotated
  `readOnlyHint: true`: `hf_whoami`, `hub_repo_search`, `hub_repo_details`, `hf_fs`. The full `tools/list`
  (names, descriptions, input schemas) is frozen in [`mcp-tools.json`](mcp-tools.json) with its sha256.
- **Few tools, full coverage.** Server 0.4.x folded the older separate tools (model, dataset, Space and paper
  search, doc search and fetch) into `hf_fs`, a filesystem-style tool over `hf://` URIs with `ls`, `cat`,
  `stat`, `find`, `search` and `attach`. It covers trending listings, papers (`search hf://papers <query>`,
  `cat hf://papers/<id>/paper.md`), daily papers, collections, repo files and cards, and documentation
  (`search hf://docs <query>`). Paper and doc search both work anonymously (checked 2026-11-25). The
  `?bouquet=` URL parameter only narrows the set (`search`, `docs`); none of the tried values adds tools.
  The extra tools in the [settings](https://huggingface.co/settings/mcp) (Contribute Repos, Sandboxes, Jobs,
  Gradio Spaces) need a logged-in account and are all write or compute tools, so they're out of scope.
- **Allow-list:** `hub_repo_search`, `hub_repo_details`, `hf_fs`. Every other tool, `hf_whoami` included, is
  hidden.
  No `bash`, `read`, `write` or `edit`, so there is no CLI or `curl` fallback; Pi's MCP resource tools and
  `codemode` are off too.
- **Pi config:** the adapter writes `mcp.json` into the episode's own `PI_CODING_AGENT_DIR`:

  ```json
  {"autoEnableCodemode": false,
   "mcpServers": {"hf": {"url": "https://huggingface.co/mcp", "exposure": "hidden", "timeout": 120,
                         "toolExposure": {"hub_repo_search": "direct", "hub_repo_details": "direct",
                                          "hf_fs": "direct"}}}}
  ```

  The server is hidden by default, so a tool the server adds later never reaches the agent. Pi runs with
  `--no-extensions -e builtin:mcp` and `--tools 'mcp__hf__hub_repo_search,mcp__hf__hub_repo_details,mcp__hf__hf_fs'`.
  `--no-extensions` also turns off Pi's built-in MCP support, which an MCP task loads back alone; codemode and
  tool search stay off. Tasks without MCP servers keep `--no-mcp`. Checked with real GLM-5.3 episodes: the model
  is offered exactly the three tools and nothing else.
- **Drift guard:** the server's tools change over time. Each episode stage first fetches `tools/list` and
  compares its hash with the snapshot; on a mismatch it refuses to start (the override is an explicit
  `--allow-mcp-drift`, which records both hashes). The endpoint, server version, tools hash and
  `pi@1.6.0` go into the run manifest, every episode record, the session header and the cards.
- **Process gate:** any tool call outside the allow-list fails the episode (`allowed_tools`), as a backstop.

## Credentials and read-only access

- The anonymous server is enough for every question here, so **no HF token is involved at all**: none in the
  agent's environment, none in `mcp.json`, none in records. The episode's `mcp.json` holds only the URL above.
- Anonymous MCP exposes only the read-only tools; the write tools (*Contribute Repos*, *Sandboxes*, *Jobs*)
  exist only for logged-in users who enable them. So the agent cannot write, like or create anything.
- The agent has no shell or file tools, so it cannot read the stage's environment or the job token; Pi still
  runs as its own unprivileged user (#8).
- **Risk: anonymous rate limits.** The server says "used anonymously and rate limits apply". The smoke run
  measures 429s at concurrency 4. If they bite, the fallback is a stage-side MCP proxy that injects the
  `huggingface` org token and passes on only the snapshotted tools. The agent would still see no token,
  but I'll ask before adding it, because a logged-in tool list depends on that account's MCP settings.

## Questions, splits and answers

The hf-cli task has 51 seed questions in one split (`seed`). 15 have a live `reference` (a top-k listing or a
fact); the other 36 are open-ended and judged by Gemini. 51 rows are too few to train on and hold out, and
judged rows aren't cheap or deterministic. Proposal:

1. **`facts` (new, shared with hf-hub-qa):** templated questions over a sampled pool of Hub entities. Each
   question has one exact answer, fetched from the Hub API when the dataset is built and **frozen with a snapshot
   date**. Only stable facts are used:
   - model: license, `pipeline_tag`, `library_name`, `base_model`, gated or not, parameter count from
     safetensors, the weights file name;
   - dataset: license, configs/splits, columns of a split;
   - paper: title and first author of an arXiv id;
   - Space: SDK;
   - multi-hop: "license of the base model of X", "which of these three models has the most parameters".

   Published as [`nielsr/hf-hub-facts`](https://huggingface.co/datasets/nielsr/hf-hub-facts) (snapshot
   2026-11-25): **1,070 train, 127 validation, 359 test** across 11 templates. All rows pass the
   oracle/no-op/canary validation. Splits are disjoint **by entity**, so test asks about
   repos and papers never seen in training. Before scoring, a row whose live value no longer matches the frozen
   one is re-checked and marked `drifted` (unscorable) instead of counting as wrong. Both tasks score through
   the *same* verifier module.
2. **`seed` (hf-hub-qa's 51 questions, verbatim):** a secondary evaluation split only, scored by hf-hub-qa's
   rubrics (live `reference` with its top-2k tolerance, `grounded`, Gemini `answer_quality`). It makes the
   numbers directly comparable with hf-hub-qa's, including the open-ended and trending questions.

**Rubrics (facts split)**, the hf-hub-qa set with only the tool-specific parts swapped:

- gates: `answered` (final message), `allowed_tools` (replaces `read_only`), `used_hub` (≥ 1 successful MCP
  call, replaces the `hf …` regex);
- `fact_correct` (programmatic): the normalized gold value appears in the answer, with aliases for licenses
  (`apache-2.0` ~ "Apache 2.0") and a ±1% tolerance on numbers;
- `grounded`: hf-hub-qa's check, reading the MCP tool outputs instead of shell output;
- metrics: `mcp_calls`, `answer_chars`.

`reward = gate(answered, gate(allowed_tools, gate(used_hub, fact_correct + 0.25·grounded)))`. An episode
passes when it is fully correct and grounded. No judge is needed for training or for the headline evaluation.

## Pipeline and budget

Run name `qwen35-4b-hf-hub-qa-mcp-glm-53-pi-160`. Every repo joins the agent-distillation collection.
Episode datasets are created private, credential-scanned, then published by `distill publish`, with Pi
sessions under `sessions/`. Stages: `traces` (GLM-5.3, up to 2 attempts per train row) → `sft` → `merge` →
`eval` (base, SFT and teacher on `test`, plus `seed`). **SFT comes first**: the first full run stops after the
SFT evaluation. `probe` → `grpo` is a second phase, run only with the user's go once the SFT numbers are in.

### Later phase: GRPO, following [harness adapters at scale](https://huggingface.co/blog/harness-adapters-at-scale)

That post runs GRPO only (no SFT) on Qwen3.8-27B with TRL `AsyncGRPOTrainer` and an OpenEnv capture proxy,
which is the same stack as our `grpo` stage. What carries over:

- **Validate every task before training** (their `openenv validate`; the
  [automation-adapter-envs](https://huggingface.co/datasets/harness-hub/automation-adapter-envs) bar). For each
  row, as a CPU test over the whole dataset:
  - the gold answer written as a sentence scores 1 (oracle);
  - an empty or "I could not find it" answer scores 0 (no-op floor);
  - a known-wrong answer scores 0 (canary), e.g. a sibling entity's value or a neighbouring parameter count;
  - rescoring gives the same result (determinism).

  Rows that fail are dropped and the counts are reported.
- **Train only on rows the model sometimes solves.** The `probe` stage runs the SFT model 4 times per train
  row and keeps rows that pass on 1–3 of the 4 attempts. Rows that always or never pass give zero-variance
  groups and no learning signal.
- **Settings:** LoRA (all-linear), LR **2e-5** (they found 1e-4 collapsed rollouts and 5e-5 unstable over longer
  runs), thinking off, reward = verifier + 0.1 tool-efficiency bonus (our `bonus_if`, as in paper-urls).
  Their group size isn't given; the automation set used 8. A batch mixes templates.
- **Keep gold answers out of training data.** They reach the model only as reward, and test entities never
  appear in train.
- **Pi context compaction** is on in Pi and off in other harnesses. Our episodes are short, but I'll record
  whether compaction ever fires.

Specific to this task:

- **The Pi GRPO path is unproven.** The earlier Pi GRPO smoke made no step, and I found no fix on any branch
  (the html-pages run made probe and GRPO opt-in). Our `grpo` stage drives Pi through the generic capture
  session, so before phase 2 a separate 2-step GRPO smoke runs and any failure gets diagnosed.
- **Rollouts hit the live MCP server.** 8 rollouts × several prompts at once is many anonymous calls. The smoke
  run measures 429s at GRPO concurrency. Rate-limited calls are retried; an episode that still fails is
  unscorable (dropped from the group baseline), never a 0.

| | Estimate | Cap (`[budget]`) |
|---|---|---|
| Smoke (`…-smoke`): 8 teacher traces, 2 SFT steps, 5-row eval × 3 variants (base, SFT, teacher) | < $1 teacher, ≈ 1.5 H200-h | $2 traces, 3 GPU-h |
| Traces: 600 train rows × up to 2 attempts. Local GLM-5.3 episodes used 1k–14k input tokens, about $0.01–0.03 each (at $1.68 / $5.28 per M) | ≈ $20 | $150 |
| Eval: teacher on 150 test (+ 51 seed, judged, later) | ≈ $5 (+ ≈ $10 judge) | in `traces_usd` |
| SFT + merge + student evals | ≈ 4 H200-h | 6 GPU-h |
| Phase 2 — probe (4 × ~400 train rows, SFT model) | ≈ 2 H200-h | 3 GPU-h |
| Phase 2 — GRPO (2× H200: vLLM + trainer; ~150 steps) | ≈ 16 H200-h | 20 GPU-h |

Each figure is checked with `--dry-run` and the budget pilot before launch.

## Smoke run (2026-11-25)

`distill smoke run.toml --set 'stages=["traces","sft","merge","eval"]' --set 'eval.variants=["base","sft","teacher"]'`
→ run `qwen35-4b-hf-hub-qa-mcp-glm-53-pi-160-smoke`.

| Stage | Result |
|---|---|
| traces (GLM-5.3, 8 train rows) | 8/8 accepted, $0.06; 1–2 MCP calls per episode |
| sft (2 steps) | adapter `…-smoke-lora` (public, in the collection) |
| merge | `…-smoke-sft-merged` |
| eval, 5 test rows | base 5/5, SFT 4/5, teacher 4/5 |

Cost: $0.12 teacher, ≈ 0.25 H200-h + 2 A100-min (≈ $1.5).

Checks:

- Pi offers exactly the three allowed MCP tools.
- Every session header carries `pi@1.6.0` and the MCP pin (server 0.4.39, tools sha256, `drift: false`).
- Sessions sit under `sessions/` for the agent-traces viewer.

Issues found and fixed:

- **Pi's `--no-extensions` also disables its built-in MCP.** The teacher got no tools and wrote fake tool calls as
  text. Fixed with `-e builtin:mcp`.
- **The first SFT upload failed** on the account's private-storage cap. Fixed by #10: models are public from
  creation.
- **The teacher's one failure was a grounding false positive.** The old checker counted the prose pair
  `text2text-generation/summarization` as a repo id. The shared stricter checker (#9) no longer counts it.
- **SFT's one failure was the hedge rule working as meant.** It answered "text classification … actually
  question-answering", which names two pipeline tags.

**Headroom risk.** The base student solved all 5 test rows. They were library and pipeline questions, which
make up about a third of the dataset. Before buying traces, a baseline of base and teacher on the 127 validation
rows (≈ $1 teacher, ≈ 0.5 H200-h) should show per-template accuracy. If base is already near the ceiling, the
traces should weight the harder templates: parameters, base model and its license, papers and first author.

## Validation baseline (2026-11-25)

Base Qwen3.5-4B and GLM-5.3 on all 127 validation rows (`distill run run.toml --stages eval --splits validation`).
Rescored with the fixed shared checker (#9) and the republished golds (`distill rescore`).

| | Passed | Mean reward | MCP calls / episode |
|---|---|---|---|
| Qwen3.5-4B (base) | **122/127 (96.1%)** | 1.061 | 1.7 |
| GLM-5.3 (teacher) | **125/127 (98.4%)** | 1.083 | 1.3 |

Every template is at or near the ceiling for the base model:

- 100%: library 21/21, license 17/17, SDK 14/14, dataset license 9/9, base model 4/4, linked paper 4/4;
- the rest: pipeline 19/20, parameters 9/10, paper title 13/14, first author 12/14.

Base failures:

- 2 first-author misreads: the paper text lists "Granite Team", or surnames only;
- 1 pipeline answer that names both tags;
- 1 call to a tool outside the allow-list;
- 1 grounding miss.

**Conclusion: v1 is too easy for SFT to show a gain.** At most about 4 points are possible, and a CLI-vs-MCP
comparison on it would compare two numbers near 97%. Single-hop lookups need 1–2 calls, and the 4B model already
does them. Proposal, v2 of hf-hub-facts: same frozen, deterministic design, harder questions, aiming for a base
pass rate of about 30–60%:

- multi-hop: the license of a model's base model; the first author of the paper a model links;
- comparison: which of three models has the most parameters or downloads (frozen counts);
- listing and aggregation: number of files, largest weight file, GGUF quantizations in a repo;
- organization questions: a model's collections, or an organization's newest model (frozen at snapshot).

## v2 validation baseline (2026-11-25)

[`nielsr/hf-hub-facts-v2`](https://huggingface.co/datasets/nielsr/hf-hub-facts-v2) (`questions_v2.py`): every question
needs at least two lookups or a repo's full file listing. 1,162 train / 149 validation / 377 test; all rows pass the
oracle, no-op and canary checks. Run `qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160`, all 149 validation rows; no row drifted.

| | Passed | MCP calls / episode | Failed MCP calls |
|---|---|---|---|
| Qwen3.5-4B (base) | **111/149 (74.5%)** | 5.8 | 36.3% |
| GLM-5.3 (teacher) | **148/152 (97.4%)** | 3.3 | 8.6% |

The teacher has 152 episodes because 3 cut-short episodes were rerun.

| Template | Base | Teacher |
|---|---|---|
| gguf_count | 7/20 | 19/20 |
| safetensors_count | 2/7 | 7/7 |
| paper_title | 19/26 | 26/26 |
| base_params | 20/29 | 29/29 |
| paper_first_author | 22/25 | 25/25 |
| base_pipeline | 16/17 | 17/17 |
| params_sum | 24/24 | 24/24 |
| base_license | 1/1 | 1/1 |

The headroom is about 23 points. It sits in file counting (listing every folder with `hf_fs`), paper lookups through a
model's tags, and the base model's size. Tool competence differs more than the pass rate: the base model fails 36% of
its MCP calls (`mcp_errors`), the teacher 9%.

## Results: v2 SFT run (2026-11-25)

Run `qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160`. Three stages:

- **Traces:** GLM-5.3 on 600 train questions, at most 2 attempts each. 599 accepted, $21.18.
- **SFT:** Qwen3.5-4B LoRA, 1 epoch, on the accepted traces with reasoning dropped (`nielsr/…-lora`).
- **Evaluation:** 150 test questions, `pi@1.6.0`, HF MCP server 0.4.39 with the pinned tools.

The student is served **without thinking** (`qwen3.5-4b-nothink`, #14), matching its reasoning-free SFT data. The
scoring is the task's rubrics with the corrected `allowed_tools` gate: invented tool names that Pi rejected count as
failed calls, not as fallbacks.

| | Passed | MCP calls / episode | Failed MCP calls |
|---|---|---|---|
| Qwen3.5-4B (base) | 110/150 (73.3%) | 5.85 | 41.2% |
| **Qwen3.5-4B + SFT** | **120/150 (80.0%)** | **3.53** | **13.0%** |
| GLM-5.3 (teacher) | 148/150 (98.7%) | 3.37 | 8.9% |

Across the 450 episodes: no reasoning leaked, no episode was truncated, and none was unscorable. On the paired rows,
SFT fixed 25 base failures and broke 15 base passes (exact McNemar p = 0.15), so the pass-rate gain is not yet
significant at 150 rows. The tool-use change is large: SFT cut the base model's failed MCP calls from 41% to 13% and
its calls per episode from 5.9 to 3.5, close to the teacher's 9% and 3.4.

| Template | n | Base | SFT | Teacher |
|---|---|---|---|---|
| paper_title | 24 | 17 | **24** | 24 |
| base_pipeline | 12 | 8 | 10 | 12 |
| base_params | 30 | 21 | 22 | 30 |
| safetensors_count | 12 | 7 | 8 | 12 |
| paper_first_author | 26 | 24 | 25 | 26 |
| base_license | 5 | 4 | 4 | 3 |
| params_sum | 26 | 23 | 22 | 26 |
| gguf_count | 15 | 6 | 5 | 15 |

What remains is counting files across folders (`gguf_count` 5/15 against the teacher's 15/15) and the base model's
size. These are the rows phase 2 (difficulty probe, then GRPO) would target.

**Before the fix: the thinking student.** The first evaluation served Qwen3.5-4B in its default thinking mode, with no
reasoning parser. Its reasoning landed in the answers: 148/150 base answers contained `</think>`. The numbers are
kept for the record:

| Scoring | Base | SFT | Teacher |
|---|---|---|---|
| as run | 76.7% | 78.7% | 98.7% |
| lenient tool gate | 77.3% | 85.3% | 98.7% |
| lenient gate, text after `</think>` only | 88.7% | 88.7% | 98.7% |

Thinking mode is not comparable with the reasoning-free SFT data: SFT shows no gain there. Hence the non-thinking
profile, and #14's guard that now flags reasoning leakage in every episode stage.

### Full test split (377 questions)

The same models on every v2 test question. The first 150 come from the run above, extended to all 377. Raw logs stayed
in the jobs (`[hub] raw_logs = false`): nielsr's private storage was full.

| | Passed | MCP calls / episode | Failed MCP calls |
|---|---|---|---|
| Qwen3.5-4B (base, non-thinking) | 272/377 (72.1%) | 6.13 | 45.0% |
| **Qwen3.5-4B + SFT** | **307/377 (81.4%)** | **3.38** | **11.9%** |
| GLM-5.3 (teacher) | 372/377 (98.7%) | 3.15 | 6.6% |

Base vs SFT on the same 377 questions: SFT fixed 72 and broke 37 (exact McNemar **p = 0.001**). The gain that was not
significant at 150 questions is significant on the full split.

| Template | n | Base | SFT | Teacher |
|---|---|---|---|---|
| paper_title | 69 | 47 | **61** | 69 |
| base_pipeline | 36 | 25 | **34** | 36 |
| paper_first_author | 67 | 56 | **66** | 67 |
| base_params | 81 | 57 | 63 | 80 |
| gguf_count | 37 | 15 | 17 | 37 |
| base_license | 15 | 14 | 14 | 13 |
| safetensors_count | 20 | 13 | 12 | 20 |
| params_sum | 52 | 45 | **40** | 50 |

SFT gains most on lookups through a model's card: the linked paper and the base model. It does not learn to count a
repo's files (`gguf_count` 17/37 against the teacher's 37/37) and loses ground on adding up three models' parameters.
Those rows are the GRPO targets.

### GRPO smoke

Run `qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160-grpo-smoke`. Two GRPO steps (TRL async GRPO, 2× H200) from the SFT
checkpoint, with the student served non-thinking and MCP rollouts through the capture proxy: both steps finished and
the adapter was pushed and merged. This is the first time Pi GRPO has made a step in this repository. On unfiltered
train rows the SFT student is right 95.5% of the time, so the in-group reward spread was only 0.04. A real GRPO run
first needs the difficulty probe, to keep the rows it solves only sometimes.

## CLI vs MCP (v2, the same 150 test questions)

Both tasks: GLM-5.3 → Qwen3.5-4B, served without thinking, in `pi@1.6.0`, on the first 150 rows of the
`nielsr/hf-hub-facts-v2` test split, scored by the shared checker (`agent_distill.rubrics.facts`). The CLI numbers are
from the hf-hub-qa run (`qwen35-4b-hf-hub-qa-v2-glm-53-pi-160`).

| Pass rate | hf CLI (`bash` + `hf` 2.4.0 + skill) | HF MCP server (3 tools) |
|---|---|---|
| Qwen3.5-4B (base) | 71/150 (47.3%) | 110/150 (73.3%) |
| Qwen3.5-4B + SFT | 117/150 (78.0%) | 120/150 (80.0%) |
| GLM-5.3 (teacher) | 148/150 (98.7%) | 148/150 (98.7%) |
| SFT vs base (fixed / broken) | 62 / 16 | 25 / 15 |

- **Base:** MCP is much easier for an untrained small model. With a shell, the base model mostly leaves `hf` for
  hand-written API calls and HTML scraping, and runs out of tool calls. It read the skill in 2% of episodes. With only
  three MCP tools it has nowhere else to go.
- **After SFT:** the gap closes. SFT teaches the CLI student to read the skill (93% of episodes) and use `hf` (+30.7
  points); the MCP student gains 6.7 points on 150 rows (+9.3 points, p = 0.001, on all 377). Both end near 80%.
- **Teacher:** GLM-5.3 is equally strong through both interfaces (98.7%).
- **What remains in both:** counting a repo's files (`gguf_count`: CLI 6/15 and MCP 5/15 after SFT, teacher 15/15)
  and the base model's parameter count.

### Caveat: SFT trained with the wrong tool list

Found after the run (PR #19). The SFT stage renders the tool definitions from `agent_distill.harnesses.tools.
harness_tools`, which for this task returned only an argument-free `submit` tool. That function knows Pi's built-in tools,
not `Task.mcp_servers`, and it gave `TextAnswer` tasks a phantom `submit`. So the 599 training conversations were
rendered with `tools = [submit]`. At inference, Pi declares the three MCP tools and no `submit`. The student learned to
call `mcp__hf__*` from the trajectories alone, without their schemas in the prompt.

- The published results stand: base, SFT and teacher were all evaluated in the real harness, with the real tool
  definitions. The 81.4% is what this SFT model does in Pi.
- The SFT gain is probably a lower bound: training with the right schemas should only help.
- GRPO is unaffected. Its rollouts run in Pi, so its token capture contains the real MCP tool definitions.
- A future SFT rerun should use main after #19, which renders the MCP tools from `Task.mcp_servers` (once #13 lands)
  and drops `submit` for `TextAnswer` tasks.

## Results: GRPO (phase 2, 2026-11-26)

The user approved phase 2 with a $150 GPU cap; it used **$74.57**.

1. **Difficulty probe** ($15.27): the SFT model (served non-thinking, temperature 0.8) answered all 1,162 v2 train
   questions 4 times each. Passes out of 4 → rows: 0 → 36, 1 → 59, 2 → 89, 3 → 202, 4 → 776. The 350 rows solved 1-3
   times form the GRPO pool. None of them shares an entity with the test split.
2. **GRPO** ($56.39, 5 h 38 min on 2× H200): TRL async GRPO in Pi with the real MCP tools, from the SFT checkpoint.
   - 100 steps, 2 prompts per step (200 draws from 350 rows, no repeats), 8 generations per prompt;
   - 16 concurrent rollouts against the anonymous HF MCP server;
   - LoRA, reward = the task's rubric reward (the fact check, gated, plus the 0.1 tool-efficiency bonus).

   Adapter: [`…-grpo-lora`](https://huggingface.co/nielsr/qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160-grpo-lora); merged
   model: [`…-grpo-merged`](https://huggingface.co/nielsr/qwen35-4b-hf-hub-qa-mcp-v2-glm-53-pi-160-grpo-merged).
3. **Evaluation** ($2.82): all 377 test questions, served non-thinking, scored by the task's rubrics.

| | Passed | MCP calls / episode | Failed MCP calls |
|---|---|---|---|
| Qwen3.5-4B (base) | 272/377 (72.1%) | 6.13 | 45.0% |
| + SFT | 307/377 (81.4%) | 3.38 | 11.9% |
| **+ SFT + GRPO** | **362/377 (96.0%)** | **1.75** | **1.2%** |
| GLM-5.3 (teacher) | 372/377 (98.7%) | 3.15 | 6.6% |

Paired on the same 377 questions (exact McNemar):

- SFT → GRPO: fixed 64, broke 9, **p = 2.4e-11**;
- base → GRPO: fixed 96, broke 6, p = 5.7e-22;
- base → SFT: fixed 72, broke 37, p = 0.001.

| Template | n | Base | SFT | GRPO | Teacher |
|---|---|---|---|---|---|
| gguf_count | 37 | 15 | 17 | **34** | 37 |
| safetensors_count | 20 | 13 | 12 | **20** | 20 |
| base_params | 81 | 57 | 63 | **79** | 80 |
| paper_title | 69 | 47 | 61 | **67** | 69 |
| params_sum | 52 | 45 | 40 | **49** | 50 |
| paper_first_author | 67 | 56 | 66 | 66 | 67 |
| base_pipeline | 36 | 25 | 34 | 33 | 36 |
| base_license | 15 | 14 | 14 | 14 | 13 |

**What GRPO learned.** The templates SFT could not fix are where GRPO gained most. For file counts the GRPO student
makes one precise call, `hf_fs find hf://models/<repo> --recursive --name "*.gguf"`, then states the count. The SFT
student listed folders one by one and lost count. Across all episodes it makes 1.75 MCP calls per question, fewer than
the teacher, and only 1.2% of them fail: the tool-efficiency bonus rewards one well-formed call over several
exploratory ones. Spot checks show legitimate answers: the count is stated first and checked against the listing, and
a miscount (32 against a gold of 34) still fails.

Caveats:

- The SFT stage trained with the wrong tool list (see above); GRPO's rollouts used the real MCP tools.
- One GRPO answer still showed `</think>` despite thinking off, as did one base answer.
