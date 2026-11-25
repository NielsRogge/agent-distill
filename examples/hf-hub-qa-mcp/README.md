# hf-hub-qa-mcp: Hub questions answered through the Hugging Face MCP server

**Status: smoke run passed; validation baseline shows hf-hub-facts v1 is near the ceiling (base 96%). A harder v2 is
proposed before the full SFT run (see "Validation baseline").**

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
