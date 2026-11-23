# agent-distill: design

Status: agreed in the design interview of 2026-11-22/23. Nothing here is implemented yet. The milestones at
the end are a proposal and need a go before work starts.

agent-distill turns a frontier agent running inside a real agent harness into a small fine-tuned open model
that runs inside the same harness. It works for any task that can be verified and uses the TRL + OpenEnv +
Hugging Face stack. The recipe is the one proven in
[url-extraction](https://github.com/NielsRogge/url-extraction): collect teacher traces in the harness, filter
them with the task's verifier, run SFT, run GRPO in the unmodified harness with the same verifier as reward,
then evaluate in the same harness. agent-distill packages that recipe as a library and CLI and makes it
general across tasks, harnesses, teachers and students. url-extraction remains the research log, and its task
becomes the first example here.

## Contents

1. [Scope and principles](#1-scope-and-principles)
2. [Package, CLI and configuration](#2-package-cli-and-configuration)
3. [Tasks](#3-tasks)
4. [Rubrics](#4-rubrics)
5. [Episodes and traces](#5-episodes-and-traces)
6. [Harness adapters](#6-harness-adapters)
7. [Pipeline stages](#7-pipeline-stages)
8. [Hub, safety and cost](#8-hub-safety-and-cost)
9. [Example 1: paper URLs (parity with url-extraction)](#9-example-1-paper-urls-parity-with-url-extraction)
10. [Example 2: stunning HTML pages](#10-example-2-stunning-html-pages)
11. [Porting from url-extraction](#11-porting-from-url-extraction)
12. [Related work and positioning](#12-related-work-and-positioning)
13. [Testing and CI](#13-testing-and-ci)
14. [Milestones](#14-milestones)
15. [Open items](#15-open-items)

## 1. Scope and principles

- **A task is** a dataset, a prompt, an answer contract, a set of rubrics, and optionally a workspace setup.
- **Teachers** are frontier models served through HF Inference Providers. **Students** are open models served
  by vLLM and trained with LoRA (SFT, then GRPO).
- **One harness per run.** Teacher traces, SFT data, GRPO rollouts and evaluation all happen in the same
  harness at the same pinned version. Training a separate LoRA per harness is a later step; the design keeps
  room for it but does not build it.
- **Harnesses at launch:** OpenCode 1.2.4, OpenCode 2.3.2 and Pi 1.6.0. Nothing else for now.
- **Compute:** HF Jobs only, for now.
- **Parity:** the same rubric set accepts teacher traces, rewards GRPO rollouts and scores evaluation. The
  only allowed differences are explicitly tagged rubrics, and tests check them.
- **Traces** are published in a form the Hub's
  [agent-traces viewer](https://huggingface.co/docs/hub/en/agent-traces) opens: Pi session format.
- **Safety:** no credentials in agent environments or written records; datasets are private until a
  credential scan passes; nothing on the Hub is ever deleted by the tool.

Out of scope for now: vision students, because TRL's async GRPO and the OpenEnv capture trace are text-only
(see §10); compute other than HF Jobs; harnesses other than OpenCode and Pi; training and merging LoRAs
across several harnesses.

## 2. Package, CLI and configuration

- The distribution is `agent-distill`, the import package is `agent_distill`, and the CLI is `distill`.
- Tooling: Python 3.12, `uv`, `hatchling`, `ruff`, `pytest`.
- License: Apache-2.0 once the repository is public. Until then it stays private with no LICENSE file.

```
agent_distill/
  task.py            Task, DatasetSpec, Limits
  episode.py         Episode, ToolEvent (normalized tool kinds), ModelCall (token capture)
  rubrics/           Rubric decorators, RubricSet, combinators, judge client and cache, compat shims
  harnesses/         base.py (Harness protocol), opencode_v1.py, opencode_v2.py, pi.py (+ pi extension, TS),
                     opencode_v1_reader.py (stored V1 messages -> Episode)
  export/            pi_session.py (Episode -> Pi v3 session), cards.py
  stages/            traces, sft, merge, probe, grpo (trl, simple), eval, publish, rescore
  jobs/              staging (wheel + task folder), launcher, manifests, budget
  profiles/          built-in TOML: teachers/, students/, harnesses/, judges/
  credentials.py     redaction + scan (ported)
  hooks.py           module:function resolution (ported)
  cli.py             distill ...
examples/
  paper-urls/        task.py, rubrics.py, system_prompt.txt, run.toml, README (results, dataset pointers)
  html-pages/        task.py, rubrics.py, concepts/, preview/ (CLI tool), gallery/, run.toml, README
docs/
  DESIGN.md
```

### Configuration

- **Tasks are Python.** A task's `task.py` exports a `task = Task(...)` object, because rubrics are code.
- **Teachers, students, harnesses and judges are TOML profiles.** The library ships built-in profiles, and
  a local `profiles/` folder in the user's project can add new ones or override built-in ones.
- **A run is a `run.toml`**, and any field can be overridden by a CLI flag:

```toml
task = "examples/paper-urls/task.py"
teacher = "glm-5.3"
student = "qwen3.5-4b"
harness = "opencode@1.2.4"
judge = "gemini-4-neon"            # only when the task has judge rubrics
stages = ["traces", "sft", "merge", "probe", "grpo", "eval", "publish"]

[hub]
namespace = "nielsr"
bill_to = "huggingface"            # X-HF-Bill-To for Inference Providers, --namespace for Jobs
collection = "nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e"

[budget]                           # see §8
traces_usd = 150
judge_usd = 100
grpo_gpu_hours = 40

[limits]                           # overrides Task.limits
timeout = 180
max_tool_calls = 15
```

Example commands:

```bash
distill run examples/paper-urls/run.toml
distill run examples/paper-urls/run.toml --stages eval --splits test --dry-run
distill rescore --task examples/paper-urls/task.py --repo nielsr/qwen35-9b-url-agent-eval
distill smoke examples/html-pages/run.toml       # 2-episode end-to-end check on HF Jobs
```

The existing url-extraction semantics carry over:

- `--stages` runs a subset of stages.
- `--dry-run` prints every job (image, flavor, full command) without launching it.
- `--wait` and detached single stages behave as in url-extraction.
- Each repo name can be overridden individually.

### Profiles

The profile schema is url-extraction's `harness/profiles.py` with the URL-specific fields removed.

**Teacher profile**

| Field | Meaning |
|---|---|
| `model` | Model id. A provider pin is part of the id, e.g. `:deepinfra`. |
| `provider` | Documentation only. |
| `capabilities` | e.g. `image_input`, for later use. |
| `[traces]` | `concurrency`, `attempts_per_row`, `episode_timeout`, `max_tool_calls`, `shards`, `flavor`, `timeout`. |
| `price` | USD per million input and output tokens, used by the budget (§8). |

**Student profile**

Unchanged from url-extraction: `[serve]`, `[lora]`, `[sft]`, `[merge]`, `[probe]` and `[grpo]`, with the same
fields. Hooks are given as `module:function` and the files they need are listed under `requires`.

**Harness profile**

| Field | Meaning |
|---|---|
| `adapter` | Which adapter implements the harness. |
| `version` | Exact pinned version. |
| `download` | URL and sha256 of the binary or npm package. |
| `answer_mode` | `native`, `submit_tool` or `final_json`. |
| `tool_map` | Native tool name to normalized tool kind. |
| `capture` | Settings for GRPO token capture. |

**Judge profile**

| Field | Meaning |
|---|---|
| `endpoint` | HF router, or the Gemini API. |
| `model` | Model id. |
| `temperature` | Fixed at 0. |
| `key_env` | The *name* of the environment variable holding the key, never the key itself. |
| `price` | Per-token price, for the budget. |
| `max_concurrency` | Concurrent judge calls. |

## 3. Tasks

```python
Task(
    name="paper-urls",
    dataset=DatasetSpec(
        repo="nielsr/paper-url-extraction-v1-validated",
        splits={"train": "train", "eval": ["test", "validation"]},
        filter=fully_scoreable,                # row -> bool
    ),
    system_prompt=Path("system_prompt.txt"),   # None keeps the harness default
    prompt=user_prompt,                        # row -> str, or a str.format template over the row
    answer=URLFindings,                        # pydantic model: JSON schema + validation; or FileAnswer("index.html")
    rubrics=RUBRICS,                           # RubricSet (§4)
    tools=["shell", "fetch", "read", "search"],  # normalized kinds; adapters map them to native tools
    setup=None,                                # optional (row, workspace: Path) -> None
    network=True,
    limits=Limits(timeout=180, max_tool_calls=15, max_model_calls=24),
    row_id=lambda row: row["arxiv_id"],
)
```

- **Answers.** `answer` is either a pydantic model or a `FileAnswer`. A pydantic model is a structured
  answer. `FileAnswer` means the answer is a file in the workspace, such as `index.html` for example 2.
  Whatever the harness and answer mode, rubrics receive the parsed answer or `None` ("no answer").
- **Workspaces.** `setup` fills a fresh temporary workspace for each episode, for example the HTML task's
  `preview` CLI.
- **Limits.** `run.toml` can override `limits` and the dataset.

## 4. Rubrics

The API is our own and small, following the conventions of the classic `verifiers` Rubric (functions plus
weights, arguments injected by parameter name). We do not depend on `verifiers` (§12).

```python
@programmatic(weight=1.0)
def urls_correct(row, answer) -> float | None: ...          # None = unscorable (dropped from GRPO baseline)

@process()
def research_protocol(row, answer, tool_events) -> float: ...  # reads normalized ToolEvents

@judge(prompt=JUDGE_PROMPT, scale=(1, 5), inputs=["screenshots"])
def visual_quality(...): ...                                  # body optional; the decorator calls the judge

@metric
def tool_calls(tool_events) -> float: ...                     # recorded, weight 0

@programmatic(train_only=True)
def shortcut_guard(tool_events) -> float: ...                 # the only kind of train/eval difference allowed
```

- **Arguments.** A rubric names the arguments it wants and gets them injected: `episode`, `row`, `answer`,
  `messages` (OpenAI format), `tool_events`, `workspace`, `artifacts` (e.g. screenshots produced by the
  task's scoring hooks) and `judge`.
- **Sync, async and errors.** Rubrics can be sync or async. An exception is logged and the rubric scores 0.
  Returning `None` marks the episode unscorable.
- **Combining scores.** A `RubricSet` holds the rubrics plus:
  - `reward(scores) -> float | None`. The default is a weighted sum, and built-in combinators cover the
    url-extraction reward: `gate(r)` (0 when the gate fails), `penalize(r, factor)` and `bonus_if(cond, r)`.
  - `passed(scores) -> bool`. The default is `reward >= threshold`.
- **What uses which.**

  | Consumer | Uses |
  |---|---|
  | Trace acceptance | `passed` |
  | Difficulty probe (prompts solved on some attempts but not all) | `passed` |
  | GRPO reward | `reward` |
  | Evaluation | every rubric, plus `reward` and pass@1 |

- **Parity.** Training and evaluation may differ only in rubrics tagged `train_only` or `eval_only`, and
  parity tests (§13) assert that.
- **Judges.** Judge profiles (§2) call the model at temperature 0. Verdicts are cached by (episode hash,
  rubric id, judge model), so trace filtering, GRPO, evaluation and rescoring never pay twice for the same
  verdict. Judge keys are visible only to the judge client, which runs in the stage process, never in the
  agent's environment. A judge rubric that is too slow for rollouts can be tagged `eval_only`.
- **Compatibility shims.**
  - `Rubric.from_callable(fn)` accepts Tinker's `RewardFn`, i.e. `async (messages) -> float | (float, metrics)`.
  - A Tinker judge `Rubric` (rubric text, extraction regex, output-format instruction) maps onto `@judge`.
  - `verifiers_compat.wrap(fn)` accepts classic verifiers functions `(prompt, completion, answer, state,
    info, **kw)`.
- **Rubric drafting (later milestone).**
  - `distill rubrics draft --task task.py --examples 20` has an LLM write `rubrics.draft.toml` (judge
    rubrics) plus stubs for programmatic checks.
  - The user reviews the draft and sets `reviewed = true`. The `traces` stage refuses an unreviewed draft.
  - `distill rubrics calibrate` runs the teacher on N rows and shows the score distribution for each rubric,
    plus the transcripts where the judge was least sure.

## 5. Episodes and traces

Every adapter returns the same harness-neutral `Episode`:

| Field | Contents |
|---|---|
| `messages` | OpenAI-style chat messages. Reasoning is kept as `reasoning_content`, and SFT drops it unless the student profile sets `keep_reasoning`. |
| `tool_events` | `ToolEvent(kind, native_name, input, success, output, error)`. `kind` is one of `shell`, `fetch`, `read`, `search`, `edit`, `write`, `submit`, `other`. Accessors such as `.command` and `.url` normalize arguments across harnesses. |
| `answer` | The parsed answer, or `None`, plus `answer_mode` and the reason it failed. |
| `usage` | Tokens and cost. |
| timing and failure | `elapsed`, `failure` (timeout, tool-call limit, infrastructure). |
| identity | `harness`, `harness_version`, `model`, task name and version, row id. |
| `model_calls` | For captured GRPO rollouts only: per call, `prompt_ids`, `completion_ids` and `logprobs`. These are the fields a Tinker `TrajectoryGroup` or a verifiers `Trace` needs. |
| `raw` | The harness-native log. |

Where an episode is written:

| What | Where | Visibility |
|---|---|---|
| Slim record (scores, metadata, the portable trajectory used for SFT) | `episodes/*.jsonl` | Private until the credential scan passes |
| Pi v3 session per episode, with `harness` in the header and the verification as a final custom message | `sessions/<prefix>/` | Opens in the agent-traces viewer |
| Raw harness logs, kept for rescoring | `raw/` | Always private |

- Pi episodes are published as Pi's own session file after redaction. OpenCode episodes go through the
  generic Episode-to-Pi exporter, which is the generalization of url-extraction's `trace_export.py`.
- Never upload episodes with full trajectories without their sessions. That rule is carried over from
  url-extraction.
- `opencode_v1_reader` turns stored OpenCode 1.2.4 messages into `Episode`s. With it, every published
  url-extraction record can be rescored.

## 6. Harness adapters

```python
class Harness(Protocol):
    name: str
    version: str

    def install(self, cache: Path) -> Path:            # exact pinned version, verified --version; never PATH
        ...

    async def run_episode(self, task, row, endpoint: Endpoint, workspace: Path, limits: Limits) -> Episode:
        ...

    def capture_factory(self, task, rows, capture_url: str, limits: Limits) -> ResourceSessionFactory:
        ...                                            # GRPO

    def native_trace(self, episode) -> list[dict] | None:   # Pi: its own session; OpenCode: None
        ...
```

- **Endpoints.** An `Endpoint` is any OpenAI-compatible endpoint: base URL, model id, and the name of the
  environment variable holding its key.
  - Teachers use `router.huggingface.co/v1`, sending `X-HF-Bill-To`.
  - Students use local vLLM, or the capture proxy during GRPO.
  - The harness process receives only the credential its endpoint needs. Student runs get none.
- **Pinned versions.** Every binary or package is downloaded at the pinned version into a per-run cache and
  its `--version` is checked. `harness@version` is recorded in every manifest, record, session header and
  card. Rescoring across versions prints a warning.
- **No mixing of versions.**
  - Repo names contain the harness.
  - The `sft` stage refuses traces from different `harness@version` pairs.
  - `grpo` and `eval` refuse to run a student in a harness other than the one its traces came from, unless
    `--allow-harness-mismatch` is passed (which still prints a warning).
- **GRPO capture.** As in url-extraction:
  - Each rollout runs one local harness process in its own workspace.
  - That process talks to the OpenEnv `CaptureServer`, using the capture session id as its API key.
  - vLLM runs with `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`, and LoRA versions are
    hot-loaded by name.
  - There is no Harbor or E2B: the HF Jobs container is the sandbox.
  - Both GRPO backends are kept: TRL `HarnessAsyncGRPOTrainer`, and the `simple` backend for students that
    TRL's packing does not support.

### Adapters

| Adapter | Source | Notes |
|---|---|---|
| `opencode@1.2.4` | Port of url-extraction `harness/opencode.py` | Answer mode `native` (`format: json_schema`, with fallback to a JSON final message). Per-request `system` and `tools`. Session permissions as today. Watchdog for the tool-call limit. `strip_credentials` is on by default for students. |
| `opencode@2.3.2` (`@opencode/cli`) | Port of the V2 runner on url-extraction `claude/opencode-v2-glm53-4b`, once it merges | Plain HTTP; the only SDK is JS/TS. Routes live under `/api/`. Prompts are asynchronous: wait with `/api/experimental/session/{id}/wait`, and page through messages. Basic auth uses a server-generated password read from stdout, so it never enters the agent's environment. The system prompt, permissions (deny `*`, then allow the needed tools) and `steps` are set in a config agent through `OPENCODE_CONFIG_CONTENT`. Answer mode `final_json`. Tool names: `shell`, `read.path`, `patch`, `subagent`. |
| `pi@1.6.0` (`@earendil-works/pi-coding-agent`) | New | Runs `pi --mode json --no-context-files --session-dir <ws>` with `--no-extensions` apart from our own. `PI_CODING_AGENT_DIR` points to a per-episode directory whose `models.json` declares the endpoint (`api: openai-completions`). Tools come from `--tools`, and the system prompt from the task. Answer mode `submit_tool`. |

**Pi extension.** The Pi adapter ships one small TypeScript extension that:

- registers a `submit` tool whose parameters are the task's answer schema, or for a `FileAnswer`, a
  `submit` tool with no arguments that marks the file as final;
- enforces `max_tool_calls` exactly, by blocking the call and terminating on the `tool_call` hook.

If the agent never calls `submit`, the adapter falls back to a JSON final message, or to the file as it is
in the workspace.

**Image inputs.** Verified in the feasibility session but not used yet:

| Harness | Images reach the model when |
|---|---|
| Pi | The model entry declares `input: ["text","image"]` |
| OpenCode 1.2.4 | `modalities.input` includes `"image"` |
| OpenCode 2.3.2 | Always, by default |

Text-only students therefore get an explicit `modalities.input: ["text"]` so that OpenCode V2 never sends
images to a text-only vLLM server.

## 7. Pipeline stages

The stage graph is the one in url-extraction's `pipeline.py`, rewritten around a `Job` abstraction.

| Stage | What it does | Output |
|---|---|---|
| `traces` | Sharded teacher episodes. Accepted when `passed`. | `<run>-traces` (episodes, sessions, raw) |
| `sft` | LoRA SFT on accepted traces of one `harness@version`, with loss on assistant turns only. | `<run>-lora` |
| `merge` | Merges the SFT adapter into the student. | `<run>-sft-merged` |
| `probe` | K attempts per row under rollout conditions; keeps rows solved on some attempts but not all. | `qualified-ids.txt` in `<run>-eval/difficulty/` |
| `grpo` | GRPO in the harness with `reward` from the rubric set, then a merge. | `<run>-grpo-lora`, `<run>-grpo-merged` |
| `eval` | Base, SFT, GRPO and teacher variants over the splits, with summaries and McNemar tests. | `<run>-eval` |
| `publish` | Credential scan, visibility changes, collection notes, cards. | (Hub settings) |
| `rescore` | Recomputes every record's scores from stored episodes. | a report |

Run naming defaults to `<student>-<task>-<teacher>-<harness>`, e.g. `qwen3.5-9b-html-pages-kimi-k3.1-pro-pi`.

**Code staging.** Before jobs are launched:

1. A wheel of the current checkout is built.
2. The wheel and the task folder are uploaded to the private `<run>-code` Hub repo, under `code/<rev>/`.
3. Every job runs `uv pip install` on that wheel.

This keeps GitHub credentials out of jobs. Once the package is on PyPI, a released version can be pinned
instead.

## 8. Hub, safety and cost

**Hub settings.**

- `[hub]` holds `namespace`, `bill_to` and an optional `collection`. Every repo the tool creates is added to
  the collection with a note naming its role (teacher traces, SFT adapter, GRPO adapter, evaluation, …).
  Nothing is hardcoded to `nielsr` or `huggingface`; the examples set those values.
- **Visibility:**
  - Models are public.
  - Datasets are created private and made public only by `publish`, after a credential scan of every text
    file passes. The scan looks for `hf_`, `sk-`, `ghp_`, `github_pat_` and Google API keys.
  - `-code` repos are always private.
- **Never deleted.** The tool never deletes Hub repos or files.

**Credentials.**

- Agent processes get no credentials they do not need. Students get none, and teachers get only their
  endpoint's key.
- Records go through `record_line`, which redacts credentials. Pi sessions are redacted before upload.
- Judge and critic keys live in the stage process only.

**Cost guards.**

- `[budget]` sets a cap per stage: USD for teacher calls and judges, GPU-hours for SFT, GRPO and eval.
- Before launch, each stage estimates its cost from a pilot of about 20 episodes and refuses to start above
  the cap unless `--over-budget` is passed.
- During traces and judging, the actual spend is tracked from token counts × profile prices, and the stage
  stops cleanly at the cap.
- Resumed stages skip episodes that are already finished, and cached judge verdicts are reused.
- Spend is written to the run manifest and to every card.

## 9. Example 1: paper URLs (parity with url-extraction)

`examples/paper-urls/` holds:

- the task: `system_prompt.txt`, `URLFindings`, `user_prompt`;
- the rubrics: URL correctness with `canonical_url` and aliases, the research protocol as a process rubric
  matching tool kinds, the tool-call limit, the shortcut guard (`train_only`), the tool-efficiency bonus,
  and label-quality filtering;
- the run configs for each pair, and a README with dataset pointers and results.

The reward reproduces url-extraction exactly:

```
reward = bonus_if(full_credit, 0.1·(1−calls/15),
         gate(shortcut_guard, penalize(protocol_ok, 0.5, url_correctness)))
```

and `passed` = every verified field is correct and the episode is complete.

**Parity tiers.** Tiers 1 and 2 gate the first release.

1. **Rescore** (CPU only). Every published url-extraction record (the evaluation repos, the teacher
   baselines and the accepted train traces in docs/pipeline.md's validation table) is rescored through
   `opencode_v1_reader` and the ported rubrics, with zero per-episode differences. This also becomes a test
   fixture.
2. **V1 evaluation re-run.** `opencode@1.2.4` evaluates the Qwen3.5-9B base and
   `qwen35-9b-url-agent-grpo-merged-kimi` on test. They must land within ±5 points of 75.0% and 82.1%.
3. **V2 cross-check**, once the url-extraction V2 experiment (GLM-5.3 → Qwen3.5-4B) publishes. Our
   rubrics must accept exactly the same teacher traces, and our V2 adapter's evaluation of its SFT and GRPO
   models must match url-extraction's numbers within noise.
4. **Full recipe** (later milestone). SFT → GRPO → eval inside agent-distill: in V1 from the 873 GLM-5.2
   traces for Qwen3.5-4B (about 85.7% on test), or in V2 from the experiment's traces. Running both also
   gives a V1-vs-V2 comparison.

## 10. Example 2: stunning HTML pages

Inspired by the "100 HTML files" experiments (e.g.
<https://miaai-lab.github.io/Claude-Fable-5.2-HTML-Files/>): ask a model for stunning, non-repetitive,
fully creative single-file HTML. The results include interactive toys, games, generative art, tools,
typography experiments and product pages. Output is visual, but **training is text-only**:

- TRL's `AsyncGRPOTrainer` rejects image data, and the capture trace has no `pixel_values`.
- So neither the teacher nor the student sees images during episodes, which keeps teacher traces and
  student rollouts identical in form.
- Only the judge looks at screenshots.

### Data

- **Concepts.** About 5k creative concepts (title, one-line idea, category), brainstormed by the teacher in
  batches under the experiment's rules. Near-duplicates are removed by text similarity, and categories are
  balanced.
- **Evaluation concepts.** Held-out concepts form validation and test.
- **Headline evaluation: the "100 HTML files" protocol.** One run produces 100 files under the original
  rules, published as a gallery that sits next to the Fable 5.2 and Opus 5.5 galleries.
- **Storage.** A private dataset `nielsr/html-concepts`, made public after review.

### Episode

- **Harness:** Pi.
- **Tools:** `read`, `write`, `edit`, `bash`, plus the `preview` CLI in the workspace.
- **Answer:** `FileAnswer("index.html")`, submitted through the Pi `submit` tool.
- **`preview`** drives headless Chromium (Playwright) and returns **text only**:
  - console errors, at load and during a scripted interaction of about 5 s (clicks, key presses, mouse
    movement);
  - whether the page is blank or near-uniform;
  - whether anything animates (frames differ);
  - horizontal overflow at 390 px and 1440 px;
  - any network requests.

### Rubrics

- **Gates** ("zero broken files", made measurable): no console errors during load or the interaction, not
  blank, no network requests, a single self-contained file, no horizontal overflow.
- **Judge (screenshots).** Desktop screenshots before and after the interaction, plus one mobile screenshot.
  Scores: visual quality, creativity, fit to the concept, and evidence of interactivity.
- **Programmatic metrics:** accessibility basics (axe-core contrast and alt text) and file size.
- **Process metric:** whether the agent previewed before submitting. Recorded only, not part of the reward.
- **Evaluation-only metrics:**
  - gate pass rate, i.e. the broken-file rate;
  - pairwise win rate against the base model and against the teacher, with left/right position randomized;
  - diversity across the 100-file run, as the spread of screenshot and DOM embeddings.

### Judge

- **Primary judge, for the training reward and for evaluation:** Gemini 4 Neon through the Gemini API, with
  Carbon as fallback.
  - Its key (`GEMINI_API_KEY`) is visible only to the judge client.
  - Status on 2026-11-23: the key is valid, but the project's prepaid credits are exhausted (HTTP 402).
    Topping up is pending.
- **Calibration:** Gemini's preferences are checked against about 100 pairs labelled by people, through a
  blind A/B page, before any GRPO spend.
- **Second evaluation judge:** Qwen3.8-27B on the HF router. It is also the training judge if Gemini stays
  unavailable.
- **Disagreement check:** large disagreement between the two judges is reported, because it suggests the
  student is exploiting one judge's preferences.
- **Terms of use:** we train only on scalar judge scores, not judge text. Gemini's terms should still be
  checked for this use.

Lessons from [pelican-env](https://huggingface.co/blog/sergiopaniego/pelican-env-openenv), which ran GRPO on
SVG drawings judged by a vision model:

| Its problem | What we do |
|---|---|
| The deterministic structure checks were gamed. | We use gates only, not shaped geometric rewards. |
| With the judge on, there was no learning signal: base-model judge scores were almost always 0. | SFT comes first, then the difficulty probe; we measure how often base and SFT scores are non-zero before any GRPO spend. |
| Scores moved as much with the judge choice as with the model being evaluated. | Two judges, plus the pairwise win rate as the headline metric. |

### Models

- **Teacher.** A baseline on about 100 validation concepts compares GLM-5.3 and Kimi-K3.1-Pro, both run
  without images. The teacher with the higher Gemini win rate and gate pass rate is chosen, weighing in cost
  per page. Traces then come from that one teacher only.
- **Students.** Qwen3.5-9B is the main student. Qwen3.5-4B is added if compute allows, to show the effect
  of student size.

### Deliverables

- A public gallery Space (base | SFT | GRPO | teacher on held-out concepts, live pages and screenshots).
- The 100-file gallery.
- A blind A/B voting page, which doubles as the human check on the judge.

### Later

- **Vision students.** Once TRL supports image observations, the agent can see its own renders. The
  feasibility findings are already in: harnesses pass images through, vLLM serves Qwen3.5 with its vision
  encoder alongside a language-model-only LoRA, and Kimi-K3.1-Pro, GLM-5.3-Flash and Qwen3.8-27B accept
  images on the router; GLM-5.3 does not.
- **OpenCode V2 as a second harness** for the same task: the first per-harness LoRA showcase.

## 11. Porting from url-extraction

**Port almost unchanged**

- `credentials.py`
- `hooks.py`
- `record_line`
- the credential scan in `publish`
- the McNemar and summary code in `report.py`
- `rollout_chains` and the `simple` GRPO backend

**Port and generalize**

| url-extraction | Becomes |
|---|---|
| `opencode.py` | the `opencode@1.2.4` adapter |
| `trace_export.py` | the Episode-to-Pi exporter |
| `profiles.py` | a profile loader without URL fields |
| `difficulty_probe.py` | the generic `probe` stage, using `passed` |
| `train_sft.py`, `train_grpo_harness.py` | stage scripts that take a task and an adapter; `OpenCodeCaptureSession` becomes the adapters' capture factories |
| `select.py` | a dataset filter in the task |

**Rewrite**

- `pipeline.py` becomes stages plus jobs, keeping the same flags.
- `report.rescore` becomes a generic rescorer driven by the rubric set.

**Move to `examples/paper-urls/`**

- the system prompt
- `URLFindings`
- `canonical_url`
- the research protocol
- the shortcut guard
- the tool-efficiency bonus
- label-quality and tier logic

**Stay only in url-extraction**

- the demo Space
- the production batch runner
- endpoint deployment
- MLX conversion
- paper mining
- label validation

## 12. Related work and positioning

**Prime Intellect `verifiers`.** We do not depend on it.

- The classic `Rubric`, `JudgeRubric` and `RubricGroup` were moved to `verifiers.legacy` in 0.3 and removed
  in 0.4.
- The v1 API (`Taskset`, `Harness`, `@vf.reward`) binds rewards to its own `Task` and `Trace`, and it
  changes often (several dev builds a day).
- It also pulls in a heavy dependency tree with Prime-specific packages.

We follow the classic calling conventions and ship a shim. Its v1 `pi`, `opencode` and interception-proxy
code are useful reference designs.

**Thinking Machines Tinker and tinker-cookbook.** We do not build on it either.

- Its `Env` is token-level, and the trainer drives the loop.
- Training runs on the paid Tinker service, not TRL on HF Jobs.
- Its `capture/proxy` is a debugging tool, and `recipes/verifiers_rl` monkeypatches `do_group_rollout`.

We accept its `RewardFn` signature and judge-rubric shape, and record per-call token ids and logprobs so
that a `TrajectoryGroup` exporter is a plain conversion, to be built if someone asks.

**What agent-distill is:** an unmodified harness, the OpenEnv capture proxy and TRL on HF Jobs, with one
rubric set for filtering, reward and evaluation, and traces viewable on the Hub. The
[harness-adapters post](https://huggingface.co/blog/harness-hub/harness-adapters) is the multi-harness
direction: one LoRA per harness on a shared base, served by name rather than merged.

## 13. Testing and CI

- **CI (GitHub Actions on the private repo):** `ruff` plus `pytest`. No GPU, no Hub writes, no network.
- **Unit tests:**
  - rubric injection and combinators;
  - tool-kind normalization;
  - the Pi-session exporter (golden files);
  - the V1 reader;
  - credential redaction and scan;
  - budget estimation;
  - stage job construction (dry run).
- **Parity tests**, the generalization of `tests/test_parity.py`:
  - recorded episodes are replayed through each adapter's `run_episode` against fake OpenCode 1.2.4,
    OpenCode 2.3.2 and Pi servers, and through the GRPO capture session's reward;
  - they must agree except for tagged rubrics;
  - they include the paper-urls rescore fixture built from published records (Tier 1).
- **`distill smoke`:** a 2-episode end-to-end run on HF Jobs, on request only.

## 14. Milestones

Each milestone ends with a PR and a short report. GPU spend is first estimated with `--dry-run` and the
budget pilot.

| # | Milestone | Done when |
|---|---|---|
| M0 | **Skeleton:** layout, `pyproject`, CLI stub, CI, AGENTS.md (Hub, billing, credential and branch rules from url-extraction), this design doc | CI green on an empty test suite |
| M1 | **Core and Tier 1 parity (CPU only):** `Task`, `Episode`/`ToolEvent` kinds, rubrics, `RubricSet` and combinators, compatibility shims, credentials, Pi exporter, `opencode_v1_reader`, the `examples/paper-urls` task and rubrics, `distill rescore` | Every published url-extraction record rescores with zero per-episode differences |
| M2 | **Jobs, OpenCode 1.2.4 and evaluation:** profiles, wheel staging, launcher, manifests, budget guard, `opencode@1.2.4` adapter, `eval` and `publish` stages | Tier 2 passes: Qwen3.5-9B base and the Kimi GRPO model within noise of 75.0% and 82.1% in V1 |
| M3 | **Training stages:** `traces`, `sft`, `merge`, `probe`, `grpo` (TRL and simple backends, V1 capture factory), `distill smoke` | A smoke run (a few traces, SFT for a few steps, 2 GRPO steps, a 5-paper eval) passes end to end on HF Jobs. **First release.** |
| M4 | **Pi and HTML task infrastructure:** `pi@1.6.0` adapter and extension, Pi capture factory, `preview` CLI, HTML gates and judge rubrics (Gemini plus Qwen3.8-27B), concept generation, blind A/B calibration page, teacher baseline (GLM-5.3 vs Kimi-K3.1-Pro on about 100 concepts) | Teacher chosen; judge calibrated on about 100 human pairs; non-zero judge rate measured for base and teacher |
| M5 | **HTML distillation run:** traces → SFT → probe → GRPO → eval for Qwen3.5-9B (and 4B if compute allows), "100 HTML files" run, gallery Space | Published models, evaluation, sessions and galleries; results in `examples/html-pages/README.md` |
| M6 | **OpenCode 2.3.2:** port the V2 runner once the url-extraction branch merges; V2 capture | Tier 3 passes; optionally HTML pages in OpenCode V2 as the second-harness LoRA |
| M7 | **Later:** rubric drafting (`draft`, `calibrate`, review gate), the full-recipe parity re-run (Tier 4), Tinker/verifiers exporters if requested, vision students once TRL supports them | As scoped then |

## 15. Open items

- **Gemini credits.** Top up the project's prepaid credits, then re-run the one-request check. Until then,
  Qwen3.8-27B is the fallback judge.
- **Terms of use.** Check Gemini's terms for training on scalar judge scores.
- **Unverified on a GPU.** The vLLM test of vision plus a language-model-only LoRA was not run (the job
  launch was denied). It is not needed until vision students.
- **OpenCode V2 runner.** Waiting for url-extraction `claude/opencode-v2-glm53-4b` to merge (M6).
- **Provider routing.** Router responses for image models reported `x-inference-provider: openrouter`.
  Check billing and pin providers in teacher and judge profiles where it matters.
