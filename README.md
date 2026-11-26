# agent-distill

Distill frontier agents running inside real, unmodified agent harnesses (**Pi**, **OpenCode**) into small fine-tuned open models using **TRL**, **OpenEnv**, and **Hugging Face Jobs**.

## How it works

You define a task in Python (dataset, prompt, answer contract, rubrics) and a run in TOML (teacher, student, pinned harness, budget). `agent-distill` runs the full distillation loop on Hugging Face Jobs:

```
traces  ->  [replay / annotate]  ->  sft  ->  merge  ->  [probe -> grpo]  ->  eval  ->  publish
```

1. **Traces:** A frontier teacher solves training tasks inside the pinned harness. Only episodes accepted by the task's rubrics are kept.
2. **SFT:** A small open student is LoRA-finetuned on accepted trajectories. For reasoning students (`qwen3.5-4b-think`), the default **cross-format SFT** recipe ([CoT-Proxy](https://arxiv.org/abs/2611.33148)) trains on teacher actions with empty think blocks and out-of-domain KL replay, then lets the student think at inference.
3. **GRPO (optional):** A difficulty probe (`probe`) selects prompts the SFT model solves inconsistently, then `grpo` trains the student directly inside the unmodified harness with the same rubric reward via OpenEnv capture.
4. **Eval & publish:** Base, SFT, GRPO, and teacher models are evaluated in the same pinned harness. `publish` scans every dataset for credentials, makes artifacts public, and adds them to a Hub collection with Pi-format traces for the [agent-traces viewer](https://huggingface.co/docs/hub/en/agent-traces). <!-- pragma: allowlist secret -->

**Core invariant:** One rubric set per task drives trace acceptance, GRPO reward, and evaluation (only explicitly tagged `train_only` / `eval_only` rubrics may differ), and every stage runs against the same pinned `harness@version`.

## Headline results

Distilling **GLM-5.3** into **Qwen3.5-4B** inside `pi@1.6.0` on multi-hop Hugging Face Hub questions (`nielsr/hf-hub-facts-v2`), comparing two tool surfaces — the HF MCP server vs. the `hf` CLI + skill:

| Model | Thinking | HF MCP (377 test) | `hf` CLI (150 test) |
| --- | :---: | ---: | ---: |
| Qwen3.5-4B (base) | off | 72.1% | 47.3% |
| + Action-only SFT | off | 81.4% | 78.0% |
| + SFT + GRPO (100 steps) | off | 96.0% | — |
| Qwen3.5-4B (base) | on | 89.1% | 86.7% |
| **+ Cross-format SFT (recipe a)** | **on** | **98.1%** | **96.7%** |
| GLM-5.3 (teacher, first record) | on | 96.3% | 98.7% |

- **Cross-format SFT** (`qwen3.5-4b-think`, ~600 traces + KL replay) matches the frontier teacher on both surfaces for ~$13 in GPU spend, without needing a difficulty probe or GRPO.
- **Non-thinking SFT + GRPO** on MCP climbs from 72.1% to 96.0% while cutting tool calls from 6.13 to 1.75 per episode (fewer than the teacher's 3.15).

All published models, evaluations, and viewable traces live in the [agent-distillation Hub collection](https://huggingface.co/collections/nielsr/agent-distillation-6b0445568a39b1d979c52e31). <!-- pragma: allowlist secret -->

## Examples

| Example | Harness | Tool surface | Highlights |
| --- | --- | --- | --- |
| [`hf-hub-qa-mcp`](examples/hf-hub-qa-mcp/README.md) | `pi@1.6.0` | HF MCP server (3 read-only tools) | 98.1% with cross-format SFT; 96.0% with non-thinking SFT + GRPO |
| [`hf-hub-qa`](examples/hf-hub-qa/README.md) | `pi@1.6.0` | `bash`, `read`, pinned `hf` CLI + skill | CLI twin of `hf-hub-qa-mcp`; 96.7% cross-format SFT, 98.7% curated full-CoT SFT |
| [`paper-urls`](examples/paper-urls/README.md) | `opencode@1.2.4`, `2.3.2` | `shell`, `fetch`, `read`, `search` | Parity port of [url-extraction](https://github.com/NielsRogge/url-extraction) (7,241 records, 0 diffs) |
| [`html-pages`](examples/html-pages/README.md) | `pi@1.6.0` | File tools + `./preview` (Chromium) | Paused experiment with browser gates and vision-judge calibration |

See [examples/README.md](examples/README.md) for a side-by-side overview and CLI-vs-MCP findings.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run distill --help
```

Preview any run without launching jobs or spending money (`--dry-run` validates configs, prints every HF Job command, and checks `[budget]` caps):

```bash
# Recommended cross-format SFT run (thinking student)
uv run distill run examples/hf-hub-qa-mcp/run-think.toml --dry-run

# Run a subset of stages
uv run distill run examples/hf-hub-qa/run-think.toml --stages eval --splits test --dry-run

# End-to-end 2-step smoke run on HF Jobs
uv run distill smoke examples/paper-urls/run-smoke.toml --dry-run

# Rescore published episode records locally (CPU only)
uv run --extra hub distill rescore --task examples/paper-urls --repo nielsr/qwen35-9b-url-agent-eval

# Scan for credentials and publish a run's datasets
uv run distill run examples/hf-hub-qa-mcp/run-think.toml --stages publish
```

## Defining a task and run

A **task** is a Python module exporting `task = Task(...)` with rubrics written as plain functions:

```python
from agent_distill.rubrics import RubricSet, gate, programmatic
from agent_distill.task import DatasetSpec, Limits, Task, TextAnswer

@programmatic
def answered(answer) -> float:
    return 1.0 if answer else 0.0

@programmatic
def exact_match(row, answer) -> float:
    return 1.0 if row["gold"] in str(answer) else 0.0

RUBRICS = RubricSet(
    rubrics=[answered, exact_match],
    reward=gate("answered", "exact_match"),
    passed=lambda s: s["answered"] == 1.0 and s["exact_match"] == 1.0,
)

task = Task(
    name="my-task",
    dataset=DatasetSpec(repo="org/my-dataset", splits={"train": "train", "eval": ["test"]}),
    prompt=lambda row: row["question"],
    answer=TextAnswer(),  # or a Pydantic BaseModel, or FileAnswer("index.html")
    rubrics=RUBRICS,
    row_id=lambda row: str(row["id"]),
    tools=("shell", "read"),
    limits=Limits(timeout=300, max_tool_calls=15),
)
```

A **run** pairs that task with built-in (or local) TOML profiles for the teacher, student, harness, and optional judge:

```toml
task = "."
teacher = "glm-5.3"
student = "qwen3.5-4b-think"
harness = "pi@1.6.0"
stages = ["traces", "replay", "sft", "eval", "publish"]

[hub]
namespace = "your-user"
bill_to = "your-org"

[budget]
traces_usd = 50
gpu_usd = 60
```

Built-in profiles live in [`agent_distill/profiles/`](agent_distill/profiles/):

- **Harnesses:** `pi@1.6.0`, `opencode@1.2.4`, `opencode@2.3.2`
- **Teachers:** `glm-5.3`, `glm-5.2`, `kimi-k3.1-pro`, `inkling-2-small`
- **Students:** `qwen3.5-4b-think` (default cross-format SFT), `qwen3.5-4b-think-fullcot`, `qwen3.5-4b-nothink`, `qwen3.5-4b`, `qwen3.5-9b-nothink`, `qwen3.5-9b`, `inkling-2-mini`
- **Judges:** `gemini-4-hydrogen`, `gemini-4-neon`, `gemini-4-carbon`, `qwen3.8-27b`

## Safety and reproducibility

- **No credentials in agent environments:** Teachers are reached through a local key-injecting proxy; students get no keys; Pi episodes run under isolated unprivileged UIDs.
- **Private-first datasets:** Episode datasets are created private and only made public by `--stages publish` after passing a credential scan. Raw harness logs stay private.
- **Pinned everything:** Harness versions, MCP tool schemas, job dependencies (`uv.lock` with hashes), and dataset snapshots are pinned and recorded in every manifest and session header.
- **Budget guards:** Every stage checks worst-case cost against `[budget]` before launching, and stops cleanly at USD caps during teacher or judge runs.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — full architecture, rubric API, harness adapters, and cross-format SFT (CoT-proxy) details.
- [examples/README.md](examples/README.md) — overview and results across all examples.
- [spaces/race/README.md](spaces/race/README.md) — the race Space: a teacher and its distilled student on the same question in the same harness, configured by run configs.
- [AGENTS.md](AGENTS.md) — repository rules for coding agents.
