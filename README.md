# agent-distill

Distill frontier agents running in real agent harnesses (OpenCode, Pi) into small fine-tuned open models,
with TRL, OpenEnv and Hugging Face Jobs.

A task (dataset, prompt, answer contract, rubrics) is run by a frontier teacher inside a pinned harness; the
traces its rubrics accept become SFT data for a small open student, which then trains with GRPO in the same
unmodified harness against the same rubrics and is evaluated there. Design and roadmap:
[docs/DESIGN.md](docs/DESIGN.md). Agent rules: [AGENTS.md](AGENTS.md).

Status: M1. Core types, rubrics, the OpenCode 1.2.4 log reader, the Pi-session exporter, `distill rescore` and
the [paper-urls example](examples/paper-urls/) work; parity tier 1 with url-extraction passes. No harness runs
episodes yet (M2).

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
uv run distill --help
```
