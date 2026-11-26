# Examples

Each example is a self-contained distillation task with its own `task.py`, `rubrics.py`, `run*.toml` configs, and results.

| Example | Harness | Status | What it demonstrates | Headline result (Qwen3.5-4B / 9B) |
| --- | --- | --- | --- | --- |
| [`hf-hub-qa-mcp/`](hf-hub-qa-mcp/README.md) | `pi@1.6.0` | Done | Hub Q&A using 3 read-only tools on the anonymous HF MCP server; deterministic rubrics + cross-format SFT and GRPO | 4B: 72.1% base (off) → **96.0%** SFT+GRPO; **98.1%** cross-format SFT (thinking on) vs 96.3% teacher (first record) |
| [`hf-hub-qa/`](hf-hub-qa/README.md) | `pi@1.6.0` | Done | CLI twin of `hf-hub-qa-mcp` using `bash`, `read`, pinned `hf` CLI, and the `hf-cli` Agent Skill | 4B: 47.3% base (off) → 78.0% SFT; **96.7%** cross-format SFT / **98.7%** curated full-CoT SFT (thinking on) vs 98.7% teacher |
| [`paper-urls/`](paper-urls/README.md) | `opencode@1.2.4`, `2.3.2` | Tier 1 & 2 pass | Port of [url-extraction](https://github.com/NielsRogge/url-extraction) (arXiv → GitHub & project URLs); structured Pydantic output + process rubrics | 7,241 published records rescored with 0 diffs; 9B GRPO 86.1% vs 79.8% base (paired p = 0.0001) |
| [`html-pages/`](html-pages/README.md) | `pi@1.6.0` | Paused | Interactive single-file HTML pages with headless Chromium gates (`./preview`) and a motion-aware vision judge | Documented teacher baseline, M5 SFT attempt, and 50-pair human judge calibration |

## CLI vs. MCP (`hf-hub-qa` vs. `hf-hub-qa-mcp`)

The two Hub Q&A examples share the exact same dataset (`nielsr/hf-hub-facts-v2`), teacher (**GLM-5.3**), student (**Qwen3.5-4B**), harness (`pi@1.6.0`), and verifier, isolating the effect of the tool interface:

- **Untrained small models strongly prefer constrained MCP tools over a shell:** with thinking off on the first 150 test questions, base Qwen3.5-4B scores **73.3%** via MCP vs. **47.3%** via CLI (where it ignores the skill and writes brittle `curl`/Python scrapers).
- **SFT closes the gap:** action-only SFT teaches the CLI model to read the skill and use `hf`, bringing both non-thinking students to ~80% (78.0% CLI vs. 80.0% MCP).
- **Cross-format SFT with thinking matches the teacher on both:** using `qwen3.5-4b-think` (`run-think.toml`) reaches **98.1%** on MCP (377 test rows) and **96.7%** on CLI (150 test rows) without GRPO.

## Structure of an example

```
examples/<name>/
  task.py          Task definition (dataset, prompts, answer contract, tools, limits)
  rubrics.py       Deterministic, process, or judge rubrics + reward and pass rules
  system_prompt.txt
  run*.toml        Run configurations (recommended route, baselines, pilots, smoke)
  README.md        Task details, commands, and full evaluation results
```

Preview any example without launching jobs:

```bash
uv run distill run examples/hf-hub-qa-mcp/run-think.toml --dry-run
uv run distill run examples/hf-hub-qa/run-think.toml --dry-run
uv run distill run examples/paper-urls/run-smoke.toml --dry-run
```

For full architecture and design notes, see [docs/DESIGN.md](../docs/DESIGN.md).
