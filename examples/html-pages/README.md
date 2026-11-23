# Example 2: stunning HTML pages

Distil a frontier agent that makes stunning, self-contained, interactive single-file HTML pages into a small open
model, in the Pi harness (`pi@1.6.0`). The design is in [docs/DESIGN.md](../../docs/DESIGN.md) §10.

## The task

- **Concepts:** `nielsr/html-concepts` (private until reviewed). The teacher brainstorms about 5k concepts in 12
  categories (`concepts.py`), and near-duplicates are removed. Per category, 5 concepts are held out for validation
  (teacher choice, judge calibration) and 10 for test.
- **Episode:** Pi with `bash`, `read`, `edit` and `write`, plus `./preview` in the workspace.
  - The agent never sees the page.
  - `./preview` loads `index.html` in headless Chromium, interacts with it for about 5 s, and reports what is
    broken as text: errors, network requests, blank rendering, animation, response to interaction, and overflow at
    390 px and 1440 px.
  - The agent submits `index.html` with Pi's `submit` tool.
- **Rubrics** (`rubrics.py`):
  - **`page_ok` (gate):** the same inspection as `./preview`: the page loads, has no errors, is not blank, makes
    no network requests and does not overflow.
  - **`page_quality` (judge):** a vision judge scores three screenshots (desktop, desktop after the interaction,
    phone) from 1 to 10 for visual quality, creativity, fit to the concept and interactivity, plus an overall score.
  - **Metrics, recorded only:** page size, basic accessibility, and whether the agent previewed before submitting.
  - **Reward and pass:** `reward = gate(page_ok, page_quality)`. An episode passes at an overall score of at least
    7/10 on a page that is not broken.
- **Judge:** Qwen3.8-27B on the HF router (`judge = "qwen3.8-27b"`). Gemini 4 Neon replaces it once its credits
  are topped up and it is calibrated against human labels.

## Running it

Chromium for `./preview`:

- in jobs, the task's `job_setup` installs it (`playwright install --with-deps chromium`);
- locally, point `AGENT_DISTILL_CHROMIUM` at a Chromium binary.

```bash
# Concepts (teacher spend capped by --budget-usd)
uv run python examples/html-pages/concepts.py --per-category 420 --bill-to huggingface \
    --output concepts.jsonl --push nielsr/html-concepts

# Teacher baseline on the 60 validation concepts (one run per teacher)
distill run examples/html-pages/run-teacher-baseline.toml --dry-run
distill run examples/html-pages/run-teacher-baseline.toml --set teacher='"kimi-k3.1-pro"' --dry-run
```

## Results so far

**Budget pilot (2026-11-23):** 6 validation concepts per teacher, in Pi, judged by Qwen3.8-27B. Costs use the
most expensive router provider's prices.

| Teacher | Passed | Cost per page | Time per page | Output cut off |
|---|---|---|---|---|
| GLM-5.3 | 3/6 | ~$0.29 | 3.5–15 min | 3/6 (reasoning exceeded 32k tokens) |
| Kimi-K3.1-Pro | 5/6 | ~$0.89 | 4–9 min | 1/6 |

- **Output limit:** every cut-off episode was the model reasoning past the 32k output limit before its first tool
  call. Teachers now get 65k tokens.
- **Previewing:** every page that passed had been previewed before it was submitted.
- **Judge resolution:** the judge gave 7 of the 8 pages it scored 8/10, so it barely discriminates. Calibrating
  it against human labels is the next step.
