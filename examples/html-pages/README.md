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
  - **`submitted` (gate):** the agent ended the episode with `submit`; a page left by a timed-out or tool-limited
    episode does not pass.
  - **`page_quality` (judge):** a vision judge watches the page in motion: an 8-frame filmstrip of a scripted
    ~10 s visit (wait, click, drag, controls, keys, scroll), the phone view and an activity log (sound, animation,
    input listeners, controls). It scores 1 to 10 for visual quality, creativity, fit to the concept and
    interactivity, plus an overall score. `page_quality_stills`, the earlier three-screenshot judge, is kept for
    comparison.
  - **Sandbox:** pages run in a browser with no files, no network (WebSockets included), no credentials and a
    120 s limit (`agent_distill/browser/sandbox.py`); the activity log comes from counters the page cannot write.
  - **Metrics, recorded only:** page size, basic accessibility, and whether the agent previewed before submitting.
  - **Reward and pass:** `reward = gate(submitted, gate(page_ok, page_quality))`. An episode passes when the agent
    submitted a page that is not broken and scores at least 7/10.
- **Judges:**
  - **Gemini 4 Hydrogen** through the Gemini API is the default (`judge = "gemini-4-hydrogen"`). Its key is read
    only by the judge client.
  - **Qwen3.8-27B** on the HF router is the second judge (`--judge qwen3.8-27b`).
  - Both are calibrated against human labels before any GRPO spend.

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
- **Judge resolution:** the judge (Qwen3.8-27B at the time) gave 7 of the 8 pages it scored 8/10, so it barely
  discriminates. Calibrating it against human labels is the next step.

**Teacher baseline (2026-11-23):** 60 validation concepts per teacher, in Pi, judged by Gemini 4 Hydrogen
(`run-teacher-baseline.toml`). Passed is the first episode per concept; reruns of episodes cut off by a timeout or
the output limit are in the records but not in this column.

| Teacher | Passed | Page broken | Judge scores (1–10) | Ended cleanly with `submit` | Cost | Median time |
|---|---|---|---|---|---|---|
| Kimi-K3.1-Pro | 56/60 | 4/60 | 7: 2, 8: 23, 9: 34 | 59/63 episodes | $52.67 | 7.0 min |
| GLM-5.3 | 42/60 | 16/60 | 5–6: 3, 7: 13, 8: 25, 9: 9 | 14/60 first episodes | $36.97 | 10.6 min |

- **Kimi-K3.1-Pro is the teacher.** It almost always previews, fixes and submits within the limits.
- **GLM-5.3 rarely finishes cleanly.** Of its first episodes, 20 hit the 15-minute timeout, 16 the tool-call limit
  and 10 the 65k output limit (reasoning before the first tool call). Many still left a good page on disk, which is
  graded, but those trajectories never submit, so they are poor SFT traces.
- **Judge resolution:** Hydrogen spreads GLM's pages over 5–9 but puts every Kimi page at 7–9, so it ranks broken or
  weak pages well and great pages poorly. Calibration against human labels comes next.
- Datasets: `nielsr/qwen35-9b-html-pages-kimi-k31-pro-pi-160-eval`, `nielsr/qwen35-9b-html-pages-glm-53-pi-160-eval`.

**Judge calibration (2026-11-24, labelling closed):** the user labelled 50 same-concept page pairs blind (Kimi vs
GLM; 36 with a preference, 14 about equal). Agreement is over the 36 decided pairs; a judge tie or an unscorable pair
counts half in the accuracy. Spearman is between the judge's difference and the label over all 50; the sign test
compares agreements with disagreements. Data and scripts: `calibration/` (`score.py` reproduces this table,
`fetch_pages.py` rebuilds the pages from the published records, `calibrate.py` and `calibrate_motion.py` re-run the
judges).

| Judge setup | Agree / disagree / tie or unscorable | Accuracy (of 36) | Spearman | Sign test p |
|---|---|---|---|---|
| Hydrogen, per page, 3 screenshots (the baseline judge) | 16 / 9 / 11 | 0.60 | 0.15 | 0.23 |
| Qwen3.8-27B, per page, 3 screenshots | 11 / 7 / 18 | 0.56 | 0.14 | 0.48 |
| Hydrogen, defect list + appeal | 14 / 15 / 7 | 0.49 | 0.04 | 1.00 |
| Hydrogen, pairwise, 3 screenshots each | 22 / 13 / 1 | 0.62 | 0.19 | 0.18 |
| **Hydrogen, per page, filmstrip + phone + activity log (current)** | 14 / 6 / 16 | 0.61 | 0.20 | 0.12 |
| Hydrogen, pairwise, filmstrip + phone + activity log | 20 / 12 / 4 | 0.61 | 0.19 | 0.22 |

- **No setup is significantly better than chance.** The current judge disagrees least often (6 of 36), mostly calling
  pairs even rather than picking the wrong page, but p = 0.12.
- **Not measured:** the user's self-agreement (1 of 10 repeat pairs labelled, with different verdicts) and agreement
  on different-concept pairs (2 of 50 same-category pairs labelled). So we do not know how close a judge could get.
- **The judge is not validated as a GRPO reward.** It reliably separates broken or weak pages from good ones (gates
  plus a 7/10 bar), which trace acceptance and the evaluation pass rate need. It barely ranks good pages against
  great ones, which is what a GRPO reward on top of SFT would push on. GRPO with it is an experiment the user must
  opt into; until then the run stops after SFT.
- The user preferred Kimi in 21 pairs and GLM in 15 (14 ties), but three of their four saved favourites were GLM
  pages the old judge had scored below the Kimi page beside them.

**Rescore with the motion-aware judge** (`distill rescore`, all episodes in the published records, reruns included):

| Teacher | Passed before → after | Judge scores before (1–10) | Judge scores after (1–10) | Mean `page_quality` |
|---|---|---|---|---|
| Kimi-K3.1-Pro | 59/63 → 58/63 | 7: 2, 8: 23, 9: 34 | 6: 1, 7: 7, 8: 28, 9: 23 | 0.838 → 0.804 |
| GLM-5.3 | 47/80 → 44/80 | 5: 1, 6: 2, 7: 13, 8: 25, 9: 9 | 5: 1, 6: 5, 7: 9, 8: 20, 9: 15 | 0.753 → 0.762 |

- The new judge is harsher on Kimi and kinder to GLM's best pages, so Kimi's lead in mean quality shrinks from 0.085
  to 0.042. Pass rates barely move (gates are unchanged; the 7/10 bar now fails one more Kimi page and three more GLM
  pages).
- Kimi stays the teacher: the gap that matters is in clean, submitted trajectories, not in judged quality.
