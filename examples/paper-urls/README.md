# Example: paper URL extraction

Given an arXiv paper, the agent finds its official GitHub repository and project page. This is the task of
[url-extraction](https://github.com/NielsRogge/url-extraction), ported to agent-distill so its published results
can be reproduced through the library (DESIGN.md §9).

| File | Contents |
|---|---|
| `task.py` | The `Task`: system prompt, user prompt, answer schema, dataset, limits (180 s, 15 tool calls) |
| `schema.py` | `URLFindings` (strict `^https?://` URLs, quoted null normalized) and the row view |
| `rubrics.py` | URL correctness (canonical URLs, reference aliases), research protocol (process rubric on normalized tool kinds), shortcut guard (`train_only`), tool-efficiency bonus; the reward and `passed` |
| `system_prompt.txt` | Identical to url-extraction's `harness/system_prompt.txt` |
| `parity/tier1.py` | Parity tier 1: both scorers on every published record |

Ported from url-extraction commit `9fb9a39` (`harness/task.py`, `harness/scoring.py`).

## Rubrics

```
reward = bonus_if(tool_efficiency, penalize(protocol, 0.5, gate(no_shortcut, url_correctness)), when reward >= 1)
passed = all_fields_correct and protocol
```

- `url_correctness`: fraction of verified requested fields that are correct (labels `gold`, `silver`,
  `verified_null`); no answer counts every verified field wrong; a row without verified fields is unscorable.
- `protocol`: the answer exists, the research protocol of the system prompt was followed (metadata inspected,
  paper read when a field stays unresolved, no project-page fetch after the GitHub URL is known, no masked
  `hf papers` exit status), and the tool-call limit was not reached.
- `no_shortcut` (`train_only`): looking the labels up on the Hub scores 0 during RL; evaluation still scores the
  answer and reports the violation.
- `tool_efficiency` (metric): `0.1 * (1 - tool_calls / 15)`, added only to a fully correct, complete episode.

## Data

- Labels: [`nielsr/paper-url-extraction-v1`](https://huggingface.co/datasets/nielsr/paper-url-extraction-v1),
  validated variant `nielsr/paper-url-extraction-v1-validated`, harder `test_hard` split in
  `nielsr/paper-url-extraction-v2`.
- Published traces and evaluations: the
  [paper URL agent collection](https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e).

## Parity with url-extraction

| Tier | Check | Status |
|---|---|---|
| 1 | Every published url-extraction record rescored by both scorers: same answer, per-field verification, completeness and reason, success criterion, and GRPO reward | **Passed** (2026-11-23): 7,241 episodes in 35 files of 7 repos, 0 differences |
| 2 | `opencode@1.2.4` re-evaluation of Qwen3.5-9B base and `qwen35-9b-url-agent-grpo-merged-kimi` within noise of 75.0% / 82.1% | M2 |
| 3 | OpenCode 2.x cross-check against url-extraction's V2 experiment | M6 |
| 4 | Full recipe re-run | M7 |

Tier 1 covers every evaluation (Qwen3.5-4B v1/v2, Qwen3.5-9B, Inkling-2-Mini incl. `test_hard`), every teacher
baseline and every trace file (GLM-5.2, Kimi-K3.1-Pro, Inkling-2-Small incl. rejected attempts), with the
tool-call limit each ran with (20 for teacher trace generation, 15 otherwise). The accepted-trace counts come out as
published (102, 873, 815, 1,215), and `distill rescore` reproduces the README numbers, e.g. Qwen3.5-9B on test:
base 75.0%, SFT 85.7%, SFT + GRPO 82.1% field accuracy with 6.8 / 3.8 / 3.0 tool calls.

Run tier 1 (CPU; downloads about 1.1 GB of public records):

```bash
git clone https://github.com/NielsRogge/url-extraction ../url-extraction
uv run --extra hub --with httpx python examples/paper-urls/parity/tier1.py --url-extraction ../url-extraction
```

Rescore published records with agent-distill alone:

```bash
uv run --extra hub distill rescore --task examples/paper-urls --repo nielsr/qwen35-9b-url-agent-eval
```
