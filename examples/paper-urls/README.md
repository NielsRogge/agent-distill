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
| `parity/make_fixture.py` | Builds `tests/fixtures/paper_urls_published.jsonl`, the tier-1 CI fixture of real published records |

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
- url-extraction's published traces and evaluations: the
  [paper URL agent collection](https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e).
- agent-distill's evaluations: the [agent-distillation collection](https://huggingface.co/collections/nielsr/agent-distillation-6b0445568a39b1d979c52e31)
  (parity tier 2: `nielsr/qwen35-9b-paper-urls-parity-opencode-124-eval`; 2.3.2 smoke test:
  `nielsr/qwen35-9b-paper-urls-smoke-opencode-232-eval`).

## Parity with url-extraction

| Tier | Check | Status |
|---|---|---|
| 1 | Every published url-extraction record rescored by both scorers: same answer, per-field verification, completeness and reason, success criterion, and GRPO reward | **Passed** (2026-11-23): 7,241 episodes in 35 files of 7 repos, 0 differences |
| 2 | `opencode@1.2.4` re-evaluation of Qwen3.5-9B base and `qwen35-9b-url-agent-grpo-merged-kimi` within noise of 75.0% / 82.1%, GRPO ahead of base on the paired papers | **Passed** (2026-11-25, mean of 3 runs in `url-eval-` workspaces): 79.8% (+4.8) / 86.1% (+4.0); GRPO-only passes 27 vs base-only 5 (sign test p = 0.0001). See below |
| 3 | OpenCode 2.x cross-check against url-extraction's V2 experiment | M6 |
| 4 | Full recipe re-run | M7 |

Tier 1 covers every evaluation (Qwen3.5-4B v1/v2, Qwen3.5-9B, Inkling-2-Mini incl. `test_hard`), every teacher
baseline and every trace file (GLM-5.2, Kimi-K3.1-Pro, Inkling-2-Small incl. rejected attempts), with the
tool-call limit each ran with (20 for teacher trace generation, 15 otherwise). The accepted-trace counts come out as
published (102, 873, 815, 1,215), and `distill rescore` reproduces the README numbers, e.g. Qwen3.5-9B on test:
base 75.0%, SFT 85.7%, SFT + GRPO 82.1% field accuracy with 6.8 / 3.8 / 3.0 tool calls.

In CI, `tests/test_parity_fixture.py` rescores 12 real published records (every outcome: passed, wrong field,
incomplete, no answer, tool-call limit; limits 15 and 20; `test_hard` labels from v2) and checks them against the
verdicts *stored in the records* (`verification.fields`, `attempt_complete`, `accepted`). When the fixture was
built, the stored verdicts agreed with both scorers on 443 of 445 records of its five source files; the 2 others
(`teacher-test-kimi-k31.jsonl`) were written by an earlier url-extraction scorer revision, and url-extraction's own
current rescoring agrees with ours on them. Tier 1 therefore compares against url-extraction's scorer at the
revision the port is taken from (`9fb9a39`), not against every stored verdict.

`distill rescore` picks each record's tool-call limit (from the record, else the task's `record_limits`: 20 for
`train*`/`distill*` trace files, 15 otherwise), reads labels with v1 first and v2 only for papers v1 lacks (as
tier 1), and prints the per-field URL accuracy pooled over all fields (the published `field_accuracy`) next to the
per-episode mean.

Run tier 1 (CPU; downloads about 1.1 GB of public records):

```bash
git clone https://github.com/NielsRogge/url-extraction ../url-extraction
uv run --extra hub --with httpx python examples/paper-urls/parity/tier1.py --url-extraction ../url-extraction
```

### Tier 2

`distill run examples/paper-urls/run.toml --stages eval` (two H200 jobs, about 30 minutes each), then
`python examples/paper-urls/parity/tier2.py --repo nielsr/qwen35-9b-paper-urls-parity-opencode-124-eval`. Test
split, 42 fully scoreable papers (84 fields), OpenCode 1.2.4, 180 s, 15 tool calls, concurrency 8, vLLM 0.33.0:

| Model | agent-distill | url-extraction | Tool calls (agent-distill / url-extraction) |
|---|---:|---:|---:|
| Qwen3.5-9B base | 78.6% | 75.0% | 4.5 / 6.8 |
| Qwen3.5-9B SFT + GRPO (Kimi-K3.1-Pro traces; merged checkpoint) | 84.5% | 82.1% | 3.4 / 3.0 |

Both are within the ±5-point tolerance. GRPO vs base: 9 papers solved only by GRPO, 2 only by base (McNemar
p = 0.065). The base model uses fewer tool calls than in url-extraction's run (4.5 vs 6.8); url-extraction's evaluator
did not pin vLLM, so the serving stack differs (here vLLM 0.33.0).

#### Repeats and the `workdir` stall (2026-11-25)

Five more runs per model (one re-run, three repeats, one GRPO-only) put run-to-run noise at about ±3 points (SD) and
showed the GRPO model ~6.5 points under 82.1% on average (75.0%), while base stayed within tolerance (77.7%). The
gap came from episodes that stalled on their first call until the 180 s limit: the GRPO model sometimes passes
`"workdir": "/tmp/url-traces-<random>"`, a directory it memorized from url-extraction's trace-generation workspaces,
and OpenCode 1.2.4 never spawns a command whose `workdir` is missing (no error, the tool stays "running"). All 17
stalls in 7 runs were such calls; none of the 58 calls naming the real workspace stalled; base never names one.
url-extraction's evaluation ran in `/tmp/url-eval-*`, which OpenCode shows the agent, and its GRPO run named that
directory correctly. Without the stalled episodes GRPO averaged 80.0%.

The fix: the task's `workspace_prefix = "url-eval-"` (the agent now sees url-extraction's workspace name), and an
OpenCode plugin (`harnesses/opencode_plugins/workdir-guard.js`, on by default, off in this parity run) that fails
such a call with `workdir does not exist: ...` instead of letting it hang. OpenCode 2.3.2 already refuses
directories outside the workspace (`external_directory: deny`).

Three runs per model with the fix (guard off, so only the workspace name changed), each 42 papers / 84 fields:

| Model | Run 1 | Run 2 | Run 3 | Mean ± SD | url-extraction |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B base | 79.8% | 81.0% | 78.6% | 79.8 ± 1.2% (+4.8) | 75.0% |
| Qwen3.5-9B SFT + GRPO | 82.1% | 86.9% | 89.3% | 86.1 ± 3.6% (+4.0) | 82.1% |

No `url-traces` directory was named (24 GRPO calls passed a `workdir`, all the real workspace), and 2 GRPO episodes
of 126 timed out (both on slow research, not stalls). On the paired papers GRPO is ahead: it alone passed 27,
base alone 5 (two-sided sign test p = 0.0001), and it had more correct fields on 23 papers, base on 9. Both means
sit inside the ±5-point band but near its top: this stack (vLLM 0.33.0) scores both models a few points above
url-extraction's unpinned one.

### OpenCode 2.3.2 smoke test

`distill run examples/paper-urls/run-opencode2.toml --stages eval`: Qwen3.5-9B base on 6 test papers ran end to end
on HF Jobs. 5 of 6 papers have every URL right, but only 1 episode follows the research protocol, and 8 of 33 tool
calls name tools that do not exist (e.g. a tool called `hf papers info 2510.00406 --format json`); in OpenCode 1.2.4
the same model made no such calls (0 of 189). A student trained in 1.2.4 is not a 2.3.2 student (DESIGN.md §6).

These smoke numbers came from the first 2.3.2 adapter, which ran all concurrent episodes on one shared OpenCode
server, workspace and data directory. url-extraction found that agents on a shared 2.x server read each other's
spilled shell output (and one `rm -rf` of it broke every episode beside it), so the adapter now starts a server
with its own workspace and XDG directories per episode. Re-run the smoke test before comparing against these
numbers.

Rescore published records with agent-distill alone:

```bash
uv run --extra hub distill rescore --task examples/paper-urls --repo nielsr/qwen35-9b-url-agent-eval
```
