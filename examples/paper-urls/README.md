# paper-urls: arXiv paper URL extraction

Given an arXiv paper, find its official GitHub repository and project page. This example ports [url-extraction](https://github.com/NielsRogge/url-extraction) (commit `9fb9a39`) into `agent-distill` to verify exact scoring and evaluation parity ([docs/DESIGN.md](../../docs/DESIGN.md#9-example-1-paper-urls-parity-with-url-extraction)).

## Parity with `url-extraction`

| Tier | Check | Status |
| --- | --- | --- |
| 1 | Rescore every published `url-extraction` record with both scorers (answer, per-field verification, completeness, pass, GRPO reward) | **Passed** — 7,241 episodes across 35 files / 7 repos, **0 differences** |
| 2 | Re-evaluate Qwen3.5-9B base and `qwen35-9b-url-agent-grpo-merged-kimi` in `opencode@1.2.4` within ±5 pts of 75.0% / 82.1% | **Passed** — mean of 3 runs: **79.8% (+4.8)** / **86.1% (+4.0)**; GRPO beats base 27–5 paired (`p = 0.0001`) |
| 3 | `opencode@2.3.2` cross-check against `url-extraction` V2 models | Pending V2 publication (base smoke test completed) |
| 4 | Full end-to-end distillation re-run | Planned |

### Tier 2 evaluation results (`opencode@1.2.4`)

Three independent runs per model on 42 fully scoreable test papers (84 fields; 180 s, 15 tool calls, concurrency 8, vLLM 0.33.0, `workspace_prefix = "url-eval-"`):

| Model | Run 1 | Run 2 | Run 3 | Mean ± SD | `url-extraction` ref |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B (base) | 79.8% | 81.0% | 78.6% | 79.8 ± 1.2% | 75.0% |
| Qwen3.5-9B (SFT + GRPO, Kimi traces) | 82.1% | 86.9% | 89.3% | **86.1 ± 3.6%** | 82.1% |

On paired papers across all 3 runs, GRPO alone passed 27 vs. base alone 5 (two-sided sign test `p = 0.0001`), and scored more correct fields on 23 papers vs. 9. Both models sit near the top of the ±5-point band because vLLM 0.33.0 scores slightly higher than `url-extraction`'s unpinned serving stack.

## Task and rubrics

| File | Role |
| --- | --- |
| `task.py` | `Task` definition (prompts, `URLFindings` schema, dataset, 180 s / 15-call limits) |
| `schema.py` | `URLFindings` Pydantic model (strict `^https?://` URLs, normalized null handling) |
| `rubrics.py` | URL correctness, research protocol, shortcut guard (`train_only`), and tool-efficiency bonus |
| `system_prompt.txt` | Exact `url-extraction` prompt for OpenCode 1.2.4 (`system_prompt_v2.txt` for 2.3.2) |
| `parity/tier1.py`, `tier2.py` | Automated Tier 1 and Tier 2 parity checks |
| `parity/make_fixture.py` | Generates `tests/fixtures/paper_urls_published.jsonl` (12 real records checked in CI) |

### Reward and pass criterion

```text
reward = bonus_if(tool_efficiency, penalize(protocol, 0.5, gate(no_shortcut, url_correctness)), when reward >= 1)
passed = all_fields_correct and protocol
```

- **`url_correctness`:** fraction of verified requested fields (`gold`, `silver`, `verified_null`) matching canonical URLs / aliases; rows without verified fields are unscorable.
- **`protocol`:** answer present, research protocol followed (metadata checked, paper read when unresolved, no project-page fetch after GitHub URL is found, no masked `hf papers` exit status), and tool-call limit not hit.
- **`no_shortcut` (`train_only`):** looking up labels on the Hub scores 0 during RL; evaluation scores the answer and reports violations.
- **`tool_efficiency` (metric):** `0.1 * (1 - tool_calls / 15)`, added only when a fully correct episode passes protocol.

## Running it

```bash
# Tier 2 evaluation in OpenCode 1.2.4
uv run distill run examples/paper-urls/run.toml --stages eval --dry-run

# End-to-end smoke run in OpenCode 2.3.2 (traces -> sft -> merge -> probe -> grpo -> eval)
uv run distill run examples/paper-urls/run-smoke.toml --dry-run

# Base smoke evaluation in OpenCode 2.3.2
uv run distill run examples/paper-urls/run-opencode2.toml --stages eval --dry-run

# Rescore published records locally (CPU only)
uv run --extra hub distill rescore --task examples/paper-urls --repo nielsr/qwen35-9b-url-agent-eval

# Run full Tier 1 parity against url-extraction (~1.1 GB download, CPU only)
git clone https://github.com/NielsRogge/url-extraction ../url-extraction
uv run --extra hub --with httpx python examples/paper-urls/parity/tier1.py --url-extraction ../url-extraction
```

## Key findings

- **Memorized `workdir` stall (OpenCode 1.2.4):** early Tier 2 repeats showed the GRPO model stalling until timeout by passing `"workdir": "/tmp/url-traces-<random>"` (memorized from training workspaces), which hangs OpenCode 1.2.4 silently. Setting `workspace_prefix = "url-eval-"` eliminated all stalls; `agent_distill` also enables `harnesses/opencode_plugins/workdir-guard.js` by default to fail missing-`workdir` calls cleanly (disabled during parity runs).
- **Harnesses are not interchangeable (`opencode@2.3.2`):** a 6-paper base smoke test in OpenCode 2.3.2 got 5/6 URLs right, but only 1/6 followed protocol and 8/33 calls invented nonexistent tools (vs. 0/189 in 1.2.4). The 2.3.2 adapter now isolates each episode in its own server, workspace, and XDG directories to prevent cross-episode interference.
- **CI parity fixture:** `tests/test_parity_fixture.py` checks 12 real published records covering every outcome (`passed`, wrong field, incomplete, no answer, call limit, limits 15/20, and `test_hard`) on every push.

## Datasets and collections

- Labels: [`nielsr/paper-url-extraction-v1`](https://huggingface.co/datasets/nielsr/paper-url-extraction-v1), `nielsr/paper-url-extraction-v1-validated`, and `nielsr/paper-url-extraction-v2` (`test_hard`). <!-- pragma: allowlist secret -->
- `url-extraction` artifacts: [paper URL agent collection](https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e). <!-- pragma: allowlist secret -->
- `agent-distill` evaluations: [agent-distillation collection](https://huggingface.co/collections/agent-distillation-hf/agent-distillation-6b09e6966000e7910fd46633) (`agent-distillation-hf/qwen35-9b-paper-urls-parity-opencode-124-eval`, `agent-distillation-hf/qwen35-9b-paper-urls-smoke-opencode-232-eval`). <!-- pragma: allowlist secret -->
