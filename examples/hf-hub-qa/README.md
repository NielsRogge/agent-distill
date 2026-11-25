# Example 3: Hugging Face Hub questions with the `hf` CLI

Distil a frontier agent that answers questions about the Hugging Face Hub with the `hf` command-line tool into a
small open model, in the Pi harness (`pi@1.6.0`). It is the CLI twin of [hf-hub-qa-mcp](../hf-hub-qa-mcp/README.md),
which answers the same questions through the HF MCP server. Both use the same teacher and student (GLM-5.3 ->
Qwen3.5-4B), dataset and checker, so only the tool surface differs.

## The task

- **Agent:** Pi with `bash` and `read`.
  - The pinned `hf` CLI (huggingface_hub 2.4.0), logged out, so it can only read from the Hub.
  - The `hf-cli` Agent Skill generated from that same CLI version (`skills/hf-cli/SKILL.md`, loaded with `--skill`).
  - No submit tool: the agent's final message, in natural language, is the answer.
- **Main task** (`task.py`): the Hub facts in `nielsr/hf-hub-facts-v2` (1,162 train / 149 validation / 377 test,
  frozen 2026-11-25), shared with the MCP twin. Every question needs at least two lookups or a file listing:
  - a model's base model's license, pipeline tag or size (asked only when the base differs);
  - the title or first author of the one paper a model links;
  - the total size of three models;
  - a repo's GGUF or safetensors file count, across every folder.

  v1 (`nielsr/hf-hub-facts`, single lookups) was too easy: base Qwen3.5-4B passed 96% of its validation questions
  through MCP.
- **Second task** (`judged.py`): 51 open-ended questions in `nielsr/hf-hub-questions` (`questions.py`), such as
  "What are the top trending models?" or "What are some good papers on knowledge distillation?". These are the
  secondary evaluation of both twins.
- **Agent isolation:** every agent runs as its own unprivileged user (`agent_distill.harnesses.isolation`). It can
  read the pinned CLI and the skill, but not the job's token or the stage's records.

## Rubrics

- **Facts** (`rubrics.py`), deterministic, the MCP twin's rubrics with the CLI's tool gates:
  - Gates (any failure scores 0):
    - answered;
    - read-only: no command that writes to the Hub or logs in;
    - used the Hub: a successful `hf` call (an answer from memory fails).
  - `fact_correct`: the shared checker (`agent_distill.rubrics.facts`). If the Hub changed since the snapshot, the
    row is unscorable, not wrong.
  - `grounded`: every Hub id the answer names appears in its own tool output (`agent_distill.rubrics.grounding`).
  - Reward: the gates over `fact_correct` and `grounded`, plus a 0.1 tool-efficiency bonus on a pass.
- **Judged questions** (`judged_rubrics.py`):
  - the same gates and grounding;
  - a live reference for the 18 ranked or factual questions (the canonical `hf` command, run when scoring);
  - a Gemini 4 Hydrogen judge of answer quality (at least 7/10 to pass).

## Running it

```bash
distill run examples/hf-hub-qa/run.toml --dry-run              # traces -> SFT -> merge -> eval (base, SFT, teacher)
distill run examples/hf-hub-qa/run-judged-pilot.toml --dry-run # the judged questions (teacher pilot)
uv run --extra hub python examples/hf-hub-qa/questions.py      # the judged questions as JSON
```

## Results so far

- **Judged questions, teacher pilots** (51 questions):

  | Teacher | Passed | Teacher spend | Median tool calls |
  | --- | --- | --- | --- |
  | GLM-5.3 | 43/51 | $2.72 | 5 |
  | Kimi-K3.1-Pro | 49/51 | $12.31 | 10 |

  - Most failures name repos the agent never looked up.
  - "Which decision models does the Transformers library support?" failed for both teachers. Both read it as the
    Decision Transformer architecture, while the Hub tag `decision-model` answers it: `hf models list --filter
    transformers --filter decision-model` equals huggingface.co/models?library=transformers&other=decision-model.
    That question now has a live reference, and the system prompt says website filters are tags.
  - A review page with both teachers' answers and the user's verdicts was used to check the rubrics.
- **Facts, GLM-5.3 teacher pilot** (127 validation questions, $3.00): 127/127 after the shared checker's hedge and
  list-gold fixes (115/127 as run).
  - The 12 failures as run were checker false negatives: correct answers that mentioned other values in context, and
    a dual license.
  - Median 2 tool calls; 100 of 127 episodes read the skill.
- **Facts v1, base Qwen3.5-4B** (127 validation questions): 118/127 (93%), against MCP's 122/127. v1 is a ceiling
  for both twins, so both moved to v2.
  - The pass rate hides how badly the base model uses the CLI. It read the skill in 2 of 127 episodes (GLM-5.3:
    100). The CLI rejected 33% of its `hf` calls (GLM-5.3: 5%): invented commands such as `hf models show`,
    `hf search` and `hf papers view`, and options such as `--arxiv`. It went around the CLI to the API (`curl`,
    Python) in 35 episodes. The `hf_errors` and `read_skill` metrics now record this in every evaluation.
  - The pinned skill itself says `--format table` is the default, which CLI 2.4.0 rejects (it accepts `agent`,
    `auto`, `human`, `json`, `quiet`); `hf skills preview` generates that line.
- **Facts v2 validation baseline** (149 questions; MCP twin in brackets):

  | | Base Qwen3.5-4B | GLM-5.3 |
  | --- | --- | --- |
  | Passed | 111/149 (74.5%) [MCP 111/149] | 149/149 (100%) [MCP 97.4%] |
  | `hf` calls per episode | 5.9 | 3.1 |
  | `hf` calls the CLI rejected | 27% | 5% |
  | Read the skill | 9/149 | 130/149 |
  | Hit the 15-call limit | 21 | 0 |

  - Per template, base vs teacher: `gguf_count` 9/20 vs 20/20, `safetensors_count` 2/7 vs 7/7, `base_params` 20/29 vs
    29/29, `paper_first_author` 18/25 vs 25/25; everything else at least 84% for base.
  - The base model's commonest rejected calls are invented commands (`hf models show` 53, `hf search` 34, `hf papers
    view` 17) and `--limit` on a file listing (20). The teacher's are mostly `--tree` with `--format json` (9).
- **Thinking off** (the user's choice; `qwen3.5-4b-nothink`, #14). The baseline above was served in Qwen3.5's
  default thinking mode, with the reasoning inside the answer (128/149 base answers contain `</think>`; scoring only
  the text after it gives 121/149). With thinking off, base Qwen3.5-4B scores **64/149 (43%)** on v2 validation, with no
  reasoning in its answers:
  - 59 episodes hit the 15-call limit (thinking: 21). Against the thinking run it lost 61 questions and gained 14.
  - It mostly abandons the CLI. It writes Python against the Hub API or scrapes HTML (`requests`, `httpx`,
    BeautifulSoup), runs into 401s on authenticated endpoints and runs out of calls. It read the skill in 13 of 149
    episodes; the CLI rejected 9% of its `hf` calls (it rarely tries `hf`).
  - The MCP twin's base, also with thinking off, scores 73.3% on its 150 test questions: without a shell it cannot
    wander off the tools.
  - Part of that was the job image, not the model: the student's `huggingface/trl` image had no `curl`, `jq` or
    `wget` (139 `curl: command not found`), while the teacher's image has `curl`. `JOB_SETUP` now installs all three
    in every image, and the task declares the commands it needs (`Task.commands`), checked in every job and as an
    agent user before the first episode. Rerun with the tools: **91/149 (61%)**; 51 questions gained, 24 lost
    (McNemar p ≈ 0.002); 29 episodes hit the call limit (was 59). It still read the skill in 1 of 149 episodes.
- **Full v2 run** (`run.toml`, `qwen35-4b-hf-hub-qa-v2-glm-53-pi-160`): 600 GLM-5.3 traces (all accepted), SFT of
  Qwen3.5-4B (thinking off) on them, evaluated on 150 test questions with the shell tools installed:

  | | Base Qwen3.5-4B | SFT | GLM-5.3 |
  | --- | --- | --- | --- |
  | Passed | 71/150 (47.3%) | **117/150 (78.0%)** | 148/150 (98.7%) |
  | Hit the 15-call limit | 40 | 19 | 0 |
  | Tool calls per episode | 8.6 | 7.2 | 4.6 |
  | `hf` calls / rejected per episode | 4.2 / 0.36 | 5.5 / 0.95 | 3.2 / 0.08 |
  | Read the skill | 2% | 93% | 93% |
  | Episodes using `curl` | 130 | 40 | 27 |

  - SFT against base, paired: 62 questions fixed, 16 broken (sign test p ≈ 1.5e-7).
  - The MCP twin on the same 150 test questions: base 73.3%, SFT 80.0%, GLM-5.3 98.7%. SFT brings the CLI student
    level with the MCP one, from a much weaker base (47% vs 73%): with a shell, the base model wanders off to
    scrapers, and SFT teaches it the CLI and the skill.
  - Per template (base / SFT / teacher): `base_params` 10 / 27 / 30 of 30, `params_sum` 11 / 22 / 26 of 26,
    `paper_first_author` 16 / 23 / 26 of 26, `paper_title` 15 / 21 / 24 of 24, `base_license` 2 / 5 / 4 of 5,
    `base_pipeline` 7 / 8 / 11 of 12, `gguf_count` 6 / 6 / 15 of 15, `safetensors_count` 4 / 5 / 12 of 12. File counts
    across folders barely moved with SFT. The student finds the right listing command but cycles through flag
    combinations the CLI rejects (`--tree` with `--format json`, `--expand files`) until it runs out of calls (11 of
    its 16 count failures). The teacher writes `--recursive --format json` to a file and counts with a short script.
    A GRPO stage on the efficiency-shaped reward is the obvious next step for these.
  - Spend: teacher traces about $21 (one shard's upload hit the account's private storage limit and was resumed),
    teacher evaluation $4.49, GPU about $12 (SFT 82 min, two evaluations and the merge on H200).
