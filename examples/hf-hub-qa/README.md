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
  - Tool mismatch in this SFT: its training examples listed `bash`, `read` and an argument-free `submit`. The
    library's tool list gave text-answer tasks Pi's captured `submit` tool (fixed in #19). In evaluation, Pi offers only
    `bash` and `read`, as in the traces. The traces never call `submit`, so the student only saw it listed. The scores
    above stand as measured. Rerun SFT or GRPO from code after #19.
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
- **SFT → GRPO on all 377 test questions** (`run-grpo.toml`, `qwen35-4b-hf-hub-qa-v2-glm-53-pi-160-grpo`):
  - SFT was rerun after #19 on the same 600 GLM-5.3 traces, so the student trained with exactly `bash` and `read`.
  - A difficulty probe then ran the SFT model 4 times on 700 train questions. 157 qualified (solved on 1-3 of the 4
    attempts), against 350 of 1,162 for the MCP twin.
  - GRPO trained from the SFT merge on those 157 questions: 78 steps, one pass over the pool, 38 minutes on 2x H200.
  - The table compares the MCP twin's run on the same 377 questions, in brackets:

  | | Base Qwen3.5-4B | SFT | SFT + GRPO | GLM-5.3 |
  | --- | --- | --- | --- | --- |
  | Passed | 206/377 (54.6%) [72.1%] | 321/377 (85.1%) [81.4%] | **331/377 (87.8%)** [96.0%] | 375/377 (99.5%) [98.7%] |
  | Hit the 15-call limit | 91 | 30 | 18 | 0 |
  | Tool calls per episode | 8.9 | 7.0 | 4.9 | 4.3 |
  | `hf` calls / rejected per episode | 4.3 / 0.50 | 5.5 / 0.93 | 3.7 / 0.22 | 3.1 / 0.10 |
  | Read the skill | 2% | 93% | 70% | 92% |

  - Paired comparisons:
    - SFT against base: 141 questions fixed, 26 broken (sign test p ≈ 2.6e-20).
    - GRPO against SFT: 46 fixed, 36 broken (p ≈ 0.32).
    - GRPO against base: 145 fixed, 20 broken.
  - On the CLI, SFT does the heavy lifting. It beats the MCP twin's SFT (85.1% vs 81.4%) from a much weaker base.
  - GRPO mostly made the student more efficient rather than more accurate:
    - it uses 30% fewer tool calls;
    - it has a quarter of SFT's rejected `hf` calls;
    - it hits the call limit 18 times instead of 30.
  - The accuracy gain from GRPO is +2.7 points and not significant, against +14.6 for the MCP twin.
  - The likely cause is the small pool. The probe qualified 22% of sampled train questions, so GRPO saw 157 questions
    once (78 steps), against 350 (100 steps) for the MCP twin.
  - Per template (base / SFT / GRPO / teacher):

    | Template | Base | SFT | GRPO | Teacher |
    | --- | --- | --- | --- | --- |
    | `base_license` | 9 | 12 | 15 | 15 of 15 |
    | `base_params` | 30 | 73 | 73 | 80 of 81 |
    | `base_pipeline` | 29 | 32 | 33 | 35 of 36 |
    | `paper_first_author` | 42 | 58 | 63 | 67 of 67 |
    | `paper_title` | 52 | 63 | 68 | 69 of 69 |
    | `params_sum` | 24 | 50 | 44 | 52 of 52 |
    | `gguf_count` | 11 | 23 | 23 | 37 of 37 |
    | `safetensors_count` | 9 | 10 | 12 | 20 of 20 |

  - File counts remain the gap: 14 of GRPO's 18 call-limit failures are count questions.
  - `params_sum` regressed by 6. Of GRPO's 8 failures there, 6 added the three sizes with a script, so the cause is
    not unscripted arithmetic; it has not been diagnosed further.
  - Of GRPO's 46 failures:
    - 18 hit the call limit;
    - 14 are wrong;
    - 13 name the right fact but also a Hub id the agent never looked up (`grounded`);
    - 1 never used the Hub.
  - Spend:
    - GPU about $43 of the approved $90: SFT $6.75, probe $11, GRPO pilot $1.90, GRPO $6.30, merges about $0.60,
      evaluations $16 (base, SFT and GRPO on H200).
    - The teacher's evaluation cost $11.80 of API calls.
