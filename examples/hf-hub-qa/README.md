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
- **Full v2 run:** not started yet.
