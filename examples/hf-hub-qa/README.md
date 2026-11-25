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
- **Main task** (`task.py`): the Hub facts in `nielsr/hf-hub-facts` (1,068 train / 127 validation / 359 test,
  frozen 2026-11-25), shared with the MCP twin. Each question has one gold value, for example:
  - a model's license, library or pipeline tag;
  - its parameter count or base model;
  - a Space's SDK;
  - a paper's title, first author or arXiv id.
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
- **Full run** (`run.toml`): in progress.
