# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor, etc.) working in this repository. The design is in
[docs/DESIGN.md](docs/DESIGN.md); read it before changing library code.

## Branches and worktrees

- Give each feature request and non-hotfix bug fix an isolated worktree and a dedicated `<agent>/<short-feature-name>` branch, where `<agent>` is the name of the coding agent doing the work (for example `codex/…`, `claude/…`, `cursor/…`).
- If the agent supplies a detached-HEAD worktree, create the branch before the first commit.
- Do not develop independent features in the primary `main` checkout.

Example:

```bash
git fetch origin main
git worktree add -b claude/new-feature ../agent-distill-new-feature origin/main
```

## Hugging Face billing

Always bill Hugging Face services to the `huggingface` org, never to a personal account:

- **Inference Providers:** send the `X-HF-Bill-To: huggingface` header (`[hub] bill_to = "huggingface"` in a `run.toml`, or `InferenceClient(bill_to="huggingface")` in ad-hoc scripts).
- **Jobs:** pass `--namespace huggingface` to `hf jobs run` / `hf jobs uv run` (the pipeline does this from `[hub] bill_to`).
- **Inference Endpoints:** create and manage them in the `huggingface` namespace (`--namespace huggingface`).

The library itself must not hardcode `huggingface` or `nielsr`: they belong in the examples' `run.toml` files.

## Credentials

- Never put credentials in an agent's environment or in written records. Agents run shell commands and may print their environment (`env`, `os.environ`).
  - Harness adapters pass an agent only the key its endpoint needs (teachers), and none to students served by local vLLM.
  - Judge and critic keys (e.g. `GEMINI_API_KEY`) are read only by the judge client in the stage process.
  - Profiles name environment variables (`key_env`), never values.
- Write episode records through the library's record writer, which redacts credentials (`hf_`, `sk-`, `ghp_`, `github_pat_`, Google API keys).
- Never print, log or commit a credential, not even partially.

## Hub artifacts

- Every artifact a run publishes on the Hugging Face Hub (models, adapters, datasets, trace and evaluation datasets, and Spaces) must be added to the run's collection (`[hub] collection` in its `run.toml`); the paper-urls example uses the url-extraction collection: https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e. If a new example has no collection yet, ask the user before creating one.
- Add it when the repo is created, with a one-line note saying what it is (teacher, student, stage, harness).
- Visibility: models and datasets are public. Spaces, code snapshots (`<run>-code` repos or buckets holding job code) and this GitHub repository stay private.
- Before making a dataset public, scan it for credentials; `distill publish` does the scan and refuses on a hit.
  - Create episode datasets (traces, evaluations) private and make them public only after the scan passes. Never pass `--public` to a job that writes episodes.
  - If a credential is found in a public repo: make the repo private at once, do not delete anything, and tell the user to revoke the token (the repo history keeps it).
- Never delete Hub repos or files without asking the user.
- Agent traces must always be uploaded in a form the Hub's [agent-traces viewer](https://huggingface.co/docs/hub/en/agent-traces) can open: one Pi-format session per episode under `sessions/` (Pi's native session for Pi episodes, the library's Episode-to-Pi exporter for OpenCode). Never upload episode JSONL with full trajectories on its own; slim summaries (e.g. difficulty-probe attempts) are exempt.
  - The `traces` and `eval` stages publish the sessions; prefer them over manual uploads.
  - New code that publishes episodes must go through the library's session publisher, and HF Jobs code staging must include it (the staged wheel does).
  - Check that the dataset page shows the traces in Data Studio before reporting the upload as done.

Example:

```python
from huggingface_hub import HfApi

HfApi().add_collection_item(
    "nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e",
    item_id="nielsr/<repo>", item_type="model",  # or "dataset" / "space"
    note="Qwen3.5-9B GRPO adapter, opencode@1.2.4 (from SFT on Kimi-K3.1-Pro traces)", exists_ok=True,
)
```

## Harness versions

- Harnesses are pinned in their profiles (`opencode@1.2.4`, `opencode@2.3.2`, `pi@1.6.0`). Never fall back to a binary that happens to be on `PATH`, and never mix traces, training and evaluation across harness versions without the explicit `--allow-harness-mismatch` override.
- Record `harness@version` in every run manifest, episode record, session header and card.

## Parity

- One rubric set per task drives trace acceptance, the GRPO reward and evaluation. The only allowed differences are rubrics tagged `train_only` / `eval_only`; keep the parity tests passing when changing rubrics, adapters or the reward.
- Changes that could move a published number must be checked by rescoring the affected published records (`distill rescore`) and reported with the before/after numbers.

## Spending

- Run `--dry-run` and the budget pilot before launching GPU jobs or large teacher/judge runs, and stay within the `[budget]` caps in the `run.toml`; ask the user before passing `--over-budget`.

## Cleanup after release

- Once the task's merged commit is live and its smoke checks pass, remove the task's worktree and delete its local and remote branch.
- First confirm the worktree is clean and that `main` contains the branch's changes (for a squash merge, compare the branch diff against the merge commit's diff).
- Clean up only the task's own worktree and branch.

Example:

```bash
git -C ../agent-distill-new-feature status --porcelain   # must be empty
git worktree remove ../agent-distill-new-feature
git branch -D claude/new-feature
git push origin --delete claude/new-feature
```
