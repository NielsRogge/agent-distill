
## Answering questions efficiently

Every command is a tool call, and fewer calls give faster answers. Plan the lookups first, then batch them.

- Put several lookups in one shell command. `--search` takes one term (a repeated `--search` keeps only the
  last), so loop over terms: `for q in alpaca dolly tulu; do hf datasets list --search "$q" --sort downloads --limit 5 --format quiet; done`.
  Do the same for `info` on several repos: `for r in org/a org/b; do hf models info "$r" --format json | head -40; done`.
- Get the fields you need from the listing itself with `--expand` instead of one `info` call per repo:
  - `--expand downloads,likes,tags` adds counts and tags. The tags include the license (`license:mit`) and the
    library.
  - On Spaces, `--expand runtime` adds the hardware: `runtime.hardware` is `zero-a10g` on ZeroGPU and
    `cpu-basic` on CPU.
- Filters that are easy to miss:
  - License: `--filter license:apache-2.0` (and `license:mit`, and so on).
  - Task: `--pipeline-tag image-segmentation`.
  - Library: `--filter transformers`.
  - Hardware: there is no hardware filter for Spaces (`zerogpu` is only a user tag). List with `--expand runtime`
    and check `runtime.hardware`.
- Count or rank with one large `--limit` and a pipe (`--format quiet | wc -l`, or `--format json | python3 -c ...`).
  Don't re-run the same listing with growing limits.
- Files in a repo: `hf spaces list OWNER/NAME --recursive --format quiet` (the same works for models and datasets).
- Most questions need a handful of calls: one or two listings, then `info` or a card for the few repos you cite.
