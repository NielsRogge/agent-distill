# Inkling-2-Mini hook modules

Verbatim copies of `inkling_adapter.py`, `inkling_train_patch.py` and `merge_adapter.py` from
[NielsRogge/url-extraction](https://github.com/NielsRogge/url-extraction) at commit `a0afe66`. The
`inkling-2-mini` student profile names them in `requires`, so `distill run` stages them with every job; the
`run-inkling.toml` configs of `hf-hub-qa` and `hf-hub-qa-mcp` add this folder to their profile lookup
(`profiles = ["../inkling-2-mini"]`). Edit them upstream and copy them back; ruff skips this folder.
