"""Parity tier 2: compare an agent-distill evaluation with url-extraction's published numbers.

Reads the evaluation summaries agent-distill published (`summaries/<label>/<split>.json`) and prints field accuracy
(fraction of verified URL fields correct), exact match, completeness and tool calls per label, next to the
url-extraction reference. Tier 2 passes when field accuracy is within +-5 points (DESIGN.md §9).

    uv run python examples/paper-urls/parity/tier2.py --repo nielsr/qwen35-9b-paper-urls-parity-opencode-124-eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# url-extraction README (Qwen3.5-9B, test, 42 papers / 84 fields, OpenCode 1.2.4)
REFERENCE = {"base": (75.0, 6.8), "grpo": (82.1, 3.0)}
TOLERANCE = 5.0


def field_accuracy(metrics: dict[str, float]) -> tuple[float, float]:
    correct = sum(v for k, v in metrics.items() if k.endswith("_correct") and k.startswith("url_correctness/"))
    scored = sum(v for k, v in metrics.items() if k.endswith("_scored") and k.startswith("url_correctness/"))
    return (100 * correct / scored if scored else float("nan")), scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(args.repo, repo_type="dataset") if f.startswith("summaries/")]
    ok = True
    for name in sorted(files):
        if not name.endswith(f"/{args.split}.json"):
            continue
        summary = json.loads(Path(hf_hub_download(args.repo, name, repo_type="dataset")).read_text())
        for label, entry in summary["labels"].items():
            accuracy, fields = field_accuracy(entry["metric_sums"])
            reference = REFERENCE.get(label)
            line = (f"{label}: field accuracy {accuracy:.1f}% ({fields:.0f} fields), pass@1 {entry['pass_at_1']:.3f}, "
                    f"tool calls {entry['mean_tool_calls']:.1f}, episodes {entry['episodes']}")  # fmt: skip
            if reference:
                delta = accuracy - reference[0]
                passed = abs(delta) <= TOLERANCE
                ok &= passed
                line += f" | url-extraction {reference[0]}% ({reference[1]} calls): {delta:+.1f} -> "
                line += "PASS" if passed else "FAIL"
            print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
