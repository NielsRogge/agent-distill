"""Agreement of each judge setup with the human pair labels (README, "Judge calibration").

    python examples/html-pages/calibration/score.py

Reads `labels.json` (the user's blind choices, -2..2, negative = left page better), `pairs.json` and
`judgements.json` (each setup's per-pair difference in [-1, 1], from `calibrate.py`). Accuracy is over the pairs the
user decided, a judge tie or an unscorable pair counting half, so every row divides by the same number.
"""

import json
from math import comb
from pathlib import Path

HERE = Path(__file__).parent
SETUPS = {
    "hydrogen_abs": "Hydrogen, per page, 3 screenshots (the baseline judge)",
    "qwen_abs": "Qwen3.8-27B, per page, 3 screenshots",
    "hydrogen_defects": "Hydrogen, defect list + appeal",
    "hydrogen_pair": "Hydrogen, pairwise, 3 screenshots each",
    "hydrogen_abs_motion": "Hydrogen, per page, filmstrip + phone + activity log (current)",
    "hydrogen_pair_motion": "Hydrogen, pairwise, filmstrip + phone + activity log",
}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va, vb = sum((x - ma) ** 2 for x in ra) ** 0.5, sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial test of wins vs losses at p = 0.5."""
    n, k = wins + losses, max(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n) if n else 1.0


def main() -> None:
    labels = json.loads((HERE / "labels.json").read_text())
    data = json.loads((HERE / "pairs.json").read_text())
    pairs = {p["id"]: p for p in data["pairs"]}
    judged = json.loads((HERE / "judgements.json").read_text())
    same = [pid for pid in labels if pairs[pid]["kind"] == "same concept" and "repeat_of" not in pairs[pid]]
    humans = {pid: labels[pid]["choice"] / 2 for pid in same}
    decided = [pid for pid in same if humans[pid] != 0]
    print(f"same-concept pairs labelled: {len(same)} ({len(decided)} with a preference, {len(same) - len(decided)} "
          f"about equal)")  # fmt: skip
    print(f"| Judge setup | Agree / disagree / tie or unscorable | Accuracy (of {len(decided)}) | Spearman | "
          "Sign test p |\n|---|---|---|---|---|")  # fmt: skip
    for key, title in SETUPS.items():
        agree = sum(1 for p in decided if (judged[p].get(key) or 0) * humans[p] > 0)
        disagree = sum(1 for p in decided if (judged[p].get(key) or 0) * humans[p] < 0)
        rest = len(decided) - agree - disagree
        rows = [(humans[p], judged[p][key]) for p in same if judged.get(p, {}).get(key) is not None]
        rho = spearman([h for h, _ in rows], [j for _, j in rows])
        accuracy = (agree + 0.5 * rest) / len(decided)
        print(f"| {title} | {agree} / {disagree} / {rest} | {accuracy:.2f} | {rho:.2f} | "
              f"{sign_test(agree, disagree):.2f} |")  # fmt: skip
    repeats = [(pid, pairs[pid]["repeat_of"]) for pid in labels if "repeat_of" in pairs[pid]]
    both = [(r, o) for r, o in repeats if o in labels]
    # A repeat shows the original pair with the sides swapped, so the same preference has the opposite sign.
    same_side = sum(1 for r, o in both if (labels[o]["choice"] > 0) - (labels[o]["choice"] < 0)
                    == (-labels[r]["choice"] > 0) - (-labels[r]["choice"] < 0))  # fmt: skip
    print(f"\nrepeat pairs labelled: {len(both)} of {sum('repeat_of' in p for p in pairs.values())}; "
          f"same verdict (A/B/equal) both times: {same_side} of {len(both)}")  # fmt: skip
    category = [pid for pid in labels if pairs[pid]["kind"] == "same category"]
    print(f"same-category pairs labelled: {len(category)} of "
          f"{sum(p['kind'] == 'same category' for p in pairs.values())} (too few to report agreement)")  # fmt: skip


if __name__ == "__main__":
    main()
