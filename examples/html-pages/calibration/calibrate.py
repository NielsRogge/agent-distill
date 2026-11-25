"""Judge calibration against the human pair labels (html-pages).

Methods, each giving a per-pair judgement d in [-1, 1] (negative = left better):
- hydrogen_abs: the baseline run's Hydrogen overall scores (right - left).
- qwen_abs: the same absolute prompt with Qwen3.8-27B.
- hydrogen_defects: Hydrogen lists defects by severity plus an appeal score; code turns them into a score.
- hydrogen_pair: Hydrogen sees both pages and picks one (5-point), run in both orders to cancel position bias.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from agent_distill.browser.preview import inspect_html
from agent_distill.rubrics.judge import make_judge_runner

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from rubrics import STILLS_PROMPT as JUDGE_PROMPT  # noqa: E402 - examples/html-pages/rubrics.py

DATA = json.loads((HERE / "pairs.json").read_text())
pairs = {p["id"]: p for p in DATA["pairs"]}
labels = {
    k: v
    for k, v in json.loads((HERE / "labels.json").read_text()).items()
    if pairs[k]["kind"] == "same concept" and "repeat_of" not in pairs[k]
}
PAGES = HERE / "pages"  # written by fetch_pages.py
CACHE = HERE / ".cache"  # screenshots, observations and judge verdicts
OUT = HERE / "judgements.json"
SHOTS = CACHE / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

DEFECT_PROMPT = """You are an exacting art director reviewing a single-file web page made for this concept:

Title: {title}
Idea: {idea}

Image 1: desktop (1440x900) right after loading. Image 2: the same page after a few seconds of clicks, key presses
and mouse movement. Image 3: phone (390 px wide). Measured: animates on its own: {animated}; changes when
interacted with: {interactive}.

First list every visible defect once, by its root cause (not once per symptom), each with a severity:
- minor: small polish issue (slightly cramped spacing, a weak colour choice, a small misalignment)
- major: clearly hurts the page (clipped or overlapping text, a broken or empty-looking region, unreadable text,
  a phone layout that does not work, the concept barely realised)
- fatal: the page does not work as the concept (blank, unusable, wrong thing entirely)
Then rate APPEAL from 1 to 10: how striking, original and delightful the page is, ignoring the defects listed,
where 5 is competent but forgettable and 10 is something people would share.

Answer exactly in this form:
DEFECT: <minor|major|fatal> - <root cause>
(one line per defect; write "DEFECT: none" if there are none)
APPEAL: <1-10>"""

PAIR_PROMPT = """You are an exacting art director comparing two single-file web pages, A and B.

Page A concept: {ta}. {ia}
Page B concept: {tb}. {ib}

Images 1-3 are page A: desktop (1440x900) after loading, desktop after a few seconds of clicks and key presses,
phone (390 px). Images 4-6 are page B in the same order. Measured: A animates: {aa}, A reacts to interaction: {ai};
B animates: {ba}, B reacts to interaction: {bi}.

Which page would you rather ship? Weigh visual quality and polish (the phone view counts), creativity, how well each
page realises its own concept, and how meaningful its interaction and motion are. Ignore which concept is more
interesting in itself.

Think briefly, then end with one line exactly in this form:
VERDICT: <A++|A+|=|B+|B++>
(A++ = A clearly better, A+ = A slightly better, = about equal, B+ = B slightly better, B++ = B clearly better)"""

VERDICT = {"A++": -1.0, "A+": -0.5, "=": 0.0, "B+": 0.5, "B++": 1.0}


async def page_info(page_id: str, lock: asyncio.Semaphore) -> dict:
    meta = SHOTS / f"{page_id}.json"
    if meta.exists():
        info = json.loads(meta.read_text())
    else:
        async with lock:
            report = await inspect_html((PAGES / f"{page_id}.html").read_text())
        for name in ("desktop", "desktop_after", "phone"):
            (SHOTS / f"{page_id}-{name}.png").write_bytes(report.screenshots[name])
        info = {
            "animated": "yes" if report.animated else "no",
            "interactive": "yes" if report.changed_by_interaction else "no",
        }
        meta.write_text(json.dumps(info))
    info["images"] = [(SHOTS / f"{page_id}-{n}.png").read_bytes() for n in ("desktop", "desktop_after", "phone")]
    return info


def last_int(pattern: str, text: str) -> float | None:
    found = re.findall(pattern, text, re.I)
    return float(found[-1]) if found else None


def defect_score(reply: str) -> float | None:
    appeal = last_int(r"APPEAL:\s*(\d+(?:\.\d+)?)", reply)
    if appeal is None:
        return None
    sev = [s.lower() for s in re.findall(r"DEFECT:\s*(minor|major|fatal)", reply, re.I)]
    if "fatal" in sev:
        return 0.0
    return max(0.0, appeal - 1.0 * sev.count("minor") - 2.0 * sev.count("major"))


async def main() -> None:
    hydro = make_judge_runner("gemini-4-hydrogen", cache_dir=CACHE)
    qwen = make_judge_runner("qwen3.8-27b", cache_dir=CACHE, bill_to="huggingface")
    hydro.client.max_tokens = 16384  # room for its reasoning before the defect list
    browser, calls = asyncio.Semaphore(3), asyncio.Semaphore(6)
    infos: dict[str, dict] = {}

    async def info(pid: str) -> dict:
        if pid not in infos:
            infos[pid] = await page_info(pid, browser)
        return infos[pid]

    async def call(runner, prompt, images):
        async with calls:
            try:
                return await runner(prompt, images)
            except Exception as exc:  # unscorable, not zero
                print("judge error:", str(exc)[:200], flush=True)
                return ""

    async def absolute(runner, side, prompt_kind):
        i = await info(side["page"])
        if prompt_kind == "abs":
            prompt = JUDGE_PROMPT.format(
                title=side["title"],
                idea=side["idea"],
                category=side["category"],
                animated=i["animated"],
                interactive=i["interactive"],
            )
            return last_int(r"SCORE:\s*(\d+(?:\.\d+)?)", await call(runner, prompt, i["images"]))
        prompt = DEFECT_PROMPT.format(
            title=side["title"], idea=side["idea"], animated=i["animated"], interactive=i["interactive"]
        )
        return defect_score(await call(runner, prompt, i["images"]))

    async def pairwise(left, right):
        a, b = await info(left["page"]), await info(right["page"])

        async def one(x, y, xi, yi):
            prompt = PAIR_PROMPT.format(
                ta=x["title"],
                ia=x["idea"],
                tb=y["title"],
                ib=y["idea"],
                aa=xi["animated"],
                ai=xi["interactive"],
                ba=yi["animated"],
                bi=yi["interactive"],
            )
            found = re.findall(
                r"VERDICT:\s*(A\+\+|A\+|=|B\+\+|B\+)", await call(hydro, prompt, xi["images"] + yi["images"])
            )
            return VERDICT[found[-1]] if found else None

        forward, backward = await asyncio.gather(one(left, right, a, b), one(right, left, b, a))
        votes = [v for v in (forward, None if backward is None else -backward) if v is not None]
        return (sum(votes) / len(votes) if votes else None), forward, backward

    async def judge_pair(pid):
        p = pairs[pid]
        left = {**p["left"], "category": p["category"]}
        right = {**p["right"], "category": p["category"]}
        ql, qr, dl, dr, pw = await asyncio.gather(
            absolute(qwen, left, "abs"),
            absolute(qwen, right, "abs"),
            absolute(hydro, left, "def"),
            absolute(hydro, right, "def"),
            pairwise(left, right),
        )
        return pid, {
            "human": labels[pid]["choice"] / 2,
            "kind": p["kind"],
            "hydrogen_abs": (
                DATA["pages"][p["right"]["page"]]["hydrogen_stills_score"]
                - DATA["pages"][p["left"]["page"]]["hydrogen_stills_score"]
            ),
            "qwen_abs": None if ql is None or qr is None else qr - ql,
            "hydrogen_defects": None if dl is None or dr is None else dr - dl,
            "hydrogen_pair": pw[0],
            "pair_forward": pw[1],
            "pair_backward": pw[2],
            "raw": {"qwen": [ql, qr], "defects": [dl, dr]},
        }

    results = dict(await asyncio.gather(*(judge_pair(pid) for pid in sorted(labels))))
    OUT.write_text(json.dumps(results, indent=1))
    print("pairs", len(results), "hydrogen usage", vars(hydro.client.usage), "qwen usage", vars(qwen.client.usage))


asyncio.run(main())
