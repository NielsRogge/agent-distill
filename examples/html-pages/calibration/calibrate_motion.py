"""Judge calibration, round 2: Hydrogen given a filmstrip, the phone view and an activity log (sound, motion,
controls), pairwise (both orders) and per page. Adds hydrogen_pair_motion / hydrogen_abs_motion to calibration.json."""

import asyncio
import json
import re
from pathlib import Path

from agent_distill.browser.observe import observe_html
from agent_distill.rubrics.judge import make_judge_runner

HERE = Path(__file__).parent
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
OBS = CACHE / "observations"
OBS.mkdir(parents=True, exist_ok=True)
results = json.load(open(OUT))

SESSION = """Each page was played in a browser for about 10 seconds by a scripted visitor: it waited, clicked the
middle, dragged across the page, used up to four controls, pressed keys and scrolled. The filmstrip shows 8 frames in
time order, each labelled with what had just happened; the activity log says what the page did, including sound,
which images cannot show."""

PAIR = """You are an exacting art director comparing two single-file web pages, A and B.

Page A concept: {ta}. {ia}
Page B concept: {tb}. {ib}

{session}

Image 1: page A filmstrip (desktop, 1440x900 frames). Image 2: page A on a phone (390 px).
Image 3: page B filmstrip. Image 4: page B on a phone.

Page A activity log:
{la}

Page B activity log:
{lb}

Which page would you rather ship? Weigh visual quality and polish (the phone view counts), creativity, how well each
page realises its own concept, and how much its motion, interaction and sound add to the experience. A page that
looks the same in every frame and ignores the visitor is static, whatever its first frame looks like. Ignore which
concept is more interesting in itself.

Think briefly, then end with one line exactly in this form:
VERDICT: <A++|A+|=|B+|B++>
(A++ = A clearly better, A+ = A slightly better, = about equal, B+ = B slightly better, B++ = B clearly better)"""

ABS = """You are an exacting art director judging a single-file web page made for this concept:

Title: {title}
Idea: {idea}

{session}

Image 1: the filmstrip (desktop, 1440x900 frames). Image 2: the page on a phone (390 px).

Activity log:
{log}

Score the page from 1 (poor) to 10 (exceptional), calibrated so that 5 is a competent but forgettable page and
9-10 is something people would share. Weigh visual quality and polish (the phone view counts), creativity, fit to the
concept, and how much its motion, interaction and sound add. Think briefly, then end with one line:
SCORE: <1-10>"""

VERDICT = {"A++": -1.0, "A+": -0.5, "=": 0.0, "B+": 0.5, "B++": 1.0}


async def main() -> None:
    hydro = make_judge_runner("gemini-4-hydrogen", cache_dir=CACHE)
    hydro.client.max_tokens = 16384
    browser, calls = asyncio.Semaphore(3), asyncio.Semaphore(6)
    pending: dict[str, asyncio.Task] = {}

    async def observe(pid: str) -> dict:
        meta = OBS / f"{pid}.json"
        if not meta.exists():
            async with browser:
                obs = await observe_html((PAGES / f"{pid}.html").read_text())
            (OBS / f"{pid}-film.png").write_bytes(obs.filmstrip)
            (OBS / f"{pid}-phone.png").write_bytes(obs.phone)
            meta.write_text(json.dumps({"log": obs.log}))
        return {
            "log": json.loads(meta.read_text())["log"],
            "images": [(OBS / f"{pid}-film.png").read_bytes(), (OBS / f"{pid}-phone.png").read_bytes()],
        }

    def info(pid: str) -> asyncio.Task:  # one observation per page, shared by every pair that shows it
        if pid not in pending:
            pending[pid] = asyncio.ensure_future(observe(pid))
        return pending[pid]

    async def call(prompt, images):
        async with calls:
            try:
                return await hydro(prompt, images)
            except Exception as exc:
                print("judge error:", str(exc)[:200], flush=True)
                return ""

    async def pairwise(left, right):
        a, b = await info(left["page"]), await info(right["page"])

        async def one(x, y, xi, yi):
            prompt = PAIR.format(
                ta=x["title"], ia=x["idea"], tb=y["title"], ib=y["idea"], session=SESSION, la=xi["log"], lb=yi["log"]
            )
            found = re.findall(r"VERDICT:\s*(A\+\+|A\+|=|B\+\+|B\+)", await call(prompt, xi["images"] + yi["images"]))
            return VERDICT[found[-1]] if found else None

        f, b_ = await asyncio.gather(one(left, right, a, b), one(right, left, b, a))
        votes = [v for v in (f, None if b_ is None else -b_) if v is not None]
        return (sum(votes) / len(votes) if votes else None), f, b_

    async def absolute(side):
        i = await info(side["page"])
        reply = await call(
            ABS.format(title=side["title"], idea=side["idea"], session=SESSION, log=i["log"]), i["images"]
        )
        found = re.findall(r"SCORE:\s*(\d+(?:\.\d+)?)", reply)
        return float(found[-1]) if found else None

    async def judge_pair(pid):
        p = pairs[pid]
        pw, al, ar = await asyncio.gather(pairwise(p["left"], p["right"]), absolute(p["left"]), absolute(p["right"]))
        results[pid].update(
            {
                "hydrogen_pair_motion": pw[0],
                "pm_forward": pw[1],
                "pm_backward": pw[2],
                "hydrogen_abs_motion": None if al is None or ar is None else ar - al,
                "raw_abs_motion": [al, ar],
            }
        )

    await asyncio.gather(*(judge_pair(pid) for pid in sorted(labels)))
    OUT.write_text(json.dumps(results, indent=1))
    print("hydrogen usage", vars(hydro.client.usage))


asyncio.run(main())
