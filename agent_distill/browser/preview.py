"""`preview`: load a single-file HTML page in headless Chromium and report, as text, what is broken (DESIGN.md §10).

    python -m agent_distill.browser.preview index.html          # what the agent's `preview` command runs
    python -m agent_distill.browser.preview index.html --json

The same inspection drives the agent's tool and the task's gates, so "the agent saw no errors" and "the gate
passed" mean the same thing. It reports:

- console errors and uncaught exceptions, at load and during a scripted ~5 s interaction (mouse movement, clicks,
  key presses, scrolling);
- whether the page is blank or nearly uniform (screenshot pixel spread);
- whether anything animates (frames taken apart differ);
- horizontal overflow at 390 px (phone) and 1440 px (desktop) widths;
- every network request (a page must be one self-contained file: no CDNs, fonts, images, fetches or WebSockets);
- a page that stops responding (an endless loop) within the time limit.

Pages run in the sandboxed browser of `sandbox.py` (served from memory, no files, no network, no credentials).

Screenshots (desktop before and after the interaction, phone) are kept for judges but never shown to the agent:
episodes are text-only. `AGENT_DISTILL_CHROMIUM` points to a Chromium binary when Playwright's own is not
installed (`playwright install chromium`).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_distill.browser.sandbox import PAGE_SECONDS, PAGE_URL, close_quietly, launch, page_context

DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}
INTERACTION_SECONDS = 5.0
# A page whose screenshot has a smaller per-channel standard deviation than this is "blank" (one flat colour).
BLANK_STDDEV = 3.0
ANIMATION_DIFF = 0.0001  # fraction of pixels (about 130 at 1440x900) that must change to count as a change
MAX_MESSAGES = 20


@dataclass
class PageReport:
    path: str
    size_bytes: int
    load_errors: list[str] = field(default_factory=list)
    interaction_errors: list[str] = field(default_factory=list)
    network_requests: list[str] = field(default_factory=list)
    blank: bool = False
    pixel_stddev: float = 0.0
    animated: bool = False
    changed_by_interaction: bool = False
    overflow_desktop_px: int = 0
    overflow_phone_px: int = 0
    title: str = ""
    load_failed: str | None = None
    screenshots: dict[str, bytes] = field(default_factory=dict, repr=False)

    @property
    def errors(self) -> list[str]:
        return self.load_errors + self.interaction_errors

    def gates(self) -> dict[str, bool]:
        """The "zero broken files" checks; every one must hold."""
        return {
            "loads": self.load_failed is None,
            "no_errors": not self.errors,
            "not_blank": not self.blank,
            "self_contained": not self.network_requests,
            "no_overflow": self.overflow_desktop_px <= 0 and self.overflow_phone_px <= 0,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("screenshots")
        data["gates"] = self.gates()
        return data

    def text(self) -> str:
        """What the agent reads."""
        gates = self.gates()
        lines = [f"preview of {self.path} ({self.size_bytes} bytes)" + (f": {self.title!r}" if self.title else "")]
        if self.load_failed:
            lines.append(f"FAILED TO LOAD: {self.load_failed}")
        failing = [name for name, ok in gates.items() if not ok]
        lines.append("verdict: " + ("PROBLEMS: " + ", ".join(failing) if failing else "OK, no problems found"))
        for label, items in (("errors at load", self.load_errors), ("errors during interaction",
                                                                       self.interaction_errors),
                             ("network requests (not allowed)", self.network_requests)):  # fmt: skip
            if items:
                counts = dict(Counter(items))
                lines.append(f"{label} ({len(items)}):")
                lines += [
                    f"  - {item}" + (f" (x{n})" if n > 1 else "") for item, n in list(counts.items())[:MAX_MESSAGES]
                ]
        lines.append(f"blank or near-uniform: {'yes' if self.blank else 'no'} (pixel spread {self.pixel_stddev:.1f})")
        lines.append(f"animates on its own: {'yes' if self.animated else 'no'}")
        lines.append(f"changes when interacted with: {'yes' if self.changed_by_interaction else 'no'}")
        lines.append(f"horizontal overflow: desktop {self.overflow_desktop_px} px, phone {self.overflow_phone_px} px")
        return "\n".join(lines)


def _stddev(png: bytes) -> float:
    from PIL import Image, ImageStat

    with Image.open(io.BytesIO(png)) as image:
        stat = ImageStat.Stat(image.convert("RGB"))
    return max(stat.stddev)


def _changed_fraction(a: bytes, b: bytes) -> float:
    from PIL import Image, ImageChops

    with Image.open(io.BytesIO(a)) as first, Image.open(io.BytesIO(b)) as second:
        if first.size != second.size:
            return 1.0
        diff = ImageChops.difference(first.convert("L"), second.convert("L")).point(lambda v: 255 if v > 8 else 0)
        histogram = diff.histogram()
        return histogram[255] / max(1, first.size[0] * first.size[1])


# The element that scrolls the page: <html> in standards mode, <body> in quirks mode (no doctype).
OVERFLOW_JS = (
    "() => Math.max(0, (document.scrollingElement || document.documentElement).scrollWidth - window.innerWidth)"
)


async def _interact(page: Any, width: int, height: int) -> None:
    """About five seconds of plausible use: move, click around the middle and the controls, press keys, scroll."""
    steps = [(0.5, 0.5), (0.3, 0.4), (0.7, 0.6), (0.5, 0.3), (0.5, 0.7)]
    for fx, fy in steps:
        await page.mouse.move(width * fx, height * fy, steps=8)
        await page.mouse.click(width * fx, height * fy)
        await page.wait_for_timeout(250)
    for key in ("Space", "ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown", "Enter", "a"):
        await page.keyboard.press(key)
        await page.wait_for_timeout(80)
    with_controls = await page.query_selector_all("button, [role=button], input, select, a[href]")
    for element in with_controls[:5]:
        try:
            await element.click(timeout=500, trial=False)
        except Exception:  # hidden or detached controls are not the page's fault
            continue
        await page.wait_for_timeout(150)
    await page.mouse.wheel(0, height)
    await page.wait_for_timeout(300)
    await page.mouse.wheel(0, -height)
    await page.wait_for_timeout(300)


async def inspect_page(path: str | Path, *, interaction: bool = True, screenshots: bool = True) -> PageReport:
    """Inspect a page file (what the agent's `preview` runs on its workspace)."""
    path = Path(path).resolve()  # noqa: ASYNC240 - one stat of a local file
    if not path.is_file():
        report = PageReport(path=path.name, size_bytes=0)
        report.load_failed = "file not found"
        return report
    html = path.read_text(encoding="utf-8", errors="replace")
    return await _inspect(html, path.name, path.stat().st_size, interaction=interaction, screenshots=screenshots)


async def _inspect(html: str, name: str, size: int, *, interaction: bool, screenshots: bool) -> PageReport:
    """Run the inspection in the sandboxed browser (`sandbox.py`), within `PAGE_SECONDS`."""
    report = PageReport(path=name, size_bytes=size)
    phase = {"name": "load"}
    try:
        await asyncio.wait_for(_run(html, report, phase, interaction, screenshots), timeout=PAGE_SECONDS)
    except TimeoutError:
        message = f"the page stopped responding (nothing finished within {PAGE_SECONDS:.0f} s; an endless loop?)"
        if phase["name"] == "load":
            report.load_failed = message
        else:
            report.interaction_errors.append(message)
    return report


async def _run(html: str, report: PageReport, phase: dict, interaction: bool, screenshots: bool) -> None:
    from playwright.async_api import async_playwright

    def on_error(text: str) -> None:
        (report.load_errors if phase["name"] == "load" else report.interaction_errors).append(text[:300])

    async with async_playwright() as playwright:
        browser = await launch(playwright)
        try:
            context = await page_context(browser, html, viewport=DESKTOP, on_blocked=report.network_requests.append)
            page = await context.new_page()
            page.on("console", lambda m: on_error(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda exc: on_error(f"uncaught {exc}"))
            try:
                await page.goto(PAGE_URL, wait_until="load", timeout=15000)
            except Exception as exc:
                report.load_failed = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                return
            await page.wait_for_timeout(1200)
            report.title = (await page.title())[:120]
            first = await page.screenshot()
            await page.wait_for_timeout(700)
            second = await page.screenshot()
            report.pixel_stddev = round(_stddev(first), 2)
            report.blank = report.pixel_stddev < BLANK_STDDEV
            report.animated = _changed_fraction(first, second) > ANIMATION_DIFF
            report.overflow_desktop_px = int(await page.evaluate(OVERFLOW_JS))
            if interaction:
                phase["name"] = "interaction"
                await _interact(page, DESKTOP["width"], DESKTOP["height"])
                after = await page.screenshot()
                report.changed_by_interaction = _changed_fraction(second, after) > ANIMATION_DIFF
                if screenshots:
                    report.screenshots["desktop_after"] = after
            if screenshots:
                report.screenshots["desktop"] = first
            await page.set_viewport_size(PHONE)
            await page.wait_for_timeout(400)
            report.overflow_phone_px = int(await page.evaluate(OVERFLOW_JS))
            if screenshots:
                report.screenshots["phone"] = await page.screenshot()
        finally:
            await close_quietly(browser)


_CACHE: dict[tuple[str, bool], PageReport] = {}
CACHE_SIZE = 16  # reports hold three full-size screenshots; one episode's gates and judge share one


async def inspect_html(html: str, *, interaction: bool = True) -> PageReport:
    """Inspect HTML text (a submitted answer); reports are cached per content within the process, so every gate
    and judge of one episode shares one browser run."""
    import hashlib

    key = (hashlib.sha256(html.encode()).hexdigest(), interaction)
    if key not in _CACHE:
        report = await _inspect(html, "index.html", len(html.encode()), interaction=interaction, screenshots=True)
        _CACHE[key] = report
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
    return _CACHE[key]


def write_tool(workspace: Path) -> Path:
    """Put the `preview` command in an episode's workspace (it runs this module with the stage's Python)."""
    tool = Path(workspace) / "preview"
    tool.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m agent_distill.browser.preview "$@"\n')
    tool.chmod(0o755)
    return tool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default="index.html")
    parser.add_argument("--json", action="store_true", help="Machine-readable report")
    parser.add_argument("--no-interaction", action="store_true")
    args = parser.parse_args(argv)
    report = asyncio.run(inspect_page(args.path, interaction=not args.no_interaction, screenshots=False))
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.text())
    return 0 if all(report.gates().values()) else 1


if __name__ == "__main__":
    sys.exit(main())
