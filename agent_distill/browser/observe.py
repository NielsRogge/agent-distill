"""What a vision judge sees of a page in motion (DESIGN.md §10): a filmstrip and an activity log.

Three still screenshots miss what makes many pages good: motion, the response to a drag or a click, and sound.
`observe_html` plays a page in headless Chromium with a scripted session (wait, drag, click the controls, press
keys, let it settle), and returns:

- `filmstrip`: one image of 8 frames in time order, labelled with what happened before each;
- `phone`: the page at 390 px;
- `log`: a text account of what the page did: Web Audio use (contexts, sources started, whether sound reaches the
  output), animation frames per second, the input events it listens to, its
  controls, and how much of the screen each step changed.

Sound cannot be shown in an image, so the log describes it. The gates (`preview.py`) are unchanged: this is only
what a judge is given.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_distill.browser.preview import DESKTOP, LOCAL_SCHEMES, PHONE, _changed_fraction

FRAME_SIZE = (720, 450)  # each frame in the filmstrip (half of 1440x900)
COLUMNS = 4

# Counts Web Audio, speech, animation frames and listeners from inside the page, before any page script runs.
PROBE_JS = r"""
(() => {
  const probe = window.__probe = {audioContexts: 0, sourcesStarted: 0, nodes: {}, toDestination: 0,
    speech: 0, raf: 0, listeners: {}, mutations: 0, audioSeconds: 0};
  const count = (map, key) => { map[key] = (map[key] || 0) + 1; };
  const AC = window.AudioContext || window.webkitAudioContext;
  if (AC) {
    const Wrapped = function (...args) {
      probe.audioContexts++;
      const ctx = new AC(...args);
      (probe._ctx = probe._ctx || []).push(ctx);
      return ctx;
    };
    Wrapped.prototype = AC.prototype;
    window.AudioContext = Wrapped; window.webkitAudioContext = Wrapped;
    for (const name of Object.getOwnPropertyNames(BaseAudioContext.prototype)) {
      if (!name.startsWith('create')) continue;
      const original = BaseAudioContext.prototype[name];
      if (typeof original !== 'function') continue;
      BaseAudioContext.prototype[name] = function (...args) {
        count(probe.nodes, name.slice(6));
        return original.apply(this, args);
      };
    }
    const start = AudioScheduledSourceNode.prototype.start;
    AudioScheduledSourceNode.prototype.start = function (...args) {
      probe.sourcesStarted++;
      return start.apply(this, args);
    };
    const connect = AudioNode.prototype.connect;
    AudioNode.prototype.connect = function (target, ...rest) {
      if (target instanceof AudioDestinationNode) probe.toDestination++;
      return connect.call(this, target, ...rest);
    };
  }
  if (window.speechSynthesis) {
    const speak = window.speechSynthesis.speak.bind(window.speechSynthesis);
    window.speechSynthesis.speak = (u) => { probe.speech++; return speak(u); };
  }
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => { probe.raf++; return raf(cb); };
  const add = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, ...rest) {
    count(probe.listeners, type);
    return add.call(this, type, ...rest);
  };
  const watch = () => new MutationObserver((m) => { probe.mutations += m.length; })
    .observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
  if (document.documentElement) watch(); else document.addEventListener('DOMContentLoaded', watch);
})();
"""

READ_PROBE_JS = r"""
() => {
  const p = window.__probe || {};
  const ctxs = p._ctx || [];
  const controls = [...document.querySelectorAll('button, [role=button], input, select, textarea, a[href]')];
  const label = (el) => (el.getAttribute('aria-label') || el.innerText || el.value || el.title || el.type || el.tagName)
    .toString().trim().replace(/\s+/g, ' ').slice(0, 30);
  return {
    audioContexts: p.audioContexts || 0, running: ctxs.filter((c) => c.state === 'running').length,
    audioSeconds: Math.max(0, ...ctxs.map((c) => c.currentTime || 0)), sourcesStarted: p.sourcesStarted || 0,
    nodes: p.nodes || {}, toDestination: p.toDestination || 0, speech: p.speech || 0, raf: p.raf || 0,
    listeners: p.listeners || {}, mutations: p.mutations || 0,
    canvases: document.querySelectorAll('canvas').length, svgs: document.querySelectorAll('svg').length,
    controls: controls.length, sliders: document.querySelectorAll('input[type=range]').length,
    controlLabels: controls.slice(0, 14).map(label).filter(Boolean),
  };
}
"""

INPUT_EVENTS = (
    "pointerdown",
    "pointermove",
    "pointerup",
    "mousedown",
    "mousemove",
    "mouseup",
    "click",
    "wheel",
    "keydown",
    "keyup",
    "touchstart",
    "touchmove",
    "input",
    "change",
    "dragstart",
    "drop",
    "scroll",
)


@dataclass
class Observation:
    filmstrip: bytes = b""
    phone: bytes = b""
    log: str = ""
    probe: dict[str, Any] = field(default_factory=dict)
    steps: list[tuple[str, float]] = field(default_factory=list)  # (what happened, fraction of pixels changed)


def _contact_sheet(frames: list[tuple[str, bytes]]) -> bytes:
    from PIL import Image, ImageDraw

    width, height = FRAME_SIZE
    rows = (len(frames) + COLUMNS - 1) // COLUMNS
    caption = 26
    sheet = Image.new("RGB", (COLUMNS * width, rows * (height + caption)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, png) in enumerate(frames):
        with Image.open(io.BytesIO(png)) as frame:
            tile = frame.convert("RGB").resize(FRAME_SIZE)
        x, y = (index % COLUMNS) * width, (index // COLUMNS) * (height + caption)
        draw.rectangle([x, y, x + width, y + caption], fill="black")
        draw.text((x + 8, y + 6), f"{index + 1}. {label}", fill="white")
        sheet.paste(tile, (x, y + caption))
    out = io.BytesIO()
    sheet.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _log(probe: dict[str, Any], steps: list[tuple[str, float]], seconds: float) -> str:
    lines = []
    if probe.get("audioContexts"):
        nodes = ", ".join(f"{n} {k}" for k, n in sorted(probe["nodes"].items(), key=lambda kv: -kv[1])[:8])
        lines.append(
            f"Sound (Web Audio): {probe['audioContexts']} audio context(s), {probe['running']} running at the end; "
            f"{probe['sourcesStarted']} sound sources started; "
            f"{'connected to the speakers' if probe['toDestination'] else 'never connected to the speakers'}. "
            f"Nodes created: {nodes or 'none'}."
        )
    else:
        lines.append("Sound: the page makes no sound (no Web Audio use).")
    if probe.get("speech"):
        lines.append(f"Speech synthesis: {probe['speech']} utterance(s).")
    fps = probe.get("raf", 0) / max(seconds, 1e-6)
    lines.append(f"Animation: about {fps:.0f} animation frames per second over the session "
                 f"({probe.get('canvases', 0)} canvas, {probe.get('svgs', 0)} svg elements; "
                 f"{probe.get('mutations', 0)} DOM changes).")  # fmt: skip
    heard = [e for e in INPUT_EVENTS if probe.get("listeners", {}).get(e)]
    lines.append("Listens for: " + (", ".join(heard) if heard else "no input events") + ".")
    if probe.get("controls"):
        lines.append(f"Controls: {probe['controls']} ({probe.get('sliders', 0)} sliders), e.g. "
                     + "; ".join(f"'{c}'" for c in probe.get("controlLabels", [])[:10]) + ".")  # fmt: skip
    else:
        lines.append("Controls: none (no buttons, inputs or links).")
    lines.append("Screen change after each step (share of pixels): "
                 + ", ".join(f"{label} {100 * frac:.0f}%" for label, frac in steps) + ".")  # fmt: skip
    return "\n".join(lines)


async def observe_page(path: str | Path) -> Observation:
    from playwright.async_api import async_playwright

    path = Path(path).resolve()  # noqa: ASYNC240 - one stat of a local file
    url = path.as_uri()
    obs = Observation()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=os.environ.get("AGENT_DISTILL_CHROMIUM") or None,
            args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
        )
        try:
            context = await browser.new_context(viewport=DESKTOP, device_scale_factor=1)

            async def route(handle: Any) -> None:
                if handle.request.url.startswith(LOCAL_SCHEMES) or handle.request.url == url:
                    await handle.continue_()
                else:
                    await handle.abort()

            await context.route("**/*", route)
            await context.add_init_script(PROBE_JS)
            page = await context.new_page()
            await page.goto(url, wait_until="load", timeout=15000)
            w, h = DESKTOP["width"], DESKTOP["height"]
            frames: list[tuple[str, bytes]] = []
            previous: bytes | None = None
            started = time.monotonic()

            async def snap(label: str) -> None:
                nonlocal previous
                shot = await page.screenshot()
                if previous is not None:
                    obs.steps.append((label, _changed_fraction(previous, shot)))
                frames.append((label, shot))
                previous = shot

            await page.wait_for_timeout(1200)
            await snap("loaded")
            await page.wait_for_timeout(1500)
            await snap("1.5 s later, untouched")
            await page.mouse.move(w * 0.5, h * 0.5)
            await page.mouse.click(w * 0.5, h * 0.5)
            await page.wait_for_timeout(600)
            await snap("click in the middle")
            await page.mouse.move(w * 0.35, h * 0.5)
            await page.mouse.down()
            for i in range(1, 13):
                await page.mouse.move(w * (0.35 + 0.025 * i), h * (0.5 - 0.012 * i))
                await page.wait_for_timeout(25)
            await page.mouse.up()
            await page.wait_for_timeout(500)
            await snap("drag/flick across the page")
            controls = await page.query_selector_all("button, [role=button], input[type=range], select")
            for element in controls[:4]:
                try:
                    box = await element.bounding_box()
                    if box and await element.get_attribute("type") == "range":
                        await page.mouse.click(box["x"] + box["width"] * 0.8, box["y"] + box["height"] / 2)
                    else:
                        await element.click(timeout=500)
                except Exception:  # hidden or detached controls are not the page's fault
                    continue
                await page.wait_for_timeout(200)
            await page.wait_for_timeout(500)
            await snap(f"{min(4, len(controls))} controls used" if controls else "no controls to use")
            for key in ("Space", "ArrowRight", "ArrowUp", "Enter", "a"):
                await page.keyboard.press(key)
                await page.wait_for_timeout(120)
            await page.wait_for_timeout(400)
            await snap("keys pressed")
            await page.mouse.wheel(0, h * 0.8)
            await page.wait_for_timeout(700)
            await snap("scrolled down")
            await page.mouse.wheel(0, -h * 0.8)
            await page.wait_for_timeout(2500)
            await snap("back up, 2.5 s to settle")
            seconds = time.monotonic() - started
            obs.probe = await page.evaluate(READ_PROBE_JS)
            obs.filmstrip = _contact_sheet(frames)
            await page.set_viewport_size(PHONE)
            await page.wait_for_timeout(500)
            obs.phone = await page.screenshot()
            obs.log = _log(obs.probe, obs.steps, seconds)
        finally:
            await browser.close()
    return obs


async def observe_html(html: str) -> Observation:
    with tempfile.TemporaryDirectory(prefix="agent-distill-observe-") as tmp:
        page = Path(tmp) / "index.html"
        page.write_text(html, encoding="utf-8")
        return await observe_page(page)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - a debugging aid
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Filmstrip and activity log of a page, as a judge sees it.")
    parser.add_argument("html", type=Path)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    obs = asyncio.run(observe_page(args.html))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "filmstrip.png").write_bytes(obs.filmstrip)
    (args.out / "phone.png").write_bytes(obs.phone)
    print(obs.log)
    print(json.dumps(obs.probe, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
