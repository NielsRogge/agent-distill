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

import asyncio
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_distill.browser.preview import DESKTOP, PHONE, _changed_fraction
from agent_distill.browser.sandbox import PAGE_SECONDS, PAGE_URL, close_quietly, launch, page_context

FRAME_SIZE = (720, 450)  # each frame in the filmstrip (half of 1440x900)
COLUMNS = 4

# Counts Web Audio, speech, animation frames and listeners from inside the page, before any page script runs. The
# counts live in a closure; the page sees only `window.__agentDistillProbe`, a frozen, non-configurable object of
# getters, so it can add to the counts only by really doing the thing (creating audio, animating, listening), never
# by writing numbers. Builtins the getters use are captured before page scripts can replace them.
PROBE_JS = r"""
(() => {
  const state = {audioContexts: 0, sourcesStarted: 0, toDestination: 0, speech: 0, raf: 0, mutations: 0};
  const nodes = new Map(), listeners = new Map(), contexts = [];
  const bump = (map, key) => map.set(String(key).slice(0, 40), (map.get(String(key).slice(0, 40)) || 0) + 1);
  const fromEntries = Object.fromEntries, freeze = Object.freeze, define = Object.defineProperty;
  const AC = window.AudioContext || window.webkitAudioContext;
  const stateOf = AC && Object.getOwnPropertyDescriptor(BaseAudioContext.prototype, 'state').get;
  const timeOf = AC && Object.getOwnPropertyDescriptor(BaseAudioContext.prototype, 'currentTime').get;
  if (AC) {
    const Wrapped = function (...args) {
      state.audioContexts++;
      const ctx = new AC(...args);
      contexts.push(ctx);
      return ctx;
    };
    Wrapped.prototype = AC.prototype;
    window.AudioContext = Wrapped; window.webkitAudioContext = Wrapped;
    for (const name of Object.getOwnPropertyNames(BaseAudioContext.prototype)) {
      if (!name.startsWith('create')) continue;
      const original = BaseAudioContext.prototype[name];
      if (typeof original !== 'function') continue;
      BaseAudioContext.prototype[name] = function (...args) {
        bump(nodes, name.slice(6));
        return original.apply(this, args);
      };
    }
    const start = AudioScheduledSourceNode.prototype.start;
    AudioScheduledSourceNode.prototype.start = function (...args) {
      state.sourcesStarted++;
      return start.apply(this, args);
    };
    const connect = AudioNode.prototype.connect;
    const Destination = AudioDestinationNode;
    AudioNode.prototype.connect = function (target, ...rest) {
      if (target instanceof Destination) state.toDestination++;
      return connect.call(this, target, ...rest);
    };
  }
  if (window.speechSynthesis) {
    const speak = window.speechSynthesis.speak.bind(window.speechSynthesis);
    window.speechSynthesis.speak = (u) => { state.speech++; return speak(u); };
  }
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => { state.raf++; return raf(cb); };
  const add = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, ...rest) {
    bump(listeners, type);
    return add.call(this, type, ...rest);
  };
  const watch = () => new MutationObserver((m) => { state.mutations += m.length; })
    .observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
  if (document.documentElement) watch(); else add.call(document, 'DOMContentLoaded', watch);
  const view = {};
  for (const key of Object.keys(state)) define(view, key, {get: () => state[key], enumerable: true});
  define(view, 'running', {get: () => contexts.filter((c) => stateOf.call(c) === 'running').length});
  define(view, 'audioSeconds', {get: () => Math.max(0, ...contexts.map((c) => timeOf.call(c) || 0))});
  define(view, 'nodes', {get: () => fromEntries(nodes)});
  define(view, 'listeners', {get: () => fromEntries(listeners)});
  define(window, '__agentDistillProbe', {value: freeze(view), writable: false, configurable: false});
})();
"""

READ_PROBE_JS = r"""
() => {
  const p = window.__agentDistillProbe;
  if (!p) return {};
  return {audioContexts: p.audioContexts, running: p.running, audioSeconds: p.audioSeconds,
    sourcesStarted: p.sourcesStarted, nodes: p.nodes, toDestination: p.toDestination, speech: p.speech, raf: p.raf,
    listeners: p.listeners, mutations: p.mutations};
}
"""

# DOM facts, read in an isolated world (its own pristine builtins, the page's DOM), so page scripts cannot patch
# querySelectorAll or innerText to change them.
READ_DOM_JS = r"""
(() => {
  const controls = [...document.querySelectorAll('button, [role=button], input, select, textarea, a[href]')];
  const label = (el) => (el.getAttribute('aria-label') || el.innerText || el.value || el.title || el.type || el.tagName)
    .toString().trim().replace(/\s+/g, ' ').slice(0, 30);
  return JSON.stringify({
    canvases: document.querySelectorAll('canvas').length, svgs: document.querySelectorAll('svg').length,
    controls: controls.length, sliders: document.querySelectorAll('input[type=range]').length,
    controlLabels: controls.slice(0, 14).map(label).filter(Boolean),
  });
})()
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


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")


def _count(value: Any, cap: int = 10**7) -> int | float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number != number or number < 0:  # NaN or negative
        return 0
    return min(number, cap) if not number.is_integer() else int(min(number, cap))


def _label(text: Any) -> str:
    """A control label as quoted data in a judge prompt: printable, short, no quotes or line breaks."""
    cleaned = "".join(ch for ch in str(text) if ch.isprintable()).replace("'", "").replace('"', "")
    return re.sub(r"\s+", " ", cleaned).strip()[:30]


def sanitize_probe(raw: dict[str, Any], dom: dict[str, Any]) -> dict[str, Any]:
    """Everything the page reported, as numbers and short names only (it is untrusted and goes into a prompt)."""

    def names(mapping: Any) -> dict[str, int]:
        if not isinstance(mapping, dict):
            return {}
        return {k: int(_count(v)) for k, v in list(mapping.items())[:200] if isinstance(k, str) and _NAME.match(k)}

    probe = {key: _count((raw or {}).get(key)) for key in
             ("audioContexts", "running", "audioSeconds", "sourcesStarted", "toDestination", "speech", "raf",
              "mutations")}  # fmt: skip
    probe["nodes"] = names((raw or {}).get("nodes"))
    probe["listeners"] = names((raw or {}).get("listeners"))
    for key in ("canvases", "svgs", "controls", "sliders"):
        probe[key] = int(_count((dom or {}).get(key)))
    labels = (dom or {}).get("controlLabels") or []
    probe["controlLabels"] = [x for x in (_label(t) for t in labels[:14] if isinstance(t, str)) if x]
    return probe


async def _read_dom(page: Any) -> dict[str, Any]:
    """`READ_DOM_JS` in an isolated world of the page's main frame (CDP)."""
    session = await page.context.new_cdp_session(page)
    try:
        tree = await session.send("Page.getFrameTree")
        world = await session.send("Page.createIsolatedWorld", {"frameId": tree["frameTree"]["frame"]["id"],
                                                                 "worldName": "agent-distill-observer"})  # fmt: skip
        result = await session.send("Runtime.evaluate", {"expression": READ_DOM_JS,
                                                         "contextId": world["executionContextId"],
                                                         "returnByValue": True})  # fmt: skip
        value = (result.get("result") or {}).get("value")
        return json.loads(value) if isinstance(value, str) else {}
    finally:
        await session.detach()


async def observe_page(path: str | Path) -> Observation:
    path = Path(path).resolve()  # noqa: ASYNC240 - one small local file
    return await observe_html(path.read_text(encoding="utf-8", errors="replace"))


async def observe_html(html: str) -> Observation:
    """Play the page in the sandboxed browser; a page that hangs raises `TimeoutError` (the judge is unscorable)."""
    obs = Observation()
    await asyncio.wait_for(_observe(html, obs), timeout=PAGE_SECONDS)
    return obs


async def _observe(html: str, obs: Observation) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await launch(playwright, ("--autoplay-policy=no-user-gesture-required",))
        try:
            context = await page_context(browser, html, viewport=DESKTOP, init_script=PROBE_JS)
            page = await context.new_page()
            await page.goto(PAGE_URL, wait_until="load", timeout=15000)
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
            obs.probe = sanitize_probe(await page.evaluate(READ_PROBE_JS), await _read_dom(page))
            obs.filmstrip = _contact_sheet(frames)
            await page.set_viewport_size(PHONE)
            await page.wait_for_timeout(500)
            obs.phone = await page.screenshot()
            obs.log = _log(obs.probe, obs.steps, seconds)
        finally:
            await close_quietly(browser)


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
