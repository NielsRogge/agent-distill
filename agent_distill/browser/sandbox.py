"""The browser every page inspection runs in (`preview`, the gates, the judge's observation).

Generated pages are untrusted code. They run in headless Chromium with:

- **No local files.** The page is served from a fake http origin (`PAGE_URL`, fulfilled from memory), never from a
  `file:` URL, so it cannot load or frame other local files (Chromium refuses `file:` from an http origin, and the
  router aborts anything that is not the page itself).
- **No network.** Every request except the page itself is recorded and aborted; WebSockets are recorded and closed;
  service workers are blocked; WebRTC may not use non-proxied UDP; and the browser's proxy is a dead local port, so
  anything that slips past the router still goes nowhere.
- **No credentials.** Chromium is started with a minimal environment (`browser_env`), not the stage's, which holds
  `HF_TOKEN` and judge keys.
- **Time limits.** A page that hangs its main thread (`while (true) {}` in a click handler) would stall Playwright
  calls forever; callers wrap a whole inspection in `asyncio.wait_for` with `PAGE_SECONDS` and close the browser
  with `close_quietly`.

Chromium runs with `--no-sandbox` because jobs run as root in containers, where its own sandbox cannot start; the
measures above are what stand between a page and the stage.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from collections.abc import Callable
from typing import Any

PAGE_URL = "http://page.agent-distill.invalid/index.html"
DEAD_PROXY = "http://127.0.0.1:9"  # the discard port: nothing listens there
PAGE_SECONDS = 120.0  # an inspection or observation that takes longer has hung
# Environment variables Chromium may see (fonts, locale); everything else, credentials included, stays behind.
BROWSER_ENV_KEEP = ("PATH", "LANG", "LANGUAGE", "LC_ALL", "TZ", "FONTCONFIG_PATH", "FONTCONFIG_FILE")
LOCAL_SCHEMES = ("data:", "blob:", "about:")
CHROMIUM_ARGS = [
    "--no-sandbox",
    "--disable-blink-features=AutomationControlled",  # pages should not see navigator.webdriver and act up
    "--proxy-bypass-list=<-loopback>",  # loopback goes to the dead proxy too
    "--force-webrtc-ip-handling-policy",
    "--webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--disable-background-networking",
    "--no-pings",
]


def browser_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in BROWSER_ENV_KEEP if key in os.environ}
    home = os.path.join(tempfile.gettempdir(), "agent-distill-browser-home")
    os.makedirs(home, exist_ok=True)
    env["HOME"] = home
    return env


async def launch(playwright: Any, extra_args: tuple[str, ...] = ()) -> Any:
    return await playwright.chromium.launch(
        executable_path=os.environ.get("AGENT_DISTILL_CHROMIUM") or None,
        args=[*CHROMIUM_ARGS, *extra_args],
        proxy={"server": DEAD_PROXY},
        env=browser_env(),
    )


async def page_context(
    browser: Any,
    html: str,
    *,
    viewport: dict[str, int],
    on_blocked: Callable[[str], None] = lambda url: None,
    init_script: str | None = None,
) -> Any:
    """A browser context that serves `html` at `PAGE_URL` and blocks (and reports) everything else."""
    context = await browser.new_context(viewport=viewport, device_scale_factor=1, service_workers="block")
    body = html.encode("utf-8")

    async def route(handle: Any) -> None:
        url = handle.request.url
        if url == PAGE_URL:
            await handle.fulfill(status=200, body=body, headers={"content-type": "text/html; charset=utf-8"})
        elif url.startswith(LOCAL_SCHEMES):
            await handle.continue_()
        else:
            on_blocked(url[:200])
            await handle.abort()

    async def websocket(ws: Any) -> None:
        on_blocked(ws.url[:200])
        await ws.close()

    await context.route("**/*", route)
    await context.route_web_socket(re.compile(".*"), websocket)
    if init_script:
        await context.add_init_script(init_script)
    return context


async def close_quietly(browser: Any) -> None:
    """Close a browser whose page may be stuck; never raise or wait long."""
    try:
        await asyncio.wait_for(browser.close(), timeout=15)
    except Exception:  # the driver tears the process down when the playwright context exits
        pass
