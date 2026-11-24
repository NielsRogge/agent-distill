"""Key-injecting proxy for teacher endpoints (DESIGN.md §6).

The harness process never receives a real credential. A teacher endpoint is reached through a `KeyProxy` that runs
in the stage process on 127.0.0.1: the harness gets the proxy's URL and a random session key, and the proxy forwards
each request to the real endpoint with the real key (read from the stage process's environment, never written
anywhere) and the endpoint's extra headers (e.g. `X-HF-Bill-To`). Responses, including server-sent event streams,
are passed through chunk by chunk. Session keys can be minted per episode and revoked when the episode ends.

    with KeyProxy("https://router.huggingface.co/v1", "HF_TOKEN", {"X-HF-Bill-To": "my-org"}) as proxy:
        key = proxy.new_key()
        # the harness uses base URL proxy.url and API key `key`
"""

from __future__ import annotations

import hmac
import os
import secrets
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

# Never forwarded in either direction: hop-by-hop headers, the harness's own credential, and lengths/encodings
# that the proxy re-frames (responses are re-sent with chunked transfer encoding).
DROPPED = {
    "authorization", "connection", "content-length", "host", "keep-alive", "proxy-authorization",
    "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade", "accept-encoding", "content-encoding",
    "x-api-key", "api-key",
}  # fmt: skip


class KeyProxy:
    """A local reverse proxy to `upstream` that swaps the session key for the key in `$key_env`."""

    def __init__(
        self,
        upstream: str,
        key_env: str,
        headers: Mapping[str, str] | None = None,
        *,
        timeout: float = 600,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not os.environ.get(key_env):
            raise RuntimeError(f"${key_env} is not set in the stage process")
        self.upstream = upstream.rstrip("/")
        self.key_env = key_env
        self.headers = dict(headers or {})
        self._keys: set[str] = set()
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30), transport=transport)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __str__(self) -> str:  # never print a key
        return f"KeyProxy({self.url} -> {self.upstream})"

    def new_key(self) -> str:
        """A random session key the proxy accepts until it is revoked (it is worthless anywhere else)."""
        key = "agent-distill-" + secrets.token_urlsafe(24)
        with self._lock:
            self._keys.add(key)
        return key

    def revoke(self, key: str) -> None:
        with self._lock:
            self._keys.discard(key)

    def accepts(self, authorization: str | None) -> bool:
        presented = (authorization or "").removeprefix("Bearer ").strip()
        with self._lock:
            keys = list(self._keys)
        return any(hmac.compare_digest(presented.encode(), key.encode()) for key in keys)

    def start(self) -> KeyProxy:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._client.close()

    def __enter__(self) -> KeyProxy:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def forward(self, method: str, path: str, headers: Mapping[str, str], body: bytes) -> httpx.Response:
        """Send the request upstream with the real key; the caller streams and closes the response."""
        outgoing = {name: value for name, value in headers.items() if name.lower() not in DROPPED}
        outgoing.update(self.headers)
        outgoing["Authorization"] = f"Bearer {os.environ[self.key_env]}"
        outgoing["Accept-Encoding"] = "identity"
        request = self._client.build_request(method, self.upstream + path, headers=outgoing, content=body)
        return self._client.send(request, stream=True)


def _handler(proxy: KeyProxy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # request lines may carry query strings; keep stderr quiet
            pass

        def _reply(self, status: int, text: str) -> None:
            data = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> bytes:  # read in full first, so the connection stays usable after a 401
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                parts = []
                while size := int(self.rfile.readline().split(b";")[0].strip() or b"0", 16):
                    parts.append(self.rfile.read(size))
                    self.rfile.readline()
                while self.rfile.readline() not in (b"\r\n", b"\n", b""):  # trailers
                    pass
                return b"".join(parts)
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _proxy(self) -> None:
            body = self._body()
            key = self.headers.get("Authorization") or (
                f"Bearer {self.headers.get('x-api-key') or self.headers.get('api-key') or ''}"
            )
            if not proxy.accepts(key):
                self._reply(401, "invalid session key")
                return
            try:
                response = proxy.forward(self.command, self.path, dict(self.headers.items()), body)
            except httpx.HTTPError as exc:
                self._reply(502, f"upstream error: {type(exc).__name__}")
                return
            try:
                self.send_response(response.status_code)
                for name, value in response.headers.multi_items():
                    if name.lower() not in DROPPED:
                        self.send_header(name, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for chunk in response.iter_raw():
                    if chunk:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (httpx.HTTPError, OSError):
                self.close_connection = True
            finally:
                response.close()

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _proxy

    return Handler
