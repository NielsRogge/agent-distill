"""The teacher key proxy: the harness holds only a session key; the proxy adds the real key and billing header and
streams responses through (DESIGN.md §6). Upstream is a fake (httpx MockTransport); no network."""

from __future__ import annotations

import threading

import httpx
import pytest

from agent_distill.harnesses.key_proxy import KeyProxy

REAL_KEY = "hf_real_key_only_the_stage_process_has"


def test_proxy_injects_the_key_and_streams(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_KEY", REAL_KEY)
    seen: list[httpx.Request] = []
    first_chunk_read = threading.Event()
    streamed: list[bool] = []

    def events():
        yield b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n'
        # The second event is produced only once the client has read the first (a buffering proxy would time out).
        streamed.append(first_chunk_read.wait(timeout=5))
        yield b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())

    proxy = KeyProxy("https://router.example/v1", "TEACHER_KEY", {"X-HF-Bill-To": "some-org"},
                     transport=httpx.MockTransport(upstream))  # fmt: skip
    with proxy:
        key = proxy.new_key()
        assert REAL_KEY not in key and REAL_KEY not in str(proxy)
        body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        lines = []
        client = httpx.Client(trust_env=False, timeout=10)
        with client.stream("POST", f"{proxy.url}/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {key}"}) as response:  # fmt: skip
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            for line in response.iter_lines():
                if line:
                    lines.append(line)
                    first_chunk_read.set()
        assert streamed == [True] and lines[-1] == "data: [DONE]" and len(lines) == 3
        (request,) = seen
        assert str(request.url) == "https://router.example/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {REAL_KEY}"
        assert request.headers["x-hf-bill-to"] == "some-org"
        assert b'"stream": true' in request.content or b'"stream":true' in request.content

        def status(headers: dict[str, str]) -> int:
            return client.post(f"{proxy.url}/chat/completions", json=body, headers=headers).status_code

        assert status({}) == 401
        assert status({"Authorization": f"Bearer {REAL_KEY}"}) == 401  # only session keys are accepted
        assert status({"Authorization": "Bearer wrong"}) == 401
        proxy.revoke(key)
        assert status({"Authorization": f"Bearer {key}"}) == 401
        assert len(seen) == 1  # nothing unauthenticated reached upstream


def test_proxy_reports_upstream_errors_without_the_key(monkeypatch) -> None:
    monkeypatch.setenv("TEACHER_KEY", REAL_KEY)

    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}")

    with KeyProxy("https://router.example/v1", "TEACHER_KEY", transport=httpx.MockTransport(upstream)) as proxy:
        with httpx.Client(trust_env=False, timeout=10) as client:
            response = client.get(f"{proxy.url}/models", headers={"Authorization": f"Bearer {proxy.new_key()}"})
        assert response.status_code == 502 and REAL_KEY not in response.text


def test_proxy_needs_the_key_in_the_stage_process(monkeypatch) -> None:
    monkeypatch.delenv("TEACHER_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEACHER_KEY"):
        KeyProxy("https://router.example/v1", "TEACHER_KEY")
