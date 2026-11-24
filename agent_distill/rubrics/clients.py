"""Judge clients (DESIGN.md §4, §10): `async complete(prompt, images) -> str` with a `model` attribute.

- `RouterJudge`: an OpenAI-compatible chat model on HF Inference Providers (e.g. Qwen3.8-27B), billed to `bill_to`.
- `GeminiJudge`: the Gemini API (`generateContent`), key sent in a header, never in the URL.

Keys are read from the environment variable a judge profile names, only here, in the stage process; agents never
see them. Transient failures (429, 5xx, timeouts) are retried with backoff; anything else raises, and the rubric
scores the episode as unscorable rather than switching judges silently (two judges' scores never mix in one
reward). Token usage and spend are tallied per client for the budget.
"""

from __future__ import annotations

import asyncio
import base64
import os
import random
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_distill.harnesses.base import HF_ROUTER

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class JudgeError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    budget_usd: float = 0.0  # 0 = no cap; past it every call fails (unscorable) and the stage's error stop ends it

    def check(self) -> None:
        if self.budget_usd and self.cost_usd >= self.budget_usd:
            raise JudgeError(f"judge budget reached: ${self.cost_usd:.2f} of ${self.budget_usd:.2f} spent")

    def add(self, tokens_in: int, tokens_out: int, price: tuple[float, float], cost: float | None = None) -> None:
        """Tally one call; `cost` is the provider's own figure (the HF router reports it), else priced tokens."""
        self.calls += 1
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        self.cost_usd += cost if cost is not None else (tokens_in * price[0] + tokens_out * price[1]) / 1e6


async def _post(url: str, *, headers: dict[str, str], body: dict[str, Any], timeout: float, retries: int) -> Any:  # noqa: ASYNC109
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            # A hard deadline on the whole request: a provider can hold a connection open without sending bytes,
            # which httpx's per-read timeout alone would wait out for every retry.
            async with asyncio.timeout(timeout), httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=body)
            if response.status_code < 400:
                return response.json()
            if response.status_code not in RETRY_STATUS or attempt == retries:
                # never echo request headers (they carry the key); the body is the provider's error message
                raise JudgeError(f"HTTP {response.status_code}: {response.text[:300]}")
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == retries:
                raise JudgeError(f"{type(exc).__name__}: {exc}") from exc
        await asyncio.sleep(delay + random.random())
        delay = min(delay * 2, 60)
    raise JudgeError("unreachable")


def _key(env: str) -> str:
    value = os.environ.get(env)
    if not value:
        raise JudgeError(f"the judge needs ${env}, which is not set")
    return value


@dataclass
class RouterJudge:
    model: str
    bill_to: str = ""
    key_env: str = "HF_TOKEN"
    base_url: str = HF_ROUTER
    price: tuple[float, float] = (0.0, 0.0)  # USD per 1M input / output tokens
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: float = 180
    retries: int = 4
    usage: Usage = field(default_factory=Usage)

    async def complete(self, prompt: str, images: list[bytes]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            data = base64.b64encode(image).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        self.usage.check()
        headers = {"Authorization": f"Bearer {_key(self.key_env)}"}
        if self.bill_to:
            headers["X-HF-Bill-To"] = self.bill_to
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_tokens, "temperature": self.temperature}  # fmt: skip
        reply = await _post(f"{self.base_url.rstrip('/')}/chat/completions", headers=headers, body=body,
                            timeout=self.timeout, retries=self.retries)  # fmt: skip
        usage = reply.get("usage") or {}
        cost = usage.get("cost")
        self.usage.add(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), self.price,
                       float(cost) if isinstance(cost, int | float) else None)  # fmt: skip
        try:
            choice = reply["choices"][0]
            text = str(choice["message"]["content"] or "")
        except (KeyError, IndexError) as exc:
            raise JudgeError(f"unexpected reply: {str(reply)[:300]}") from exc
        if not text.strip():
            raise JudgeError(f"empty reply (finish_reason {choice.get('finish_reason')})")
        return text


@dataclass
class GeminiJudge:
    model: str  # e.g. "gemini-4-neon"
    key_env: str = "GEMINI_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    price: tuple[float, float] = (0.0, 0.0)
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: float = 180
    retries: int = 4
    usage: Usage = field(default_factory=Usage)

    async def complete(self, prompt: str, images: list[bytes]) -> str:
        self.usage.check()
        parts: list[dict[str, Any]] = [{"text": prompt}]
        parts += [{"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode()}}
                  for image in images]  # fmt: skip
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"maxOutputTokens": self.max_tokens, "temperature": self.temperature}}  # fmt: skip
        reply = await _post(f"{self.base_url}/models/{self.model}:generateContent",
                            headers={"x-goog-api-key": _key(self.key_env)}, body=body, timeout=self.timeout,
                            retries=self.retries)  # fmt: skip
        usage = reply.get("usageMetadata") or {}
        # Thinking tokens are billed as output (`thoughtsTokenCount` is not part of `candidatesTokenCount`).
        output = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
        self.usage.add(int(usage.get("promptTokenCount") or 0), output, self.price)
        try:
            candidate = reply["candidates"][0]
            text = "".join(part.get("text", "") for part in (candidate.get("content") or {}).get("parts", []))
        except (KeyError, IndexError) as exc:
            raise JudgeError(f"unexpected reply: {str(reply)[:300]}") from exc
        if not text.strip():
            raise JudgeError(f"empty reply (finishReason {candidate.get('finishReason')})")
        return text


JUDGE_BUDGET_ENV = "AGENT_DISTILL_JUDGE_BUDGET_USD"  # set per job by the pipeline from `[budget] judge_usd`


def judge_client(profile: Any, bill_to: str = "", budget_usd: float | None = None) -> RouterJudge | GeminiJudge:
    """A client for a judge profile (`profiles/judges/*.toml`), capped at `budget_usd` (default: the job's
    `AGENT_DISTILL_JUDGE_BUDGET_USD`, 0 = no cap)."""
    price = (profile.price.input, profile.price.output)
    if budget_usd is None:
        budget_usd = float(os.environ.get(JUDGE_BUDGET_ENV) or 0)
    if budget_usd and profile.api == "gemini" and not any(price):
        raise ValueError(f"judge {profile.name} has no price in its profile, so its budget cannot be enforced")
    if profile.api == "gemini":
        key = profile.key_env or "GEMINI_API_KEY"
        client: RouterJudge | GeminiJudge = GeminiJudge(model=profile.model, key_env=key, price=price,
                                                        max_tokens=profile.max_tokens)  # fmt: skip
    elif profile.api == "router":
        client = RouterJudge(model=profile.model, bill_to=bill_to, key_env=profile.key_env or "HF_TOKEN", price=price,
                             max_tokens=profile.max_tokens)  # fmt: skip
    else:
        raise ValueError(f"judge {profile.name}: unknown api {profile.api!r}")
    client.usage.budget_usd = budget_usd
    return client
