"""A dependency-free client for talking to a deployed SGLang server.

Uses only the standard library, so it runs anywhere `modal` runs — including
inside a `@app.local_entrypoint()` used to smoke-test a deployment.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

__all__ = ["wait_for_server", "chat", "stream_chat"]

MINUTES = 60  # seconds


def _request(
    url: str,
    path: str,
    *,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    session_id: str | None = None,
) -> urllib.request.Request:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    if session_id is not None:
        # Modal Servers route requests sharing this header to the same replica,
        # which keeps a multi-turn conversation warm in the KV cache.
        request_headers["Modal-Session-ID"] = session_id
    return urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=request_headers,
    )


def wait_for_server(
    url: str,
    *,
    timeout: float = 15 * MINUTES,
    headers: dict[str, str] | None = None,
) -> None:
    """Block until the server answers, tolerating cold-start 503s.

    A Modal Server returns 503 while it has no live replica, so a fresh
    deployment legitimately looks down for as long as the model takes to load.
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                _request(url, "/health", headers=headers), timeout=10
            ) as response:
                if response.status < 400:
                    return
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code < 500:
                raise
        except Exception as error:  # noqa: BLE001 — cold start, keep polling
            last_error = error
        time.sleep(5)
    raise TimeoutError(f"{url} was not ready within {timeout}s ({last_error})")


def chat(
    url: str,
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = 256,
    temperature: float | None = None,
    timeout: float = 5 * MINUTES,
    headers: dict[str, str] | None = None,
    session_id: str | None = None,
    **extra,
) -> str:
    """Send an OpenAI-style chat completion and return the reply text."""
    payload: dict = {"messages": messages, "stream": False}
    for key, value in (
        ("model", model),
        ("max_tokens", max_tokens),
        ("temperature", temperature),
    ):
        if value is not None:
            payload[key] = value
    payload |= extra  # the escape hatch wins, as with SGLangConfig.extra_flags

    request = _request(
        url, "/v1/chat/completions", payload=payload, headers=headers, session_id=session_id
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"]


def stream_chat(
    url: str,
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = 256,
    timeout: float = 5 * MINUTES,
    headers: dict[str, str] | None = None,
    session_id: str | None = None,
    **extra,
) -> Iterator[str]:
    """Stream a chat completion, yielding text deltas as they arrive."""
    payload: dict = {"messages": messages, "stream": True}
    for key, value in (("model", model), ("max_tokens", max_tokens)):
        if value is not None:
            payload[key] = value
    payload |= extra

    request = _request(
        url,
        "/v1/chat/completions",
        payload=payload,
        headers={"Accept": "text/event-stream", **(headers or {})},
        session_id=session_id,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError:  # keepalive
                continue
            delta = (event.get("choices") or [{}])[0].get("delta") or {}
            if chunk := delta.get("content"):
                yield chunk
