"""Ollama provider - the zero-cost default.

Uses the OpenAI-compatible /v1/chat/completions endpoint that Ollama exposes,
which gives us tool calling on models that support it (llama3.1+, qwen2.5,
mistral-nemo) without a second code path.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import httpx

from app.config import settings
from app.llm.base import (
    Completion,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
    estimate_tokens,
)

log = logging.getLogger("dvaa.llm")

# Timeouts. A local model on a laptop is slow, but what used to fail here was
# not slowness, it was *length*: an unstreamed request has a single deadline for
# the whole answer, so a long generation and a wedged server are the same event.
# Ask a 7B model on CPU for two thousand tokens and it blows a budget a
# fifty-token answer never comes near, through no fault of the server.
#
# Streaming splits that one deadline into two questions that are actually
# different. CONNECT_TIMEOUT and the stall timeout ask "is anything happening?",
# and answer it from the gap between chunks - so an answer that keeps arriving
# is never cut off for being long. The generation ceiling then asks "is this
# ever going to end?", which is the only question a length should be judged
# against. LLM06 exists to let people ask for runaway generations; the provider
# should not be the thing that stops them.
CONNECT_TIMEOUT = 5.0


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        # (fetched_at, models) for the model listing below.
        self._tags_cache: tuple[float, list[str]] | None = None

    # How long a model listing is reused before the server is asked again.
    # Every page render needs this to draw the picker, and the call is a
    # blocking one on an async request path: with Ollama down it costs the
    # timeout, per page, and the event loop waits through all of it. Models are
    # pulled by hand every few minutes at most, so a short cache is free.
    _TAGS_TTL_SECONDS = 30.0

    def installed_models(self) -> list[str]:
        """Models this Ollama actually has pulled.

        Asked of the server rather than kept in a list here, so the picker can
        never offer something that would 404 on first use - but cached, so
        "asked of the server" does not mean once per page.
        """
        now = time.monotonic()
        cached = self._tags_cache
        if cached is not None and now - cached[0] < self._TAGS_TTL_SECONDS:
            return cached[1]

        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError):
            models: list[str] = []
        else:
            models = sorted(m.get("name", "") for m in data.get("models") or [] if m.get("name"))
        self._tags_cache = (now, models)
        return models

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # Ollama reports usage on the final event when asked for it. Older
            # builds ignore the field and send none; the estimate at the bottom
            # covers them, and LLM06 bills from that estimate as it always did.
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]

        # The read timeout is what makes this work: on a streamed response httpx
        # applies it per read, so it measures the gap between chunks rather than
        # the length of the answer. It still has to be generous - a cold model
        # loads into memory before the first token, and that is the longest
        # legitimate gap in an otherwise healthy request.
        timeout = httpx.Timeout(settings.ollama_stall_timeout, connect=CONNECT_TIMEOUT)
        deadline = time.monotonic() + settings.ollama_max_seconds

        text_parts: list[str] = []
        # Tool call fragments, keyed by the index each delta carries. A streamed
        # call arrives spread over many events: the name in one, then the
        # arguments a few characters at a time.
        partial_calls: dict[int, dict[str, Any]] = {}
        raw_usage: dict[str, Any] = {}
        cut_off = False

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions", json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        self._raise_for_response(resp)

                    async for line in resp.aiter_lines():
                        if time.monotonic() > deadline:
                            # Leaving the block closes the connection, which is
                            # how Ollama is told to stop generating.
                            cut_off = True
                            break
                        _consume_event(line, text_parts, partial_calls, raw_usage)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? "
                f"Try: ollama serve && ollama pull {self.model}"
            ) from exc
        except httpx.TimeoutException as exc:
            # Now this really is a stall, not a long answer: the server took the
            # request and then sent nothing at all for the whole window.
            raise ProviderError(
                f"Ollama accepted the request but sent nothing for "
                f"{settings.ollama_stall_timeout:.0f}s. '{self.model}' may be too large "
                "for this machine to load - try a smaller model from the picker, or "
                "raise DVAA_OLLAMA_STALL_TIMEOUT."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? "
                f"Try: ollama serve && ollama pull {self.model}"
            ) from exc

        text = "".join(text_parts)
        if cut_off:
            # Kept rather than raised. An answer that ran out of clock has still
            # produced real output and really cost the tokens it produced, and a
            # truncated answer teaches more than an error page does.
            log.warning(
                "ollama generation hit the %.0fs ceiling; returning %d chars",
                settings.ollama_max_seconds,
                len(text),
            )
            text += (
                f"\n\n[cut off after {settings.ollama_max_seconds:.0f}s - the model was "
                "still generating. Raise DVAA_OLLAMA_MAX_SECONDS to let it run longer.]"
            )

        if settings.debug_provider:
            # The reassembled turn, before any of our parsing touches the tool
            # calls. This is what settles whether a call arrived with empty
            # arguments because the model sent none, or because a fragment was
            # dropped on the way through.
            log.warning(
                "ollama streamed message: %s",
                json.dumps({"content": text, "tool_calls": list(partial_calls.values())})[:4000],
            )

        calls = [
            ToolCall(
                id=slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                name=slot["name"],
                arguments=_decode_arguments(slot["arguments"]),
            )
            for _, slot in sorted(partial_calls.items())
            if slot["name"]
        ]

        usage = Usage(
            prompt_tokens=raw_usage.get("prompt_tokens")
            or sum(estimate_tokens(m.content) for m in messages),
            completion_tokens=raw_usage.get("completion_tokens") or estimate_tokens(text),
        )

        return Completion(text=text, tool_calls=calls, usage=usage)

    def _raise_for_response(self, resp: httpx.Response) -> None:
        """Turn an error response into the ProviderError it deserves."""
        if resp.status_code == 404:
            raise ProviderError(
                f"Ollama has no model named '{self.model}'. Pull it with: ollama pull {self.model}"
            )
        raise ProviderError(f"Ollama returned {resp.status_code}: {resp.text[:400]}")


def _consume_event(
    line: str,
    text_parts: list[str],
    partial_calls: dict[int, dict[str, Any]],
    raw_usage: dict[str, Any],
) -> None:
    """Fold one server-sent event into the turn being assembled."""
    line = line.strip()
    if not line.startswith("data:"):
        return  # keep-alives, comments, and the blank line between events
    body = line[len("data:") :].strip()
    if not body or body == "[DONE]":
        return
    try:
        event = json.loads(body)
    except json.JSONDecodeError:
        return

    if event.get("usage"):
        raw_usage.update(event["usage"])

    for choice in event.get("choices") or []:
        # "message" for the odd build that answers a stream request with one
        # whole non-streamed choice.
        delta = choice.get("delta") or choice.get("message") or {}
        if delta.get("content"):
            text_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            slot = partial_calls.setdefault(
                tc.get("index", len(partial_calls)),
                {"id": "", "name": "", "arguments": ""},
            )
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            args = fn.get("arguments")
            if isinstance(args, str):
                slot["arguments"] += args
            elif args:
                # Some builds send the whole object at once rather than
                # streaming it as text fragments.
                slot["arguments"] = args


def _decode_arguments(raw: Any) -> dict[str, Any]:
    """A tool call's arguments, however they arrived."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}
