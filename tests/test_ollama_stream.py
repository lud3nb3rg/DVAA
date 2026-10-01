"""The Ollama provider's streamed transport.

The bug these guard against: a non-streamed request has one deadline for the
whole answer, so asking a local 7B model for a long one - which LLM06 exists to
make people do - failed as if the server were down. Streaming makes the timeout
measure the gap between chunks instead, and a separate wall clock decides
whether a generation is genuinely endless.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.llm.base import Message, ProviderError
from app.llm.ollama import OllamaProvider


def _event(**delta) -> bytes:
    payload = {"choices": [{"index": 0, "delta": delta}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _mock(handler, monkeypatch) -> list[dict]:
    """Point the provider at a MockTransport. Returns the requests it sends."""
    import app.llm.ollama as mod

    sent: list[dict] = []

    async def _handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    transport = httpx.MockTransport(_handle)
    real = httpx.AsyncClient

    def _client(**kw):
        kw.pop("transport", None)
        return real(transport=transport, **kw)

    monkeypatch.setattr(mod.httpx, "AsyncClient", _client)
    return sent


def _streaming(*chunks: bytes):
    async def _body():
        for chunk in chunks:
            yield chunk

    return lambda request: httpx.Response(200, content=_body())


async def test_it_asks_for_a_stream(monkeypatch):
    """The regression guard. `stream: False` is what reintroduces the deadline."""
    sent = _mock(_streaming(_event(content="hi"), b"data: [DONE]\n\n"), monkeypatch)

    await OllamaProvider().chat([Message(role="user", content="hello")])

    assert sent[0]["stream"] is True
    assert sent[0]["stream_options"] == {"include_usage": True}


async def test_a_long_answer_is_reassembled_in_order(monkeypatch):
    _mock(
        _streaming(
            _event(content="The digits "),
            b": keep-alive\n\n",
            _event(content="of pi "),
            _event(content="are 3.14159"),
            b'data: {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2048}}\n\n',
            b"data: [DONE]\n\n",
        ),
        monkeypatch,
    )

    out = await OllamaProvider().chat([Message(role="user", content="pi")])

    assert out.text == "The digits of pi are 3.14159"
    # Billed from what the server reported, not from our estimate. LLM06's meter
    # is only honest if this survives the switch to streaming.
    assert out.usage.prompt_tokens == 11
    assert out.usage.completion_tokens == 2048


async def test_usage_falls_back_to_an_estimate(monkeypatch):
    """Older Ollama builds ignore stream_options and report nothing."""
    _mock(_streaming(_event(content="x" * 400), b"data: [DONE]\n\n"), monkeypatch)

    out = await OllamaProvider().chat([Message(role="user", content="hello")])

    assert out.usage.completion_tokens == 100  # 400 chars / 4
    assert out.usage.prompt_tokens > 0


async def test_tool_calls_are_assembled_from_fragments(monkeypatch):
    """A streamed call arrives as a name, then arguments a few chars at a time."""

    def frag(**fn):
        return _event(tool_calls=[{"index": 0, "id": fn.pop("id", None), "function": fn}])

    _mock(
        _streaming(
            frag(id="call_1", name="send_email", arguments=""),
            frag(arguments='{"to": "ops@'),
            frag(arguments='northwind.example"}'),
            b"data: [DONE]\n\n",
        ),
        monkeypatch,
    )

    out = await OllamaProvider().chat([Message(role="user", content="mail ops")])

    assert out.wants_tools
    (call,) = out.tool_calls
    assert call.id == "call_1"
    assert call.name == "send_email"
    assert call.arguments == {"to": "ops@northwind.example"}


async def test_an_endless_generation_is_cut_off_and_kept(monkeypatch):
    """The wall clock, not the answer's length, is what ends a runaway job.

    And what it produced is returned rather than raised: the tokens were really
    spent, so LLM06 should really bill them.
    """
    import app.llm.ollama as mod

    # A clock that jumps 500s per reading: the deadline is set on the first,
    # survives one chunk, and has passed by the second.
    ticks = iter(range(0, 100_000, 500))
    monkeypatch.setattr(mod.time, "monotonic", lambda: float(next(ticks)))
    monkeypatch.setattr(mod.settings, "ollama_max_seconds", 900.0)
    _mock(
        _streaming(*[_event(content="3.14159 ") for _ in range(50)]),
        monkeypatch,
    )

    out = await OllamaProvider().chat([Message(role="user", content="pi forever")])

    assert out.text.startswith("3.14159 ")
    assert "cut off after 900s" in out.text
    assert out.usage.completion_tokens > 0


async def test_silence_still_reports_a_stall(monkeypatch):
    """A server that accepts and then says nothing is a different failure."""

    def _handler(request):
        raise httpx.ReadTimeout("no data", request=request)

    _mock(_handler, monkeypatch)

    with pytest.raises(ProviderError) as err:
        await OllamaProvider().chat([Message(role="user", content="hello")])
    assert "sent nothing" in str(err.value)
    assert "DVAA_OLLAMA_STALL_TIMEOUT" in str(err.value)


async def test_a_dead_server_still_says_so(monkeypatch):
    def _handler(request):
        raise httpx.ConnectError("refused", request=request)

    _mock(_handler, monkeypatch)

    with pytest.raises(ProviderError) as err:
        await OllamaProvider().chat([Message(role="user", content="hello")])
    assert "Is it running?" in str(err.value)


async def test_a_missing_model_still_says_so(monkeypatch):
    _mock(lambda request: httpx.Response(404, text="model not found"), monkeypatch)

    with pytest.raises(ProviderError) as err:
        await OllamaProvider(model="nope:1b").chat([Message(role="user", content="hi")])
    assert "ollama pull nope:1b" in str(err.value)
