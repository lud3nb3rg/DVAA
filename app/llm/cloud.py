"""Optional cloud providers.

Both are opt-in: `pip install -e ".[cloud]"` and set DVAA_PROVIDER plus the
matching API key. Note that LLM06 (Unbounded Consumption) is a *real* denial-of-
wallet challenge - running it against a metered API costs actual money. The
challenge warns about this at runtime.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from app.config import settings
from app.llm.base import (
    Completion,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)


def anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Map the shared transcript onto Anthropic's content-block format.

    Split out from the provider so it can be tested without an API key: the
    pairing between a `tool_use` block and the `tool_result` that answers it is
    the part that is easy to get wrong, and Anthropic rejects a `tool_result`
    whose `tool_use_id` matches no `tool_use`.
    """
    convo: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content,
            }
            # One assistant turn can emit several tool_use blocks, and Anthropic
            # wants every result for that batch in the SINGLE user turn that
            # follows it - a second user message carrying the second result is
            # rejected as a tool_use with no result. The loop hands them over one
            # at a time, so they are coalesced back together here.
            if convo and convo[-1]["role"] == "user" and isinstance(convo[-1]["content"], list):
                convo[-1]["content"].append(result)
            else:
                convo.append({"role": "user", "content": [result]})
        elif m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks += [
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in m.tool_calls
            ]
            convo.append({"role": m.role, "content": blocks})
        else:
            convo.append({"role": m.role, "content": m.content})
    return convo


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError(
                'Anthropic support not installed. Run: pip install -e ".[cloud]"'
            ) from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        self.client = AsyncAnthropic()
        self.model = model or settings.anthropic_model

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = anthropic_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.to_anthropic() for t in tools]

        resp = await self.client.messages.create(**kwargs)

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return Completion(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=resp.usage.input_tokens,
                completion_tokens=resp.usage.output_tokens,
            ),
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError(
                'OpenAI support not installed. Run: pip install -e ".[cloud]"'
            ) from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not set.")
        self.client = AsyncOpenAI()
        self.model = model or settings.openai_model

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]

        resp = await self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(
                ToolCall(
                    id=tc.id or f"call_{uuid.uuid4().hex[:8]}",
                    name=tc.function.name,
                    arguments=args if isinstance(args, dict) else {"_raw": args},
                )
            )

        return Completion(
            text=msg.content or "",
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            ),
        )
