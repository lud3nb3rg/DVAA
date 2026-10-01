"""Provider-agnostic LLM interface.

Every challenge talks to this and nothing lower. Adding a provider never
requires touching challenge code, and the test suite swaps in `FakeProvider`
so exploits are reproducible without a model running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    # The calls an assistant turn made. These have to travel back to the model
    # with the turn: a `tool` message is a *reply* to one of them, matched by
    # tool_call_id, and a transcript that carries the replies without the calls
    # shows the model a pile of results it cannot attribute to anything it did.
    # It then re-runs tools it has already run and invents arguments it should
    # have read out of an earlier result.
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = [c.to_openai() for c in self.tool_calls]
        return d


@dataclass
class ToolSpec:
    """A tool offered to the model."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Completion:
    """One model turn."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # Some challenges (LLM02) deliberately expose this as a side channel.
    reasoning: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ProviderError(RuntimeError):
    """Raised when the backing model is unreachable or misconfigured."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion: ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate.

    Good enough for the LLM06 cost meter; not accurate enough for billing.
    """
    return max(1, len(text) // 4)
