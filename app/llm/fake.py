"""Deterministic scripted provider.

Used by the test suite so every challenge's level-outcomes can be asserted
in CI with no model running. It is not offered as a user-facing mode:
a scripted model teaches nothing about prompt injection.

The script is a list of rules. The first rule whose `match` appears in the
most recent user/tool content wins and yields its canned completion.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from app.llm.base import Completion, Message, ToolCall, ToolSpec, Usage, estimate_tokens


@dataclass
class Rule:
    """A scripted response.

    `match` is a regex tested against the latest non-system message content.
    Either `text` or `tool_calls` (or both) is produced when it matches.
    """

    match: str
    text: str = ""
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    reasoning: str = ""


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        rules: list[Rule] | None = None,
        default: str = "I'm not sure how to help with that.",
        responder: Callable[[list[Message]], Completion] | None = None,
    ) -> None:
        self.rules = rules or []
        self.default = default
        self.responder = responder
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        self.calls.append(list(messages))

        if self.responder is not None:
            return self.responder(messages)

        haystack = "\n".join(m.content for m in messages if m.role != "system")

        for rule in self.rules:
            if re.search(rule.match, haystack, re.IGNORECASE | re.DOTALL):
                return self._build(rule, messages)

        return Completion(
            text=self.default,
            usage=Usage(
                prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
                completion_tokens=estimate_tokens(self.default),
            ),
        )

    @staticmethod
    def _build(rule: Rule, messages: list[Message]) -> Completion:
        return Completion(
            text=rule.text,
            reasoning=rule.reasoning,
            tool_calls=[
                ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args)
                for name, args in rule.tool_calls
            ],
            usage=Usage(
                prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
                completion_tokens=estimate_tokens(rule.text),
            ),
        )
