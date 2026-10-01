"""A small provider-agnostic tool-calling agent.

This is the shared machinery for the agent-based challenges (LLM02, LLM03).
It is deliberately faithful to how a real ReAct-style loop works, because the
vulnerabilities being taught live in that structure:

  * tool outputs re-enter the context window (LLM01's propagation property),
  * the model chooses which tool to call and with what arguments,
  * and whether a call is *allowed* is a decision someone has to make.

Where that decision is made - in the loop, or delegated to the model - is the
entire subject of LLM03 Excessive Agency. The loop therefore exposes a
`policy` hook so a challenge can enforce authorization deterministically, or
deliberately fail to.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.llm import Completion, Message, ToolCall, ToolSpec, get_provider


@dataclass
class Tool:
    """A callable the agent may invoke."""

    spec: ToolSpec
    # Called with the parsed arguments; returns a string observation.
    run: Callable[[dict[str, Any]], Awaitable[str] | str]

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class PolicyDecision:
    allow: bool
    reason: str = ""
    # If set, the agent records this instead of running the tool.
    substitute_observation: str | None = None


# A policy sees the pending call and the run context, and rules on it.
Policy = Callable[[ToolCall, "AgentRun"], PolicyDecision]


def allow_all(call: ToolCall, run: AgentRun) -> PolicyDecision:
    """The null policy: whatever the model asks for, it gets."""
    return PolicyDecision(allow=True)


_REPEAT_NOTE = (
    "\n\n[harness] You already made this exact call in this run and the result "
    "has not changed. Repeating it will not get you anything new — call a "
    "different tool, call this one with different arguments, or give your final "
    "answer now."
)


def _nudge(observation: str, repeated: bool) -> str:
    """Tell the model, in-band, that it is going in circles.

    A model stuck in a loop is not reasoning about the observation any more; it
    is pattern-matching on a context window that looks the same every turn. The
    cheapest way out is to make the window stop looking the same.
    """
    return observation + _REPEAT_NOTE if repeated else observation


@dataclass
class Step:
    kind: str  # "tool_call" | "observation" | "final" | "blocked"
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""


@dataclass
class AgentRun:
    """The transcript and running state of one agent invocation.

    Challenges read `steps` to render the trace, and stash scenario state (the
    acting user, whether a sensitive tool fired) in `context`.
    """

    messages: list[Message]
    steps: list[Step] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    final_text: str = ""
    tool_calls_made: int = 0

    def tool_names_called(self) -> list[str]:
        return [s.name for s in self.steps if s.kind == "tool_call"]


async def run_agent(
    system_prompt: str,
    user_message: str,
    tools: list[Tool],
    *,
    policy: Policy = allow_all,
    context: dict[str, Any] | None = None,
    max_steps: int = 6,
    max_tool_calls: int = 16,
    max_repeated_calls: int = 3,
    temperature: float = 0.2,
) -> AgentRun:
    """Drive the model/tool loop to a final answer, or to one of its ceilings.

    Three ceilings, because a small model finds three different ways to not
    finish. `max_steps` bounds model turns. `max_tool_calls` bounds calls, which
    is a different number entirely once a model starts emitting a batch of them
    per turn. `max_repeated_calls` bounds calling the *same thing* over and over,
    which no step budget catches: the model is spending its turns and making no
    progress, and the run needs to say so rather than quietly produce an answer
    built on nothing. That last one is counted per call signature - alternating
    between two repeated calls is still two loops, not one long one.
    """
    provider = get_provider()
    tool_map = {t.name: t for t in tools}
    specs = [t.spec for t in tools]

    def stop(run: AgentRun, why: str) -> AgentRun:
        run.final_text = f"[agent stopped: {why}]"
        run.steps.append(Step(kind="final", content=run.final_text))
        return run

    run = AgentRun(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ],
        # `context if context is not None` - not `context or {}` - so a caller
        # that passes an empty dict keeps sharing that same dict (challenges
        # rely on tool closures writing into it).
        context=context if context is not None else {},
    )

    # How many times the model has gone back to each call signature it has
    # already made. Counted per signature, not in total: a model working through
    # three tools and revisiting each once is making progress, and a single
    # shared counter would stop it as though it were stuck in one loop.
    repeats: dict[str, int] = {}

    for _ in range(max_steps):
        completion: Completion = await provider.chat(
            run.messages, tools=specs, temperature=temperature, max_tokens=800
        )

        if not completion.wants_tools:
            run.final_text = completion.text or ""
            run.steps.append(Step(kind="final", content=run.final_text))
            return run

        # Record the assistant's tool-call turn *including the calls*, so the
        # tool results appended below are replies to something the model can see
        # it asked for.
        run.messages.append(
            Message(
                role="assistant",
                content=completion.text or "",
                tool_calls=list(completion.tool_calls),
            )
        )

        for call in completion.tool_calls:
            if run.tool_calls_made >= max_tool_calls:
                return stop(run, f"tool-call budget of {max_tool_calls} exhausted")

            signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
            repeated = signature in repeats
            repeats[signature] = repeats.get(signature, 0) + 1
            if repeated and repeats[signature] > max_repeated_calls:
                return stop(
                    run,
                    f"the model repeated a tool call it had already made "
                    f"{max_repeated_calls} times without making progress",
                )

            run.steps.append(
                Step(kind="tool_call", name=call.name, arguments=call.arguments)
            )
            run.tool_calls_made += 1

            decision = policy(call, run)
            if not decision.allow:
                observation = decision.substitute_observation or (
                    f"[blocked by policy] {decision.reason}"
                )
                observation = _nudge(observation, repeated)
                run.steps.append(Step(kind="blocked", name=call.name, content=observation))
            else:
                tool = tool_map.get(call.name)
                if tool is None:
                    observation = (
                        f"[error] no such tool: {call.name}. Available tools: "
                        f"{', '.join(tool_map)}"
                    )
                else:
                    try:
                        result = tool.run(call.arguments)
                        observation = await result if hasattr(result, "__await__") else result
                    except Exception as exc:
                        # A tool that raises is a tool *result*, not a 500. Models
                        # pass malformed arguments constantly, and a loop that dies
                        # on the first one takes the whole scenario with it - the
                        # player never finds out whether their attack would have
                        # worked, because the agent never got that far.
                        observation = (
                            f"[error] {call.name} failed: {exc.__class__.__name__}: {exc}"
                        )
                observation = _nudge(str(observation), repeated)
                run.steps.append(
                    Step(kind="observation", name=call.name, content=observation)
                )

            run.messages.append(
                Message(
                    role="tool",
                    content=str(observation),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )

    run.final_text = "[agent stopped: step limit reached]"
    run.steps.append(Step(kind="final", content=run.final_text))
    return run


def tool_from(
    name: str,
    description: str,
    parameters: dict[str, Any],
    run: Callable[[dict[str, Any]], Awaitable[str] | str],
) -> Tool:
    return Tool(spec=ToolSpec(name=name, description=description, parameters=parameters), run=run)


def dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, indent=indent)
