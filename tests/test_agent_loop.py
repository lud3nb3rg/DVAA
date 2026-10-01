"""The agent loop's contract with the model.

These are the tests that were missing. Every LLM03 challenge test scripts the
model's next move directly, so the loop could send the provider a transcript the
model could not possibly follow and every one of them still passed. What the
loop *hands back to the model* needs asserting on its own.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agent import run_agent, tool_from
from app.llm.base import Completion, Message, ToolCall, Usage
from app.llm.fake import FakeProvider
from app.llm.registry import set_provider


def _tc(name, args, cid="call_1"):
    return ToolCall(id=cid, name=name, arguments=args)


def _drive(responder, tools, **kw):
    set_provider(FakeProvider(responder=responder))
    try:
        return asyncio.run(run_agent("system", "go", tools, **kw))
    finally:
        set_provider(None)


PING = tool_from("ping", "pings", {"type": "object"}, lambda _a: "pong")


# --------------------------------------------------------------------------
# The transcript the model gets back
# --------------------------------------------------------------------------


def test_the_assistant_turn_carries_the_calls_it_made():
    """A tool result is a reply. The thing it replies to has to be there too.

    Without this the model receives a bare `assistant: ""` followed by tool
    output whose tool_call_id matches nothing in the conversation. It cannot
    tell what produced the output, or that it already ran the tool - so it runs
    it again, and invents arguments it should have read out of the last result.
    """
    seen: list[list[Message]] = []

    def responder(messages):
        seen.append(list(messages))
        if any(m.role == "tool" for m in messages):
            return Completion(text="done", usage=Usage(1, 1))
        return Completion(text="", tool_calls=[_tc("ping", {"n": 1})], usage=Usage(1, 1))

    _drive(responder, [PING])

    final_turn = seen[-1]
    assistant = [m for m in final_turn if m.role == "assistant"]
    tool_msgs = [m for m in final_turn if m.role == "tool"]
    assert assistant and assistant[0].tool_calls, "assistant turn lost its tool calls"
    assert tool_msgs
    # Every tool reply resolves to a call the assistant is shown making.
    call_ids = {c.id for m in assistant for c in m.tool_calls}
    assert {m.tool_call_id for m in tool_msgs} <= call_ids


def test_the_wire_format_pairs_calls_with_their_results():
    """The OpenAI-shaped dict is what Ollama and OpenAI actually receive."""
    assistant = Message(
        role="assistant", content="", tool_calls=[_tc("ping", {"n": 1}, "call_x")]
    )
    wire = assistant.to_dict()
    assert wire["tool_calls"][0]["id"] == "call_x"
    assert wire["tool_calls"][0]["function"]["name"] == "ping"
    # Arguments go over the wire as a JSON string, not a dict.
    assert json.loads(wire["tool_calls"][0]["function"]["arguments"]) == {"n": 1}

    reply = Message(role="tool", content="pong", tool_call_id="call_x", name="ping")
    assert reply.to_dict()["tool_call_id"] == wire["tool_calls"][0]["id"]


def test_a_turn_without_tool_calls_stays_a_plain_message():
    assert "tool_calls" not in Message(role="assistant", content="hi").to_dict()


def test_anthropic_turns_tool_calls_into_tool_use_blocks():
    """Anthropic rejects a tool_result whose tool_use_id has no tool_use."""
    from app.llm.cloud import anthropic_messages

    convo = anthropic_messages(
        [
            Message(role="system", content="sys"),
            Message(role="user", content="go"),
            Message(
                role="assistant", content="thinking", tool_calls=[_tc("ping", {}, "c1")]
            ),
            Message(role="tool", content="pong", tool_call_id="c1", name="ping"),
        ]
    )

    def blocks_of(kind):
        return [
            b
            for m in convo
            if isinstance(m["content"], list)
            for b in m["content"]
            if b.get("type") == kind
        ]

    tool_use, tool_result = blocks_of("tool_use"), blocks_of("tool_result")
    assert tool_use and tool_result
    assert tool_result[0]["tool_use_id"] == tool_use[0]["id"]
    assert tool_use[0]["name"] == "ping"
    # The system prompt goes in its own parameter, not the conversation.
    assert all(m["role"] != "system" for m in convo)


# --------------------------------------------------------------------------
# The ceilings
# --------------------------------------------------------------------------


def test_a_parallel_batch_of_calls_all_come_back_addressed():
    """Models emit several calls per turn; each result must name its own call."""
    seen: list[list[Message]] = []

    def responder(messages):
        seen.append(list(messages))
        if any(m.role == "tool" for m in messages):
            return Completion(text="done", usage=Usage(1, 1))
        return Completion(
            text="",
            tool_calls=[_tc("ping", {"n": i}, f"call_{i}") for i in range(3)],
            usage=Usage(1, 1),
        )

    _drive(responder, [PING])

    final_turn = seen[-1]
    assistant = next(m for m in final_turn if m.role == "assistant" and m.tool_calls)
    replies = [m for m in final_turn if m.role == "tool"]
    assert len(assistant.tool_calls) == 3
    assert [m.tool_call_id for m in replies] == ["call_0", "call_1", "call_2"]


@pytest.mark.parametrize("per_turn", [1, 5])
def test_a_no_progress_loop_ends_the_run(per_turn):
    def responder(messages):
        return Completion(
            text="",
            tool_calls=[_tc("ping", {}, f"c{i}") for i in range(per_turn)],
            usage=Usage(1, 1),
        )

    run = _drive(responder, [PING], max_steps=20)
    assert "agent stopped" in run.final_text
    assert run.tool_calls_made <= 16


# --------------------------------------------------------------------------
# Repeat detection counts repeats of a call, not repeats in general
# --------------------------------------------------------------------------


def test_alternating_calls_are_not_one_loop():
    """Two tools, each called twice, is not a model going in circles.

    The counter used to be a single number for the whole run, so a model working
    through several tools and revisiting each one hit the ceiling for repeating
    "a tool call" it had never repeated more than once.
    """
    calls = {"n": 0}

    def responder(messages):
        calls["n"] += 1
        if calls["n"] > 6:
            return Completion(text="done", usage=Usage(1, 1))
        name = "ping" if calls["n"] % 2 else "pong"
        return Completion(
            text="", tool_calls=[_tc(name, {"n": 1})], usage=Usage(1, 1)
        )

    run = _drive(
        responder,
        [PING, tool_from("pong", "pongs", {"type": "object"}, lambda _a: "ping")],
        max_steps=10,
        max_repeated_calls=3,
    )
    assert "repeated a tool call" not in run.final_text
    assert run.final_text == "done"


def test_one_call_repeated_past_its_budget_still_stops_the_run():
    def responder(messages):
        return Completion(text="", tool_calls=[_tc("ping", {"n": 1})], usage=Usage(1, 1))

    run = _drive(responder, [PING], max_steps=10, max_repeated_calls=3)
    assert "repeated a tool call" in run.final_text


def test_a_batch_of_tool_calls_answers_in_one_user_turn():
    """Anthropic wants every result for one assistant turn in the turn after it.

    The loop hands results over one message at a time, so a model that emitted
    three tool_use blocks produced three separate user messages - and the API
    rejects the first of them as a batch with two results missing.
    """
    from app.llm.cloud import anthropic_messages

    convo = anthropic_messages(
        [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                tool_calls=[_tc("a", {}, "c1"), _tc("b", {}, "c2"), _tc("c", {}, "c3")],
            ),
            Message(role="tool", content="ra", tool_call_id="c1", name="a"),
            Message(role="tool", content="rb", tool_call_id="c2", name="b"),
            Message(role="tool", content="rc", tool_call_id="c3", name="c"),
        ]
    )

    assistant = next(m for m in convo if m["role"] == "assistant")
    following = convo[convo.index(assistant) + 1]
    assert following["role"] == "user"
    ids = [b["tool_use_id"] for b in following["content"] if b["type"] == "tool_result"]
    assert ids == ["c1", "c2", "c3"]
    # And nothing trails behind it.
    assert convo.index(following) == len(convo) - 1


def test_a_later_user_turn_does_not_absorb_stray_results():
    """Coalescing must not reach back past an assistant turn."""
    from app.llm.cloud import anthropic_messages

    convo = anthropic_messages(
        [
            Message(role="assistant", content="", tool_calls=[_tc("a", {}, "c1")]),
            Message(role="tool", content="ra", tool_call_id="c1", name="a"),
            Message(role="assistant", content="", tool_calls=[_tc("a", {}, "c2")]),
            Message(role="tool", content="rb", tool_call_id="c2", name="a"),
        ]
    )
    user_turns = [m for m in convo if m["role"] == "user"]
    assert len(user_turns) == 2
    assert [b["tool_use_id"] for m in user_turns for b in m["content"]] == ["c1", "c2"]
