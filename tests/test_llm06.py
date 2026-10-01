"""LLM06 Unbounded Consumption — two-sided check.

The meter bills the model's reported usage on the chunk it really processed, so
these tests script a model with a known appetite: `summarizer(n)` answers every
call with n completion tokens. That is the only thing the fake decides — the
chunk counts, the caps, the refinement chain and the ceiling are all the
challenge's own arithmetic.
"""

from __future__ import annotations

import pytest

from app.challenges.llm06_unbounded_consumption.challenge import (
    BUDGET_USD,
    CHUNK_CAP,
    MAX_JOBS,
    RATE_LIMIT,
    SHAPE_CEILING,
    SHAPE_MAX_CHUNKS,
    shape_score,
)
from app.core.flags import flag_for
from app.core.levels import Level
from app.llm.base import Completion, Usage
from app.llm.fake import FakeProvider
from app.llm.registry import set_provider
from tests.conftest import set_level

CHALLENGE = "llm06"
PATH = f"/challenge/{CHALLENGE}"

# A document long enough to be split into several chunks on its own.
DOCUMENT = "Northwind field service log entry. " * 40

# What an attacker writes when they want every chunk to cost the maximum: the
# instruction is the lever the request-size validators never look at.
GREEDY = (
    "For every clause, write an exhaustive analysis, then restate it in three "
    "other registers, then critique each restatement at length."
)
HONEST = "Summarize this chunk in one sentence."

# DOCUMENT splits into 4 chunks, so `copies` multiplies by four.
BLUNT_COPIES = 40  # 160 chunks - four times what routine traffic ever produces
ROUTINE_COPIES = 10  # 40 chunks - exactly the shape ceiling, and no more


@pytest.fixture
def summarizer():
    """Install a model with a chosen output appetite. Returns the installer."""
    installed = []

    def install(completion_tokens: int):
        provider = FakeProvider(
            responder=lambda msgs: Completion(
                text="summary " * 20,
                usage=Usage(prompt_tokens=150, completion_tokens=completion_tokens),
            )
        )
        set_provider(provider)
        installed.append(provider)
        return provider

    yield install
    set_provider(None)


def run(client, **data) -> str:
    payload = {"document": DOCUMENT, "instruction": HONEST, "copies": 1, "jobs": 1}
    payload.update(data)
    resp = client.post(f"{PATH}/run", data=payload)
    assert resp.status_code == 200
    return resp.text


def session_id_of(client) -> str:
    """The session id this client is carrying.

    Read out of the session cookie rather than guessed from the database, so a
    test with two clients in it can tell them apart.
    """
    import base64
    import json

    from app.core.session import SESSION_KEY

    raw = client.cookies.get("session", "")
    payload = raw.split(".")[0]
    data = json.loads(base64.urlsafe_b64decode(payload + "=="))
    return data[SESSION_KEY]


def spend(client) -> float:
    """The client's current spend on the level it is set to."""
    from app.challenges.llm06_unbounded_consumption.challenge import _acct
    from app.core.levels import get_level

    sid = session_id_of(client)
    return _acct(sid, get_level(sid, CHALLENGE))["usd"]


# --------------------------------------------------------------------------
# The asymmetry itself
# --------------------------------------------------------------------------


def test_an_honest_job_costs_almost_nothing(client, summarizer):
    """The baseline the attack has to be measured against."""
    set_level(client, CHALLENGE, "low")
    summarizer(40)
    out = run(client, instruction=HONEST)
    assert flag_for(CHALLENGE, Level.LOW) not in out
    assert spend(client) < 0.05


# --------------------------------------------------------------------------
# LOW — one job, written to be expensive, blows the budget
# --------------------------------------------------------------------------


def test_low_a_single_greedy_batch_blows_the_budget(client, summarizer):
    set_level(client, CHALLENGE, "low")
    summarizer(900)
    out = run(client, instruction=GREEDY, copies=BLUNT_COPIES)
    assert flag_for(CHALLENGE, Level.LOW) in out


def test_low_the_instruction_is_what_makes_it_expensive(client, summarizer):
    """Same batch, terse instruction: the fan-out alone does not do it."""
    set_level(client, CHALLENGE, "low")
    summarizer(4)
    out = run(client, instruction=HONEST, copies=2)
    assert flag_for(CHALLENGE, Level.LOW) not in out


# --------------------------------------------------------------------------
# MEDIUM — the dashboard sees the batch, and prices nothing
# --------------------------------------------------------------------------


def test_medium_refuses_the_blunt_job_that_beat_low(client, summarizer):
    """The shape monitor does the one thing it can: it sees the batch."""
    set_level(client, CHALLENGE, "medium")
    summarizer(900)
    out = run(client, instruction=GREEDY, copies=BLUNT_COPIES)
    assert flag_for(CHALLENGE, Level.MEDIUM) not in out
    assert "batch-shape alarm" in out
    assert spend(client) == 0.0  # held before a token was spent


def test_medium_falls_to_generation_inside_a_routine_shape(client, summarizer):
    """The documented bypass: the dashboard prices nothing, so put the cost
    where it has no column - in what the model generates per chunk."""
    set_level(client, CHALLENGE, "medium")
    summarizer(900)
    out = run(client, instruction=GREEDY, copies=ROUTINE_COPIES, jobs=2)
    assert flag_for(CHALLENGE, Level.MEDIUM) in out


def test_medium_volume_alone_cannot_reach_the_budget(client, summarizer):
    """Routine-shaped jobs with a terse instruction, run until the rate limit
    stops them: the batch is not the lever any more."""
    set_level(client, CHALLENGE, "medium")
    summarizer(40)
    out = run(client, instruction=HONEST, copies=ROUTINE_COPIES, jobs=MAX_JOBS)
    assert flag_for(CHALLENGE, Level.MEDIUM) not in out
    assert spend(client) < BUDGET_USD


def test_the_shape_score_is_the_batch_against_routine_traffic(client):
    assert shape_score(SHAPE_MAX_CHUNKS) == SHAPE_CEILING
    assert shape_score(SHAPE_MAX_CHUNKS * 2) == SHAPE_CEILING * 2
    assert shape_score(1) < SHAPE_CEILING


def test_low_is_not_monitored_at_all(client, summarizer):
    """The blunt job the monitor stops at Medium is exactly what Low permits."""
    set_level(client, CHALLENGE, "low")
    summarizer(900)
    out = run(client, instruction=GREEDY, copies=BLUNT_COPIES)
    assert "batch-shape alarm" not in out
    assert flag_for(CHALLENGE, Level.LOW) in out


def test_medium_rate_limit_still_refuses_a_long_loop(client, summarizer):
    """The control it does implement, doing the one thing it can."""
    set_level(client, CHALLENGE, "medium")
    summarizer(2)
    out = run(client, instruction=HONEST, jobs=MAX_JOBS)
    assert "rate limit" in out
    assert flag_for(CHALLENGE, Level.MEDIUM) not in out


# --------------------------------------------------------------------------
# HIGH — every job is bounded; the number of jobs is not
# --------------------------------------------------------------------------


def test_high_bounds_the_job_that_beat_medium(client, summarizer):
    set_level(client, CHALLENGE, "high")
    summarizer(900)
    out = run(client, instruction=GREEDY, copies=ROUTINE_COPIES, jobs=2)
    assert flag_for(CHALLENGE, Level.HIGH) not in out
    # The circuit breaker is visible in the result, not just in the arithmetic.
    assert f"{CHUNK_CAP} processed" in out or "circuit breaker" in out
    assert spend(client) < BUDGET_USD


def test_high_falls_to_accumulation_across_jobs(client, summarizer):
    set_level(client, CHALLENGE, "high")
    summarizer(900)
    out = run(client, instruction=GREEDY, copies=ROUTINE_COPIES, jobs=MAX_JOBS)
    assert flag_for(CHALLENGE, Level.HIGH) in out


def test_high_shape_monitor_never_fires_on_a_stream_of_routine_jobs(client, summarizer):
    """Why an alert is not a control: the monitor scores each job on its own, so
    a workload that is anomalous only in aggregate is, to it, not anomalous."""
    set_level(client, CHALLENGE, "high")
    summarizer(900)
    out = run(client, instruction=GREEDY, copies=ROUTINE_COPIES, jobs=MAX_JOBS)
    assert "batch-shape alarm" not in out
    assert spend(client) >= BUDGET_USD


def test_high_caps_output_however_greedy_the_instruction(client, summarizer):
    set_level(client, CHALLENGE, "high")
    summarizer(9_000)
    run(client, instruction=GREEDY, copies=ROUTINE_COPIES)
    from app.challenges.llm06_unbounded_consumption.challenge import MAX_TOKENS_CAPPED

    # A model that ignored the budget still cannot be billed past it.
    assert f"{MAX_TOKENS_CAPPED} completion tokens" in run(
        client, instruction=GREEDY, copies=1
    )


# --------------------------------------------------------------------------
# IMPOSSIBLE — the ceiling halts inference before the tokens are spent
# --------------------------------------------------------------------------


def test_impossible_ceiling_cannot_be_spent_through(client, summarizer):
    set_level(client, CHALLENGE, "impossible")
    summarizer(900)
    out = ""
    for _ in range(6):
        out = run(client, instruction=GREEDY, copies=ROUTINE_COPIES, jobs=MAX_JOBS)
    assert flag_for(CHALLENGE, Level.IMPOSSIBLE) not in out
    assert "ceiling" in out
    assert spend(client) < BUDGET_USD


def test_impossible_refuses_before_calling_the_model(client, summarizer):
    """A ceiling that bills you and then complains is an alert, not a control."""
    set_level(client, CHALLENGE, "impossible")
    provider = summarizer(900)
    for _ in range(6):
        run(client, instruction=GREEDY, copies=ROUTINE_COPIES, jobs=MAX_JOBS)
    calls_before = len(provider.calls)
    run(client, instruction=GREEDY, copies=ROUTINE_COPIES, jobs=MAX_JOBS)
    assert len(provider.calls) == calls_before  # refused without a model call


# --------------------------------------------------------------------------
# The meter belongs to one session, on one level
# --------------------------------------------------------------------------


def test_each_level_bills_its_own_tab(client, summarizer):
    set_level(client, CHALLENGE, "low")
    summarizer(900)
    run(client, instruction=GREEDY, copies=BLUNT_COPIES)
    assert spend(client) >= BUDGET_USD

    set_level(client, CHALLENGE, "impossible")
    # Spend run up where there was no control must not carry into the level
    # whose control is the point.
    assert spend(client) == 0.0


def test_one_players_meter_is_not_another_players(client, summarizer):
    from fastapi.testclient import TestClient

    from app.main import app

    set_level(client, CHALLENGE, "low")
    summarizer(900)
    run(client, instruction=GREEDY, copies=BLUNT_COPIES)

    with TestClient(app) as other:
        set_level(other, CHALLENGE, "low")
        out = run(other, instruction=HONEST, copies=1)
        # A fresh session starts at zero however much the first one spent.
        assert flag_for(CHALLENGE, Level.LOW) not in out
        assert "$0.0" in out or "$0.00" in out


def test_reset_clears_only_the_active_level(client, summarizer):
    set_level(client, CHALLENGE, "low")
    summarizer(900)
    run(client, instruction=GREEDY, copies=BLUNT_COPIES)
    assert spend(client) >= BUDGET_USD
    client.post(f"{PATH}/reset")
    assert spend(client) == 0.0


def test_the_rate_limit_constant_is_what_the_level_doc_claims(client):
    from app.core.challenge import registry

    doc = registry.get(CHALLENGE).level_doc(Level.MEDIUM)
    assert str(RATE_LIMIT) in doc.defense
