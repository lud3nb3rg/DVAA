"""The run summary.

A leaderboard is dead weight on a lab with one player on it, so a run is scored
against itself: tries per capture, how long a rung sat unsolved, and how many
captures were clean. These assert the counter behind those numbers counts the
right posts and only those.
"""

from __future__ import annotations

import pytest

from app.core import scoreboard
from app.core.flags import flag_for
from app.core.levels import Level
from tests.conftest import set_level
from tests.test_llm06 import session_id_of

CHALLENGE = "llm01"
PATH = f"/challenge/{CHALLENGE}"


def file_ticket(client, body: str = "hello") -> None:
    """One try: the player's own attack surface on llm01."""
    resp = client.post(f"{PATH}/ticket", data={"subject": "s", "body": body})
    assert resp.status_code == 200


def tries(client, challenge_id: str = CHALLENGE, level: str = "low") -> int:
    rows = scoreboard.attempts_for(session_id_of(client))
    row = rows.get((challenge_id, level))
    return row.count if row else 0


def summary(client) -> scoreboard.Run:
    return scoreboard.run_summary(session_id_of(client))


# --------------------------------------------------------------------------
# What counts as a try
# --------------------------------------------------------------------------


def test_an_attack_post_is_counted(client):
    set_level(client, CHALLENGE, "low")
    file_ticket(client)
    file_ticket(client)
    assert tries(client) == 2


def test_housekeeping_posts_are_not_counted(client):
    """Switching level or resetting the scenario is not an attempt on it."""
    set_level(client, CHALLENGE, "low")
    client.post(f"{PATH}/reset-scenario")
    set_level(client, CHALLENGE, "low")
    assert tries(client) == 0


def test_a_get_is_not_counted(client):
    set_level(client, CHALLENGE, "low")
    client.get(PATH)
    assert tries(client) == 0


def test_tries_are_counted_per_level(client):
    set_level(client, CHALLENGE, "low")
    file_ticket(client)
    set_level(client, CHALLENGE, "impossible")  # never gated
    file_ticket(client)
    assert tries(client, level="low") == 1
    assert tries(client, level="impossible") == 1


@pytest.mark.parametrize(
    "path,counts",
    [
        ("/challenge/llm01/ticket", True),
        ("/challenge/llm01/triage", True),
        ("/challenge/llm01/level", False),
        ("/challenge/llm01/reset-scenario", False),
        ("/challenge/llm06/reset", False),
        ("/challenge/llm05/newsession", False),
        ("/challenge/llm01", False),
        ("/flag", False),
        ("/levels", False),
        ("/challenge/nope/ticket", False),
    ],
)
def test_the_path_classifier(client, path, counts):
    assert scoreboard.is_attempt(path)[1] is counts


# --------------------------------------------------------------------------
# The summary itself
# --------------------------------------------------------------------------


def test_an_untouched_rung_is_not_in_the_run(client):
    client.get("/")  # a session exists; nothing has been attempted in it
    assert summary(client).rows == []


def test_an_open_rung_shows_its_tries(client):
    set_level(client, CHALLENGE, "low")
    file_ticket(client)
    row = summary(client).rows[0]
    assert row.challenge_id == CHALLENGE
    assert row.level is Level.LOW
    assert row.tries == 1
    assert not row.captured
    assert row.minutes is None


def test_a_capture_in_three_tries_or_fewer_is_clean(client):
    set_level(client, CHALLENGE, "low")
    for _ in range(3):
        file_ticket(client)
    client.post("/flag", data={"flag": flag_for(CHALLENGE, Level.LOW), "return_to": "/"})
    row = summary(client).rows[0]
    assert row.captured
    assert row.clean
    assert row.minutes is not None and row.minutes >= 0


def test_a_grind_is_captured_but_not_clean(client):
    set_level(client, CHALLENGE, "low")
    for _ in range(4):
        file_ticket(client)
    client.post("/flag", data={"flag": flag_for(CHALLENGE, Level.LOW), "return_to": "/"})
    row = summary(client).rows[0]
    assert row.captured and not row.clean


def test_efficiency_is_captures_per_hundred_tries(client):
    set_level(client, CHALLENGE, "low")
    for _ in range(4):
        file_ticket(client)
    client.post("/flag", data={"flag": flag_for(CHALLENGE, Level.LOW), "return_to": "/"})
    run = summary(client)
    assert run.tries == 4
    assert run.captures == 1
    assert run.efficiency == 25


def test_an_empty_run_does_not_divide_by_zero(client):
    client.get("/")
    assert summary(client).efficiency == 0


def test_the_scoreboard_page_renders_the_run(client):
    set_level(client, CHALLENGE, "low")
    file_ticket(client)
    page = client.get("/scoreboard").text
    assert "This run" in page
    assert "SupportBot Ticket Triage" in page


def test_resetting_progress_clears_the_run(client):
    set_level(client, CHALLENGE, "low")
    file_ticket(client)
    client.post("/reset")
    assert summary(client).rows == []
    assert tries(client) == 0
