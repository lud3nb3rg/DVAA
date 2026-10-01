"""The level ladder.

The scored rungs open one at a time, each when the rung below it is captured.
That is a game-design rule, not a security control: a player who clicks
straight to High never sees their Low payload die, and that moment is the only
thing this lab is really teaching.

Every other test file runs with DVAA_UNLOCK_ALL=1 (see conftest) because those
tests assert what a *level* does. This one turns the gate back on.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.core.challenge import registry
from app.core.flags import flag_for
from app.core.levels import Level

CHALLENGE = "llm01"
PATH = f"/challenge/{CHALLENGE}"


@pytest.fixture
def ladder(monkeypatch):
    """Run with the gate on, the way a player meets it."""
    registry.discover()  # tests that never touch the client still need the hub
    monkeypatch.setattr(settings, "unlock_all", False)
    yield


def capture(client, challenge_id: str, level: Level) -> None:
    resp = client.post(
        "/flag", data={"flag": flag_for(challenge_id, level), "return_to": "/"}
    )
    assert resp.status_code == 200


def active_level(page_text: str) -> str:
    """The rung the challenge page says it is on."""
    marker = "Active defense &mdash; "
    if marker not in page_text:
        marker = "Active defense — "
    return page_text.split(marker, 1)[1].split("<", 1)[0].strip()


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_a_fresh_player_has_only_low_and_impossible(ladder):
    challenge = registry.get(CHALLENGE)
    assert challenge.unlocked_levels(set()) == [Level.LOW, Level.IMPOSSIBLE]


def test_medium_is_refused_before_low_is_captured(ladder, client):
    resp = client.post(f"{PATH}/level", data={"level": "medium"})
    assert "Medium is locked" in resp.text
    assert active_level(client.get(PATH).text) == "Low"


def test_capturing_low_unlocks_medium(ladder, client):
    capture(client, CHALLENGE, Level.LOW)
    challenge = registry.get(CHALLENGE)
    assert Level.MEDIUM in challenge.unlocked_levels({"low"})

    resp = client.post(f"{PATH}/level", data={"level": "medium"})
    assert active_level(resp.text) == "Medium"


def test_high_still_needs_medium(ladder, client):
    capture(client, CHALLENGE, Level.LOW)
    resp = client.post(f"{PATH}/level", data={"level": "high"})
    assert "High is locked" in resp.text
    assert active_level(resp.text) == "Low"

    capture(client, CHALLENGE, Level.MEDIUM)
    resp = client.post(f"{PATH}/level", data={"level": "high"})
    assert active_level(resp.text) == "High"


def test_impossible_is_never_gated(ladder, client):
    """It yields nothing, so opening it early costs the player no discovery."""
    resp = client.post(f"{PATH}/level", data={"level": "impossible"})
    assert active_level(resp.text) == "Impossible"


def test_a_locked_rung_is_rendered_disabled(ladder, client):
    page = client.get(PATH).text
    medium = page.split('value="medium"', 1)[1].split(">", 1)[0]
    assert "disabled" in medium


def test_the_locked_rung_does_not_leak_its_defense_text(ladder, client):
    """Knowing a defense exists is fair. Reading it before you have earned the
    rung below hands over most of the puzzle."""
    challenge = registry.get(CHALLENGE)
    page = client.get(PATH).text
    assert challenge.level_doc(Level.MEDIUM).defense[:60] not in page


# --------------------------------------------------------------------------
# The hub-wide switch cannot walk around it
# --------------------------------------------------------------------------


def test_set_all_levels_clamps_to_what_is_unlocked(ladder, client):
    client.post("/levels", data={"level": "high"})
    for challenge in registry.all():
        assert active_level(client.get(challenge.path).text) == "Low"


def test_set_all_levels_follows_the_ladder_up(ladder, client):
    for challenge in registry.all():
        capture(client, challenge.id, Level.LOW)
    client.post("/levels", data={"level": "high"})
    for challenge in registry.all():
        # Everyone reaches Medium; nobody reaches High without capturing Medium,
        # and llm09 ships no High at all.
        assert active_level(client.get(challenge.path).text) == "Medium"


def test_set_all_levels_says_when_it_held_challenges_back(ladder, client):
    resp = client.post("/levels", data={"level": "high"})
    assert "hardest open rung" in resp.text


# --------------------------------------------------------------------------
# The escape hatch
# --------------------------------------------------------------------------


def test_unlock_all_opens_every_rung(client, monkeypatch):
    monkeypatch.setattr(settings, "unlock_all", True)
    challenge = registry.get(CHALLENGE)
    assert challenge.unlocked_levels(set()) == challenge.level_order
    resp = client.post(f"{PATH}/level", data={"level": "high"})
    assert active_level(resp.text) == "High"


# --------------------------------------------------------------------------
# Challenges that skip a rung
# --------------------------------------------------------------------------


def test_a_skipped_rung_does_not_block_the_ladder(ladder):
    """llm09 ships no High. Its Medium must still be the top of its ladder
    rather than a rung that can never open."""
    challenge = registry.get("llm09")
    assert Level.HIGH not in challenge.level_order
    assert challenge.unlocked_levels({"low"}) == [Level.LOW, Level.MEDIUM, Level.IMPOSSIBLE]
    assert challenge.resolve_unlocked_level(Level.HIGH, {"low"}) is Level.MEDIUM
