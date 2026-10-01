"""Player session helpers.

A "session" here is just a scoreboard identity - it is not an auth boundary and
must not be treated as one. Several challenges hand the LLM a `current_user`
precisely so learners can watch what happens when authorization is delegated to
a model instead of enforced deterministically.
"""

from __future__ import annotations

import uuid

from fastapi import Request

SESSION_KEY = "dvaa_sid"
PLAYER_KEY = "dvaa_player"


def session_id(request: Request) -> str:
    sid = request.session.get(SESSION_KEY)
    if not sid:
        sid = uuid.uuid4().hex
        request.session[SESSION_KEY] = sid
    return sid


def player_name(request: Request) -> str:
    return request.session.get(PLAYER_KEY) or "anonymous"


def set_player_name(request: Request, name: str) -> None:
    cleaned = (name or "").strip()[:32] or "anonymous"
    request.session[PLAYER_KEY] = cleaned


# The scoreboard identity a player parked when they asked for a fresh session.
# LLM05 hands out a new session id to demonstrate that poisoned memory outlives
# a session - which used to mean the player's solves became unreachable, because
# the id they were scored under was overwritten and never shown to them again.
# Keeping the old one in the cookie makes that demonstration reversible.
PREV_SESSION_KEY = "dvaa_prev_sid"


def rotate_session(request: Request) -> str:
    """Issue a fresh scoreboard identity, parking the current one.

    The new id is genuinely new - nothing is carried across, which is the whole
    point when a challenge is showing you that its state is not session-scoped.
    The old id is remembered so `restore_session` can put the player back.
    """
    previous = request.session.get(SESSION_KEY)
    if previous:
        request.session[PREV_SESSION_KEY] = previous
    fresh = uuid.uuid4().hex
    request.session[SESSION_KEY] = fresh
    return fresh


def parked_session_id(request: Request) -> str:
    """The identity waiting to be resumed, or "" if there is none."""
    return request.session.get(PREV_SESSION_KEY) or ""


def restore_session(request: Request) -> str:
    """Return to the parked identity. Returns it, or "" if there was none."""
    parked = request.session.pop(PREV_SESSION_KEY, "")
    if parked:
        request.session[SESSION_KEY] = parked
    return parked
