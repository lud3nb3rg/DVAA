"""Difficulty levels.

A level is not a difficulty multiplier - it is a real implementation of a
progressively better OWASP mitigation. Medium and High are *deliberately*
bypassable via a specific, documented weakness; that bypass is the lesson.
"""

from __future__ import annotations

import json
from enum import StrEnum

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.config import settings
from app.db import ChallengeState, get_session


class Level(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMPOSSIBLE = "impossible"

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @property
    def blurb(self) -> str:
        return {
            Level.LOW: "No defense at all. The naive build.",
            Level.MEDIUM: "A plausible defense with a known weakness.",
            Level.HIGH: "A serious defense that still leaves a gap.",
            Level.IMPOSSIBLE: "The architectural fix. Its flag should be unreachable.",
        }[self]

    @classmethod
    def parse(cls, value: str | None, default: Level | None = None) -> Level:
        try:
            return cls(str(value).lower())
        except ValueError:
            return default or cls(settings.default_level)


ALL_LEVELS = [Level.LOW, Level.MEDIUM, Level.HIGH, Level.IMPOSSIBLE]


def _row(s, session_id: str, challenge_id: str) -> ChallengeState | None:
    return s.exec(
        select(ChallengeState)
        .where(ChallengeState.session_id == session_id)
        .where(ChallengeState.challenge_id == challenge_id)
        .order_by(ChallengeState.id)
    ).first()


def _update_state(session_id: str, challenge_id: str, **fields) -> None:
    """Write fields onto a session's row for a challenge, creating it if absent.

    Read-then-write, so two requests that arrive together can both find nothing
    and both insert. The unique index in `app/db.py` is what actually decides
    the race; this catches the loser's IntegrityError and re-runs it as the
    update it always meant to be.
    """
    for attempt in (1, 2):
        with get_session() as s:
            row = _row(s, session_id, challenge_id)
            if row is None:
                row = ChallengeState(
                    session_id=session_id,
                    challenge_id=challenge_id,
                    level=settings.default_level,
                )
            for key, value in fields.items():
                setattr(row, key, value)
            s.add(row)
            try:
                s.commit()
                return
            except IntegrityError:
                s.rollback()
                if attempt == 2:
                    raise


def get_level(session_id: str, challenge_id: str) -> Level:
    with get_session() as s:
        row = _row(s, session_id, challenge_id)
        if row:
            return Level.parse(row.level)
    return Level.parse(settings.default_level)


def set_level(session_id: str, challenge_id: str, level: Level) -> None:
    _update_state(session_id, challenge_id, level=level.value)


def set_all_levels(session_id: str, level: Level) -> None:
    """Set the level for every registered challenge at once (hub control).

    A challenge that does not ship the requested level is clamped to the
    hardest rung it does ship below it, rather than being left behind - and
    then clamped again to the hardest rung this player has actually unlocked,
    so the hub control cannot walk around the ladder the per-challenge switch
    enforces.
    """
    # Imported here rather than at module scope: `flags` imports this module.
    from app.core.challenge import registry
    from app.core.flags import solved_levels

    for challenge in registry.all():
        solved = solved_levels(session_id, challenge.id)
        set_level(session_id, challenge.id, challenge.resolve_unlocked_level(level, solved))


# --------------------------------------------------------------------------
# Per-challenge scratch state
# --------------------------------------------------------------------------


def get_state(session_id: str, challenge_id: str) -> dict:
    with get_session() as s:
        row = _row(s, session_id, challenge_id)
        if not row:
            return {}
        try:
            return json.loads(row.state_json)
        except json.JSONDecodeError:
            return {}


def set_state(session_id: str, challenge_id: str, state: dict) -> None:
    _update_state(session_id, challenge_id, state_json=json.dumps(state))
