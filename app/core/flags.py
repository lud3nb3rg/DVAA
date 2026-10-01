"""CTF flags.

Flags are derived by HMAC from a per-install salt, so two workshop instances
never share answers, and a flag cannot be guessed from the challenge id alone.

`IMPOSSIBLE` has a flag like every other level, but nothing in DVAA is supposed
to emit it. The architectural fix removes the capability rather than filtering
it, so the intended path yields nothing at all. That makes the Impossible flag a
bug bounty: if you ever hold one, you found a way through a control that was
built to have no way through - an unintended vulnerability in this lab, not a
solution we planted. `tests/test_impossible.py` asserts that the intended paths
stay silent.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.config import settings
from app.core.levels import Level
from app.db import Solve, get_session

FLAG_RE = re.compile(r"DVAA\{[a-z0-9_]+\}")

_NOISE_RE = re.compile(r"[^a-z0-9]+")


def normalize(submitted: str) -> str:
    """Reduce a flag to the characters that actually carry it.

    Several levels are *designed* to be beaten by getting a model to emit the
    secret in a shape a filter does not recognise - spaced out, one character
    per line, reversed out of base64. Those transformations are lossy in the
    ways models are lossy: the braces go missing, the underscores get eaten.
    A player who has clearly recovered the secret should not then be told their
    flag is invalid because the model dropped a brace, so submissions are
    compared on their alphanumerics. The digest is the entropy; the punctuation
    around it was never the secret.
    """
    return _NOISE_RE.sub("", submitted.strip().lower())


def flag_for(challenge_id: str, level: Level | str) -> str:
    """The flag for a challenge at a level.

    Every level has one, IMPOSSIBLE included. Holding the Impossible flag means
    the architectural control leaked - see the module docstring.
    """
    level = Level.parse(str(level))

    digest = hmac.new(
        settings.resolved_salt().encode(),
        f"{challenge_id}:{level.value}".encode(),
        hashlib.sha256,
    ).hexdigest()[:16]
    return f"DVAA{{{challenge_id}_{level.value}_{digest}}}"


def verify(challenge_id: str, level: Level | str, submitted: str) -> bool:
    return hmac.compare_digest(
        normalize(flag_for(challenge_id, level)), normalize(submitted)
    )


def identify(submitted: str) -> tuple[str, Level] | None:
    """Match a submitted flag against every challenge/level pair.

    Lets the scoreboard accept a flag captured at one level even if the player
    has since switched the challenge to another. Only levels the challenge
    actually ships are matched: `flag_for` will happily derive a flag for a rung
    that does not exist, and awarding points for one would be free money.
    """
    from app.core.challenge import registry

    submitted = submitted.strip()
    for challenge in registry.all():
        for level in challenge.level_order:
            if verify(challenge.id, level, submitted):
                return challenge.id, level
    return None


def record_solve(session_id: str, challenge_id: str, level: Level, player: str) -> bool:
    """Record a solve. Returns False if it was already recorded.

    The check and the insert are two statements, so the unique index on
    (session, challenge, level) is what actually makes "already recorded" true:
    a second submission racing the first loses at commit time and is reported
    as the duplicate it is, rather than scoring twice.
    """
    with get_session() as s:
        existing = s.exec(
            select(Solve)
            .where(Solve.session_id == session_id)
            .where(Solve.challenge_id == challenge_id)
            .where(Solve.level == level.value)
        ).first()
        if existing:
            return False
        s.add(
            Solve(
                session_id=session_id,
                challenge_id=challenge_id,
                level=level.value,
                player=player,
            )
        )
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
            return False
    return True


def solved_levels(session_id: str, challenge_id: str) -> set[str]:
    with get_session() as s:
        rows = s.exec(
            select(Solve)
            .where(Solve.session_id == session_id)
            .where(Solve.challenge_id == challenge_id)
        ).all()
    return {r.level for r in rows}


def redact_flags(text: str) -> str:
    """Strip anything flag-shaped from text.

    Used where a challenge must echo untrusted content back without handing
    over a flag the player has not actually earned.
    """
    return FLAG_RE.sub("DVAA{redacted}", text)
