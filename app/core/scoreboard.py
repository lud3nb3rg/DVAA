"""Scoreboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.core.challenge import registry
from app.core.levels import Level
from app.db import Attempt, Solve, get_session

# LOW is the warm-up; HIGH requires defeating a real defense. IMPOSSIBLE is not
# part of the course - it is a bounty, worth more than the rest of a challenge
# put together, because capturing one means finding a hole nobody planted.
POINTS = {Level.LOW: 10, Level.MEDIUM: 25, Level.HIGH: 50, Level.IMPOSSIBLE: 100}


# Posts that are not attempts on the challenge: switching level, resetting the
# scenario or a meter, taking or giving back a throwaway session. Matched on the
# last path segment, so a challenge adding its own housekeeping route only has
# to name it here rather than opt into the counter.
NOT_AN_ATTEMPT = {
    "level",
    "reset-scenario",
    "reset",
    "newsession",
    "endsession",
    "clear",
}


def is_attempt(path: str) -> tuple[str, bool]:
    """(challenge_id, whether this POST counts as a try) for a request path."""
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2 or parts[0] != "challenge":
        return "", False
    challenge_id = parts[1]
    if registry.get(challenge_id) is None:
        return "", False
    if len(parts) < 3:  # the challenge page itself
        return challenge_id, False
    return challenge_id, parts[-1] not in NOT_AN_ATTEMPT


def record_attempt(session_id: str, challenge_id: str, level: Level) -> None:
    """Count one try. Read-then-write, so the unique index settles a race."""
    for attempt in (1, 2):
        with get_session() as s:
            row = s.exec(
                select(Attempt)
                .where(Attempt.session_id == session_id)
                .where(Attempt.challenge_id == challenge_id)
                .where(Attempt.level == level.value)
            ).first()
            if row is None:
                row = Attempt(
                    session_id=session_id, challenge_id=challenge_id, level=level.value
                )
            row.count += 1
            row.last_at = datetime.now(UTC)
            s.add(row)
            try:
                s.commit()
                return
            except IntegrityError:
                s.rollback()
                if attempt == 2:
                    raise


def attempts_for(session_id: str) -> dict[tuple[str, str], Attempt]:
    with get_session() as s:
        rows = s.exec(select(Attempt).where(Attempt.session_id == session_id)).all()
    return {(r.challenge_id, r.level): r for r in rows}


@dataclass
class RunRow:
    """One rung of the run summary."""

    challenge_id: str
    title: str
    level: Level
    tries: int
    captured: bool
    minutes: float | None  # first try to capture; None if never captured

    @property
    def clean(self) -> bool:
        """Captured in three tries or fewer - the badge worth chasing again."""
        return self.captured and self.tries <= 3


@dataclass
class Run:
    rows: list[RunRow]

    @property
    def tries(self) -> int:
        return sum(r.tries for r in self.rows)

    @property
    def captures(self) -> int:
        return sum(1 for r in self.rows if r.captured)

    @property
    def clean(self) -> int:
        return sum(1 for r in self.rows if r.clean)

    @property
    def efficiency(self) -> int:
        """Captures per hundred tries. Your number to beat next run."""
        return round(100 * self.captures / self.tries) if self.tries else 0


def run_summary(session_id: str) -> Run:
    """The run scored against itself: tries per capture, and time spent stuck.

    This is what replaces a leaderboard on a lab with one player on it. A
    leaderboard of one is a list with your name on it; a run summary is
    something you can beat.
    """
    attempts = attempts_for(session_id)
    with get_session() as s:
        solves = s.exec(select(Solve).where(Solve.session_id == session_id)).all()
    solved_at = {(x.challenge_id, x.level): x.solved_at for x in solves}

    rows: list[RunRow] = []
    for challenge in registry.all():
        for level in challenge.level_order:
            key = (challenge.id, level.value)
            attempt = attempts.get(key)
            when = solved_at.get(key)
            minutes = None
            if when is not None and attempt is not None:
                first = attempt.first_at
                if first.tzinfo is None:
                    first = first.replace(tzinfo=UTC)
                stamp = when if when.tzinfo else when.replace(tzinfo=UTC)
                minutes = max(0.0, (stamp - first).total_seconds() / 60)
            if attempt is None and when is None:
                continue  # never touched; not part of the run yet
            rows.append(
                RunRow(
                    challenge_id=challenge.id,
                    title=challenge.title,
                    level=level,
                    tries=attempt.count if attempt else 0,
                    captured=when is not None,
                    minutes=minutes,
                )
            )
    return Run(rows=rows)


@dataclass
class Row:
    player: str
    solves: int
    points: int
    last_solve: datetime | None


@dataclass
class Progress:
    solved: int
    total: int

    @property
    def percent(self) -> int:
        return round(100 * self.solved / self.total) if self.total else 0


def max_points() -> int:
    """Points obtainable by solving the lab as designed.

    IMPOSSIBLE is excluded: it is a bounty on top, so a player who lands one
    finishes above 100%. Challenges that ship fewer rungs contribute fewer
    points, so 100% stays reachable.
    """
    return sum(
        POINTS[level]
        for challenge in registry.all()
        for level in challenge.scored_levels
    )


def bounties_for(session_id: str) -> list[str]:
    """Challenge ids where this player captured the IMPOSSIBLE flag."""
    with get_session() as s:
        solves = s.exec(select(Solve).where(Solve.session_id == session_id)).all()
    return sorted(s.challenge_id for s in solves if s.level == Level.IMPOSSIBLE.value)


def leaderboard() -> list[Row]:
    with get_session() as s:
        solves = s.exec(select(Solve)).all()

    by_player: dict[str, Row] = {}
    for solve in solves:
        row = by_player.setdefault(solve.player, Row(solve.player, 0, 0, None))
        row.solves += 1
        row.points += POINTS.get(Level.parse(solve.level), 0)
        if row.last_solve is None or solve.solved_at > row.last_solve:
            row.last_solve = solve.solved_at

    return sorted(by_player.values(), key=lambda r: (-r.points, r.last_solve or datetime.max))


def progress_for(session_id: str) -> Progress:
    """How many challenges the player has solved at least once."""
    with get_session() as s:
        solves = s.exec(select(Solve).where(Solve.session_id == session_id)).all()
    return Progress(solved=len({solve.challenge_id for solve in solves}), total=len(registry.all()))


def last_capture_for(session_id: str) -> dict[str, datetime]:
    """challenge_id -> the most recent capture on it."""
    with get_session() as s:
        solves = s.exec(select(Solve).where(Solve.session_id == session_id)).all()
    latest: dict[str, datetime] = {}
    for solve in solves:
        if solve.challenge_id not in latest or solve.solved_at > latest[solve.challenge_id]:
            latest[solve.challenge_id] = solve.solved_at
    return latest


def points_by_level() -> dict[str, int]:
    """POINTS keyed by the level's plain string, for templates.

    Enum members hash by name, so a template doing `points['low']` would miss.
    """
    return {level.value: points for level, points in POINTS.items()}


def solves_for(session_id: str) -> dict[str, set[str]]:
    """challenge_id -> set of solved level names, for the hub grid."""
    with get_session() as s:
        solves = s.exec(select(Solve).where(Solve.session_id == session_id)).all()
    out: dict[str, set[str]] = {}
    for solve in solves:
        out.setdefault(solve.challenge_id, set()).add(solve.level)
    return out


def points_for(session_id: str) -> int:
    with get_session() as s:
        solves = s.exec(select(Solve).where(Solve.session_id == session_id)).all()
    return sum(POINTS.get(Level.parse(solve.level), 0) for solve in solves)
