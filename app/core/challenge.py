"""Challenge plugin contract and auto-discovery.

Each OWASP category lives in its own package under `app/challenges/` and
exports a module-level `CHALLENGE`. `discover()` imports every such package and
builds the hub from what it finds, so categories can land one at a time without
touching the framework.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.core.levels import ALL_LEVELS, Level


@dataclass
class LevelDoc:
    """What a level implements, and how it fails.

    `bypass` is empty for LOW (nothing to bypass) and for IMPOSSIBLE (nothing
    that should work).
    """

    defense: str
    bypass: str = ""


@dataclass
class Challenge:
    """One OWASP category's scenario."""

    id: str  # "llm01"
    owasp_id: str  # "LLM01:2026"
    owasp_title: str  # "Prompt Injection"
    title: str  # "SupportBot Ticket Triage"
    tagline: str
    scenario: str  # markdown, shown above the challenge UI
    objective: str  # what the learner must achieve
    # LOW and IMPOSSIBLE are mandatory - the undefended build and the
    # architectural fix. MEDIUM and HIGH are each optional, because a mitigation
    # step only earns its place if it leaves a gap the player can actually get
    # through. A rung whose flag no payload can reach is not a hard level, it is
    # a dead end, and the honest fix is to not ship it.
    levels: dict[Level, LevelDoc]
    router: APIRouter
    template_dir: Path
    # Set when the scenario stands in for a train-time or distribution-time risk
    # that cannot be authentically exploited inside a web app.
    surrogate_note: str = ""
    tags: list[str] = field(default_factory=list)
    # Clears anything the player added to the scenario - tickets they filed,
    # mail the agent sent, examples they poisoned - so the next attempt starts
    # from the seeded world. Called on every level switch: a payload left over
    # from Low is otherwise still sitting there when the player reaches High.
    reset_scenario: Callable[[], None] | None = None

    @property
    def index(self) -> int:
        """Numeric rank, for hub ordering."""
        return int(self.id.removeprefix("llm"))

    @property
    def level_order(self) -> list[Level]:
        """The levels this challenge actually offers, easiest first."""
        return [lvl for lvl in ALL_LEVELS if lvl in self.levels]

    @property
    def scored_levels(self) -> list[Level]:
        """Offered levels that count toward the course (IMPOSSIBLE is a bounty)."""
        return [lvl for lvl in self.level_order if lvl is not Level.IMPOSSIBLE]

    def offers(self, level: Level) -> bool:
        return level in self.levels

    def unlocked_levels(self, solved: set[str]) -> list[Level]:
        """The rungs this player may switch to, given what they have captured.

        The scored rungs form a ladder: each one opens when the rung below it
        has been captured. That is not artificial difficulty - a player who
        jumps straight to High never experiences their Low payload dying, and
        that moment is the only thing this lab is actually teaching.

        IMPOSSIBLE is outside the ladder. It is a bounty with nothing to give
        up, so opening it early costs the player no discovery, and the
        architectural fix is worth reading whenever someone wants to.
        """
        if settings.unlock_all:
            return list(self.level_order)

        out: list[Level] = []
        cleared_below = True
        for lvl in self.level_order:
            if lvl is Level.IMPOSSIBLE:
                out.append(lvl)
                continue
            if cleared_below:
                out.append(lvl)
            cleared_below = cleared_below and lvl.value in solved
        return out

    def unlocked(self, level: Level, solved: set[str]) -> bool:
        return level in self.unlocked_levels(solved)

    def resolve_level(self, level: Level) -> Level:
        """The nearest level this challenge offers, at or below `level`.

        The hub's "set every challenge to X" has to mean something for a
        challenge that skips X. Clamping down keeps that control honest: ask for
        High on a challenge with no High and you land on its hardest rung below
        it, rather than silently staying wherever you were.
        """
        if self.offers(level):
            return level
        below = [lvl for lvl in self.level_order if ALL_LEVELS.index(lvl) < ALL_LEVELS.index(level)]
        return below[-1] if below else self.level_order[0]

    def resolve_unlocked_level(self, level: Level, solved: set[str]) -> Level:
        """`resolve_level`, then clamped again to what this player has unlocked.

        The hub-wide "set every challenge to X" has to stay honest for a player
        who has not climbed that far yet: asking for High across the board puts
        each challenge on its hardest *open* rung rather than either refusing
        the whole action or handing out a level the ladder has not reached.
        """
        target = self.resolve_level(level)
        unlocked = self.unlocked_levels(solved)
        if target in unlocked:
            return target
        below = [lvl for lvl in unlocked if ALL_LEVELS.index(lvl) < ALL_LEVELS.index(target)]
        return below[-1] if below else self.level_order[0]

    @property
    def full_title(self) -> str:
        return f"{self.owasp_id} {self.owasp_title}"

    @property
    def path(self) -> str:
        return f"/challenge/{self.id}"

    def level_doc(self, level: Level) -> LevelDoc:
        return self.levels.get(level, LevelDoc(defense="Not documented."))

    def reset(self) -> bool:
        """Restore the seeded scenario. Returns False if there is nothing to do."""
        if self.reset_scenario is None:
            return False
        self.reset_scenario()
        return True


class ChallengeRegistry:
    def __init__(self) -> None:
        self._challenges: dict[str, Challenge] = {}
        self._discovered = False

    def register(self, challenge: Challenge) -> None:
        if challenge.id in self._challenges:
            raise ValueError(f"Duplicate challenge id: {challenge.id}")
        missing = [
            lvl.value
            for lvl in (Level.LOW, Level.IMPOSSIBLE)
            if lvl not in challenge.levels
        ]
        if missing:
            raise ValueError(
                f"Challenge {challenge.id} is missing level docs for: {', '.join(missing)}"
            )
        if not any(lvl in challenge.levels for lvl in (Level.MEDIUM, Level.HIGH)):
            raise ValueError(
                f"Challenge {challenge.id} offers no defended level between "
                "Low and Impossible"
            )
        self._challenges[challenge.id] = challenge

    def get(self, challenge_id: str) -> Challenge | None:
        return self._challenges.get(challenge_id)

    def all(self) -> list[Challenge]:
        return sorted(self._challenges.values(), key=lambda c: c.index)

    def template_dirs(self) -> list[Path]:
        return [c.template_dir for c in self._challenges.values() if c.template_dir.is_dir()]

    def discover(self) -> None:
        """Import every package under app/challenges and register its CHALLENGE."""
        if self._discovered:
            return
        self._discovered = True

        import app.challenges as pkg

        for mod in pkgutil.iter_modules(pkg.__path__):
            if not mod.ispkg:
                continue
            module = importlib.import_module(f"app.challenges.{mod.name}.challenge")
            challenge = getattr(module, "CHALLENGE", None)
            if challenge is None:
                raise ImportError(
                    f"app.challenges.{mod.name}.challenge defines no CHALLENGE"
                )
            self.register(challenge)


registry = ChallengeRegistry()
