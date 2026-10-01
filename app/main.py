"""DVAA application entrypoint."""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import select
from starlette.middleware.sessions import SessionMiddleware

from app.config import BANNER, REPO_ROOT, enforce_safety_gate, settings
from app.core import flags, scoreboard
from app.core.challenge import registry
from app.core.levels import Level, get_level, set_all_levels, set_level
from app.core.rendering import render, render_challenge
from app.core.session import (
    SESSION_KEY,
    player_name,
    restore_session,
    session_id,
    set_player_name,
)
from app.db import init_db, reset_progress
from app.llm.registry import restore_model
from app.seed import seed_all

log = logging.getLogger("dvaa")

enforce_safety_gate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(BANNER)
    init_db()
    seed_all()
    try:
        restore_model()  # the player's last model choice, if it is still pulled
    except Exception as exc:  # a provider that cannot be reached is not fatal
        log.warning("Could not restore the selected model: %s", exc)
    # Printed on every boot so a stale container is visible in the logs too,
    # not only on the About page.
    log.info(
        "Registered %d challenges (code %s)", len(registry.all()), code_fingerprint()
    )
    print(f"code fingerprint: {code_fingerprint()}")
    yield


app = FastAPI(title="DVAA", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.middleware("http")
async def count_attempts(request: Request, call_next):
    """Count each try at a challenge, for the run summary.

    One place rather than ten: every challenge's attack surface is a POST under
    its own prefix, and the handful of housekeeping posts that are not attempts
    are named in `scoreboard.NOT_AN_ATTEMPT`. A challenge that adds a new
    housekeeping route adds it there; anything else counts, which is the right
    default for a metric whose whole job is to notice effort.

    It reads the session rather than creating one, so a request that arrives
    with no identity is simply not counted instead of minting a scoreboard
    identity from a middleware.
    """
    response = await call_next(request)
    if request.method != "POST" or response.status_code >= 400:
        return response
    challenge_id, counts = scoreboard.is_attempt(request.url.path)
    if not counts:
        return response
    sid = request.session.get(SESSION_KEY)
    if sid:
        scoreboard.record_attempt(sid, challenge_id, get_level(sid, challenge_id))
    return response


# Added last, so it wraps everything above: the counter above runs inside it and
# can see request.session.
app.add_middleware(SessionMiddleware, secret_key=settings.resolved_secret_key())
app.mount("/static", StaticFiles(directory=REPO_ROOT / "app" / "static"), name="static")

# Challenge routers are mounted at import time, *before* the generic
# /challenge/{challenge_id} fallback below. Starlette matches in registration
# order, so a fallback declared first would shadow every challenge.
registry.discover()
for _challenge in registry.all():
    app.include_router(_challenge.router, prefix=_challenge.path)


def _local_path(candidate: str, fallback: str = "/") -> str:
    """Reduce a form-supplied `return_to` to a path on this application.

    Every redirect target in DVAA comes back from a form field, and a field that
    can carry "https://elsewhere.example" is an open redirect - a real one, in
    the framework rather than in a challenge. Nothing here needs more than a
    path, so anything that could name another origin (a scheme, a "//" prefix,
    a backslash Windows browsers normalise to one) is discarded rather than
    parsed cleverly.
    """
    candidate = (candidate or "").strip()
    if not candidate.startswith("/"):
        return fallback
    if candidate.startswith("//") or candidate.startswith("/\\"):
        return fallback
    # Keep the path only: the message and its kind are appended below, and a
    # caller-supplied query string would just be a second place to inject them.
    return candidate.split("?", 1)[0].split("#", 1)[0] or fallback


def _back(path: str, text: str, kind: str = "info") -> RedirectResponse:
    target = _local_path(path)
    return RedirectResponse(f"{target}?{urlencode({'msg': text, 'kind': kind})}", status_code=303)


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """A hash of the source this process is actually running.

    `docker compose up` reuses an existing image, so editing a challenge and
    bringing the lab back up runs the old code with no indication that anything
    is stale - and a bug you already fixed keeps reproducing. This is the number
    that settles it: rebuild, and if it has not moved, the image did not either.
    """
    digest = hashlib.sha256()
    for path in sorted(REPO_ROOT.joinpath("app").rglob("*")):
        if path.suffix in (".py", ".html", ".js", ".css") and path.is_file():
            digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def lab_identity(request: Request) -> dict:
    """The three things a lost run can be lost by, printed side by side.

    A run disappears if the browser cookie changes or if the data directory is
    recreated, and from the hub those are indistinguishable - both just show
    zero solves. The salt fingerprint pins down which one happened: it is stable
    for the life of a data directory and different for a new one. It is a hash
    prefix, never the salt, because the salt is what every flag is derived from.
    """
    from app.db import Solve, get_session

    sid = session_id(request)
    salt = hashlib.sha256(settings.resolved_salt().encode()).hexdigest()[:12]
    with get_session() as s:
        solves = list(s.exec(select(Solve)).all())
    return {
        "session_id": sid,
        "salt_fingerprint": salt,
        "code_fingerprint": code_fingerprint(),
        "db_path": str(settings.db_file),
        "solves_here": sum(1 for row in solves if row.session_id == sid),
        "solves_total": len(solves),
    }


# --------------------------------------------------------------------------
# Hub
# --------------------------------------------------------------------------


@app.get("/")
async def hub(request: Request):
    sid = session_id(request)
    return render(
        request,
        "hub.html",
        {
            "levels_by_challenge": {c.id: get_level(sid, c.id) for c in registry.all()},
            "solves": scoreboard.solves_for(sid),
            "progress": scoreboard.progress_for(sid),
            "points": scoreboard.points_for(sid),
            "max_points": scoreboard.max_points(),
        },
    )


@app.get("/scoreboard")
async def show_scoreboard(request: Request):
    sid = session_id(request)
    rows = scoreboard.leaderboard()
    return render(
        request,
        "scoreboard.html",
        {
            # A leaderboard of one is a list with your name on it. Only show it
            # once there is actually a field to be part of; the run summary is
            # what a solo player is actually playing against.
            "rows": rows if len(rows) > 1 else [],
            "run": scoreboard.run_summary(sid),
            "solves": scoreboard.solves_for(sid),
            "last_capture": scoreboard.last_capture_for(sid),
            "points": scoreboard.points_for(sid),
            "max_points": scoreboard.max_points(),
            "progress": scoreboard.progress_for(sid),
            "bounties": scoreboard.bounties_for(sid),
            "level_points": scoreboard.points_by_level(),
        },
    )


@app.get("/about")
async def about(request: Request):
    return render(request, "about.html", {"lab": lab_identity(request)})


# --------------------------------------------------------------------------
# Player / levels / progress
# --------------------------------------------------------------------------


@app.post("/player")
async def update_player(request: Request, name: str = Form("")):
    set_player_name(request, name)
    return RedirectResponse("/", status_code=303)


@app.post("/model")
async def update_model(request: Request, model: str = Form(""), return_to: str = Form("/")):
    """Switch the model every challenge runs against.

    Lab-wide rather than per-session: there is one Ollama behind this and one
    model loaded in it, so pretending otherwise would only mean two players
    fighting over which weights are resident.
    """
    from app.llm.registry import set_model

    if set_model(model.strip()):
        return _back(return_to, f"Now running on {model.strip()}.", "ok")
    return _back(return_to, f"'{model.strip()}' is not installed on this Ollama.", "bad")


@app.post("/levels")
async def update_all_levels(request: Request, level: str = Form(...)):
    sid = session_id(request)
    lvl = Level.parse(level)
    targets = {
        c.id: c.resolve_unlocked_level(lvl, flags.solved_levels(sid, c.id))
        for c in registry.all()
    }
    changed = [c for c in registry.all() if get_level(sid, c.id) != targets[c.id]]
    set_all_levels(sid, lvl)
    for challenge in changed:
        challenge.reset()
    held_back = sum(1 for c in registry.all() if targets[c.id] is not c.resolve_level(lvl))
    if held_back:
        return _back(
            "/",
            f"Set to {lvl.label} where it is unlocked. {held_back} challenge(s) "
            "stayed on their hardest open rung.",
            "info",
        )
    return _back("/", f"All challenges set to {lvl.label}.", "info")


_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@app.post("/session")
async def resume_session(request: Request, sid: str = Form("")):
    """Adopt an existing session id, to pick a run back up.

    A session here is a scoreboard identity and nothing else - it guards no data
    and crosses no trust boundary (see app/core/session.py). Being able to type
    one in is how a player recovers a run after a lost cookie, and how they carry
    one between browsers.
    """
    candidate = sid.strip().lower()
    if not _SESSION_ID_RE.match(candidate):
        return _back("/about", "That is not a session id. Expected 32 hex characters.", "bad")
    request.session[SESSION_KEY] = candidate
    return _back("/", "Resumed that session.", "ok")


@app.post("/session/restore")
async def restore_parked_session(request: Request, return_to: str = Form("/")):
    """Go back to the run parked by a challenge that issued a fresh session.

    LLM05 hands out a new scoreboard identity to show that poisoned memory is
    not session-scoped. That is a real demonstration and it really does take
    your solves off the board - so it has to be one click to undo, not a lost
    run.
    """
    resumed = restore_session(request)
    if not resumed:
        return _back(return_to, "There is no parked run to go back to.", "info")
    return _back(return_to, "Back on your original run.", "ok")


@app.post("/reset")
async def reset(request: Request):
    reset_progress(session_id(request))
    for challenge in registry.all():
        challenge.reset()
    return _back("/", "Progress cleared and every scenario reset.", "info")


@app.post("/challenge/{challenge_id}/level")
async def update_level(request: Request, challenge_id: str, level: str = Form(...)):
    challenge = registry.get(challenge_id)
    if challenge is None:
        return RedirectResponse("/", status_code=303)
    sid = session_id(request)
    # Clamped, so a URL naming a level this challenge does not ship cannot park
    # a player on a rung with no page behind it.
    new_level = challenge.resolve_level(Level.parse(level))
    # The ladder. The template renders locked rungs as disabled, so reaching
    # here with one means a hand-posted form - answer it the same way, with the
    # reason rather than a silent clamp.
    solved = flags.solved_levels(sid, challenge_id)
    if not challenge.unlocked(new_level, solved):
        below = challenge.level_order[challenge.level_order.index(new_level) - 1]
        return _back(
            challenge.path,
            f"{new_level.label} is locked. Capture {below.label} first.",
            "bad",
        )
    # Reset on a *change* of level: a ticket filed at Low, or mail the agent
    # sent, should not still be sitting there at High. Re-selecting the level
    # you are already on is not a change, and must not throw away your work.
    if get_level(sid, challenge_id) != new_level:
        challenge.reset()
    set_level(sid, challenge_id, new_level)
    return RedirectResponse(challenge.path, status_code=303)


@app.post("/challenge/{challenge_id}/reset-scenario")
async def reset_scenario(request: Request, challenge_id: str):
    challenge = registry.get(challenge_id)
    if challenge is None:
        return RedirectResponse("/", status_code=303)
    if challenge.reset():
        return _back(challenge.path, "Scenario reset to its seeded state.", "info")
    return _back(challenge.path, "This challenge has no scenario state to reset.", "info")


# --------------------------------------------------------------------------
# Flag submission
# --------------------------------------------------------------------------


@app.post("/flag")
async def submit_flag(request: Request, flag: str = Form(""), return_to: str = Form("/")):
    sid = session_id(request)
    match = flags.identify(flag)

    if match is None:
        return _back(return_to, "That is not a valid flag.", "bad")

    challenge_id, level = match
    challenge = registry.get(challenge_id)
    label = challenge.owasp_id if challenge else challenge_id

    if flags.record_solve(sid, challenge_id, level, player_name(request)):
        pts = scoreboard.POINTS[level]
        return _back(return_to, f"Captured {label} at {level.label} — +{pts} points.", "ok")
    return _back(return_to, f"Already captured {label} at {level.label}.", "info")


# --------------------------------------------------------------------------
# Fallback for challenges that register no root handler of their own
# --------------------------------------------------------------------------


@app.get("/challenge/{challenge_id}")
async def challenge_index(request: Request, challenge_id: str):
    challenge = registry.get(challenge_id)
    if challenge is None:
        return RedirectResponse("/", status_code=303)
    return render_challenge(request, challenge, "challenge_shell.html")


def run() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    run()
