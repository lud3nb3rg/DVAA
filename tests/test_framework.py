"""Framework-level tests: pages render, levels switch, flags verify."""

from __future__ import annotations

import re

import pytest

from app.core.challenge import registry
from app.core.flags import flag_for, identify, verify
from app.core.levels import ALL_LEVELS, Level, get_level
from tests.conftest import set_level


def test_hub_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "OWASP GenAI/LLM Top 10" in resp.text


def test_static_pages_render(client):
    for url in ("/scoreboard", "/about"):
        assert client.get(url).status_code == 200


# --------------------------------------------------------------------------
# Lab identity: a run must be recoverable, and its salt must never be printed
# --------------------------------------------------------------------------


def _session_id_of(client) -> str:
    from app.core.session import SESSION_KEY

    client.get("/")  # any page mints the session cookie if there is none
    return _read_session(client)[SESSION_KEY]


def _read_session(client) -> dict:
    """Decode the signed session cookie the app handed this client."""
    import base64
    import json

    import itsdangerous

    from app.config import settings

    signer = itsdangerous.TimestampSigner(settings.resolved_secret_key())
    raw = signer.unsign(client.cookies["session"].encode(), max_age=None)
    return json.loads(base64.b64decode(raw))


def test_about_page_prints_the_lab_identity(client):
    from app.config import settings
    from app.main import code_fingerprint

    body = client.get("/about").text
    assert _session_id_of(client) in body
    # The fingerprint, never the salt itself - every flag is derived from it.
    assert settings.resolved_salt() not in body
    assert code_fingerprint() in body


def test_the_code_fingerprint_tracks_the_source(tmp_path):
    """The number that says whether a container is running current code.

    It has to change when a challenge changes, or it is worse than useless: it
    would confirm a stale image as fresh.
    """
    import hashlib

    from app.config import REPO_ROOT
    from app.main import code_fingerprint

    def recompute(skip: str = "") -> str:
        digest = hashlib.sha256()
        for path in sorted(REPO_ROOT.joinpath("app").rglob("*")):
            if path.suffix not in (".py", ".html", ".js", ".css") or not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel == skip:
                continue
            digest.update(rel.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()[:12]

    assert code_fingerprint() == recompute()
    # Drop one challenge from the hash and it must land somewhere else.
    assert recompute(skip="app/challenges/llm03_excessive_agency/challenge.py") != (
        code_fingerprint()
    )


def test_a_run_can_be_resumed_from_its_session_id(client):
    """The recovery path for a lost cookie, and the only reason the id is shown."""
    from app.core.flags import record_solve, solved_levels

    original = _session_id_of(client)
    record_solve(original, "llm01", Level.LOW, "tester")

    client.cookies.clear()
    assert _session_id_of(client) != original

    resp = client.post("/session", data={"sid": original})
    assert resp.status_code == 200
    assert _session_id_of(client) == original
    assert "low" in solved_levels(original, "llm01")


def test_a_parked_run_is_advertised_and_recoverable_from_any_page(client):
    """A challenge that hands out a throwaway identity must not lose the run.

    LLM05 rotates the session id to show that poisoned memory is not
    session-scoped. The scoreboard going empty is the demonstration; the run
    becoming unreachable was a bug, and the banner is what makes the difference
    visible while the player is standing in it.
    """
    from app.core.flags import record_solve

    original = _session_id_of(client)
    record_solve(original, "llm01", Level.LOW, "tester")

    client.post("/challenge/llm05/newsession")
    assert _session_id_of(client) != original

    hub = client.get("/").text
    assert original in hub  # the parked id, printed where the player can see it
    assert "go back to my run" in hub

    resp = client.post("/session/restore", data={"return_to": "/"})
    assert resp.status_code == 200
    assert _session_id_of(client) == original
    assert "go back to my run" not in client.get("/").text


def test_restoring_with_nothing_parked_is_a_no_op(client):
    original = _session_id_of(client)
    client.post("/session/restore", data={"return_to": "/"})
    assert _session_id_of(client) == original


@pytest.mark.parametrize("bad", ["", "not-hex", "abc", "z" * 32])
def test_a_malformed_session_id_is_refused(client, bad):
    original = _session_id_of(client)
    client.post("/session", data={"sid": bad})
    assert _session_id_of(client) == original


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------


class _SwitchableProvider:
    """Stands in for Ollama: has a model, and knows what is installed."""

    name = "ollama"

    def __init__(self):
        self.model = "small:1b"

    def installed_models(self):
        return ["big:7b", "small:1b"]

    async def chat(self, messages, tools=None, **kw):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def switchable():
    from app.llm.registry import set_provider

    provider = _SwitchableProvider()
    set_provider(provider)
    yield provider
    set_provider(None)


def test_the_model_picker_offers_what_is_installed(client, switchable):
    body = client.get("/").text
    assert 'action="/model"' in body
    assert "big:7b" in body and "small:1b" in body


def test_switching_model_takes_effect_and_persists(client, switchable):
    from app.db import Setting, get_session
    from app.llm.registry import MODEL_SETTING_KEY, restore_model

    client.post("/model", data={"model": "big:7b", "return_to": "/"})
    assert switchable.model == "big:7b"

    # Written down, so a restarted container comes back on the same model
    # instead of silently reverting and making a challenge behave differently.
    with get_session() as s:
        assert s.get(Setting, MODEL_SETTING_KEY).value == "big:7b"

    switchable.model = "small:1b"
    restore_model()
    assert switchable.model == "big:7b"


def test_a_model_that_is_not_installed_is_refused(client, switchable):
    resp = client.post("/model", data={"model": "hallucinated:70b", "return_to": "/"})
    assert switchable.model == "small:1b"
    assert "not installed" in resp.text


def test_a_provider_with_no_choice_shows_no_picker(client):
    """The fake provider, and any cloud one, have nothing to offer."""
    body = client.get("/").text
    assert 'action="/model"' not in body


def test_every_challenge_page_renders(client):
    for challenge in registry.all():
        resp = client.get(challenge.path)
        assert resp.status_code == 200, f"{challenge.id} failed to render"
        assert challenge.title in resp.text


def test_unknown_challenge_redirects_home(client):
    resp = client.get("/challenge/does-not-exist")
    assert resp.status_code == 200
    assert "OWASP GenAI/LLM Top 10" in resp.text


# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------


def test_flag_is_stable_and_scoped():
    a = flag_for("llm01", Level.LOW)
    assert a == flag_for("llm01", Level.LOW)
    assert a != flag_for("llm01", Level.MEDIUM)
    assert a != flag_for("llm02", Level.LOW)
    assert verify("llm01", Level.LOW, a)
    assert not verify("llm01", Level.MEDIUM, a)


def test_a_flag_mangled_by_the_model_still_verifies():
    """The High levels are beaten by making a model reshape the secret, and
    models lose punctuation when they do. Recovering the value is the win; the
    braces are not part of it."""
    flag = flag_for("llm02", Level.HIGH)
    digits = flag.removeprefix("DVAA{").removesuffix("}")

    variants = [
        " ".join(flag),  # one character per line / spaced out
        flag.replace("_", ""),  # underscores eaten
        flag.rstrip("}"),  # closing brace dropped
        flag.upper(),
        f"  {flag}\n",
        "DVAA{" + digits.replace("_", " ") + "}",
    ]
    for variant in variants:
        assert verify("llm02", Level.HIGH, variant), f"rejected {variant!r}"
        assert identify(variant) == ("llm02", Level.HIGH)


def test_a_wrong_digest_is_still_rejected_however_it_is_spelled():
    flag = flag_for("llm02", Level.HIGH)
    wrong = flag[:-2] + ("aa" if not flag.endswith("aa}") else "bb")
    assert not verify("llm02", Level.HIGH, wrong)
    assert not verify("llm02", Level.HIGH, "")
    assert not verify("llm02", Level.HIGH, "   ")
    # And a flag for the wrong level does not slide through normalization.
    assert not verify("llm02", Level.HIGH, flag_for("llm02", Level.MEDIUM))


def test_impossible_has_a_flag_that_verifies():
    """Impossible is a bounty, not an absence.

    The flag exists and scores; what makes it Impossible is that no intended
    path produces it. That half of the contract is tests/test_impossible.py.
    """
    for challenge in registry.all():
        flag = flag_for(challenge.id, Level.IMPOSSIBLE)
        assert flag.startswith("DVAA{")
        assert verify(challenge.id, Level.IMPOSSIBLE, flag)
        assert not verify(challenge.id, Level.IMPOSSIBLE, "DVAA{anything}")


def test_identify_matches_any_challenge_level():
    for challenge in registry.all():
        for level in challenge.level_order:
            assert identify(flag_for(challenge.id, level)) == (challenge.id, level)
    assert identify("DVAA{not_a_real_flag}") is None


def test_identify_rejects_a_level_the_challenge_does_not_ship():
    """`flag_for` derives a flag for any rung; only shipped rungs may score."""
    for challenge in registry.all():
        for level in ALL_LEVELS:
            if challenge.offers(level):
                continue
            assert identify(flag_for(challenge.id, level)) is None, (
                f"{challenge.id} scores a flag for {level}, which it does not ship"
            )


def test_flag_submission_records_a_solve(client):
    flag = flag_for("llm01", Level.LOW)
    resp = client.post("/flag", data={"flag": flag, "return_to": "/"})
    assert resp.status_code == 200
    assert "Captured" in resp.text or "10" in resp.text

    resp = client.post("/flag", data={"flag": flag, "return_to": "/"})
    assert "Already captured" in resp.text


def test_bad_flag_is_rejected(client):
    resp = client.post("/flag", data={"flag": "DVAA{nope}", "return_to": "/"})
    assert "not a valid flag" in resp.text


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("level", [lvl.value for lvl in ALL_LEVELS])
def test_level_switching_persists(client, level):
    challenge = registry.all()[0]
    set_level(client, challenge.id, level)
    resp = client.get(challenge.path)
    assert f'badge-{level}">{Level(level).label}' in resp.text or level in resp.text


def test_global_level_switch_applies_to_all(client):
    client.post("/levels", data={"level": "high"})
    for challenge in registry.all():
        resp = client.get(challenge.path)
        # Clamped down for a challenge that ships no High.
        assert challenge.resolve_level(Level.HIGH).label in resp.text


def test_a_level_a_challenge_does_not_ship_is_clamped(client):
    """Posting an unshipped level must not park the player on a dead rung."""
    for challenge in registry.all():
        for level in ALL_LEVELS:
            if challenge.offers(level):
                continue
            set_level(client, challenge.id, level.value)
            landed = get_level(_session_id_of(client), challenge.id)
            assert challenge.offers(landed)
            assert landed == challenge.resolve_level(level)


def test_bypass_text_is_absent_until_the_level_is_solved(client):
    """The weakness is the answer, so it is not in the page - at all.

    Not in a collapsed spoiler, not in the page source. It appears only after
    the level has been captured, as the write-up.
    """
    from markupsafe import escape

    for challenge in registry.all():
        for level in challenge.level_order:
            if not challenge.levels[level].bypass:
                continue
            set_level(client, challenge.id, level.value)
            body = client.get(challenge.path).text
            assert str(escape(challenge.levels[level].bypass)) not in body, (
                f"{challenge.id}/{level} ships its own answer in the page source"
            )


def test_bypass_text_appears_once_the_level_is_solved(client):
    challenge = registry.all()[0]
    set_level(client, challenge.id, "medium")
    client.post(
        "/flag",
        data={"flag": flag_for(challenge.id, Level.MEDIUM), "return_to": challenge.path},
    )

    from markupsafe import escape

    body = client.get(challenge.path).text
    assert str(escape(challenge.levels[Level.MEDIUM].bypass)) in body


def test_every_challenge_documents_every_level_it_ships():
    """Every rung a challenge ships is documented, and every middle rung has a gap.

    A challenge may skip Medium or High - see Challenge.levels - but it may not
    ship one without saying how it breaks, because a defended level with no way
    through is an Impossible under another name.
    """
    for challenge in registry.all():
        assert challenge.offers(Level.LOW)
        assert challenge.offers(Level.IMPOSSIBLE)
        for level in challenge.level_order:
            doc = challenge.levels[level]
            assert doc.defense.strip(), f"{challenge.id}/{level} has no defense doc"
        # Medium and High must state the gap - that gap is the lesson.
        for level in (Level.MEDIUM, Level.HIGH):
            if not challenge.offers(level):
                continue
            assert challenge.levels[level].bypass.strip(), (
                f"{challenge.id}/{level} documents no bypass"
            )
        # Low and Impossible should not claim one.
        assert not challenge.levels[Level.LOW].bypass
        assert not challenge.levels[Level.IMPOSSIBLE].bypass


# --------------------------------------------------------------------------
# The flag must never be sitting in the page source
# --------------------------------------------------------------------------


def test_flags_are_not_leaked_in_page_source(client):
    """A challenge page must not contain its own answer."""
    for challenge in registry.all():
        for level in challenge.level_order:
            set_level(client, challenge.id, level.value)
            body = client.get(challenge.path).text
            for probe_level in ALL_LEVELS:
                flag = flag_for(challenge.id, probe_level)
                assert flag not in body, (
                    f"{challenge.id} leaks its {probe_level} flag in page source "
                    f"while set to {level}"
                )


# --------------------------------------------------------------------------
# Redirect targets
#
# `return_to` is a form field on two POST routes. A form field that can name
# another origin is an open redirect, and this one is in the framework rather
# than in a challenge - nothing here is supposed to be exploitable.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "https://attacker.example/phish",
        "//attacker.example/phish",
        "http:/attacker.example",
        "javascript:alert(1)",
        "/\attacker.example",
    ],
)
def test_a_hostile_return_to_never_leaves_the_app(client, hostile):
    resp = client.post(
        "/flag",
        data={"flag": "DVAA{nope}", "return_to": hostile},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    location = resp.headers["location"]
    assert location.startswith("/")
    assert not location.startswith("//")
    assert "attacker.example" not in location


def test_a_local_return_to_is_preserved(client):
    resp = client.post(
        "/flag",
        data={"flag": "DVAA{nope}", "return_to": "/challenge/llm03"},
        follow_redirects=False,
    )
    assert resp.headers["location"].startswith("/challenge/llm03?")


# --------------------------------------------------------------------------
# One row per thing
#
# Both of these are check-then-write, so the database has to be the thing that
# decides the race. Two submissions of one flag must score once, and two level
# switches must leave one state row rather than two the reader picks between.
# --------------------------------------------------------------------------


def test_a_solve_cannot_be_recorded_twice(client):
    from sqlmodel import select

    from app.core.flags import record_solve
    from app.db import Solve, get_session

    sid = _session_id_of(client)
    assert record_solve(sid, "llm02", Level.LOW, "tester") is True
    assert record_solve(sid, "llm02", Level.LOW, "tester") is False

    with get_session() as s:
        rows = list(
            s.exec(
                select(Solve)
                .where(Solve.session_id == sid)
                .where(Solve.challenge_id == "llm02")
                .where(Solve.level == Level.LOW.value)
            ).all()
        )
    assert len(rows) == 1


def test_the_database_refuses_a_duplicate_solve_outright(client):
    """Not just the check above it: the constraint itself."""
    from sqlalchemy.exc import IntegrityError

    from app.core.flags import record_solve
    from app.db import Solve, get_session

    sid = _session_id_of(client)
    record_solve(sid, "llm04", Level.MEDIUM, "tester")

    with pytest.raises(IntegrityError):
        with get_session() as s:
            s.add(
                Solve(
                    session_id=sid,
                    challenge_id="llm04",
                    level=Level.MEDIUM.value,
                    player="tester",
                )
            )
            s.commit()


def test_repeated_level_changes_leave_one_state_row(client):
    from sqlmodel import select

    from app.db import ChallengeState, get_session

    sid = _session_id_of(client)
    for level in ("low", "high", "medium", "impossible", "low"):
        set_level(client, "llm07", level)

    with get_session() as s:
        rows = list(
            s.exec(
                select(ChallengeState)
                .where(ChallengeState.session_id == sid)
                .where(ChallengeState.challenge_id == "llm07")
            ).all()
        )
    assert len(rows) == 1


def test_state_and_level_share_one_row(client):
    """`set_state` must not create a second row beside the level's."""
    from sqlmodel import select

    from app.core.levels import get_level, get_state, set_state
    from app.db import ChallengeState, get_session

    sid = _session_id_of(client)
    set_level(client, "llm08", "medium")
    set_state(sid, "llm08", {"note": "scratch"})

    with get_session() as s:
        rows = list(
            s.exec(
                select(ChallengeState)
                .where(ChallengeState.session_id == sid)
                .where(ChallengeState.challenge_id == "llm08")
            ).all()
        )
    assert len(rows) == 1
    assert get_state(sid, "llm08") == {"note": "scratch"}
    assert get_level(sid, "llm08") is Level.MEDIUM


# --------------------------------------------------------------------------
# Scenario state belongs to the level it was made at
#
# `reset_scenario` is the framework's one hook for player-added state, and it
# fires on every level change and from the hub's reset. A challenge that lets a
# player add to its *scenario* and does not supply one silently carries that
# work across a level switch, so the next level is not the seeded world it
# claims to be. (Per-session, per-level scratch state - LLM06's meter, LLM08's
# transcript - is a different thing: it is already keyed by level, and its own
# button is the right place for it.)
# --------------------------------------------------------------------------


def test_injected_documents_do_not_survive_a_level_change(client, echo_context):
    """LLM09 was the challenge that had a reset button and no reset hook.

    The marker is chosen so it appears nowhere but the injected document itself:
    the query is echoed back on the page, so probing with the marker as the
    query would find it whether or not the document was still indexed.
    """
    marker = "POISONMARKER77"
    document = f"Zebra quartz ledger reconciliation notes {marker}"
    probe = {"q": "Zebra quartz ledger"}

    set_level(client, "llm09", "low")
    client.post("/challenge/llm09/poison", data={"text": document})
    assert marker in client.post("/challenge/llm09/query", data=probe).text

    set_level(client, "llm09", "impossible")
    assert marker not in client.post("/challenge/llm09/query", data=probe).text


def test_the_hub_reset_clears_injected_documents(client, echo_context):
    marker = "POISONMARKER88"
    set_level(client, "llm09", "low")
    client.post(
        "/challenge/llm09/poison",
        data={"text": f"Zebra quartz ledger reconciliation notes {marker}"},
    )
    client.post("/reset")
    resp = client.post("/challenge/llm09/query", data={"q": "Zebra quartz ledger"})
    assert marker not in resp.text


# --------------------------------------------------------------------------
# The over-permissioned SQL tool reads. It does not write.
#
# LLM03's run_report is meant to teach what an unscoped query can reach. A
# read-write handle would also let an injected DROP take the solves table with
# it: sqlite3 rolls an uncommitted DELETE back when the connection closes, but
# DDL runs in autocommit, so the damage would be permanent - collateral that
# teaches nothing the SELECT does not already teach.
# --------------------------------------------------------------------------


def test_the_raw_sql_handle_reads_everything_and_writes_nothing(client):
    import sqlite3

    from app.db import raw_connection

    with raw_connection() as conn:
        # Reading any table, unscoped, still works - that is the lesson.
        assert conn.execute("SELECT count(*) FROM hr_records").fetchone()[0] >= 0
        for destructive in (
            "DROP TABLE solves",
            "DELETE FROM solves",
            "UPDATE hr_records SET national_id = 'x'",
            "CREATE TABLE pwned (i int)",
        ):
            with pytest.raises(sqlite3.OperationalError):
                conn.execute(destructive)


def test_run_report_reports_a_write_attempt_instead_of_performing_it(client):
    from sqlmodel import select as sqlmodel_select

    from app.agent import AgentRun
    from app.challenges.llm03_excessive_agency.challenge import build_tools
    from app.db import Solve, get_session

    tools = {t.name: t for t in build_tools(Level.LOW, AgentRun(messages=[]))}
    assert "[sql error]" in tools["run_report"].run({"sql": "DROP TABLE solves"})

    # The table is still there to query, which it would not be if the DDL had
    # gone through: sqlite3 commits DDL immediately.
    with get_session() as s:
        s.exec(sqlmodel_select(Solve)).all()


# --------------------------------------------------------------------------
# Inline event handlers die under the CSP two challenge levels serve
# --------------------------------------------------------------------------

_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)
_INLINE_HANDLER = re.compile(r"\son[a-z]+\s*=", re.IGNORECASE)


def test_the_shared_chrome_uses_no_inline_event_handlers():
    """LLM10 serves its page under `script-src 'self'`.

    Every challenge page renders base.html and challenge_shell.html, so an on*=
    attribute in either works everywhere except there - and fails silently when
    it does, which is the worst way for the header's model picker to stop
    working. Page-specific templates that are never served under a policy are
    not covered by this.
    """
    from app.config import REPO_ROOT

    shared = REPO_ROOT / "app" / "templates"
    offenders = []
    for name in ("base.html", "challenge_shell.html"):
        body = _JINJA_COMMENT.sub("", (shared / name).read_text(encoding="utf-8"))
        for number, line in enumerate(body.splitlines(), 1):
            if _INLINE_HANDLER.search(line):
                offenders.append(f"{name}:{number}")
    assert not offenders, f"inline event handlers in the shared chrome: {offenders}"
