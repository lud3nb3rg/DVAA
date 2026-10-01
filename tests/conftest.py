"""Test harness.

Every test runs against a scratch database and a scripted `FakeProvider`, so
every level-outcome is asserted deterministically with no model running.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Must be set before app.config is imported anywhere.
os.environ["DVAA_I_UNDERSTAND"] = "1"
os.environ["DVAA_FLAG_SALT"] = "test-salt-do-not-use-in-anger"
os.environ["DVAA_PROVIDER"] = "fake"
# The per-challenge tests assert what each *level* does, so they jump straight
# to the rung under test. The ladder that gates those rungs in play is asserted
# on its own, in tests/test_progression.py, which turns this back off.
os.environ["DVAA_UNLOCK_ALL"] = "1"

_tmpdir = tempfile.mkdtemp(prefix="dvaa-test-")
os.environ["DVAA_DB_PATH"] = str(Path(_tmpdir) / "test.db")
os.environ["DVAA_CHROMA_PATH"] = str(Path(_tmpdir) / "chroma")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.llm.fake import FakeProvider, Rule  # noqa: E402
from app.llm.registry import set_provider  # noqa: E402


@pytest.fixture
def client():
    """A TestClient with a fresh session, lifespan run (db init + seed)."""
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake():
    """Install a FakeProvider; tests append rules to it."""
    provider = FakeProvider()
    set_provider(provider)
    yield provider
    set_provider(None)


@pytest.fixture
def echo_context():
    """A provider that replies with everything it was given.

    The bluntest possible attacker: if a secret reached the context window,
    this surfaces it. Useful for asserting that IMPOSSIBLE levels keep secrets
    out of context entirely rather than relying on the model to stay quiet.
    """

    def responder(messages):
        from app.llm.base import Completion, Usage

        dump = "\n".join(f"[{m.role}] {m.content}" for m in messages)
        return Completion(text=dump, usage=Usage(1, 1))

    provider = FakeProvider(responder=responder)
    set_provider(provider)
    yield provider
    set_provider(None)


def set_level(client: TestClient, challenge_id: str, level: str) -> None:
    resp = client.post(f"/challenge/{challenge_id}/level", data={"level": level})
    assert resp.status_code == 200


def rule(match: str, text: str = "", **kw) -> Rule:
    return Rule(match=match, text=text, **kw)
