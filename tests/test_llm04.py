"""LLM04 Supply Chain — two-sided check.

Exercises the real (sandboxed) pickle path: the player publishes to the registry
and Northwind's install job either resolves to their artifact and runs its
payload, or refuses. No model needed.
"""

from __future__ import annotations

import pytest
from sqlmodel import select

from app.challenges.llm04_supply_chain import registry as reg
from app.challenges.llm04_supply_chain.challenge import digest_of, lockfile
from app.core.flags import flag_for
from app.core.levels import Level
from app.db import RegistryArtifact, get_session
from tests.conftest import set_level

CHALLENGE = "llm04"
PATH = f"/challenge/{CHALLENGE}"

OFFICIAL = "northwind-tools"  # seeded, signed, safetensors, requires tensor-utils
TRANSITIVE = "tensor-utils"  # seeded, signed, safetensors, in no manifest
PINNED = "2.3.1"
TRANSITIVE_PINNED = "0.9.4"


@pytest.fixture(autouse=True)
def clean_registry(client):
    """Withdraw player artifacts between tests.

    Every test in this file publishes to one shared registry, and a package left
    behind by an earlier test is exactly the kind of thing that would make a
    resolver assertion pass for the wrong reason.
    """
    from app.challenges.llm04_supply_chain.challenge import reset_scenario

    reg.reset_marker()
    reset_scenario()
    yield
    reset_scenario()


def artifact_by_name(name: str) -> RegistryArtifact:
    with get_session() as s:
        row = s.exec(select(RegistryArtifact).where(RegistryArtifact.name == name)).first()
    assert row is not None, f"registry has no artifact {name}"
    return row


def publish(client, **fields) -> str:
    """Publish an artifact as the player. Returns the rendered registry."""
    data = {
        "name": "evil-pack",
        "publisher": "ml-tooling-collective",
        "version": "9.9.9",
        "fmt": "pickle",
        "payload_kind": "exfil_env",
        "requires": "",
        "description": "",
    }
    data.update(fields)
    resp = client.post(f"{PATH}/publish", data=data)
    assert resp.status_code == 200
    return resp.text


def install(client) -> str:
    resp = client.post(f"{PATH}/install")
    assert resp.status_code == 200
    return resp.text


# --------------------------------------------------------------------------
# The primitive: unpickling a forged artifact runs its payload
# --------------------------------------------------------------------------


def test_forged_pickle_executes_on_load():
    reg.reset_marker()
    blob = reg.forge_artifact("DVAA{unit_probe}", "reverse_shell", "me/pack@1.0")
    assert not reg.code_ran()
    reg.load_pickle(blob)
    assert reg.code_ran()
    assert reg.captured_flag() == "DVAA{unit_probe}"
    assert "reverse shell" in reg.payload_story()
    assert reg.payload_artifact() == "me/pack@1.0"


def test_safetensors_has_no_code_path():
    reg.reset_marker()
    blob = reg.forge_safetensors({"artifact": "x", "w": 1.0})
    assert reg.load_safetensors(blob)["w"] == 1.0
    assert not reg.code_ran()


def test_artifacts_do_not_share_a_digest(client):
    """A lockfile pin is only worth something if two packages hash differently."""
    seen = {digest_of(artifact_by_name(n), Level.IMPOSSIBLE) for n in (OFFICIAL, TRANSITIVE)}
    assert len(seen) == 2


def test_version_ordering_is_numeric():
    assert reg.version_key("10.0.0") > reg.version_key("9.9.9")
    assert reg.version_key("2.3.1") == reg.version_key("2.3.1.0")


# --------------------------------------------------------------------------
# The registry will not let you claim somebody else's namespace
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "publisher", ["northwind-robotics", "Northwind_Robotics", "NORTHWIND ROBOTICS"]
)
def test_reserved_publisher_namespaces_are_refused(client, publisher):
    set_level(client, CHALLENGE, "high")
    out = publish(client, name="impostor", publisher=publisher, signed="1")
    assert "reserved publisher namespace" in out
    with get_session() as s:
        assert s.exec(
            select(RegistryArtifact).where(RegistryArtifact.name == "impostor")
        ).first() is None


# --------------------------------------------------------------------------
# LOW — the manifest names a package with no version constraint
# --------------------------------------------------------------------------


def test_low_installs_the_official_pack_untouched(client):
    set_level(client, CHALLENGE, "low")
    out = install(client)
    assert "DVAA{" not in out


def test_low_falls_to_a_higher_version_under_the_same_name(client):
    set_level(client, CHALLENGE, "low")
    publish(client, name=OFFICIAL, version="99.0.0")
    out = install(client)
    assert flag_for(CHALLENGE, Level.LOW) in out
    assert "highest version" in out


def test_low_ignores_a_lower_version(client):
    set_level(client, CHALLENGE, "low")
    publish(client, name=OFFICIAL, version="0.0.1")
    out = install(client)
    assert flag_for(CHALLENGE, Level.LOW) not in out


# --------------------------------------------------------------------------
# MEDIUM — versions pinned, publishers not: publish at the coordinate
# --------------------------------------------------------------------------


def test_medium_ignores_a_higher_version(client):
    """The Low payload stops working: the pin means a bigger number is nothing."""
    set_level(client, CHALLENGE, "medium")
    publish(client, name=OFFICIAL, version="99.0.0")
    out = install(client)
    assert flag_for(CHALLENGE, Level.MEDIUM) not in out


def test_medium_falls_to_publishing_at_the_pinned_coordinate(client):
    set_level(client, CHALLENGE, "medium")
    publish(client, name=OFFICIAL, version=PINNED)
    out = install(client)
    assert flag_for(CHALLENGE, Level.MEDIUM) in out
    assert "most recent" in out


# --------------------------------------------------------------------------
# HIGH — direct deps signed and allowlisted; transitive ones are not
# --------------------------------------------------------------------------


def test_high_blocks_the_medium_payload(client):
    set_level(client, CHALLENGE, "high")
    publish(client, name=OFFICIAL, version=PINNED, signed="1")
    out = install(client)
    assert flag_for(CHALLENGE, Level.HIGH) not in out
    assert "allowlisted publisher" in out


def test_high_falls_to_the_transitive_dependency(client):
    """`tensor-utils` is in no manifest, so no allowlist covers it."""
    set_level(client, CHALLENGE, "high")
    publish(client, name=TRANSITIVE, version=TRANSITIVE_PINNED)
    out = install(client)
    assert flag_for(CHALLENGE, Level.HIGH) in out
    assert "transitive" in out


def test_high_transitive_dependency_is_still_version_pinned(client):
    set_level(client, CHALLENGE, "high")
    publish(client, name=TRANSITIVE, version="99.0.0")
    out = install(client)
    assert flag_for(CHALLENGE, Level.HIGH) not in out


# --------------------------------------------------------------------------
# IMPOSSIBLE — every node pinned to a digest, data-only formats only
# --------------------------------------------------------------------------


def test_impossible_installs_the_locked_graph_cleanly(client):
    set_level(client, CHALLENGE, "impossible")
    out = install(client)
    assert "DVAA{" not in out
    assert "lockfile pin" in out


@pytest.mark.parametrize(
    ("name", "version"),
    [
        (OFFICIAL, PINNED),
        (OFFICIAL, "99.0.0"),
        (TRANSITIVE, TRANSITIVE_PINNED),
        (TRANSITIVE, "99.0.0"),
    ],
)
def test_impossible_refuses_everything_the_player_publishes(client, name, version):
    set_level(client, CHALLENGE, "impossible")
    publish(client, name=name, version=version, signed="1")
    out = install(client)
    assert flag_for(CHALLENGE, Level.IMPOSSIBLE) not in out
    assert "DVAA{" not in out


def test_impossible_lockfile_pins_the_vendor_bytes_only(client):
    """Republishing at a locked coordinate must not move the pin."""
    set_level(client, CHALLENGE, "impossible")
    before = lockfile(Level.IMPOSSIBLE)
    publish(client, name=OFFICIAL, version=PINNED, fmt="safetensors", signed="1")
    assert lockfile(Level.IMPOSSIBLE) == before


def test_impossible_refuses_a_dependency_with_no_lock_entry(client):
    """A pack that drags in something nobody reviewed is refused, not resolved."""
    set_level(client, CHALLENGE, "impossible")
    publish(client, name="unlocked-dep", version="1.0.0")
    from app.challenges.llm04_supply_chain.challenge import run_install_job

    result = run_install_job(Level.IMPOSSIBLE)
    steps = [s for s in result["steps"] if s["requirement"].startswith("unlocked-dep")]
    assert all(not s["ok"] for s in steps)
    assert result["loot"] is None


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------


def test_reset_withdraws_only_player_artifacts(client):
    from app.challenges.llm04_supply_chain.challenge import reset_scenario

    set_level(client, CHALLENGE, "low")
    publish(client, name="throwaway")
    reset_scenario()

    with get_session() as s:
        rows = list(s.exec(select(RegistryArtifact)).all())
    assert not any(a.from_player for a in rows)
    assert any(a.name == OFFICIAL for a in rows)
