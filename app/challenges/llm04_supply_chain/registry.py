"""The mock package registry: artifact bytes, and how a name is resolved to one.

SAFETY: the only pickle this module ever loads is one DVAA builds itself, in
`forge_artifact`, whose `__reduce__` runs a benign, self-contained callback - it
writes a marker into an in-process dict. The player chooses what the payload
*claims* to do; what it actually does is record that choice. There is no network
fetch, no arbitrary payload, and no eval of anything the player typed. The
vulnerability being taught is real (unpickling an untrusted artifact executes its
`__reduce__`), but the blast radius is a flag string, and the whole path is
reachable only from inside this challenge.

Everything here is pure: it takes lists of artifact rows and returns decisions.
The database, the levels and the wiring live in challenge.py.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import struct
from typing import Any

# What a payload can claim to do. Every one of them is the same benign function;
# the label is the player's choice of story, recorded as proof their artifact -
# not ours - is what ran.
PAYLOADS = {
    "exfil_env": "read the host's environment and phone it home",
    "drop_marker": "write a file into the model cache directory",
    "reverse_shell": "open a reverse shell to the publisher",
}

# The "code execution" side effect. A real malicious package would do the thing
# in the label; this records that arbitrary code ran, and stashes the flag it
# found on the host it ran on.
_EXECUTION_MARKER: dict[str, str] = {}


def reset_marker() -> None:
    _EXECUTION_MARKER.clear()


def code_ran() -> bool:
    return "flag" in _EXECUTION_MARKER


def captured_flag() -> str | None:
    return _EXECUTION_MARKER.get("flag")


def payload_story() -> str:
    return _EXECUTION_MARKER.get("story", "")


def payload_artifact() -> str:
    return _EXECUTION_MARKER.get("artifact", "")


def _payload(flag: str, kind: str, ident: str = "") -> None:
    """The benign stand-in for malicious code carried by the artifact."""
    _EXECUTION_MARKER["flag"] = flag
    _EXECUTION_MARKER["kind"] = kind
    _EXECUTION_MARKER["artifact"] = ident
    _EXECUTION_MARKER["story"] = PAYLOADS.get(kind, PAYLOADS["drop_marker"])
    _EXECUTION_MARKER["proof"] = "arbitrary code executed during unpickling"


class _MaliciousArtifact:
    """An object whose unpickling runs code, via __reduce__.

    This is the canonical unsafe-deserialization primitive: pickle calls the
    callable returned by __reduce__ at load time, before you can inspect what
    you loaded.
    """

    def __init__(self, flag: str, kind: str, ident: str) -> None:
        self.flag = flag
        self.kind = kind
        self.ident = ident

    def __reduce__(self):
        return (_payload, (self.flag, self.kind, self.ident))


def forge_artifact(flag: str, kind: str = "drop_marker", ident: str = "") -> bytes:
    """Serialize a poisoned artifact. Stands in for a malicious file on a hub.

    `ident` is the artifact's coordinate, and it is part of the pickled bytes so
    that two different packages never share a digest. A registry where every
    artifact hashes the same would make the lockfile at Impossible meaningless.
    """
    return pickle.dumps(_MaliciousArtifact(flag, kind, ident))


def forge_safetensors(payload: dict[str, Any]) -> bytes:
    """A minimal, safe, data-only artifact.

    Real safetensors is a JSON header + raw tensor bytes with no code path. This
    toy version keeps that essential property: it is pure data, so loading it
    can never execute anything.
    """
    header = json.dumps(payload, sort_keys=True).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00" * 8


def load_safetensors(blob: bytes) -> dict:
    (hlen,) = struct.unpack("<Q", blob[:8])
    return json.loads(blob[8 : 8 + hlen].decode())


def load_pickle(blob: bytes) -> str:
    """Deserialize an artifact. THIS is the vulnerability, on purpose."""
    obj = pickle.loads(blob)  # noqa: S301
    return type(obj).__name__


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------
# Names, versions and namespaces
# --------------------------------------------------------------------------


def normalize_publisher(name: str) -> str:
    """Fold the harmless spelling variations of a publisher name."""
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def is_reserved(publisher: str, reserved: set[str]) -> bool:
    """Does this name claim a namespace somebody else owns?

    A registry that let anyone upload as `northwind-robotics` would make the
    publisher allowlist at High a formality, so this is an *exact* check after
    normalisation - the one place in this challenge where the comparison is not
    the weak link.
    """
    return normalize_publisher(publisher) in {normalize_publisher(r) for r in reserved}


def version_key(version: str) -> tuple[int, ...]:
    """Order versions the way a resolver does: numerically, left to right."""
    parts: list[int] = []
    for chunk in version.split(".")[:4]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (4 - len(parts)))


def parse_requirement(text: str) -> tuple[str, str]:
    """`"tensor-utils==0.9.4"` -> `("tensor-utils", "0.9.4")`. No pin -> `""`."""
    name, sep, version = text.partition("==")
    return name.strip(), (version.strip() if sep else "")


def declared_requirements(requires: str) -> list[str]:
    return [r.strip() for r in (requires or "").split(",") if r.strip()]


# --------------------------------------------------------------------------
# Resolution: one strategy per level's install job
#
# Each returns (artifact, note). A None artifact means nothing was installed and
# the note says why.
# --------------------------------------------------------------------------


def resolve_highest_version(candidates: list) -> tuple[Any, str]:
    """No constraint at all: whatever is newest under that name wins.

    This is dependency confusion in one line. The job asked for a name; the
    registry answered with whoever last claimed the biggest number.
    """
    if not candidates:
        return None, "no artifact published under that name"
    winner = max(candidates, key=lambda a: (version_key(a.version), a.id or 0))
    return winner, (
        f"no version constraint — resolved to the highest version in the registry: "
        f"{winner.version} from '{winner.publisher}'"
    )


def resolve_pinned_version(candidates: list, want: str) -> tuple[Any, str]:
    """Version pinned, publisher not.

    A pin names a coordinate, not an artifact and not an author. Anyone may
    publish at that coordinate, and the registry serves whatever landed there
    most recently.
    """
    if not want:
        return resolve_highest_version(candidates)
    matches = [a for a in candidates if a.version.strip() == want]
    if not matches:
        return None, f"no artifact published at the pinned version {want}"
    winner = max(matches, key=lambda a: a.id or 0)
    extra = (
        f" — {len(matches)} artifacts sit at this coordinate; served the most recent"
        if len(matches) > 1
        else ""
    )
    return winner, f"version pin {want} satisfied by '{winner.publisher}'{extra}"


def resolve_trusted(
    candidates: list, want: str, allowlist: set[str]
) -> tuple[Any, str]:
    """Version pinned, signature required, publisher must be on the allowlist."""
    artifact, note = resolve_pinned_version(candidates, want)
    if artifact is None:
        return None, note
    matches = [a for a in candidates if not want or a.version.strip() == want]
    trusted = [
        a
        for a in matches
        if a.signature == "valid" and normalize_publisher(a.signed_by) in allowlist
    ]
    if not trusted:
        return None, (
            f"rejected: nothing at {want or 'any version'} is signed by an allowlisted "
            f"publisher ({', '.join(sorted(allowlist))})"
        )
    winner = max(trusted, key=lambda a: a.id or 0)
    return winner, f"signed by allowlisted publisher '{winner.signed_by}'; pin {want} satisfied"


def resolve_locked(
    candidates: list, locked_digest: str, digest_of
) -> tuple[Any, str]:
    """Only the exact bytes the lockfile pinned, and only data-only formats."""
    if not locked_digest:
        return None, "refused: no lockfile entry for this dependency"
    for artifact in candidates:
        if digest_of(artifact) != locked_digest:
            continue
        if artifact.fmt != "safetensors":
            return None, (
                f"refused: '{artifact.fmt}' is a code-bearing format; the lockfile "
                "admits data-only artifacts"
            )
        return artifact, f"digest matched the lockfile pin {locked_digest[:16]}…"
    return None, (
        f"refused: nothing under this name hashes to the pinned digest "
        f"{locked_digest[:16]}…"
    )
