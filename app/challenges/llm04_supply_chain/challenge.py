"""LLM04:2026 Supply Chain — the install job (surrogate).

You cannot ship a real poisoned model from a real hub inside a web app, so this
reproduces the mechanism that actually gets people: nobody compromises the
platform team. They publish something the platform team's *resolver* decides to
install.

Northwind's CI runs an install job. It reads a manifest, resolves each entry
against a public registry, loads what comes back, and then does the same for
whatever that artifact declares as its own dependencies. The player publishes to
the registry and never touches the manifest. Each level tightens what the
resolver will accept, and each tightening leaves exactly one documented gap:

  * LOW       — the manifest names a package with no version constraint.
  * MEDIUM    — versions are pinned, so publish at the pinned coordinate.
  * HIGH      — direct dependencies are signed and allowlisted; the transitive
                ones are not, because the manifest never listed them.
  * IMPOSSIBLE— a lockfile pins a digest for every node in the graph, direct and
                transitive, and admits only data-only formats.

The lesson is model and tool artifacts as first-class attack surface (LLM04),
unsafe deserialization specifically (Scenario #7), and dependency resolution as
the thing that keeps deciding trust on the platform team's behalf.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from sqlmodel import select

from app.core.challenge import Challenge, LevelDoc
from app.core.flags import flag_for
from app.core.levels import Level
from app.core.rendering import challenge_partial, current_level, render_challenge, template_dir_for
from app.db import RegistryArtifact, get_session

from . import registry as reg

router = APIRouter()

CHALLENGE_ID = "llm04"
FORMATS = ("pickle", "safetensors")

# Publisher namespaces Northwind owns. Nobody else may publish under them, and
# from HIGH the install job will only take a direct dependency from one of them.
TRUSTED_PUBLISHERS = {"northwind-robotics"}

MANIFEST_NAME = "northwind-agent.manifest"
LOCKFILE_NAME = "northwind-agent.lock"

# What the agent's manifest asks for, per level. Only the top-level entries are
# ever written down; `tensor-utils` reaches the install job because
# `northwind-tools` declares it, which is the whole of the HIGH lesson.
MANIFEST = {
    Level.LOW: ["northwind-tools"],
    Level.MEDIUM: ["northwind-tools==2.3.1"],
    Level.HIGH: ["northwind-tools==2.3.1"],
    Level.IMPOSSIBLE: ["northwind-tools==2.3.1"],
}

# The dependency graph the lockfile pins at IMPOSSIBLE: every node, not just the
# ones a human wrote in the manifest.
LOCKED_PACKAGES = {"northwind-tools": "2.3.1", "tensor-utils": "0.9.4"}

MAX_DEPTH = 4


# --------------------------------------------------------------------------
# Artifact bytes and digests
# --------------------------------------------------------------------------


def _artifacts() -> list[RegistryArtifact]:
    with get_session() as s:
        return list(s.exec(select(RegistryArtifact).order_by(RegistryArtifact.id)).all())


def _coordinate(artifact: RegistryArtifact) -> str:
    return f"{artifact.publisher}/{artifact.name}@{artifact.version}"


def blob_for(artifact: RegistryArtifact, level: Level) -> bytes:
    """Produce the artifact's bytes.

    Pickle artifacts carry the level flag in their payload; a safetensors
    artifact is pure data and has nowhere to put one. Both are keyed by the
    artifact's coordinate so that no two packages share a digest.
    """
    ident = _coordinate(artifact)
    if artifact.fmt == "pickle":
        return reg.forge_artifact(
            flag_for(CHALLENGE_ID, level), artifact.payload_kind or "drop_marker", ident
        )
    return reg.forge_safetensors({"artifact": ident, "w0": 0.1, "w1": 0.2})


def digest_of(artifact: RegistryArtifact, level: Level) -> str:
    """The digest the registry computes for itself when the artifact lands."""
    return reg.sha256(blob_for(artifact, level))


def lockfile(level: Level) -> dict[str, str]:
    """name -> pinned digest, built from the vendor's own published artifacts.

    A lockfile records the bytes that were reviewed. It is generated from what
    the vendor shipped, not from whatever the registry is serving today, so
    republishing at the same coordinate cannot change it.
    """
    pinned: dict[str, str] = {}
    for artifact in _artifacts():
        want = LOCKED_PACKAGES.get(artifact.name)
        if (
            want
            and artifact.version.strip() == want
            and reg.normalize_publisher(artifact.signed_by) in TRUSTED_PUBLISHERS
        ):
            pinned[artifact.name] = digest_of(artifact, level)
    return pinned


# --------------------------------------------------------------------------
# The install job
# --------------------------------------------------------------------------


def _resolve(
    level: Level,
    name: str,
    pin: str,
    candidates: list[RegistryArtifact],
    direct: bool,
    locked: dict[str, str],
):
    """Pick the artifact this level's resolver would install for a requirement."""
    if level is Level.LOW:
        # No constraint anywhere in the graph.
        return reg.resolve_highest_version(candidates)

    if level is Level.MEDIUM:
        # Versions pinned, publisher never considered.
        return reg.resolve_pinned_version(candidates, pin)

    if level is Level.HIGH:
        if direct:
            return reg.resolve_trusted(candidates, pin, TRUSTED_PUBLISHERS)
        # Transitive dependencies were never written into the manifest, so the
        # allowlist has nothing to say about them and they fall back to the
        # previous policy: pinned version, publisher unchecked.
        return reg.resolve_pinned_version(candidates, pin)

    return reg.resolve_locked(
        candidates, locked.get(name, ""), lambda a: digest_of(a, level)
    )


def _load(artifact: RegistryArtifact, level: Level) -> tuple[bool, str]:
    """Load a resolved artifact. A pickle runs its payload; safetensors cannot."""
    blob = blob_for(artifact, level)
    if artifact.fmt == "pickle":
        reg.load_pickle(blob)
        return True, (
            "deserialized as pickle — its __reduce__ ran during the load, before "
            "anything could inspect what had been fetched"
        )
    reg.load_safetensors(blob)
    return True, "loaded data-only artifact; no code path"


def run_install_job(level: Level) -> dict:
    """Run Northwind's install job at this level and return its log."""
    reg.reset_marker()

    catalogue = _artifacts()
    locked = lockfile(level) if level is Level.IMPOSSIBLE else {}
    steps: list[dict] = []
    seen: set[str] = set()

    def install(requirement: str, direct: bool, depth: int) -> None:
        name, pin = reg.parse_requirement(requirement)
        if not name or name in seen or depth > MAX_DEPTH:
            return
        seen.add(name)

        candidates = [a for a in catalogue if a.name == name]
        artifact, note = _resolve(level, name, pin, candidates, direct, locked)
        steps.append(
            {
                "kind": "resolve",
                "requirement": requirement,
                "direct": direct,
                "depth": depth,
                "artifact": artifact,
                "coordinate": _coordinate(artifact) if artifact else "",
                "detail": note,
                "ok": artifact is not None,
            }
        )
        if artifact is None:
            return

        try:
            ok, detail = _load(artifact, level)
        except Exception as exc:  # a broken artifact is a result too
            ok, detail = False, f"load failed: {exc}"
        steps.append(
            {
                "kind": "load",
                "requirement": requirement,
                "direct": direct,
                "depth": depth,
                "artifact": artifact,
                "coordinate": _coordinate(artifact),
                "detail": detail,
                "ok": ok,
                "executed": artifact.fmt == "pickle" and ok,
            }
        )
        if not ok:
            return

        for dep in reg.declared_requirements(artifact.requires):
            install(dep, direct=False, depth=depth + 1)

    for requirement in MANIFEST[level]:
        install(requirement, direct=True, depth=0)

    executed = reg.code_ran()
    return {
        "steps": steps,
        "manifest": MANIFEST[level],
        "locked": locked,
        "executed": executed,
        "loot": reg.captured_flag() if executed else None,
        "story": reg.payload_story() if executed else "",
        "ran_from": reg.payload_artifact() if executed else "",
    }


def reset_scenario() -> None:
    """Withdraw everything the player published."""
    with get_session() as s:
        for artifact in s.exec(
            select(RegistryArtifact).where(RegistryArtifact.from_player == True)  # noqa: E712
        ).all():
            s.delete(artifact)
        s.commit()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


def _page_context(request: Request) -> dict:
    level = current_level(request, CHALLENGE_ID)
    artifacts = _artifacts()
    return {
        "challenge": CHALLENGE,
        "artifacts": artifacts,
        # A registry shows you the digest it computed for what it stored. So
        # does this one - in full, because at Impossible you are meant to be
        # able to compare it against the lockfile yourself.
        "digests": {a.id: digest_of(a, level) for a in artifacts},
        "trusted": sorted(TRUSTED_PUBLISHERS),
        "formats": FORMATS,
        "payloads": reg.PAYLOADS,
        "manifest": MANIFEST[level],
        "manifest_name": MANIFEST_NAME,
        "lockfile_name": LOCKFILE_NAME,
        "locked": lockfile(level) if level is Level.IMPOSSIBLE else {},
        "level": level,
    }


@router.get("")
async def index(request: Request):
    return render_challenge(request, CHALLENGE, "llm04.html", _page_context(request))


@router.post("/publish")
async def publish_route(
    request: Request,
    name: str = Form(""),
    publisher: str = Form(""),
    version: str = Form("1.0.0"),
    fmt: str = Form("pickle"),
    signed: str = Form(""),
    payload_kind: str = Form("drop_marker"),
    requires: str = Form(""),
    description: str = Form(""),
):
    """Publish to the registry. Anyone may; that is the point.

    The one thing you may not do is upload under somebody else's namespace. A
    registry that let you sign as the vendor would collapse every level into
    the first one, and it is also the one control real registries do enforce.
    """
    publisher = publisher.strip()[:80] or "anonymous"
    if reg.is_reserved(publisher, TRUSTED_PUBLISHERS):
        context = _page_context(request)
        context["publish_error"] = (
            f"'{publisher}' is a reserved publisher namespace, owned by Northwind. "
            "Pick a name you can actually claim."
        )
        return challenge_partial(request, "llm04_registry.html", context)

    artifact = RegistryArtifact(
        name=name.strip()[:80] or "unnamed-artifact",
        publisher=publisher,
        version=version.strip()[:20] or "1.0.0",
        fmt=fmt if fmt in FORMATS else "pickle",
        # The registry signs on publish with the key the publisher registered
        # for themselves. Every public registry works this way: a signature says
        # who uploaded it, never whether they should be trusted.
        signature="valid" if signed else "",
        signed_by=publisher if signed else "",
        requires=requires.strip()[:200],
        description=description.strip()[:200],
        payload_kind=payload_kind if payload_kind in reg.PAYLOADS else "drop_marker",
        from_player=True,
    )
    with get_session() as s:
        s.add(artifact)
        s.commit()

    return challenge_partial(request, "llm04_registry.html", _page_context(request))


@router.post("/install")
async def install_route(request: Request):
    level = current_level(request, CHALLENGE_ID)
    result = run_install_job(level)

    note = {
        Level.LOW: (
            "The manifest names a package and nothing else. The resolver picked the "
            "highest version published under that name."
        ),
        Level.MEDIUM: "Every requirement is version-pinned. Publishers are not checked.",
        Level.HIGH: (
            "Direct dependencies must be signed by an allowlisted publisher. Transitive "
            "ones are resolved by version pin alone."
        ),
        Level.IMPOSSIBLE: (
            "Every dependency in the graph is pinned to a digest by the lockfile, and "
            "only data-only artifacts are admitted."
        ),
    }[level]

    return challenge_partial(
        request,
        "llm04_result.html",
        {"result": result, "defense_note": note, "level": level},
    )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

CHALLENGE = Challenge(
    id=CHALLENGE_ID,
    owasp_id="LLM04",
    owasp_title="Supply Chain",
    title="The Install Job",
    tagline="Nobody breaks into the platform team. They publish something it agrees to install.",
    tags=["dependency confusion", "unsafe deserialization", "pickle", "transitive deps", "lockfile"],
    surrogate_note=(
        "Real supply-chain compromise means a poisoned model on a real hub, which a web "
        "app cannot honestly stage. This is a runtime surrogate: a local registry serves "
        "locally-forged artifacts, and the only pickle ever loaded is one DVAA builds "
        "itself. You choose what your payload claims to do; what actually executes is a "
        "benign function that records your choice and the flag. The vulnerability is "
        "real — unpickling an untrusted artifact runs its __reduce__ before you can "
        "inspect it — but the blast radius is a marker in a dict, and the whole path is "
        "reachable only from this challenge inside the container."
    ),
    scenario="""
Northwind's platform team does not install anything by hand. CI runs an **install
job**: it reads `northwind-agent.manifest`, resolves each entry against the
public registry, loads what comes back, and then resolves whatever *that*
artifact declares as its own dependencies.

Their pack, `northwind-tools`, ships as safetensors — pure data — and is signed.
It depends on `tensor-utils`, which is not written in the manifest anywhere;
it arrives because the pack asks for it.

Anyone can publish to the same registry, and the install job runs against
whatever is listed at the moment it runs. Loading a pickle executes whatever
code its author put in `__reduce__`, before anyone can look at what they loaded.
""",
    objective="""
Get the install job to **load an artifact you published** and execute its
payload. When a poisoned pickle is deserialized its payload runs and reports the
flag.

You control everything about what you publish — the package name, your publisher
name, the version, the format, whether it is signed, what it depends on, and
what the payload claims to do. You do not control the manifest, and you cannot
publish under Northwind's own publisher namespace.

Read the install log. It tells you exactly which requirement was resolved to
which artifact, and why.
""",
    levels={
        Level.LOW: LevelDoc(
            defense=(
                "None worth the name. The manifest asks for `northwind-tools` with no "
                "version constraint, and the resolver installs the highest version "
                "published under that name, from anyone."
            ),
        ),
        Level.MEDIUM: LevelDoc(
            defense=(
                "Every requirement is version-pinned: the manifest asks for "
                "`northwind-tools==2.3.1`, and packages pin their own dependencies the "
                "same way. Publishing a higher version no longer wins anything."
            ),
            bypass=(
                "A version pin names a coordinate — a name and a number — not an artifact "
                "and not an author. Publish *at* the pinned coordinate: same package name, "
                "same version, your bytes. Two artifacts now sit there and the registry "
                "serves the one that landed most recently, which is yours. Pinning without "
                "an identity or a digest behind it just tells the attacker which number to "
                "type."
            ),
        ),
        Level.HIGH: LevelDoc(
            defense=(
                "The install job requires every manifest entry to be signed by an "
                "allowlisted publisher, and the registry will not let you upload under a "
                "reserved namespace. `northwind-tools` can now only resolve to Northwind's "
                "own signed pack."
            ),
            bypass=(
                "The allowlist was written from the manifest, and the manifest only lists "
                "*direct* dependencies. `tensor-utils` is not in it — it arrives because "
                "`northwind-tools` declares it — so nothing checks its publisher or its "
                "signature; it is resolved by version pin alone, exactly as at Medium. "
                "Publish `tensor-utils` at `0.9.4` and the vendor's own signed, "
                "audited pack pulls your pickle in behind it. Most real dependency graphs "
                "are mostly edges nobody wrote down."
            ),
        ),
        Level.IMPOSSIBLE: LevelDoc(
            defense=(
                "A lockfile pins a SHA-256 for every node in the resolved graph, direct "
                "and transitive alike, and it is generated from the artifacts that were "
                "actually reviewed. A requirement with no lockfile entry is refused, an "
                "artifact whose bytes do not hash to the pinned digest is refused, and "
                "code-bearing formats are refused outright — so even a digest collision "
                "would land on something with no code path to execute. Resolution stops "
                "deciding trust; the review did."
            ),
        ),
    },
    router=router,
    template_dir=template_dir_for(__file__),
    reset_scenario=reset_scenario,
)
