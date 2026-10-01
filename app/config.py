"""Configuration and the safety gate.

This is one of the few modules in DVAA that is *not* deliberately vulnerable.
Everything here exists to keep an intentionally-broken app from ending up
somewhere it can hurt someone.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
# Kept for installs that already generated one here before secrets moved into
# the data directory.
LEGACY_SALT_FILE = REPO_ROOT / ".dvaa_salt"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DVAA_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Safety
    i_understand: bool = False
    allow_network_bind: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    # Secrets. Both are generated on first boot and persisted next to the
    # database, so nobody has to invent one in docker-compose.yml and no two
    # installs share flags. Set them explicitly only to pin a workshop's flags
    # across rebuilt containers.
    flag_salt: str = ""
    secret_key: str = ""

    # Provider
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    # Both in seconds, both explained at the top of app/llm/ollama.py. The stall
    # timeout is how long the server may say *nothing* - it has to cover loading
    # a cold model, not producing an answer. The ceiling is the wall clock a
    # single generation may run for once it has started producing; raise it on a
    # slow machine, or when a challenge is meant to run a long job to completion.
    ollama_stall_timeout: float = 120.0
    ollama_max_seconds: float = 900.0
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o-mini"

    # Storage
    db_path: str = "./data/dvaa.db"
    chroma_path: str = "./data/chroma"

    # Defaults
    default_level: str = "low"

    # Progression. A challenge's middle rungs are locked until the rung below
    # them is captured, because "my payload worked at Low, why did it die at
    # Medium?" is the whole lesson and a level dropdown lets a player skip it.
    # Set DVAA_UNLOCK_ALL=1 to open every rung at once - for an instructor
    # demonstrating one level, or for re-testing a challenge without replaying
    # the ladder. IMPOSSIBLE is never gated either way: it yields nothing, so
    # visiting it spoils nothing, and reading the architectural fix is the
    # point of shipping it.
    unlock_all: bool = False

    # Log the raw JSON every provider call returns. Off by default because it
    # prints whole model responses to the console; on when a model is behaving
    # in a way the trace cannot explain, which is the only way to tell "the
    # model sent nothing" from "we parsed it wrong".
    debug_provider: bool = False

    @property
    def db_file(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def chroma_dir(self) -> Path:
        p = Path(self.chroma_path)
        return p if p.is_absolute() else REPO_ROOT / p

    def _persisted_secret(self, filename: str, configured: str) -> str:
        """Read a generated secret, creating it on first boot.

        It lives in the data directory because that is the path a container
        mounts as a volume: a salt written to the image's working directory
        would be regenerated every time the container is recreated, silently
        invalidating every flag a workshop had already captured.
        """
        if configured:
            return configured

        path = self.db_file.parent / filename
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value

        path.parent.mkdir(parents=True, exist_ok=True)

        if filename == ".dvaa_salt" and LEGACY_SALT_FILE.exists():
            value = LEGACY_SALT_FILE.read_text(encoding="utf-8").strip()
            if value:
                # Copy it forward rather than just reading it. Returning the
                # legacy value without writing it left the migration permanently
                # half-done: the data directory - the thing a container actually
                # mounts, and the thing the About page tells you to keep - still
                # had no salt in it, so deleting a stray file at the repo root
                # silently rotated every flag the workshop had captured.
                path.write_text(value, encoding="utf-8")
                return value

        generated = secrets.token_hex(16)
        path.write_text(generated, encoding="utf-8")
        return generated

    def resolved_salt(self) -> str:
        """The flag salt, generated and persisted on first boot."""
        return self._persisted_secret(".dvaa_salt", self.flag_salt)

    def resolved_secret_key(self) -> str:
        """The session-signing key, generated and persisted on first boot."""
        return self._persisted_secret(".dvaa_session_key", self.secret_key)


settings = Settings()

BANNER = r"""
  ____  __     __    _        _
 |  _ \ \ \   / /   / \      / \      Damn Vulnerable AI Agent
 | | | | \ \ / /   / _ \    / _ \     OWASP GenAI/LLM Top 10 (2026)
 | |_| |  \ V /   / ___ \  / ___ \
 |____/    \_/   /_/   \_\/_/   \_\   *** DELIBERATELY VULNERABLE ***
"""

_REFUSAL = """
DVAA refuses to start.

This application is deliberately vulnerable. It contains working prompt
injection, data exfiltration, unsafe deserialization, SQL injection and XSS
by design. Running it where others can reach it puts you and them at risk.

To continue, confirm you understand by setting:

    DVAA_I_UNDERSTAND=1

in your .env file (copy .env.example to .env), or in the environment.
"""

_NETWORK_REFUSAL = """
DVAA refuses to start.

DVAA_HOST is set to {host}, which exposes this deliberately vulnerable
application beyond loopback. If that is genuinely what you want on an isolated
lab network, also set:

    DVAA_ALLOW_NETWORK_BIND=1

Do not do this on a network you share with anyone else.
"""


def enforce_safety_gate() -> None:
    """Abort startup unless the operator has acknowledged the risk.

    Called from main.py at import time and from __main__ before uvicorn binds.
    """
    if not settings.i_understand:
        print(BANNER, file=sys.stderr)
        print(_REFUSAL, file=sys.stderr)
        raise SystemExit(2)

    if settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.allow_network_bind:
        print(BANNER, file=sys.stderr)
        print(_NETWORK_REFUSAL.format(host=settings.host), file=sys.stderr)
        raise SystemExit(2)
