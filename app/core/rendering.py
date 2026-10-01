"""Template rendering.

The Jinja loader searches the shared template directory plus every challenge's
own `templates/`, so a challenge can ship `llm01/chat.html` alongside its code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates
from jinja2 import ChoiceLoader, FileSystemLoader
from markdown_it import MarkdownIt

from app.config import REPO_ROOT
from app.core.challenge import Challenge, registry
from app.core.flags import solved_levels
from app.core.levels import ALL_LEVELS, Level, get_level
from app.core.session import parked_session_id, player_name, session_id

SHARED_TEMPLATES = REPO_ROOT / "app" / "templates"

_md = MarkdownIt("commonmark").enable("table")
_templates: Jinja2Templates | None = None


def markdown(text: str) -> str:
    """Render trusted, author-written markdown (scenario text, level docs).

    This is for content DVAA itself ships. Model output is never rendered
    through here - LLM10 has its own deliberately unsafe renderer.
    """
    return _md.render(text or "")


def flash_messages(request: Request) -> list[dict[str, str]]:
    """One-shot notices carried across a redirect in the query string."""
    msg = request.query_params.get("msg")
    if not msg:
        return []
    kind = request.query_params.get("kind", "info")
    if kind not in ("ok", "bad", "warn", "info"):
        kind = "info"
    return [{"text": msg, "kind": kind}]


def get_templates() -> Jinja2Templates:
    global _templates
    if _templates is None:
        registry.discover()
        loaders = [FileSystemLoader(str(SHARED_TEMPLATES))]
        loaders += [FileSystemLoader(str(d)) for d in registry.template_dirs()]
        templates = Jinja2Templates(directory=str(SHARED_TEMPLATES))
        templates.env.loader = ChoiceLoader(loaders)
        templates.env.filters["markdown"] = markdown
        _templates = templates
    return _templates


def render(request: Request, template: str, context: dict[str, Any] | None = None):
    """Render a page with the globals every template expects."""
    from app.llm.registry import describe_provider

    ctx: dict[str, Any] = {
        "request": request,
        "challenges": registry.all(),
        "player": player_name(request),
        "all_levels": ALL_LEVELS,
        "messages": flash_messages(request),
        # Set only while a challenge has handed the player a throwaway
        # identity; the banner it drives is how they get their run back.
        "parked_sid": parked_session_id(request),
    }
    from app.llm.registry import available_models, current_model

    try:
        ctx["provider_label"] = describe_provider()
        # Empty whenever the provider has no choice to offer (a cloud provider,
        # the test fake, or an Ollama we cannot reach), which hides the picker.
        ctx["models"] = available_models()
        ctx["active_model"] = current_model()
    except Exception as exc:  # provider misconfigured - surface it, don't crash
        ctx["provider_label"] = f"unavailable ({exc.__class__.__name__})"
        ctx["models"] = []
        ctx["active_model"] = ""
    ctx.update(context or {})
    return get_templates().TemplateResponse(request, template, ctx)


def render_challenge(
    request: Request,
    challenge: Challenge,
    template: str,
    context: dict[str, Any] | None = None,
):
    """Render a challenge page inside the shared challenge shell."""
    sid = session_id(request)
    level = get_level(sid, challenge.id)
    solved = solved_levels(sid, challenge.id)
    ctx: dict[str, Any] = {
        "challenge": challenge,
        "level": level,
        "level_doc": challenge.level_doc(level),
        "solved": solved,
        # Only the rungs this challenge ships - see Challenge.level_order.
        "challenge_levels": challenge.level_order,
        # The ladder: rungs open at or below the player's progress. A locked
        # rung is rendered, disabled, with its defense withheld - knowing a
        # defense exists is fine, reading its description before you have met
        # the one below it is a spoiler you did not ask for.
        "unlocked_levels": challenge.unlocked_levels(solved),
        "levels": {lvl: challenge.level_doc(lvl) for lvl in challenge.level_order},
    }
    ctx.update(context or {})
    return render(request, template, ctx)


def challenge_partial(
    request: Request, template: str, context: dict[str, Any] | None = None
):
    """Render an HTMX fragment (no shell, no globals beyond request)."""
    ctx: dict[str, Any] = {"request": request}
    ctx.update(context or {})
    return get_templates().TemplateResponse(request, template, ctx)


def template_dir_for(module_file: str) -> Path:
    """Convenience for challenge packages: `template_dir_for(__file__)`."""
    return Path(module_file).resolve().parent / "templates"


def current_level(request: Request, challenge_id: str) -> Level:
    return get_level(session_id(request), challenge_id)
