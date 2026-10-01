"""Output-handling defenses for LLM10.

The whole category is about what happens to model output *after* generation, so
these functions run over text the model produced and decide how it reaches the
page, the database, and the collector.
"""

from __future__ import annotations

import re

import bleach

# --------------------------------------------------------------------------
# MEDIUM: a <script> blocklist
# --------------------------------------------------------------------------

# The regex everyone writes first. It strips the obvious tag and misses every
# other way a browser runs script.
_SCRIPT_RE = re.compile(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)
_SCRIPT_OPEN_RE = re.compile(r"<\s*/?\s*script[^>]*>", re.IGNORECASE)


def strip_script_tags(html: str) -> str:
    html = _SCRIPT_RE.sub("", html)
    return _SCRIPT_OPEN_RE.sub("", html)


# --------------------------------------------------------------------------
# HIGH: a real HTML sanitizer
# --------------------------------------------------------------------------

# bleach removes scripts and event handlers properly. It intentionally allows
# images, because "render the markdown a user asked for" is a legitimate
# feature — and that allowance is the remaining gap: an <img> whose src points
# at the collector still fires an outbound request when the browser paints it.
_ALLOWED_TAGS = [
    "p", "br", "strong", "em", "code", "pre", "ul", "ol", "li",
    "a", "img", "h1", "h2", "h3", "blockquote", "table", "thead",
    "tbody", "tr", "th", "td",
]
_ALLOWED_ATTRS = {
    "a": ["href", "title"],
    "img": ["src", "alt", "title"],
}


def sanitize_html(html: str) -> str:
    return bleach.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=["http", "https", "mailto"],
        strip=True,
    )


# --------------------------------------------------------------------------
# IMPOSSIBLE: context-aware encoding + no remote fetch
# --------------------------------------------------------------------------

_IMG_TAG_RE = re.compile(r"<\s*img[^>]*>", re.IGNORECASE)
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def strip_remote_images(html: str) -> str:
    """Remove images entirely so nothing auto-fetches.

    The Impossible renderer treats model output as text: it is escaped and shown
    as-is, and any construct that would make the browser reach out to a remote
    host is dropped before rendering.
    """
    html = _IMG_TAG_RE.sub("[image removed]", html)
    html = _MD_IMG_RE.sub("[image removed]", html)
    return html
