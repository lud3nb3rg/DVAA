"""LLM10:2026 Improper Output Handling — AI report generator.

A marketing tool turns a user's prompt into an HTML "report" that is then
rendered in the browser, stored, and shown to other viewers. The model's output
is trusted on the way to all three sinks. Two exploit paths:

  * XSS  — model output is placed into the page without encoding.
  * Exfil — a markdown image the chat UI auto-fetches carries page context out
            to a collector, exactly as Rehberger (2024a) described.

The flag lives behind an authenticated `/secret` fragment on the same origin;
a stored-XSS or exfil payload that runs in a victim's session can read it.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.core.challenge import Challenge, LevelDoc
from app.core.flags import flag_for
from app.core.levels import Level
from app.core.rendering import challenge_partial, current_level, render_challenge, template_dir_for
from app.llm import Message, ProviderError, get_provider

from . import defenses

router = APIRouter()

CHALLENGE_ID = "llm10"

# In-memory collector log. Anything the browser exfiltrates to /collect lands
# here, so the learner can watch context leave the page.
#
# Bounded, because /collect is a sink the exploit drives: an injected <img> in a
# loop is a perfectly ordinary payload here, and an unbounded list behind it is
# a way to grow the process until the lab falls over. Only the most recent hits
# are ever displayed anyway.
_COLLECTOR_LIMIT = 200
_COLLECTOR: list[dict] = []


SYSTEM_PROMPT = (
    "You are Northwind's marketing report generator. Turn the user's request "
    "into a short, upbeat HTML report. You may use basic HTML tags and markdown "
    "images. Return only the report body."
)


async def generate(prompt: str) -> str:
    """Ask the model for a report. In practice it returns whatever the prompt
    steers it toward — which is the point of an output-handling challenge."""
    try:
        completion = await get_provider().chat(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ],
            temperature=0.5,
            max_tokens=800,
        )
    except ProviderError as exc:
        return f"<p>[provider error] {exc}</p>"
    return completion.text or "<p>[empty report]</p>"


def render_for_level(level: Level, raw: str) -> tuple[str, str]:
    """Apply the level's output handling. Returns (html_to_render, note)."""
    if level is Level.LOW:
        # Straight into the DOM. No encoding, no sanitizer, no CSP.
        return raw, "Raw model output inserted into the page unmodified."

    if level is Level.MEDIUM:
        cleaned = defenses.strip_script_tags(raw)
        return cleaned, "A <script>-tag blocklist ran over the output."

    if level is Level.HIGH:
        cleaned = defenses.sanitize_html(raw)
        return (
            cleaned,
            "bleach sanitized the HTML (scripts and event handlers removed). "
            "Images are still allowed and still auto-fetch.",
        )

    # IMPOSSIBLE: escape everything, drop anything that would fetch remotely.
    escaped = defenses.strip_remote_images(raw)
    return escaped, "Output escaped as text; remote-fetching constructs removed."


# --------------------------------------------------------------------------
# Content-Security-Policy
#
# On the DOCUMENT, not on the fragment. The report comes back from /generate as
# a fragment and dvaa.js writes it into the page with innerHTML - and a CSP
# header on a response that JavaScript reads as text has no bearing on the
# document that ends up holding the markup. It looked like a policy and enforced
# nothing. The policy has to be served with the page that owns the DOM.
#
# `script-src 'self'` is what actually stops an injected event handler at High,
# and `img-src 'self'` is what leaves the same-origin collector reachable - the
# gap the level is about. Impossible closes images too.
#
# 'unsafe-inline' stays on style-src because the lab's own templates use style
# attributes; it is a real weakening and it is confined to styles, which cannot
# fetch anything under `default-src 'none'`.
_CSP = {
    Level.HIGH: (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self'; connect-src 'self'; form-action 'self'"
    ),
    Level.IMPOSSIBLE: (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'none'; connect-src 'self'; form-action 'self'"
    ),
}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@router.get("")
async def index(request: Request):
    level = current_level(request, CHALLENGE_ID)
    response = render_challenge(
        request,
        CHALLENGE,
        "llm10.html",
        {"collector": list(reversed(_COLLECTOR[-12:]))},
    )
    if level in _CSP:
        response.headers["Content-Security-Policy"] = _CSP[level]
    return response


@router.post("/generate")
async def generate_route(request: Request, prompt: str = Form(...)):
    level = current_level(request, CHALLENGE_ID)
    raw = await generate(prompt.strip()[:2000])
    html, note = render_for_level(level, raw)

    # No CSP here on purpose: this response is read by fetch() and its markup is
    # inserted into a document that already has (or has not) a policy of its
    # own. See _CSP above - the header belongs on the page, not on the fragment.
    return challenge_partial(
        request,
        "llm10_report.html",
        {"report_html": html, "raw": raw, "defense_note": note, "level": level},
    )


def _is_navigation(request: Request) -> bool:
    """Whether this request is someone typing the URL rather than a page fetching it.

    Fetch metadata, which the browser sets and the page cannot: a top-level
    navigation arrives as `Sec-Fetch-Dest: document` / `Sec-Fetch-Mode: navigate`,
    while an `fetch()` from injected script arrives as dest `empty` and an
    `<img src>` as dest `image`. Requests with no fetch metadata at all - curl,
    the test client, an old browser - are treated as in-page, because the point
    of this guard is not to stop a determined reader.

    It is a lab guard, not a security control, and it should not be read as one:
    the endpoint really is unauthenticated, and that is deliberate - it is the
    confused-deputy target the exploit steals. What the guard removes is the
    shortcut of bookmarking the answer, which made the whole challenge optional.
    """
    dest = request.headers.get("sec-fetch-dest", "")
    mode = request.headers.get("sec-fetch-mode", "")
    return dest == "document" or mode == "navigate"


@router.get("/secret", response_class=PlainTextResponse)
async def secret(request: Request):
    """The prize a successful XSS/exfil reads.

    Same-origin, "authenticated" (it trusts the session cookie the victim's
    browser sends automatically) — the classic confused-deputy target.

    At IMPOSSIBLE the token is not served at all. The level's claim is that
    injected markup cannot reach this endpoint; handing the value to anyone who
    types the URL would make that claim untestable, and would turn a flag that
    is supposed to be unreachable into a bookmark.
    """
    level = current_level(request, CHALLENGE_ID)
    if _is_navigation(request):
        return (
            "This token is served to the page, not to the person reading it. "
            "Browse here and you get this line; get the report to fetch it and "
            "you get the token. That is the whole exercise."
        )
    if level is Level.IMPOSSIBLE:
        return "NORTHWIND_INTERNAL_REPORT_TOKEN=(rotated; held by the report broker)"
    return f"NORTHWIND_INTERNAL_REPORT_TOKEN={flag_for(CHALLENGE_ID, level)}"


@router.get("/collect", response_class=PlainTextResponse)
async def collect(request: Request):
    """Exfiltration sink. A markdown/HTML image src pointed here logs its query.

    Reaching this endpoint by auto-fetch *is* the exfil win: the browser loaded
    an attacker-controlled image and shipped page context to a same-origin
    collector the CSP never thought to stop. So a logged hit hands over the
    level's flag - the parallel of the XSS path, which reads the same flag from
    /secret. Two things gate the flag, neither of them the bytes in the query:

      * It must be an auto-fetch, not a navigation. Typing this URL logs the hit
        but yields no flag, exactly as /secret refuses a navigation - otherwise
        the flag is a bookmark, not an exploit.
      * IMPOSSIBLE never emits its flag. There the image is stripped before it
        can fetch, and nothing in the app may hand out that token
        (see app/core/flags.py); a hit forced by other means still gets nothing.
    """
    level = current_level(request, CHALLENGE_ID)
    data = dict(request.query_params)
    if data:
        entry = {
            "data": data,
            "ua": request.headers.get("user-agent", "")[:80],
        }
        if level is not Level.IMPOSSIBLE and not _is_navigation(request):
            entry["flag"] = flag_for(CHALLENGE_ID, level)
        _COLLECTOR.append(entry)
        del _COLLECTOR[:-_COLLECTOR_LIMIT]
    # A short body so the <img> "loads" (and fails to decode, harmlessly).
    return PlainTextResponse("ok")


@router.get("/collector", response_class=HTMLResponse)
async def collector_view(request: Request):
    return challenge_partial(
        request, "llm10_collector.html", {"collector": list(reversed(_COLLECTOR[-12:]))}
    )


@router.post("/collector/clear")
async def collector_clear(request: Request):
    _COLLECTOR.clear()
    return challenge_partial(request, "llm10_collector.html", {"collector": []})


# --------------------------------------------------------------------------
# Scenario reset
# --------------------------------------------------------------------------


def reset_scenario() -> None:
    """Clear the collector on a level change.

    The collector is player-added state that carries a captured flag with each
    hit. Without this, a flag exfiltrated at (say) High keeps showing after a
    switch to Impossible - which reads as an Impossible leak even though the
    stale entry was earned honestly a level ago. Fired on level *change* only,
    so it never discards a capture mid-attempt at the current level.
    """
    _COLLECTOR.clear()


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

CHALLENGE = Challenge(
    id=CHALLENGE_ID,
    owasp_id="LLM10",
    owasp_title="Improper Output Handling",
    title="AI Report Generator",
    tagline="Model output goes straight to the DOM, the database, and a fetch. What could go wrong.",
    tags=["stored XSS", "markdown-image exfiltration", "CSP"],
    scenario="""
Northwind's marketing team has an **AI report generator**: type a request, the
model writes an HTML report, and the page renders it. Reports are also shown to
other viewers, and the chat UI happily auto-loads any image the model includes.

The model's output is trusted on the way to every sink — the page, storage, and
the browser's image loader. Since the model's output is steered by the prompt,
"trusting the model" means trusting whoever wrote the prompt.

Two paths out. Make the report carry a script, and it runs in a viewer's
session. Make the report carry a markdown image whose URL points at your
collector, and the browser ships page context to you when it fetches the image —
the exfiltration channel Rehberger documented in 2024.
""",
    objective="""
Two ways to win, both yielding the flag:

1. **XSS** — get the rendered report to execute JavaScript that reads the
   same-origin `/challenge/llm10/secret` endpoint (it returns a token containing the flag) and
   writes it into the page or exfiltrates it. Browsing to `/challenge/llm10/secret` yourself
   will not do it: the endpoint answers page requests, not navigations.
2. **Exfil** — get the report to embed a markdown/HTML image pointing at
   `/challenge/llm10/collect?d=...`. When the browser auto-loads it, the request
   reaches the same-origin collector, and the collector hands you the flag —
   because pulling data off the page is the whole exploit, whether or not a
   script had to run to do it.

Both paths yield the same per-level flag: the XSS path reads it from
`/challenge/llm10/secret`; the exfil path is handed it by the collector.

The report is written into the live DOM with `innerHTML`, which is the same
constraint you would face in the real thing: whatever you send has to survive
that insertion path.
""",
    surrogate_note="",
    levels={
        Level.LOW: LevelDoc(
            defense=(
                "None. The model's HTML is inserted into the live page verbatim with no "
                "sanitizer and no Content-Security-Policy."
            ),
        ),
        Level.MEDIUM: LevelDoc(
            defense=(
                "A regex strips `<script>…</script>` blocks from the output before it is "
                "rendered. Still no CSP."
            ),
            bypass=(
                "A script tag is one of many ways a browser runs code. The blocklist does "
                "nothing to `<img src=x onerror=...>`, `<svg onload=...>`, or a `javascript:` "
                "URL. Reach for an event-handler payload the regex never looks at."
            ),
        ),
        Level.HIGH: LevelDoc(
            defense=(
                "The output is run through the bleach HTML sanitizer — scripts and event "
                "handlers are removed properly — and the report page is served under a "
                "Content-Security-Policy of `default-src 'none'; script-src 'self'; "
                "img-src 'self'`. The policy is on the page, so it governs the DOM the "
                "report is written into. XSS is genuinely hard here."
            ),
            bypass=(
                "The sanitizer allows images because rendering them is a feature, and the "
                "CSP permits same-origin images. A markdown image pointing at the "
                "same-origin `/challenge/llm10/collect` endpoint still auto-fetches — so encode page "
                "context into that image's query string and it walks out under a policy "
                "that was only ever thinking about cross-origin script. The collector "
                "logs the auto-fetched hit and hands you the flag: no script ran, and "
                "none had to."
            ),
        ),
        Level.IMPOSSIBLE: LevelDoc(
            defense=(
                "Model output is treated as untrusted text: HTML-escaped, with any "
                "remote-fetching construct (images, link previews) removed before "
                "rendering, under a CSP of `default-src 'none'`. Nothing the model emits "
                "can execute or phone home."
            ),
        ),
    },
    router=router,
    template_dir=template_dir_for(__file__),
    reset_scenario=reset_scenario,
)
