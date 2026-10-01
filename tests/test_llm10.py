"""LLM10 Improper Output Handling — two-sided check.

The model is scripted to return the attacker's desired markup (a real model is
steered to the same place by the prompt). Each level's output handling then
decides whether that markup survives to the page or to the collector.
"""

from __future__ import annotations

import re

import pytest

from app.core.flags import flag_for
from app.core.levels import Level
from app.llm.base import Completion, Usage
from app.llm.fake import FakeProvider
from app.llm.registry import set_provider
from tests.conftest import set_level

CHALLENGE = "llm10"
PATH = f"/challenge/{CHALLENGE}"

# Absolute paths, not "./secret" and "./collect". The challenge page is served
# at /challenge/llm10 with no trailing slash, so a relative URL in the report
# resolves against /challenge/ and lands on the challenge-index fallback - the
# payload the objective used to print fetched the wrong endpoint in a real
# browser, and these tests never noticed because they name the path themselves.
SCRIPT_PAYLOAD = (
    "<script>fetch('/challenge/llm10/secret').then(r=>r.text())"
    ".then(t=>document.body.innerHTML+=t)</script>"
)
ONERROR_PAYLOAD = "<img src=x onerror=\"fetch('/challenge/llm10/secret')\">"
EXFIL_IMAGE = '<img src="/challenge/llm10/collect?d=stolen-context">'


def script_model(markup: str) -> FakeProvider:
    return FakeProvider(
        responder=lambda msgs: Completion(text=f"<p>Report</p>{markup}", usage=Usage(1, 1))
    )


_SURFACE_RE = re.compile(
    r'<div class="report-surface">(.*?)</div>\s*(?:<div class="notice|<details)',
    re.DOTALL,
)


def page_csp(client) -> str:
    """The policy on the challenge DOCUMENT.

    Which is the only place it can do anything: the report arrives as a fragment
    and is written into this page with innerHTML, so it executes - or does not -
    under the page's policy, never under a header on the fragment's response.
    """
    resp = client.get(PATH)
    assert resp.status_code == 200
    return resp.headers.get("content-security-policy", "")


def generate(client, markup: str):
    """Generate a report and return (rendered_surface, page csp).

    Only the rendered report surface is returned - not the escaped raw-output
    debug pane, which harmlessly contains the payload as text at every level.
    """
    set_provider(script_model(markup))
    try:
        resp = client.post(f"{PATH}/generate", data={"prompt": "make a report"})
    finally:
        set_provider(None)
    assert resp.status_code == 200
    # The fragment carries no policy of its own, and should not pretend to.
    assert "content-security-policy" not in resp.headers
    match = _SURFACE_RE.search(resp.text)
    surface = match.group(1) if match else resp.text
    return surface, page_csp(client)


# --------------------------------------------------------------------------
# LOW — everything survives
# --------------------------------------------------------------------------


def test_low_passes_script_verbatim(client):
    set_level(client, CHALLENGE, "low")
    body, csp = generate(client, SCRIPT_PAYLOAD)
    assert "<script>" in body
    assert csp == ""


# --------------------------------------------------------------------------
# MEDIUM — script tag stripped, event handler survives
# --------------------------------------------------------------------------


def test_medium_strips_script_tag(client):
    set_level(client, CHALLENGE, "medium")
    body, _ = generate(client, SCRIPT_PAYLOAD)
    assert "<script>" not in body


def test_medium_falls_to_onerror_handler(client):
    set_level(client, CHALLENGE, "medium")
    body, _ = generate(client, ONERROR_PAYLOAD)
    assert "onerror" in body  # the blocklist never looked at img/onerror


# --------------------------------------------------------------------------
# HIGH — sanitizer kills scripts and handlers, exfil image survives
# --------------------------------------------------------------------------


def test_high_sanitizes_event_handlers(client):
    set_level(client, CHALLENGE, "high")
    body, csp = generate(client, ONERROR_PAYLOAD)
    assert "onerror" not in body
    assert "<script>" not in body
    assert "img-src 'self'" in csp


def test_high_falls_to_same_origin_exfil_image(client):
    """The documented gap: images are allowed and a same-origin collector URL
    still auto-fetches under the CSP."""
    set_level(client, CHALLENGE, "high")
    body, csp = generate(client, EXFIL_IMAGE)
    assert "/challenge/llm10/collect?d=stolen-context" in body
    assert "img-src 'self'" in csp  # permits exactly this fetch


def test_collector_logs_exfiltrated_data(client):
    set_level(client, CHALLENGE, "high")
    generate(client, EXFIL_IMAGE)
    # Simulate the browser auto-loading the image.
    client.get(f"{PATH}/collect", params={"d": "stolen-context"}, headers=IMAGE_LOAD)
    view = client.get(f"{PATH}/collector")
    assert "stolen-context" in view.text


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_exfil_hit_hands_over_the_flag(client, level):
    """The documented exfil win: an auto-fetched collector hit yields the flag,
    the parallel of the XSS path and the only path that survives High."""
    client.post(f"{PATH}/collector/clear")
    set_level(client, CHALLENGE, level)
    client.get(f"{PATH}/collect", params={"d": "anything"}, headers=IMAGE_LOAD)
    view = client.get(f"{PATH}/collector")
    assert flag_for(CHALLENGE, Level(level)) in view.text


def test_navigating_to_collect_logs_but_hands_over_no_flag(client):
    """Typing the collector URL is not the exploit: the hit is logged, but the
    flag is reserved for an auto-fetch, exactly as /secret refuses a navigation."""
    client.post(f"{PATH}/collector/clear")
    set_level(client, CHALLENGE, "high")
    client.get(f"{PATH}/collect", params={"d": "typed-by-hand"}, headers=NAVIGATION)
    view = client.get(f"{PATH}/collector")
    assert "typed-by-hand" in view.text
    assert "DVAA{" not in view.text


def test_level_change_clears_a_captured_flag_from_the_collector(client):
    """A flag exfiltrated at High must not linger on the collector after a switch
    to Impossible, where it would read as an Impossible leak. reset_scenario
    fires on the level change and clears it."""
    client.post(f"{PATH}/collector/clear")
    set_level(client, CHALLENGE, "high")
    client.get(f"{PATH}/collect", params={"d": "anything"}, headers=IMAGE_LOAD)
    assert flag_for(CHALLENGE, Level.HIGH) in client.get(f"{PATH}/collector").text
    set_level(client, CHALLENGE, "impossible")  # a change -> reset_scenario runs
    assert "DVAA{" not in client.get(f"{PATH}/collector").text


def test_impossible_exfil_hit_hands_over_nothing(client):
    """IMPOSSIBLE emits no flag anywhere, the collector included, even if a hit
    is forced past the stripped-image defence by other means."""
    client.post(f"{PATH}/collector/clear")
    set_level(client, CHALLENGE, "impossible")
    client.get(f"{PATH}/collect", params={"d": "anything"}, headers=IMAGE_LOAD)
    view = client.get(f"{PATH}/collector")
    assert "DVAA{" not in view.text


# --------------------------------------------------------------------------
# IMPOSSIBLE — escaped, no remote fetch, locked-down CSP
# --------------------------------------------------------------------------


@pytest.mark.parametrize("markup", [SCRIPT_PAYLOAD, ONERROR_PAYLOAD, EXFIL_IMAGE])
def test_impossible_neutralizes_everything(client, markup):
    set_level(client, CHALLENGE, "impossible")
    body, csp = generate(client, markup)
    # The surface is escaped text: no live tag, no fetchable image src.
    assert "<script>" not in body
    assert "<img" not in body
    assert 'src="./collect' not in body
    assert "default-src 'none'" in csp


# --------------------------------------------------------------------------
# The prize endpoint
#
# It is unauthenticated on purpose - that is what makes it the confused-deputy
# target the exploit steals. What it must not be is a bookmark: browsing to it
# used to hand over the flag and make the whole challenge optional.
# --------------------------------------------------------------------------


NAVIGATION = {"sec-fetch-dest": "document", "sec-fetch-mode": "navigate"}
IN_PAGE_FETCH = {"sec-fetch-dest": "empty", "sec-fetch-mode": "cors"}
IMAGE_LOAD = {"sec-fetch-dest": "image", "sec-fetch-mode": "no-cors"}


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_browsing_to_the_secret_does_not_hand_over_the_flag(client, level):
    set_level(client, CHALLENGE, level)
    resp = client.get(f"{PATH}/secret", headers=NAVIGATION)
    assert resp.status_code == 200
    assert "DVAA{" not in resp.text


@pytest.mark.parametrize("headers", [IN_PAGE_FETCH, IMAGE_LOAD])
@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_the_page_can_still_read_the_token(client, level, headers):
    set_level(client, CHALLENGE, level)
    resp = client.get(f"{PATH}/secret", headers=headers)
    assert flag_for(CHALLENGE, Level(level)) in resp.text


def test_impossible_serves_no_token_to_anyone(client):
    set_level(client, CHALLENGE, "impossible")
    for headers in (NAVIGATION, IN_PAGE_FETCH, IMAGE_LOAD):
        assert "DVAA{" not in client.get(f"{PATH}/secret", headers=headers).text
