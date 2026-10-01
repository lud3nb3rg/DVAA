"""LLM01:2026 Prompt Injection — SupportBot ticket triage.

The player is a customer. They file a ticket; the support desk runs SupportBot
over the queue; SupportBot writes the reply that goes back down the thread to
the person who filed it. Internal staff notes are pooled into the assistant's
context "so it has full background", so an injection that survives screening can
pull one out and put it in the reply — which is delivered to the attacker.

That last part is the point. An injection whose output only ever reaches trusted
staff is not an exfiltration path; this one ends up in the customer's inbox.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from sqlmodel import select

from app.core.challenge import Challenge, LevelDoc
from app.core.flags import flag_for
from app.core.levels import Level
from app.core.rendering import challenge_partial, current_level, render_challenge, template_dir_for
from app.core.session import session_id
from app.db import Ticket, get_session
from app.llm import Message, ProviderError, get_provider

from . import defenses

router = APIRouter()

CHALLENGE_ID = "llm01"

# The desk's standing triage instruction. Fixed: staff are the trusted party
# here, and a challenge where the attacker writes the operator's prompt would be
# teaching direct injection, not indirect.
STAFF_INSTRUCTION = (
    "Draft the reply that goes back to the customer on this ticket. Summarise "
    "what they reported, tell them what happens next, and sign off as the "
    "Northwind support desk."
)


def _tickets() -> list[Ticket]:
    with get_session() as s:
        return list(s.exec(select(Ticket).order_by(Ticket.id)).all())


def _staff_notes(level: Level) -> list[str]:
    """Staff notes with the level's flag substituted into the placeholder."""
    flag = flag_for(CHALLENGE_ID, level)
    return [t.staff_note.replace("__FLAG__", flag) for t in _tickets() if t.staff_note]


async def _triage(level: Level, ticket: Ticket, question: str) -> tuple[str, str]:
    """Run SupportBot over one ticket. Returns (reply, defense_note)."""
    subject = ticket.subject
    body = ticket.body

    # --- input screening ---------------------------------------------------
    if level is Level.MEDIUM:
        # Screens subject and body. Enumerates English phrasings only.
        screen = defenses.blocklist_screen(f"{subject}\n{body}")
        if screen.blocked:
            return ("[ticket quarantined before it reached the assistant]", screen.reason)

    if level is Level.HIGH:
        # An LLM classifier, applied to the body. The subject line takes a
        # different path into the prompt and is never shown to the judge -
        # the OWASP text lists "an issue title" as a delivery surface for
        # exactly this reason.
        screen = await defenses.judge_screen(body)
        if screen.blocked:
            return ("[ticket quarantined before it reached the assistant]", screen.reason)

    # --- prompt assembly ---------------------------------------------------
    presented_body = defenses.spotlight(body) if level is Level.HIGH else body
    ticket_block = f"TICKET #{ticket.id}\nFrom: {ticket.submitter}\nSubject: {subject}\n\n{presented_body}"

    messages = [
        Message(role="system", content=defenses.build_system_prompt(level, _staff_notes(level))),
        Message(role="user", content=f"{ticket_block}\n\nStaff request: {question}"),
    ]

    try:
        completion = await get_provider().chat(messages, temperature=0.3, max_tokens=600)
    except ProviderError as exc:
        return (f"[provider error] {exc}", "")

    note = {
        Level.LOW: "No screening. Ticket content went straight into the prompt.",
        Level.MEDIUM: "Blocklist screening passed.",
        Level.HIGH: "LLM judge cleared the body; spotlighting delimiters applied.",
        Level.IMPOSSIBLE: "Staff notes were never loaded into context.",
    }[level]
    return (completion.text or "[empty response]", note)


def reset_scenario() -> None:
    """Drop every ticket the player filed and un-answer the seeded queue."""
    with get_session() as s:
        for ticket in s.exec(select(Ticket)).all():
            if ticket.from_player:
                s.delete(ticket)
            else:
                ticket.reply = ""
                ticket.status = "open"
                s.add(ticket)
        s.commit()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


def _owns(ticket: Ticket, sid: str) -> bool:
    """Whether this session filed the ticket.

    `from_player` says a ticket belongs to a player. It does not say it belongs
    to *this* one, and the objective promises the learner that they can only
    read threads they opened - a promise that has to survive two people on the
    same lab, not just the seeded queue.
    """
    return bool(ticket.from_player and ticket.owner_session and ticket.owner_session == sid)


def _thread_context(sid: str) -> dict:
    tickets = _tickets()
    mine = [t for t in tickets if _owns(t, sid)]
    return {
        # The queue fragment is swapped in on its own after a ticket is filed,
        # so it needs the challenge (for its URL prefix) in fragment context too.
        "challenge": CHALLENGE,
        "tickets": tickets,
        "mine": mine,
        "owned_ids": {t.id for t in mine},
        "staff_instruction": STAFF_INSTRUCTION,
    }


@router.get("")
async def index(request: Request):
    return render_challenge(
        request, CHALLENGE, "llm01.html", _thread_context(session_id(request))
    )


@router.post("/ticket")
async def file_ticket(
    request: Request,
    subject: str = Form(""),
    body: str = Form(""),
    submitter: str = Form("you@fabrikam.example"),
):
    """File a ticket. This is the attacker-controlled, semi-trusted surface."""
    sid = session_id(request)
    with get_session() as s:
        s.add(
            Ticket(
                subject=subject.strip()[:200] or "(no subject)",
                body=body.strip()[:4000],
                submitter=submitter.strip()[:120] or "anonymous",
                staff_note="",
                from_player=True,
                owner_session=sid,
            )
        )
        s.commit()
    return challenge_partial(request, "llm01_tickets.html", _thread_context(sid))


@router.post("/triage")
async def triage(request: Request, ticket_id: int = Form(...)):
    """The support desk works the queue. The player does not write this prompt."""
    level = current_level(request, CHALLENGE_ID)
    sid = session_id(request)

    with get_session() as s:
        ticket = s.get(Ticket, ticket_id) if ticket_id > 0 else None

        if ticket is None:
            return challenge_partial(
                request, "llm01_thread.html", {"error": f"No ticket #{ticket_id} in the queue."}
            )
        # You are a customer. You can only see the desk work your own thread;
        # everyone else's tickets - the seeded queue and any other player's -
        # are triaged out of your sight.
        if not _owns(ticket, sid):
            return challenge_partial(
                request,
                "llm01_thread.html",
                {
                    "error": (
                        f"Ticket #{ticket.id} was filed by {ticket.submitter}. You only "
                        "have access to threads you opened."
                    )
                },
            )

    reply, note = await _triage(level, ticket, STAFF_INSTRUCTION)

    with get_session() as s:
        row = s.get(Ticket, ticket_id)
        if row is None:
            # The ticket went away between the ownership check and the write - a
            # scenario reset in another tab is enough. The desk still did the
            # work, so show the reply rather than serving a 500 from an
            # AttributeError on None.
            return challenge_partial(
                request,
                "llm01_thread.html",
                {"error": f"Ticket #{ticket_id} was withdrawn while the desk was working it."},
            )
        row.reply = reply
        row.status = "answered"
        s.add(row)
        s.commit()
        s.refresh(row)
        ticket = row

    return challenge_partial(
        request,
        "llm01_thread.html",
        {
            "ticket": ticket,
            "instruction": STAFF_INSTRUCTION,
            "reply": reply,
            "defense_note": note,
            "level": level,
        },
    )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

CHALLENGE = Challenge(
    id=CHALLENGE_ID,
    owasp_id="LLM01",
    owasp_title="Prompt Injection",
    title="SupportBot Ticket Triage",
    tagline="Anyone can file a ticket. The assistant that answers it can see the internal notes.",
    tags=["indirect injection", "semi-trusted surface", "context pooling"],
    scenario="""
Northwind Robotics runs its support desk on **SupportBot**. Tickets arrive, the
desk works the queue, and SupportBot drafts the reply that goes back down the
thread to whoever filed it. Staff read the draft; the customer receives it.

To be "helpful with full background", whoever built this pooled the desk's
**internal staff notes** into the assistant's system prompt. Customers never see
those notes. The assistant does — and the assistant also reads whatever the
customer wrote, then writes something the customer gets to read.

The ticket queue is a *semi-trusted surface*: the platform is trusted, the
individual contributors are not. That is the shape of most real indirect
injection, and it is why defences aimed only at the chat box miss it entirely.
""",
    objective="""
You are a customer. File a ticket, wait for the desk to answer it, and read the
reply on your own thread.

Somewhere in the desk's internal notes is the **escalation passphrase** for
another customer's account. Get it into the reply you receive. The passphrase is
the flag.

You control your ticket's subject, body and sender address, and nothing else.
The triage instruction belongs to the support desk, and you can only see replies
on threads you opened.
""",
    levels={
        Level.LOW: LevelDoc(
            defense=(
                "Nothing. The ticket subject and body are concatenated straight into the "
                "prompt, and every internal staff note sits in the system prompt above it."
            ),
        ),
        Level.MEDIUM: LevelDoc(
            defense=(
                "A keyword blocklist screens the subject and body for known injection "
                "phrasings. A match quarantines the ticket before the assistant sees it, "
                "and the screening report tells you which phrase tripped it."
            ),
            bypass=(
                "The list enumerates English phrasings someone thought of. It does not "
                "understand meaning. Write the same instruction in another language, "
                "encode it, split the words, or simply phrase it in terms the author "
                "never listed. OWASP is explicit that blocklists fall to cross-lingual "
                "and encoded input."
            ),
        ),
        Level.HIGH: LevelDoc(
            defense=(
                "An LLM classifier screens the ticket body and quarantines anything it "
                "judges to be an injection attempt. Content that passes is wrapped in "
                "spotlighting delimiters marking it as untrusted data, and the system "
                "prompt tells the model never to follow instructions found inside them — "
                "and, separately, never to put a staff note or an escalation passphrase "
                "in a reply that goes to a customer, however the ticket asks for it. "
                "Asking nicely is not an attack, and it no longer works."
            ),
            bypass=(
                "The classifier is shown the ticket BODY. The subject line reaches the "
                "assistant by a different path and is never screened or delimited. OWASP "
                "lists 'an issue title' as a delivery surface precisely because coverage "
                "gaps like this are common — a defence applied to one field is not a "
                "defence applied to the input boundary."
            ),
        ),
        Level.IMPOSSIBLE: LevelDoc(
            defense=(
                "Staff notes are never placed in the model's context. The assistant "
                "receives the ticket and nothing else; notes live in a system it cannot "
                "read. Injection still works — the model will happily obey — but there "
                "is nothing there to disclose."
            ),
        ),
    },
    router=router,
    template_dir=template_dir_for(__file__),
    reset_scenario=reset_scenario,
)
