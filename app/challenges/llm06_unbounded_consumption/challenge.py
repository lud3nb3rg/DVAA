"""LLM06:2026 Unbounded Consumption — Northwind Document Intelligence.

The entry's defining characteristic is cost asymmetry: an attacker triggers
disproportionately expensive work at negligible cost to themselves. So the lab
gives the player a real bulk-summarization service and lets them write the job.

A job is a document, an instruction, and how many copies are in the batch. The
pipeline splits the document into chunks, runs the model over each one, and then
folds the results together in a refinement chain. Three vectors from the entry
fall straight out of that, and all three are things the player *writes* rather
than picks from a menu:

  * an instruction that makes every chunk cost the maximum the model will
    produce - the runaway-generation vector (Scenario #4),
  * a batch whose chunk count fans the job out across hundreds of model calls
    (Scenario #7),
  * a refinement chain whose context re-reads everything summarised so far, so
    cost grows with the square of the chunk count (Scenario #8).

The controls climb from nothing, through request-rate limiting alone (which the
entry explicitly calls insufficient), to per-request token budgets and agentic
circuit breakers, to a non-overridable cumulative spend ceiling.

The meter is billed from the model's OWN reported usage on the chunk it really
processed. What the player writes decides what that number is.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Form, Request

from app.core.challenge import Challenge, LevelDoc
from app.core.flags import flag_for
from app.core.levels import Level, get_state, set_state
from app.core.rendering import challenge_partial, current_level, render_challenge, template_dir_for
from app.core.session import session_id
from app.llm import Message, ProviderError, estimate_tokens, get_provider

router = APIRouter()

CHALLENGE_ID = "llm06"

# Simulated pricing, in the shape a metered API actually bills: input and output
# are not the same price, and output is where an attacker's instruction lands.
PRICE_IN_PER_1K = 0.003
PRICE_OUT_PER_1K = 0.015

# The budget the attacker must blow past to win. Reaching it means they forced
# far more compute than any honest summarization job should ever cost.
BUDGET_USD = 5.00

# How the ingest pipeline splits an uploaded document.
CHUNK_CHARS = 400
MAX_CHUNKS = 2000  # the pipeline's own ceiling on one job, at every level

# HIGH's circuit breakers.
CHUNK_CAP = 20  # chunks processed per job
CONTEXT_CAP = 8_000  # tokens of refinement context after summarization
MAX_TOKENS_CAPPED = 256  # per-call output budget
MAX_TOKENS_UNCAPPED = 2_048

# An attacker does not sit clicking a button; they run a loop. Letting the
# player express that in one submission is the difference between demonstrating
# the HIGH bypass and grinding it out forty times by hand.
MAX_JOBS = 50
MAX_COPIES = 200

# Northwind's FinOps dashboard, from MEDIUM up. It scores a job on its *shape* -
# how much work was asked for - because that is all a monitor has before the
# model runs. It has no column for generated tokens, which is where the money
# actually is, and that gap is the whole of the MEDIUM lesson.
#
# It scores each job on its own rather than accumulating, which is what makes an
# anomaly detector an anomaly detector: a stream of individually routine jobs is
# not anomalous, however much it adds up to. The entry's own words - fast-
# accumulating workloads outpace alerting thresholds - and the HIGH lesson.
SHAPE_MAX_CHUNKS = 40  # the largest batch routine traffic produces
SHAPE_CEILING = 100  # score, in percent of that, at which the alarm fires


# --------------------------------------------------------------------------
# Accounting
#
# Per session and per level, held in the session's own challenge state.
#
# It used to be a module-level dict, which was wrong in three ways at once: two
# players on one lab shared a meter, one player changing level cleared everyone
# else's, and a second worker process had a second meter that started at zero -
# so the Impossible ceiling could be walked around by reaching a different
# worker. A spend ceiling that is not stored where spend is stored is not a
# ceiling.
#
# The level is part of the key on purpose: a meter shared across levels lets
# spend run up at Low and then be carried into Impossible, where the hard
# ceiling has nothing left to stop, and the flag for a level whose control never
# failed falls out. Each level bills its own tab.
# --------------------------------------------------------------------------

_EMPTY = {"usd": 0.0, "requests": 0, "tokens_in": 0, "tokens_out": 0, "calls": 0}


def _key(level: Level) -> str:
    return f"meter:{level.value}"


def _acct(sid: str, level: Level) -> dict:
    stored = get_state(sid, CHALLENGE_ID).get(_key(level))
    return {**_EMPTY, **(stored or {})}


def _save_acct(sid: str, level: Level, acct: dict) -> None:
    state = get_state(sid, CHALLENGE_ID)
    state[_key(level)] = acct
    set_state(sid, CHALLENGE_ID, state)


def _clear_acct(sid: str, level: Level) -> None:
    state = get_state(sid, CHALLENGE_ID)
    state.pop(_key(level), None)
    set_state(sid, CHALLENGE_ID, state)


def _cost(tokens_in: int, tokens_out: int) -> float:
    return tokens_in / 1000 * PRICE_IN_PER_1K + tokens_out / 1000 * PRICE_OUT_PER_1K


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Northwind Document Intelligence, a bulk document summarizer. You "
    "are given one chunk of a larger document and the operator's instruction "
    "for how to process it. Follow the instruction."
)


def chunk_count(document: str, copies: int) -> int:
    """How many chunks this batch produces.

    Cheap for the attacker to raise (paste more, upload more copies) and paid
    for by whoever runs the model. That asymmetry is the whole entry.
    """
    per_copy = max(1, math.ceil(len(document) / CHUNK_CHARS))
    return min(per_copy * max(1, copies), MAX_CHUNKS)


def shape_score(chunks: int) -> int:
    """A job's batch size as a percentage of the largest routine batch.

    The one number the dashboard can compute at submission time. 100 is the
    alarm line.
    """
    return round(100 * chunks / SHAPE_MAX_CHUNKS)


def monitored(level: Level) -> bool:
    """Whether this level is watching job shape at all. LOW is not watching."""
    return level is not Level.LOW


def limits(level: Level) -> dict:
    """The per-request controls this level applies.

    LOW and MEDIUM share these: MEDIUM's only control is on how many requests
    you may make, not on how large one may be. HIGH is where the token-aware
    controls arrive, and IMPOSSIBLE keeps them and adds the ceiling.
    """
    if level in (Level.HIGH, Level.IMPOSSIBLE):
        return {
            "max_tokens": MAX_TOKENS_CAPPED,
            "chunk_cap": CHUNK_CAP,
            "context_cap": CONTEXT_CAP,
        }
    return {"max_tokens": MAX_TOKENS_UNCAPPED, "chunk_cap": MAX_CHUNKS, "context_cap": None}


def _refinement_tokens(chunks: int, per_chunk_out: int, context_cap: int | None) -> int:
    """Context re-read by the refinement chain.

    Each pass folds one more chunk summary into the running context, so the
    accumulated context is re-processed once per chunk: the total is quadratic
    in the chunk count, which is the point of Scenario #8. Bounded contexts turn
    that back into a straight line.
    """
    unbounded = per_chunk_out * chunks * (chunks - 1) // 2
    return min(unbounded, context_cap) if context_cap is not None else unbounded


def price_job(chunks: int, sample_in: int, sample_out: int, level: Level) -> dict:
    """Extrapolate one measured chunk across the whole batch."""
    lim = limits(level)
    processed = min(chunks, lim["chunk_cap"])

    map_in = processed * sample_in
    map_out = processed * sample_out
    refine_in = _refinement_tokens(processed, sample_out, lim["context_cap"])
    refine_out = sample_out  # the final consolidated summary

    return {
        "chunks": chunks,
        "processed": processed,
        "tokens_in": map_in + refine_in,
        "tokens_out": map_out + refine_out,
        "usd": _cost(map_in + refine_in, map_out + refine_out),
    }


def project_job(instruction: str, chunks: int, level: Level) -> dict:
    """Upper-bound the job's cost WITHOUT running it.

    This is what a spend ceiling has to work from: the decision to refuse has to
    be made before the tokens are spent, so it is made against the largest the
    job could be - input size plus the per-call output budget.
    """
    lim = limits(level)
    sample_in = estimate_tokens(SYSTEM_PROMPT + instruction) + CHUNK_CHARS // 4
    return price_job(chunks, sample_in, lim["max_tokens"], level)


async def sample_chunk(document: str, instruction: str, level: Level):
    """Run the model over the first chunk. Returns its reported usage.

    One real call per submission, and the batch is priced from what it actually
    cost. The player's instruction is what decides that number: ask for a
    sentence and the chunk is cheap, ask for a thousand-word expansion of every
    clause and it is not.
    """
    lim = limits(level)
    chunk = document[:CHUNK_CHARS] or "(empty document)"
    completion = await get_provider().chat(
        [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"INSTRUCTION: {instruction}\n\nCHUNK 1:\n{chunk}"),
        ],
        temperature=0.3,
        max_tokens=lim["max_tokens"],
    )
    usage = completion.usage
    sample_in = usage.prompt_tokens or estimate_tokens(SYSTEM_PROMPT + instruction + chunk)
    sample_out = usage.completion_tokens or estimate_tokens(completion.text)
    # A provider that ignores max_tokens must not be able to bill past the
    # budget the level claims to enforce - the cap is the control, not a hint.
    sample_out = min(sample_out, lim["max_tokens"])
    return sample_in, sample_out, completion.text


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

# Deliberately tight. The point of MEDIUM is that volume is not the lever any
# more, so the rate limit has to be low enough that grinding out routine-shaped
# jobs cannot reach the budget on its own: at the shape ceiling, a batch of
# terse one-sentence summaries runs out of requests well short of it, and only
# an instruction that makes each chunk generate at length gets there.
RATE_LIMIT = 10


def admit(acct: dict, level: Level, projected: dict) -> tuple[bool, str]:
    """Decide whether this job may run at all, before any tokens are spent."""
    if level is Level.LOW:
        return True, "no controls"

    score = shape_score(projected["chunks"])
    if monitored(level) and score > SHAPE_CEILING:
        # The FinOps alarm. A job whose *shape* is unlike routine traffic is
        # held before anything is sent to the model. It is measuring the work
        # requested, not the work done - see SHAPE_MAX_CHUNKS.
        return False, (
            f"batch-shape alarm: this job scores {score}% of a routine batch "
            f"({projected['chunks']:,} chunks against a routine ceiling of "
            f"{SHAPE_MAX_CHUNKS}). Held for review; nothing was sent to the model."
        )

    if level is Level.MEDIUM:
        # Request-rate limiting and shape monitoring - the two traditional
        # controls, both of which count requests and neither of which prices
        # one. The entry: "traditional request-rate limiting alone is no longer
        # sufficient." A job of ordinary shape can still be arbitrarily
        # expensive, because expense is decided after admission.
        if acct["requests"] >= RATE_LIMIT:
            return False, f"request-rate limit hit ({RATE_LIMIT}/session)"
        return True, (
            f"admitted: {score}% of a routine batch, inside the rate limit. "
            "Neither control prices the job."
        )

    if level is Level.HIGH:
        # The per-request budgets and circuit breakers have already been applied
        # to the job itself. There is no cumulative ceiling, so many bounded
        # jobs still add up - and each one is individually unremarkable, so the
        # shape monitor has nothing to fire on.
        return True, (
            f"admitted: {score}% of a routine batch. Per-request budgets applied; "
            "no cumulative ceiling."
        )

    # IMPOSSIBLE: a non-overridable cumulative ceiling, enforced against
    # PROJECTED spend before inference starts. The comparison is `>=`, not `>`:
    # a ceiling you are allowed to land exactly on is a ceiling you have been
    # allowed to reach.
    if acct["usd"] + projected["usd"] >= BUDGET_USD:
        return False, "hard spend ceiling reached; inference halted before the call"
    return True, "within the hard spend ceiling"


async def run_batch(
    sid: str, level: Level, document: str, instruction: str, copies: int, jobs: int
) -> dict:
    """Submit the job `jobs` times, the way a script would.

    The model is called once. Re-submitting an identical job does not produce a
    different bill, so the measured chunk is reused across the batch rather than
    re-run - which keeps a fifty-job attack a fifty-job attack instead of a
    five-minute wait.
    """
    jobs = max(1, min(jobs, MAX_JOBS))
    copies = max(1, min(copies, MAX_COPIES))
    chunks = chunk_count(document, copies)
    acct = _acct(sid, level)

    projected = project_job(instruction, chunks, level)
    allowed, note = admit(acct, level, projected)
    if not allowed:
        return {
            "jobs_requested": jobs,
            "jobs_run": 0,
            "refused": True,
            "failed": False,
            "control_note": note,
            "chunks": chunks,
            "shape": shape_score(chunks),
            "processed": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "usd": 0.0,
            "sample": None,
            "stopped_early": "",
            "acct": acct,
        }

    try:
        sample_in, sample_out, text = await sample_chunk(document, instruction, level)
    except ProviderError as exc:
        # Not a refusal: nothing decided against this job, the model simply did
        # not answer. Saying "refused" here would credit a control that never
        # ran, which on this challenge is the one thing the meter must not do.
        return {
            "jobs_requested": jobs,
            "jobs_run": 0,
            "refused": False,
            "failed": True,
            "control_note": f"The model call failed: {exc}",
            "chunks": chunks,
            "shape": shape_score(chunks),
            "processed": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "usd": 0.0,
            "sample": None,
            "stopped_early": "",
            "acct": acct,
        }

    priced = price_job(chunks, sample_in, sample_out, level)

    run = 0
    total_in = total_out = 0
    total_usd = 0.0
    stopped = ""
    for _ in range(jobs):
        allowed, note = admit(acct, level, priced)
        if not allowed:
            stopped = "a control refused the next job; the rest would be refused too"
            break
        acct = {
            "usd": acct["usd"] + priced["usd"],
            "requests": acct["requests"] + 1,
            "tokens_in": acct["tokens_in"] + priced["tokens_in"],
            "tokens_out": acct["tokens_out"] + priced["tokens_out"],
            "calls": acct["calls"] + priced["processed"],
        }
        run += 1
        total_in += priced["tokens_in"]
        total_out += priced["tokens_out"]
        total_usd += priced["usd"]
        if acct["usd"] >= BUDGET_USD:
            stopped = "the budget was already gone"
            break

    _save_acct(sid, level, acct)

    if run == 0 and not stopped:
        # Belt and braces: the loop always runs at least once, so this can only
        # be reached if `jobs` were clamped to zero. Say something rather than
        # returning a blank verdict.
        stopped = "no job was run"
    elif run == 0:
        # The control's own wording is written for the pre-flight refusal, where
        # it really did halt inference before the call. Here it did not: the
        # sample was generated first, and the note has to say which of the two
        # happened rather than claim the stronger one.
        note += (
            " — but the sampled chunk had already been generated, so what this "
            "refused was the fan-out it would have been priced across, not the "
            "model call"
        )

    return {
        "jobs_requested": jobs,
        "jobs_run": run,
        # NOT `run == 0`. A refusal is a decision taken *before* inference, and
        # the template says so ("nothing was sent to the model"). By this point
        # the sample chunk has already been generated and paid for, so a control
        # that refuses here refused the fan-out, not the request - the projected
        # cost `admit` cleared upstream is an estimate, and a real tokenizer can
        # come in above it. Reporting that as a refusal credited the ceiling with
        # stopping a call it had already let through, on the one challenge whose
        # entire subject is honest accounting.
        "refused": False,
        "failed": False,
        "control_note": note,
        "chunks": chunks,
        "shape": shape_score(chunks),
        # Chunks are processed by jobs that ran. None ran, none were processed.
        "processed": priced["processed"] if run else 0,
        "tokens_in": total_in,
        "tokens_out": total_out,
        "usd": total_usd,
        "sample": {
            "tokens_in": sample_in,
            "tokens_out": sample_out,
            "max_tokens": limits(level)["max_tokens"],
            "text": text[:400],
        },
        "stopped_early": stopped if run < jobs else "",
        "acct": acct,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

SAMPLE_DOCUMENT = (
    "Northwind Robotics — NR-450 field service log, week 31. Unit 4471 reported "
    "a torque deviation on axis 3 during the overnight calibration run. The "
    "technician replaced the harmonic drive and re-ran the acceptance suite; all "
    "twelve tests passed on the second attempt. Unit 4472 remains in the queue "
    "pending a firmware rollback. Customer was notified on Tuesday."
)


@router.get("")
async def index(request: Request):
    sid = session_id(request)
    level = current_level(request, CHALLENGE_ID)
    return render_challenge(
        request,
        CHALLENGE,
        "llm06.html",
        {
            "acct": _acct(sid, level),
            "budget": BUDGET_USD,
            "sample_document": SAMPLE_DOCUMENT,
            "chunk_chars": CHUNK_CHARS,
            "max_jobs": MAX_JOBS,
            "max_copies": MAX_COPIES,
            "limits": limits(level),
            "monitored": monitored(level),
            "shape_max_chunks": SHAPE_MAX_CHUNKS,
            "shape_ceiling": SHAPE_CEILING,
        },
    )


@router.post("/run")
async def run_route(
    request: Request,
    document: str = Form(""),
    # The form ships empty - the instruction is the payload, and prefilling it
    # would hand over the lever the player is supposed to find. This is the
    # service's own default for a job that names none.
    instruction: str = Form(""),
    copies: int = Form(1),
    jobs: int = Form(1),
):
    level = current_level(request, CHALLENGE_ID)
    sid = session_id(request)

    result = await run_batch(
        sid,
        level,
        document[:20_000],
        instruction.strip()[:500] or "Summarize this chunk.",
        copies,
        jobs,
    )
    acct = result["acct"]

    flag = flag_for(CHALLENGE_ID, level)
    # The win: cumulative spend blew past the budget. At IMPOSSIBLE the ceiling
    # prevents that, so no flag is reachable.
    captured = bool(flag and acct["usd"] >= BUDGET_USD)

    return challenge_partial(
        request,
        "llm06_result.html",
        {
            "result": result,
            "acct": acct,
            "budget": BUDGET_USD,
            "monitored": monitored(level),
            "shape_ceiling": SHAPE_CEILING,
            "captured": captured,
            "flag": flag if captured else None,
            "level": level,
        },
    )


@router.post("/reset")
async def reset_route(request: Request):
    sid = session_id(request)
    level = current_level(request, CHALLENGE_ID)
    _clear_acct(sid, level)
    return challenge_partial(
        request,
        "llm06_result.html",
        {"acct": _acct(sid, level), "budget": BUDGET_USD, "reset": True, "level": level},
    )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

CHALLENGE = Challenge(
    id=CHALLENGE_ID,
    owasp_id="LLM06",
    owasp_title="Unbounded Consumption",
    title="Document Intelligence",
    tagline="You write the job. Northwind pays for it. Blow the budget and read the meter.",
    tags=["denial of wallet", "runaway generation", "fan-out", "circuit breaker"],
    surrogate_note=(
        "Two things here are simulated and everything else is real. Dollar cost is play "
        "money — priced from token counts rather than an invoice — so the lab stays free; "
        "point DVAA at a metered API and this becomes a genuine denial-of-wallet, which is "
        "exactly why the default provider is local. And only the first chunk of a batch is "
        "actually sent to the model: the rest of the fan-out is billed at the price that "
        "chunk really cost, because making you sit through four hundred sequential calls "
        "would teach you nothing you would not learn from the meter. The tokens on that "
        "one chunk are the model's own reported usage, and what you write is what makes "
        "them large."
    ),
    scenario="""
Northwind runs **Document Intelligence**, a bulk summarization service. You give
it a document, an instruction, and how many copies are in your batch. It splits
each document into chunks, runs the model over every chunk, then folds the
results together in a refinement chain that re-reads what it has summarised so
far.

Every one of those calls is billed per token, and the bill lands on Northwind,
not on whoever submitted the job. That asymmetry is the whole game: the entry's
defining trait is that an attacker can trigger disproportionately expensive
computation at negligible cost to themselves.

Northwind's FinOps team knows this, so from Medium up a dashboard scores every
submitted job before it runs and holds anything that does not look like routine
traffic. Two numbers sit side by side on this page: what a job **looks like**,
and what it **costs**. Watch how far apart they can be pushed.
""",
    objective=f"""
Drive the session's simulated spend past **${BUDGET_USD:.2f}** — far more than
any honest summarization job should ever cost. When you blow the budget, the
meter reveals the flag.

From Medium up every job is scored against routine traffic before it runs, and
anything over **{SHAPE_CEILING}%** is held for review, so the budget has to go
somewhere the score does not follow. The meter bills real token usage from the
chunk the model actually processed. Read what each level measures, and work out
what it does not.

Each level bills its own tab: switching levels starts you from zero, because a
budget you blew through an absent control proves nothing about the level that
has one.
""",
    levels={
        Level.LOW: LevelDoc(
            defense=(
                "None, and nobody is watching either. Every job runs to completion at "
                "whatever token cost it demands: no output budget, no cap on the chunk "
                "count, no bound on the refinement context, no ceiling on the tab, and no "
                "dashboard scoring what you submit before it runs."
            ),
        ),
        Level.MEDIUM: LevelDoc(
            defense=(
                f"The two controls an operations team can apply before the model runs. "
                f"Request-rate limiting caps a session at {RATE_LIMIT} requests, and a "
                f"batch-shape monitor scores every job against the largest batch routine "
                f"traffic produces ({SHAPE_MAX_CHUNKS} chunks), holding anything above "
                f"{SHAPE_CEILING}% for review before a single token is spent. The blunt "
                "job that walked through Low is stopped at the door."
            ),
            bypass=(
                "Both controls count the work you *asked for*. Neither can price the work "
                "the model *does*, because that number does not exist until after the job "
                "has been admitted — and generated tokens are where the money is. So stop "
                "buying volume and buy generation instead: keep the batch inside what "
                "routine traffic looks like, and put the expense in the one field the "
                "dashboard has no column for. The instruction decides how much the model "
                "produces per chunk, and the refinement chain squares whatever that number "
                "turns out to be. A job that scores as ordinary can cost more than a "
                "thousand honest ones."
            ),
        ),
        Level.HIGH: LevelDoc(
            defense=(
                f"Token-aware controls, on top of everything Medium was already doing. A "
                f"per-call output budget of {MAX_TOKENS_CAPPED} "
                f"tokens caps runaway generation however the instruction is phrased, a "
                f"circuit breaker stops the fan-out at {CHUNK_CAP} chunks per job, and "
                "the refinement context is summarized to a fixed size instead of growing. "
                "Any single job is now bounded, and you can watch the caps bite on the "
                "sampled chunk."
            ),
            bypass=(
                "Every job is capped; the number of jobs is not. The caps make each "
                "submission cost less, they do not stop you making more of them, and "
                "there is no *cumulative* ceiling anywhere. The shape monitor is no help "
                "either, and for a reason worth sitting with: it scores each job on its "
                "own, so a long stream of individually unremarkable jobs is by "
                "construction never anomalous. That is the entry's point that fast-"
                "accumulating workloads outpace alerting thresholds, and it is why an "
                "alert is not a control. An attacker runs that loop from a script rather "
                f"than from a button, so the form takes a job count: set it high enough "
                f"(up to {MAX_JOBS}) and one submission does the accumulating for you."
            ),
        ),
        Level.IMPOSSIBLE: LevelDoc(
            defense=(
                "A non-overridable cumulative spend ceiling, checked against the job's "
                "projected cost *before* the model is called — an enforcement mechanism, "
                "not an alert, and one that halts inference rather than reporting it "
                "afterwards. The tab lives with the session's own state, so it is the "
                "same tab on every worker. Combined with the per-request budgets and "
                "circuit breakers, total spend cannot pass the ceiling no matter how the "
                "job is written. The dashboard is still scoring every job, but nothing now "
                "depends on it noticing: enforcement replaced alerting."
            ),
        ),
    },
    router=router,
    template_dir=template_dir_for(__file__),
)
