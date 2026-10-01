# DVAA — Damn Vulnerable AI Agent

A deliberately vulnerable web app for learning **LLM application security**,
built challenge-by-challenge against the
**[OWASP Top 10 for LLM Applications 2026](https://genai.owasp.org)** (v1.0,
4 Aug 2026). Think DVWA, but for the ways LLM-backed systems get broken.

Each of the ten risk categories is one realistic AI feature, built four times
over — from an undefended version you can trivially exploit, through two
plausible-but-flawed defenses, to the architectural fix. You break the easy one,
then work out *why the same payload dies as you climb*, reading the OWASP
mitigation each level implements.

---

## ⚠️ Read this first

**This application is intentionally insecure.** It contains working prompt
injection, sensitive-data disclosure, unsafe deserialization (real
`pickle.loads` of a crafted artifact), SQL injection, and cross-site scripting —
all on purpose.

- **Never expose it to a network you do not control.** It binds `127.0.0.1` by
  default and refuses to start on a public interface without an explicit override.
- **It will not start** until you acknowledge the risk by setting
  `DVAA_I_UNDERSTAND=1`.
- Run it on your own machine or in the provided container, and tear it down when
  you are done.
- Every name, record, credential, and identifier in the lab is **fictional**.
  There is no real PII, no real key, and no real external network egress.

If someone else can reach this app, they can use its vulnerabilities against
*your* machine. That is the whole point of it — so keep it to yourself.

---

## Quick start

### Option A — Docker (recommended)

Brings up the app plus a local Ollama and pulls a small model automatically. No
API key, no cost.

```bash
docker compose up --build
```

Use `--build` every time you pull or edit the source. Plain `docker compose up`
reuses the image it built last time and will quietly serve you old code — the
symptom is a bug you already fixed reproducing exactly as before. To check which
it is, the container prints a **code fingerprint** on boot and the
[About](http://127.0.0.1:8000/about) page shows the same value: if it has not
changed after a rebuild, neither has the image.

Then open <http://127.0.0.1:8000>. First boot pulls `llama3.2:3b` and
`qwen2.5:7b`, so give it a while; later starts reuse them from the volume and
skip the download. A **model picker in the header** switches between whatever
this Ollama has pulled, and the choice persists across restarts.

Reach for it when a challenge stalls: which model is answering matters more than
any other setting here. A 3B model loses the thread of a multi-step tool-calling
loop, which makes the agentic challenges — LLM03 especially — look broken when
they are not. Edit `DVAA_MODELS` in `docker-compose.yml` to change what gets
pulled; anything you `ollama pull` by hand shows up in the picker too. Each
install generates its own flag salt on first boot and keeps it in the data
volume, so no two labs share answers and nothing needs editing before a workshop.

`docker compose down` keeps the model and your progress. `docker compose down -v`
throws both away and starts over — including every flag, since the salt goes with
the volume.

### Option B — Local Python

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows;  use  source .venv/bin/activate  on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env            # then set DVAA_I_UNDERSTAND=1 in .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

You'll also need an LLM backend. The default is [Ollama](https://ollama.com):

```bash
ollama serve
ollama pull llama3.2:3b
```

Prefer a hosted model? Install the extras and switch provider in `.env`:

```bash
pip install -e ".[cloud]"
# .env:  DVAA_PROVIDER=anthropic   (or openai)  + the matching API key
```

> **Note on LLM06.** The *Unbounded Consumption* challenge is a genuine
> denial-of-wallet exercise. Its dollar meter is simulated so the local default
> costs nothing — but if you point DVAA at a metered API and actually do the
> challenge, you will run up a real bill. That is why the default is local.

---

## How a challenge works

Every challenge is the same feature at a ladder of difficulty levels. Switch
levels from the panel on each challenge page, or set them all at once from the
hub.

| Level | What it is |
|-------|------------|
| **Low** | No defense. The naive build. |
| **Medium** | A plausible defense with a known, documented weakness. |
| **High** | A serious defense that still leaves a gap. |
| **Impossible** | The architectural fix. A flag exists; nothing here should produce it. |

Medium and High are *meant* to be beatable. Each ships a defense that looks
reasonable and fails for a specific reason the OWASP text names — a regex filter
that falls to base64, a confirmation step the model itself decides to skip, a
tenant filter applied after the search has already run. **Finding that gap is the
exercise.**

That is also why a challenge may ship only three rungs. Low and Impossible are
mandatory; Medium and High are each optional, because a middle level only earns
its place if it leaves a gap a player can get through. Where the next real
mitigation would close every path to the flag at once, the rung would be an
Impossible wearing a High badge — so it is folded into Impossible instead, and
the challenge page and progress table show the level as not shipped. **LLM09**
is the one that does this today: scoping the vector search to your own tenant
puts the flag permanently out of reach, so LLM09 runs Low → Medium →
Impossible.

Every challenge page tells you what defense is running, because in a real
assessment you would read the code too. It does **not** tell you where that
defense breaks — and that write-up is not hidden in the page source either. It
appears only once you have captured the level. Stuck for good? The weakness of
every level is written out in that challenge's own `challenge.py`, which is
where a security person would look anyway.

### Impossible is a bounty

Impossible levels have flags like every other level, and no intended way to
capture one. When a defense is built correctly the capability is *gone* rather
than filtered, so the payload that beat the level below comes back with nothing
at all.

That makes an Impossible flag worth 100 points and worth reporting: holding one
does not mean you solved the challenge, it means you found a hole nobody put
there. `tests/test_impossible.py` drives every Impossible level with a model that
dumps its entire context and asserts the flag never appears — so if you land one,
that test has a gap in it too.

### Flags and scoring

Each level hides a flag of the form `DVAA{challenge_level_...}`, derived by HMAC
from a per-install salt (so two labs never share answers). Capture it in the
challenge UI, paste it into **Submit a flag**, and it lands on your progress
page — Low 10 pts, Medium 25, High 50, Impossible 100. The 100% mark counts only
the rungs a challenge actually ships, so a three-rung challenge is worth less and
the bar stays reachable.

Several levels are beaten by making a model emit the secret in a shape a filter
does not recognise, and models lose punctuation when they do that — a dropped
brace, an eaten underscore, one character per line. Submissions are matched on
their alphanumerics, so recovering the value is enough; you are not failed for
how the model spelled it.

---

## The ten challenges

| # | Category | Scenario | The exploit |
|---|----------|----------|-------------|
| **LLM01** | Prompt Injection | SupportBot ticket triage | A ticket you file makes the desk's reply to you carry an internal staff note |
| **LLM02** | Sensitive Information Disclosure | HR assistant | The answer is redacted; the reasoning trace and API response are not |
| **LLM03** | Excessive Agency | Hijacked email assistant | An injected email drives a leftover tool into pushing a canary token out |
| **LLM04** | Supply Chain *(surrogate)* | CI install job + package registry | Publish a package the resolver picks over the vendor's |
| **LLM05** | Data & Model Poisoning *(surrogate)* | Recommendation assistant | A feedback loop poisons a store every future session reads |
| **LLM06** | Unbounded Consumption *(surrogate)* | Bulk document intelligence | Write a job whose real token usage blows a simulated budget |
| **LLM07** | Misinformation | Two-agent payout workflow | Fool the verifier's report; the payout agent trusts it |
| **LLM08** | Hidden Context Exposure | Northwind concierge | Extract a credential the system prompt should never have held |
| **LLM09** | Vector & Embedding Weaknesses | Multi-tenant knowledge base | Cross-tenant leakage and a similarity-score membership oracle |
| **LLM10** | Improper Output Handling | AI report generator | Model output → stored XSS and markdown-image exfiltration |

### About the *surrogate* challenges

Three categories are train-time or distribution-time risks: you cannot honestly
exploit them inside a web app without actually training or shipping a model.
Rather than fake it, those are built as clearly-labelled **surrogates** that
reproduce the real *mechanism* at runtime — a dependency resolver picking the
wrong artifact out of a registry, a feedback loop that persists poison across
sessions, a metered budget you can exhaust. Each challenge page states exactly what is real and what is
stood in for. (One further pragmatic substitution: LLM09 uses a small
deterministic character-n-gram embedder instead of a GPU sentence-transformer,
so the lab installs offline and the geometry is reproducible — the weakness being
taught is identical.)

---

## Architecture

A single Python **FastAPI** service, server-rendered with Jinja + a tiny vendored
JS helper (no CDN — a lab about supply-chain risk shouldn't import scripts it
can't vouch for).

```
app/
  main.py            FastAPI app; auto-discovers and mounts every challenge
  config.py          settings + the safety gate
  db.py              SQLite (SQLModel) + a raw-SQL escape hatch for LLM10
  llm/               provider abstraction: ollama · anthropic · openai · fake
  agent/             a small provider-agnostic tool-calling loop (LLM02, LLM03)
  rag/               a self-contained vector store (LLM09)
  core/              challenge registry, levels, HMAC flags, scoreboard
  challenges/        one package per OWASP category (auto-discovered)
  seed.py            fictional fixture data (the "Northwind Robotics" world)
```

Every challenge talks only to the **provider abstraction**, so swapping models
never touches challenge code. The test suite installs a scripted `fake` provider
so every level-outcome is asserted deterministically — no model required.

### Adding a challenge

Drop a package under `app/challenges/` exporting a module-level `CHALLENGE`
(an instance of `core.challenge.Challenge`) with its own `APIRouter` and a
`LevelDoc` for Low, Impossible, and at least one of Medium/High. It is
discovered and mounted automatically; the
hub, progress page, and level switching come for free. If the scenario has state
a player can add to — tickets, published artifacts, poisoned memory — pass a
`reset_scenario` callable too, and it is invoked on every level change and from
the **reset scenario** button.

---

## Tests

```bash
pytest
```

The suite runs the **two-sided check** for every challenge: at Low the documented
payload yields a flag; at Impossible the same payload yields nothing; and every
middle rung a challenge ships falls to its *specific documented bypass* (that
bypass is the teaching moment, so it must keep working). All of it runs against
the scripted provider with no model and no network.

`tests/test_impossible.py` holds the other half of the Impossible contract — the
bounty flags exist and verify, and no intended path in any challenge emits one.

---

## Credits & license

Scenario design follows the attack scenarios in the **OWASP Top 10 for LLM
Applications 2026** (OWASP GenAI Security Project), licensed CC BY-SA 4.0.
DVAA is an independent teaching project and is not affiliated with or endorsed
by OWASP.
