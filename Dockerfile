# DVAA — Damn Vulnerable AI Agent
#
# This image contains a DELIBERATELY VULNERABLE application. The container is the
# blast-radius boundary for the LLM04 unsafe-deserialization and LLM10 SQL/XSS
# challenges: run it, keep it on loopback or an isolated lab network, and throw
# it away when you are done. Do not deploy it anywhere reachable by others.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DVAA_HOST=0.0.0.0 \
    DVAA_PORT=8000

WORKDIR /app

# Dependencies first, for layer caching.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .

COPY app ./app

# Non-root runtime.
RUN useradd --create-home --uid 10001 dvaa \
    && mkdir -p /app/data && chown -R dvaa:dvaa /app
USER dvaa

EXPOSE 8000

# The safety gate still applies inside the container: the operator must set
# DVAA_I_UNDERSTAND=1. Binding 0.0.0.0 inside the container is expected (it is
# published to the host), so ALLOW_NETWORK_BIND is set here; the host-side port
# mapping in docker-compose keeps it on 127.0.0.1.
ENV DVAA_ALLOW_NETWORK_BIND=1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
