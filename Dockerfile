# syntax=docker/dockerfile:1
# =============================================================================
# Primus — production container image (multi-stage).
#
# Runs the full Gradio UI + dual-model agent stack against an Ollama service.
# Identical app to `uv run python admin_assistant.py`; only the launch surface differs.
#
# State lives under /data (PRIMUS_DATA_DIR) and /config (PRIMUS_CONFIG_DIR) — mount volumes
# there (host bind mounts for local dev, named volumes for cloud) to persist the knowledge
# base, memories, chats, agent outputs, config.json, and logs across restarts.
# =============================================================================

# ---- Stage 1: build wheels into an isolated venv (keeps build tools out of runtime) ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ---- Stage 2: lean runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # App data/config (overridable; compose sets the same values explicitly).
    PRIMUS_DATA_DIR=/data \
    PRIMUS_CONFIG_DIR=/config \
    OLLAMA_URL=http://ollama:11434 \
    # Keep model/HF/general caches on the writable data volume so they persist and work
    # regardless of which UID the container runs as (host-UID match in local mode).
    XDG_CACHE_HOME=/data/.cache \
    HF_HOME=/data/.cache/huggingface

# Runtime libs only: ffmpeg + libgomp1 for faster-whisper/ctranslate2; curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application: the entrypoint host module + the primus/ package (world-readable, so the
# container still works when run as an arbitrary host UID in local bind-mount mode).
COPY admin_assistant.py ./
COPY primus ./primus

# Non-root user. /data and /config are owned by it, so fresh named volumes (cloud) inherit
# writable ownership on first mount. For host bind mounts (local), compose overrides `user:`
# to your host UID/GID so the shared ~/.primus stays writable.
RUN useradd --create-home --uid 1000 primus \
    && mkdir -p /data /config "$XDG_CACHE_HOME" "$HF_HOME" \
    && chown -R primus:primus /app /data /config

USER primus

VOLUME ["/data", "/config"]
EXPOSE 7860

# Liveness: the Gradio app answers on /gradio_api/startup-events once the UI is fully built.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:7860/gradio_api/startup-events || exit 1

# Headless server: bind all interfaces inside the container (host port mapping controls real
# exposure), no system tray. Data/config dirs come from the env vars above.
CMD ["python", "admin_assistant.py", "--host", "0.0.0.0", "--port", "7860", "--no-tray"]
