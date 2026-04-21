# syntax=docker/dockerfile:1.7
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=20

WORKDIR /app

# Dependencias del sistema requeridas por opencv-python/mediapipe
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && grep -viE '^\s*tensorflow([<>=!~].*)?\s*$' /tmp/requirements.txt > /tmp/requirements.base.txt \
    && if [ -s /tmp/requirements.base.txt ]; then \
        python -m pip install --timeout 120 --retries 20 --resume-retries 20 -r /tmp/requirements.base.txt; \
    fi \
    && TF_REQ="$(grep -iE '^\s*tensorflow([<>=!~].*)?\s*$' /tmp/requirements.txt || true)" \
    && if [ -n "$TF_REQ" ]; then \
        python -m pip install --timeout 180 --retries 30 --resume-retries 30 "$TF_REQ"; \
    fi \
    && python -m pip uninstall -y jax jaxlib || true

COPY . /app

CMD ["python", "--version"]
