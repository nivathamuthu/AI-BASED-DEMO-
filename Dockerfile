# DocIntel Lab application image. Django, Celery and pytest all run from this image.
# The repo is mounted over /app in docker-compose, so code edits do not need a rebuild.
# Rebuild only when requirements change: docker compose build
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ requirements/
# dev.txt includes base.txt, so pytest is available inside the container.
# The default image has no torch. Local embeddings and the cross-encoder are
# opt-in: docker compose build --build-arg INSTALL_ML=1
RUN pip install --no-cache-dir --retries 12 --timeout 60 -r requirements/dev.txt \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
# Separate layer, so a slow or failed torch download does not invalidate the base install.
# The pip cache mount keeps finished wheels across failed builds; pip>=25.1 resumes broken downloads.
ARG INSTALL_ML=0
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_ML" = "1" ]; then \
      pip install "pip>=25.1" \
      && pip install --retries 12 --timeout 120 --resume-retries 10 -r requirements/ml.txt; \
    fi

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
