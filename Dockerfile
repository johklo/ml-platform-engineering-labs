# Runtime image for the Kubeflow movie pipeline components (`ml-labs tools kfp compile --image`).
# Dependencies come from the committed uv.lock (core + `pipeline` extra); the repo code is installed
# as the `ml_labs` package so every component imports the same code that CI tests.
FROM python:3.12.12-slim-bookworm

ARG SOURCE_REVISION
RUN test -n "$SOURCE_REVISION" || (echo 'Pass --build-arg SOURCE_REVISION=<git commit sha>' >&2; exit 1)

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ML_LABS_CODE_REVISION=${SOURCE_REVISION}
LABEL org.opencontainers.image.source="https://github.com/johklo/ml-platform-engineering-labs" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

WORKDIR /opt/ml-labs
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes --extra pipeline -o requirements.txt \
    && uv pip install --system --no-cache -r requirements.txt
COPY src ./src
RUN uv pip install --system --no-cache --no-deps . \
    && python -c "import kfp, ml_labs.movies, ml_labs.data; print('ml_labs image ready')"

WORKDIR /workspace