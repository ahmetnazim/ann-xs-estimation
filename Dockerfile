# Reproducibility capsule (docs/3 §13.10; P8-D11 (a), P8-D12 (a), P8-D13 (a)). CPU only, linux/arm64.
#
#   build    python base + the pinned uv binary; `uv sync --frozen --no-dev --group ops`
#   runtime  the capsule: python:3.12-slim-bookworm (digest-pinned) + libgomp1 (XGBoost),
#            non-root uid 1000, the locked venv and the source tree; no data, no caches
#   check    runtime + uv, git and the dev group, for `scripts/container_check.sh`
#            (ci_local.sh inside the container); not the capsule itself
#
# The production run executes natively on the M4 Max (binding environment: uv.lock plus
# scripts/dev_env.sh). The venv lives at the path dev_env.sh exports for HOME=/home/app, so
# ci_local.sh runs unchanged inside the check image. Bit identity across macOS and Linux is not
# claimed (BLAS, libm); the uv.lock SHA-256 of the image equals the run's (container_check.sh).
# Base digests: multi-arch index digests resolved 2026-10-13 with `docker buildx imagetools`.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------------------------------- build
# The uv image is distroless (uv binary only); the venv is built on the runtime's own Python so
# its interpreter path (/usr/local/bin/python3.12) exists unchanged in the runtime stage.
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/app/.venvs/ANN_XS_Estimation
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group ops --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group ops

# -------------------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/app app
ENV HOME=/home/app \
    VIRTUAL_ENV=/home/app/.venvs/ANN_XS_Estimation \
    PATH=/home/app/.venvs/ANN_XS_Estimation/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY --from=build --chown=1000:1000 /home/app/.venvs /home/app/.venvs
COPY --from=build --chown=1000:1000 /app /app
WORKDIR /app
USER 1000:1000
LABEL org.opencontainers.image.title="ann-xs-estimation capsule" \
      org.opencontainers.image.description="Phase 8 reproducibility capsule (docs/3 13.10); CPU, linux/arm64"
CMD ["python", "-c", "import src, torch, xgboost; print('capsule ok', torch.__version__, xgboost.__version__)"]

# ---------------------------------------------------------------------------------------- check
FROM runtime AS check
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
USER 1000:1000
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/app/.venvs/ANN_XS_Estimation
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --all-groups
# uv keys the editable project's freshness on the src/ inode, which differs in every container,
# so the offline `uv sync` of ci_local.sh rebuilds the project. Rebuilding it once here without
# the cache mount leaves the build backend (hatchling) in this image's own uv cache, so that
# offline rebuild succeeds. Only the check image carries this cache; the capsule carries none.
RUN uv sync --frozen --all-groups --reinstall-package ann-xs-estimation
