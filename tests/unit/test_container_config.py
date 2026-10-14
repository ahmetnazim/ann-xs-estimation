"""Capsule configuration contract (docs/3 §13.10; P8-D11 (a), P8-D13 (a)). Static checks only.

Sources: docs/3 section 13.10 (multi-stage CPU image for linux/arm64; build stage on the uv
0.12.1 image with ``uv sync --frozen --no-dev --group ops``; runtime ``python:3.12-slim-bookworm``
pinned by digest with ``libgomp1``, non-root uid 1000, no data or caches inside; compose mounts
``data/raw`` and ``data/processed`` read-only and the run root and ``reports/`` read-write;
``scripts/container_check.sh`` = build, ``ci_local.sh`` inside, ``verify_models.py --quick``,
one pilot fit; image ID, base digests and the uv.lock SHA-256 recorded). PHASE_08 P8-D11 (a)
(``.dockerignore`` entries; ``PYTHONDONTWRITEBYTECODE=1``) and P8-D13 (a) (dormant docker job,
setup-uv cache keyed on ``uv.lock``, every action pinned by commit SHA). No image is built here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
COMPOSE = REPO_ROOT / "docker-compose.yml"
CHECK = REPO_ROOT / "scripts" / "container_check.sh"
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DIGEST = r"@sha256:[0-9a-f]{64}"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _stages(text: str) -> dict[str, str]:
    """Stage name -> body (lines until the next FROM)."""
    parts = re.split(r"^FROM ", text, flags=re.M)[1:]
    out = {}
    for part in parts:
        head, _, body = part.partition("\n")
        name = head.split(" AS ")[-1].strip() if " AS " in head else head.strip()
        out[name] = head + "\n" + body
    return out


def test_bases_are_pinned_by_digest() -> None:
    """13.10: python:3.12-slim-bookworm and ghcr.io/astral-sh/uv:0.12.1, both by digest."""
    text = _dockerfile()
    assert re.search(rf"^ARG PYTHON_IMAGE=python:3\.12-slim-bookworm{DIGEST}$", text, re.M)
    assert re.search(rf"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:0\.12\.1{DIGEST}$", text, re.M)
    for line in re.findall(r"^FROM (\S+)", text, re.M):
        assert line.startswith("${") or line in _stages(text), f"unpinned base {line}"


def test_stages_and_sync_commands() -> None:
    """13.10: build syncs frozen without dev (+ops); runtime is non-root with libgomp1."""
    stages = _stages(_dockerfile())
    assert {"uv", "build", "runtime", "check"} <= set(stages)
    build = stages["build"]
    assert "uv sync --frozen --no-dev --group ops" in build
    assert "UV_PYTHON_DOWNLOADS=never" in build
    runtime = stages["runtime"]
    assert "libgomp1" in runtime
    assert "--uid 1000" in runtime
    assert re.search(r"^USER 1000:1000$", runtime, re.M)
    assert "PYTHONDONTWRITEBYTECODE=1" in runtime
    assert "uv sync" not in runtime  # the runtime only copies the locked venv
    assert "/home/app/.venvs/ANN_XS_Estimation" in runtime  # = dev_env.sh with HOME=/home/app
    check = stages["check"]
    assert check.startswith("runtime AS check")
    assert "uv sync --frozen --all-groups" in check
    assert re.search(r"^USER 1000:1000$", check, re.M)


def test_no_data_enters_the_image() -> None:
    """P8-D11 (a): .dockerignore excludes data, the DVC cache and the non-code trees."""
    entries = {ln.strip() for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()}
    for want in ("data/", ".dvc/cache", ".agents/", "literature/", "Originals/", "docs/review/"):
        assert want in entries, f".dockerignore lacks {want}"
    for cache in (".git/", ".venv/", ".dvc/config.local"):
        assert cache in entries
    assert not re.search(r"^(COPY|ADD)\s+.*\bdata/", _dockerfile(), re.M)


def test_compose_mounts() -> None:
    """13.10: data (and the DVC cache its symlinks point into) ro; run root and reports rw."""
    svc = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["capsule"]
    assert svc["build"]["target"] == "runtime"
    assert svc["platform"] == "linux/arm64"
    assert svc["user"] == "1000:1000"
    mounts = {v.split(":")[1]: v for v in svc["volumes"] if v.startswith("./")}
    assert mounts["/app/data/raw"].endswith(":ro")
    assert mounts["/app/data/processed"].endswith(":ro")
    assert not mounts["/app/reports"].endswith(":ro")
    cache = [v for v in svc["volumes"] if "dvc_store/ANN_XS_Estimation/cache" in v]
    assert len(cache) == 1
    assert cache[0].endswith(":ro")
    runs = [v for v in svc["volumes"] if v.endswith(":/runs")]
    assert len(runs) == 1
    assert svc["environment"]["ANNXS_RUN_ROOT"] == "/runs"


def test_container_check_script() -> None:
    """P8-D13 (a): executable, strict, and runs the four checks of 13.10 plus the record."""
    assert os.access(CHECK, os.X_OK), "scripts/container_check.sh must be mode 755"
    text = CHECK.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert "source scripts/dev_env.sh" in text
    for want in (
        "--target runtime",
        "--target check",
        '--platform "$PLATFORM"',
        "bash scripts/ci_local.sh",
        "scripts/verify_models.py --quick",
        "scripts/train.py fit",
        "uv.lock",
        "reports/container_check.json",
        "--network none",
    ):
        assert want in text, f"container_check.sh lacks {want!r}"
    assert not re.search(r"docker\s+(image\s+)?push|--push\b", text)  # never uploads an image


def test_workflow_docker_job_and_pins() -> None:
    """P8-D13 (a): dormant docker job (no push), setup-uv cache on uv.lock, SHA-pinned actions."""
    data = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))
    uses = [s["uses"] for job in data["jobs"].values() for s in job["steps"] if "uses" in s]
    assert uses, "no actions found"
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), f"not SHA-pinned: {ref}"
    verify = data["jobs"]["verify"]["steps"]
    uv = next(s for s in verify if s.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert uv["with"]["enable-cache"] is True
    assert uv["with"]["cache-dependency-glob"] == "uv.lock"
    docker = data["jobs"]["docker"]
    build = next(s for s in docker["steps"] if s.get("uses", "").startswith("docker/build-push"))
    assert build["with"]["push"] is False
    assert build["with"]["platforms"] == "linux/arm64"
    assert build["with"]["target"] == "check"
