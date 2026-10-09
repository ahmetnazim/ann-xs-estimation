"""Run logging; every run records Settings.params_sha256 (ADL-052).

Imported as src.utils.logging, so it does not shadow the standard library module.

Sources: ADL-052 (``params_sha256`` of the exact ``params.yaml`` bytes in every run record),
P6-D6 (a) (``run.json``: params and table hashes, git commit, seeds, wall time). The header is a
plain JSON-serializable dict; package versions are read with ``importlib.metadata`` so that no
heavy package (torch, xgboost) is imported to report its version (ADL-051).

Owning phase: 5 (implemented in Phase 6, Batch D).
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Final

__all__ = ["get_logger", "git_state", "run_header"]

_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_PACKAGES: Final[tuple[str, ...]] = ("numpy", "pandas", "torch", "xgboost", "pydantic")


def get_logger(name: str = "annxs", level: str = "INFO") -> logging.Logger:
    """Return a logger writing to stderr with one handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def git_state(repo_root: Path) -> dict[str, Any]:
    """``{"commit": HEAD sha or "unknown", "dirty": bool}`` of ``repo_root`` (never raises)."""
    try:
        head = subprocess.run(  # fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}
    return {"commit": head, "dirty": bool(status.strip())}


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "absent"


def run_header(
    params_sha256: str,
    repo_root: Path,
    **ids: object,
) -> dict[str, Any]:
    """Provenance header of one run: params hash, git state, platform, versions and ``ids``."""
    if len(params_sha256) != 64:
        raise ValueError("params_sha256 must be a SHA-256 hex digest (ADL-052)")
    return {
        "params_sha256": params_sha256,
        "git": git_state(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in _PACKAGES},
        **ids,
    }
