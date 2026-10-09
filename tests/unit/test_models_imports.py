"""Import-isolation tests for src/models (P6-D7 (a), ADL-051); each runs a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_MODULES = (
    "backbone",
    "heads",
    "losses",
    "m2_mlp",
    "m3_physics",
    "m4_soft",
    "m5_hard",
    "m6_two_stage",
    "ensemble",
)


def run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    return result.stdout.strip()


def test_baselines_imports_neither_torch_nor_xgboost() -> None:
    code = (
        "import sys, src.models.baselines; print('torch' in sys.modules, 'xgboost' in sys.modules)"
    )
    assert run(code) == "False False"


def test_torch_modules_do_not_import_xgboost() -> None:
    mods = ", ".join(f"src.models.{m}" for m in TORCH_MODULES)
    assert run(f"import sys, {mods}; print('xgboost' in sys.modules)") == "False"


def test_models_package_imports_nothing() -> None:
    code = (
        "import sys, src.models; "
        "print(sorted(m for m in sys.modules if m.startswith('src.models.')), "
        "'torch' in sys.modules)"
    )
    assert run(code) == "[] False"


def test_xgb_worker_cli_is_torch_free() -> None:
    code = (
        "import sys; from src.models.baselines import main; "
        "import src.models.baselines as b; print('torch' in sys.modules)"
    )
    assert run(code) == "False"
    result = subprocess.run(
        [sys.executable, "-m", "src.models.baselines", "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert "--xgb-fit" in result.stdout
