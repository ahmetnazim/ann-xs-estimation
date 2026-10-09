"""CI configuration contract (P7-D3 (a), P7-D9 (c); ADL-019). Static checks only, no CI run.

Sources: PHASE_07 P7-D3 (a): ``scripts/ci_local.sh`` is the binding gate and the dormant
``.github/workflows/ci.yml`` carries the identical command list, validated by parsing; the pytest
selector is ``-m "not integration and not slow"``. P7-D9 (c): per-package coverage floors
physics 95, models 87, evaluation 90, data 74, compared at integer precision (PI 2026-10-09).
Expected values are those decision texts, not the files under test.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_LOCAL = REPO_ROOT / "scripts" / "ci_local.sh"
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
SELECTOR = '-m "not integration and not slow"'
FLOORS = {"physics": 95, "models": 87, "evaluation": 90, "data": 74}  # P7-D9 (c)


def local_commands() -> list[str]:
    text = CI_LOCAL.read_text(encoding="utf-8")
    block = text.split("# CI-COMMANDS-BEGIN\n", 1)[1].split("# CI-COMMANDS-END", 1)[0]
    cmds = [ln.strip() for ln in block.splitlines() if ln.strip()]
    assert all(c.startswith("step ") for c in cmds), f"P7-D3: non-step line in block: {cmds}"
    return [c.removeprefix("step ") for c in cmds]


def workflow() -> dict:
    data = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "P7-D3: ci.yml must parse to a mapping"
    return data


def workflow_commands() -> list[str]:
    steps = workflow()["jobs"]["verify"]["steps"]
    return [s["run"] for s in steps if "run" in s]


def test_command_lists_are_identical() -> None:
    """P7-D3 (a): ci_local.sh and ci.yml run the same commands in the same order."""
    local, remote = local_commands(), workflow_commands()
    assert local == remote, f"P7-D3: command lists differ\nlocal:  {local}\nci.yml: {remote}"


def test_workflow_parses_and_is_minimal() -> None:
    """P7-D3 (a): ci.yml parses; ubuntu, Python 3.12, read-only token, push and PR triggers."""
    data = workflow()
    triggers = data.get("on", data.get(True))  # YAML 1.1 reads the key `on` as True
    assert set(triggers) == {"push", "pull_request"}, f"P7-D3: triggers {triggers}"
    assert data["permissions"] == {"contents": "read"}
    job = data["jobs"]["verify"]
    assert job["runs-on"] == "ubuntu-latest"
    uv = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("astral-sh/setup-uv"))
    assert uv["with"]["python-version"] == "3.12"
    assert job["env"]["HYPOTHESIS_PROFILE"] == "ci"


def test_pytest_selector_and_sync() -> None:
    """P7-D3 (a): selector excludes integration and slow; uv sync is frozen with all groups."""
    cmds = local_commands()
    pytest_cmds = [c for c in cmds if c.startswith("uv run pytest")]
    assert len(pytest_cmds) == 1, f"P7-D3: expected one pytest command, got {pytest_cmds}"
    assert SELECTOR in pytest_cmds[0]
    assert "--cov=src" in pytest_cmds[0]
    assert "--cov-branch" in pytest_cmds[0]
    assert cmds[0] == "uv sync --frozen --all-groups"
    for tool in ("uv run ruff check .", "uv run ruff format --check .", "uv run mypy"):
        assert tool in cmds, f"P7-D3: missing {tool!r}"


def test_per_package_floors_match_p7_d9() -> None:
    """P7-D9 (c): one coverage report per package with the decided floor, at precision 0."""
    found = {}
    for c in local_commands():
        m = re.fullmatch(r'uv run coverage report --include="src/(\w+)/\*" --fail-under=(\d+)', c)
        if m:
            found[m.group(1)] = int(m.group(2))
    assert found == FLOORS, f"P7-D9 (c): floors {found}, want {FLOORS}"
    cov = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]
    assert cov["coverage"]["report"]["precision"] == 0
    assert cov["coverage"]["run"]["branch"] is True


def test_ci_local_is_executable_and_strict() -> None:
    """P7-D3 (a): the binding script is executable, fails fast and uses the Drive-safe env."""
    assert os.access(CI_LOCAL, os.X_OK), "P7-D3: scripts/ci_local.sh must be mode 755"
    text = CI_LOCAL.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert "source scripts/dev_env.sh" in text
    assert "export HYPOTHESIS_PROFILE=ci" in text
