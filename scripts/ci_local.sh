#!/usr/bin/env bash
# scripts/ci_local.sh -- binding CI gate (P7-D3 (a); ADL-019: no git remote, so the GitHub
# workflow .github/workflows/ci.yml is dormant and this script is the gate evidence).
#
# Usage (from anywhere):  bash scripts/ci_local.sh
#
# The commands between CI-COMMANDS-BEGIN and CI-COMMANDS-END must equal, in order, the `run:`
# steps of .github/workflows/ci.yml (enforced by tests/unit/test_ci_config.py).
# - pytest selector `-m "not integration and not slow"`: the slow A3.1 study and the real-data
#   integration tests run separately (`-m slow` once at the gate; `-m integration` locally).
# - Coverage floors per package (P7-D9 (c)): physics 95, models 87, evaluation 90, data 74,
#   compared at integer precision ([tool.coverage.report] precision = 0; PI 2026-10-09).
# - `uv sync --frozen --all-groups` keeps the ops group (DVC) installed; a plain
#   `uv sync --frozen` would uninstall it.
# - Every cache and the coverage data file stay outside Google Drive (scripts/dev_env.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/dev_env.sh
source scripts/dev_env.sh
export HYPOTHESIS_PROFILE=ci

step() {
    printf '\n==> %s\n' "$*"
    local t0=$SECONDS
    "$@"
    printf '<== ok in %ss: %s\n' "$((SECONDS - t0))" "$*"
}

start=$SECONDS
# CI-COMMANDS-BEGIN
step uv sync --frozen --all-groups
step uv run ruff check .
step uv run ruff format --check .
step uv run mypy
step uv run pytest -m "not integration and not slow" --cov=src --cov-branch --cov-report=term-missing:skip-covered
step uv run coverage report --include="src/physics/*" --fail-under=95
step uv run coverage report --include="src/models/*" --fail-under=87
step uv run coverage report --include="src/evaluation/*" --fail-under=90
step uv run coverage report --include="src/data/*" --fail-under=74
# CI-COMMANDS-END
printf '\nci_local.sh: all checks passed in %ss (commit %s)\n' \
    "$((SECONDS - start))" "$(git rev-parse --short HEAD)"
