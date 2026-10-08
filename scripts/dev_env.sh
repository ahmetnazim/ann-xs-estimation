# shellcheck shell=bash
# scripts/dev_env.sh -- SOURCE this file before any uv/ruff/mypy/pytest command:
#
#     source scripts/dev_env.sh
#
# Keeps the virtualenv and every tool cache OUTSIDE the Google Drive sync folder
# (P4-D2 (a); ADL-051). Exports are scoped to the current shell session only; nothing
# is written to ~/.zshrc. Works in bash >= 3.2 and zsh.
if (return 0 2>/dev/null); then :; else
    echo "dev_env.sh: source this file instead of executing it: source scripts/dev_env.sh" >&2
    exit 1
fi

_annxs="ANN_XS_Estimation"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/${_annxs}"
export PYTHONPYCACHEPREFIX="${HOME}/.cache/pycache/${_annxs}"
export RUFF_CACHE_DIR="${HOME}/.cache/ruff/${_annxs}"
export MYPY_CACHE_DIR="${HOME}/.cache/mypy/${_annxs}"
export COVERAGE_FILE="${HOME}/.cache/${_annxs}.coverage"
unset _annxs

echo "dev_env.sh: UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
