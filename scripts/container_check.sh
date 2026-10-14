#!/usr/bin/env bash
# scripts/container_check.sh -- reproducibility-capsule verification (docs/3 §13.10; P8-D13 (a)).
#
# Usage (from anywhere, Docker running):  bash scripts/container_check.sh
#
# 1. Build the `runtime` (capsule) and `check` targets for linux/arm64 (local images only;
#    nothing is uploaded to a registry).
# 2. The image's uv.lock SHA-256 must equal the repository's (the production run's lockfile).
# 3. `scripts/ci_local.sh` inside the check image, offline (--network none, UV_OFFLINE=1); the
#    repository .git is mounted read-only for the commit line and the git-based unit tests.
# 4. `scripts/verify_models.py --quick` inside the capsule (synthetic data; report kept in the
#    container).
# 5. One pilot fit inside the capsule through docker-compose.yml (real frozen tables mounted
#    read-only; artefacts in a scratch run root outside the repo and outside Google Drive).
# 6. Record image IDs, base digests, versions and step timings in reports/container_check.json.
#
# ci_local.sh stays the binding native gate; this script is the capsule's gate evidence.
# No metric is printed; the pilot fit's run.json stays in the scratch run root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/dev_env.sh
source scripts/dev_env.sh

IMAGE="ann-xs-estimation"
PLATFORM="linux/arm64"
PILOT_ROOT="${HOME}/dvc_store/ANN_XS_Estimation/capsule_check"
RECORD="reports/container_check.json"
PY="${UV_PROJECT_ENVIRONMENT}/bin/python"
declare -a STEPS=()

step() {
    local name=$1
    shift
    printf '\n==> %s: %s\n' "$name" "$*"
    local t0=$SECONDS
    "$@"
    local dt=$((SECONDS - t0))
    printf '<== ok in %ss: %s\n' "$dt" "$name"
    STEPS+=("${name}=${dt}")
}

start=$SECONDS
step build_runtime docker buildx build --platform "$PLATFORM" --target runtime \
    -t "${IMAGE}:capsule" --load .
step build_check docker buildx build --platform "$PLATFORM" --target check \
    -t "${IMAGE}:check" --load .

host_lock=$(shasum -a 256 uv.lock | cut -d' ' -f1)
image_lock=$(docker run --rm --network none "${IMAGE}:capsule" sha256sum /app/uv.lock | cut -d' ' -f1)
if [[ "$host_lock" != "$image_lock" ]]; then
    echo "container_check: uv.lock SHA-256 differs (host ${host_lock}, image ${image_lock})" >&2
    exit 1
fi
printf '\nuv.lock sha256 %s (host == image)\n' "$host_lock"

step ci_local docker run --rm --network none -e UV_OFFLINE=1 \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' \
    -v "$PWD/.git:/app/.git:ro" "${IMAGE}:check" bash scripts/ci_local.sh
step verify_models docker run --rm --network none "${IMAGE}:capsule" \
    python scripts/verify_models.py --quick --out /tmp/verify_models.json

rm -rf "$PILOT_ROOT"
mkdir -p "$PILOT_ROOT"
step pilot_fit env ANNXS_RUN_ROOT_HOST="$PILOT_ROOT" docker compose run --rm --pull never capsule \
    python scripts/train.py fit --model M5 --family S-ID --fold 0 --kind inner --index 0 \
    --grid 0 --checkpoint-dir /runs/pilot --log-level WARNING
pilot_runs=$(find "$PILOT_ROOT/pilot" -name run.json | wc -l | tr -d ' ')
if [[ "$pilot_runs" != "1" ]]; then
    echo "container_check: pilot fit wrote ${pilot_runs} run.json files, expected 1" >&2
    exit 1
fi

versions=$(docker run --rm --network none "${IMAGE}:capsule" python -c \
    "import platform, torch, xgboost, numpy; print(platform.python_version(), torch.__version__, xgboost.__version__, numpy.__version__)")
capsule_id=$(docker image inspect --format '{{.Id}}' "${IMAGE}:capsule")
check_id=$(docker image inspect --format '{{.Id}}' "${IMAGE}:check")
python_base=$(sed -n 's/^ARG PYTHON_IMAGE=//p' Dockerfile)
uv_base=$(sed -n 's/^ARG UV_IMAGE=//p' Dockerfile)

"$PY" - "$RECORD" <<EOF
import json, subprocess, sys
steps = dict(s.split("=") for s in "${STEPS[*]}".split())
py, torch, xgb, np = "${versions}".split()
record = {
    "schema_version": 1,
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "platform": "${PLATFORM}",
    "docker": subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True).stdout.strip(),
    "images": {"capsule": "${capsule_id}", "check": "${check_id}"},
    "base_images": {"python": "${python_base}", "uv": "${uv_base}"},
    "uv_lock_sha256": "${host_lock}",
    "uv_lock_sha256_image": "${image_lock}",
    "versions": {"python": py, "torch": torch, "xgboost": xgb, "numpy": np},
    "steps_s": {k: int(v) for k, v in steps.items()},
    "pilot_fit": {"model": "M5", "family": "S-ID", "fold": 0, "kind": "inner", "grid": 0, "run_json": int("${pilot_runs}")},
    "result": "pass",
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=1, sort_keys=True)
    fh.write("\n")
EOF
printf '\ncontainer_check.sh: all checks passed in %ss (capsule %s)\n' \
    "$((SECONDS - start))" "${capsule_id:7:12}"
