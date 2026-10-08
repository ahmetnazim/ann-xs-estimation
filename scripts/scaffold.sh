#!/usr/bin/env bash
# scripts/scaffold.sh -- create the Phase 4 package tree (PHASE_04.md section 5 Step 2).
#
# P4-D3 (a): `src` is the import root.  P4-D8 (a): placeholder modules carry a docstring that
# names their protocol/proof sections and owning phase; dvc.yaml stage modules expose only
# main() -> None raising NotImplementedError("Phase 5"). No numerical, physics or ML logic.
#
# Guarantees:
#   * portable: bash >= 3.2 (macOS), Linux, WSL; no associative arrays, no mapfile;
#   * idempotent: `mkdir -p` and create-if-absent writes only; never truncates or overwrites
#     (noclobber is set as a second safety net); a second run reports "skipped" for everything;
#   * never touches data/, .agents/, literature/, .dvc/, .git/ or the research state file,
#     and never writes README.md, CITATION.cff, dvc.yaml, dvc.lock, .dvcignore, params.yaml,
#     pyproject.toml or src/config.py (they already exist or are written by reviewed commits);
#   * no network access.
#
# Usage (from the git root):  bash scripts/scaffold.sh
set -euo pipefail
set -o noclobber

top=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "scaffold.sh: not inside a git work tree" >&2
    exit 1
}
top=$(cd "$top" && pwd -P)
here=$(pwd -P)
if [ "$here" != "$top" ]; then
    echo "scaffold.sh: run from the git root: $top" >&2
    exit 1
fi
for sentinel in dvc.yaml docs/3_experimental_protocol.md; do
    if [ ! -f "$sentinel" ]; then
        echo "scaffold.sh: $sentinel not found; refusing to run in this repository" >&2
        exit 1
    fi
done

created=0
skipped=0

guard() {
    case "$1" in
        data | data/* | .agents | .agents/* | literature | literature/* | .dvc | .dvc/* | \
            .git | .git/* | *research_state* | README.md | CITATION.cff | dvc.yaml | dvc.lock | \
            .dvcignore | params.yaml | pyproject.toml | uv.lock | src/config.py)
            echo "scaffold.sh: refusing protected path: $1" >&2
            exit 1
            ;;
    esac
}

mkd() {
    local d
    for d in "$@"; do
        guard "$d"
        if [ -d "$d" ]; then
            skipped=$((skipped + 1))
            printf 'skipped  %s/\n' "$d"
        else
            mkdir -p "$d"
            created=$((created + 1))
            printf 'created  %s/\n' "$d"
        fi
    done
}

# put <path> <line>...  : create <path> with the given lines only if it does not exist.
put() {
    local path=$1
    shift
    guard "$path"
    if [ -e "$path" ]; then
        skipped=$((skipped + 1))
        printf 'skipped  %s\n' "$path"
        return 0
    fi
    printf '%s\n' "$@" >"$path"
    created=$((created + 1))
    printf 'created  %s\n' "$path"
}

# keep <path> : create an empty .gitkeep only if absent.
keep() {
    local path=$1
    guard "$path"
    if [ -e "$path" ]; then
        skipped=$((skipped + 1))
        printf 'skipped  %s\n' "$path"
        return 0
    fi
    : >"$path"
    created=$((created + 1))
    printf 'created  %s\n' "$path"
}

# py_doc <phase> <path> <summary> <detail>... : docstring-only module.
py_doc() {
    local phase=$1 path=$2 summary=$3
    shift 3
    put "$path" "\"\"\"${summary}" "" "$@" \
        "Owning phase: ${phase}. Placeholder (P4-D8 (a)); no numerical, physics or ML logic yet." \
        '"""'
}

# py_main <phase> <path> <summary> <detail>... : module with a main() that fails loudly.
py_main() {
    local phase=$1 path=$2 summary=$3
    shift 3
    put "$path" "\"\"\"${summary}" "" "$@" \
        "Owning phase: ${phase}. Placeholder (P4-D8 (a)): main() raises until Phase ${phase}, so" \
        "a premature \`dvc repro\` or CLI call fails loudly instead of writing empty outputs." \
        '"""' \
        "" \
        "" \
        "def main() -> None:" \
        "    \"\"\"Run this entry point; not implemented until Phase ${phase}.\"\"\"" \
        "    raise NotImplementedError(\"Phase ${phase}\")" \
        "" \
        "" \
        'if __name__ == "__main__":' \
        "    main()"
}

# Short labels keep generated lines under ruff's 100-character limit; defined in src/__init__.py.
P="protocol"
M="proofs"

# ------------------------------------------------------------------ directories
mkd src src/data src/physics src/models src/evaluation src/utils \
    scripts tests tests/unit tests/physics reports \
    docs/figures docs/tables docs/review docker .github/workflows

# ------------------------------------------------------------------ src
py_doc 4 src/__init__.py \
    "ANN_XS_Estimation successor study: importable package root (P4-D3 (a))." \
    "Layout: config (Pydantic settings), data (dvc.yaml stages), physics (Phi), models (ladder)," \
    "evaluation (metrics, bootstrap, audit, hypotheses) and utils." \
    "In module docstrings, 'protocol' = docs/3_experimental_protocol.md (a98feff) and" \
    "'proofs' = docs/2_mathematical_proofs.md (0d61c36); both are frozen."

# ------------------------------------------------------------------ src/data (Phase 5)
py_doc 5 src/data/__init__.py \
    "Data ETL: one module per dvc.yaml stage, plus the raw-unit schemas." \
    "Sources: $P sections 1.5 (stages), 2 (schemas, QC) and 3.3 (fold table)." \
    "Only the split stage creates folds; all later code reads the frozen fold table."

py_main 5 src/data/acquire.py \
    "Stage acquire@<unit>: fetch one manifest unit into data/raw/<unit> and verify it." \
    "Sources: $P sections 1.1 (manifest), 1.2 (checksums, immutability), 1.5; ADL-043." \
    "Units: the 11 dvc.yaml foreach items. Reads params acquire. Writes MANIFEST.sha256;" \
    "fails on a checksum mismatch and never overwrites."

py_main 5 src/data/validate_raw.py \
    "Stage validate_raw: rules V1-V5 on every raw unit." \
    "Sources: $P section 2.3 (raw-unit schemas). Writes reports/validate_raw.json;" \
    "a unit fails only on V1, V2 or V3."

py_doc 5 src/data/schemas.py \
    "Pandera DataFrame models for the protocol schemas (P4-D6 (a); closes P3-D3)." \
    "Sources: $P section 2.3: hydroswot_raw, site_table, fold_assignment (3.3), ehydro_ind and" \
    "every raw auxiliary unit. Fields marked as unverified are asserted at acquisition; a" \
    "deviation needs an ADL amendment, never a silent remap."

py_main 5 src/data/unit_map.py \
    "Stage unit_map: sentinels to missing, then the exact unit map U to SI." \
    "Sources: $P section 2.1; $M section 1.2 (U), Propositions 1.1-1.3." \
    "Reads params domain.s_floor."

py_main 5 src/data/qc.py \
    "Stage qc: row-wise, deterministic QC rules R1-R14 with flags retained." \
    "Sources: $P sections 2.2 (bounds) and 2.4 (rules). Reads params qc." \
    "Writes reports/qc_counts.json with row and site counts per rule."

py_main 5 src/data/link.py \
    "Stage link: site_no to NHDPlusV2.1 COMID via GageLoc, snap fallback, drainage-area check." \
    "Sources: $P section 2.5; ADL-011. Reads params link." \
    "Writes reports/link_counts.json, including the GageLoc vs k-NN disagreement rate."

py_main 5 src/data/condition_extract.py \
    "Stage condition_extract: one mean-flow row per site, argmin |ln(Q_obs / Q_E)|." \
    "Sources: $P section 2.6. Reads params condition."

py_main 5 src/data/features.py \
    "Stage features: join VAA, EROM, StreamCat, USGS attributes, D50 and the aridity index." \
    "Sources: $P sections 2.3 (site_table), 2.7 (no imputation) and 3.5 (L4 banned list)." \
    "Writes reports/feature_missing.json."

py_main 5 src/data/split.py \
    "Stage split: site-COMID groups and the frozen fold-assignment table." \
    "Sources: $P sections 3.1 (groups), 3.2 (families), 3.3 (table). Reads params seeds, splits." \
    "The ONLY module allowed to create folds; its SHA-256 enters the protocol by ADL in Phase 5."

py_main 5 src/data/ind_prepare.py \
    "Stage ind_prepare: eHydro independent set, deduplicated against training reaches." \
    "Sources: $P sections 1.1 (ehydro, O-5 (c)), 3.2 (S-IND) and 3.5 (L8)." \
    "Reads params splits.ind_dedupe_m."

# ------------------------------------------------------------------ src/physics (Phase 6)
py_doc 6 src/physics/__init__.py \
    "Physics layer Phi: hard Manning + continuity map from (W, r, n, Q_E, S) to geometry." \
    "Sources: $M sections 2-3. Rules: Phi acts row-wise (Proposition 3.4); SI units only" \
    "(Propositions 1.2, 3.5); CPU float64 (protocol 4.2, O-8)."

py_doc 6 src/physics/manning_layer.py \
    "Phi depth solve: Picard iteration from the wide-channel depth, implicit-function gradients." \
    "Sources: $M section 3 (Propositions 3.1-3.6). Tolerances from params solver" \
    "(protocol 4.2, P3-12): tau_l, K_max, per-row stopping. No unrolled autograd."

py_doc 6 src/physics/shape.py \
    "Power-law section geometry: kappa = 1/F(beta, r), wetted perimeter and shape target t." \
    "Sources: $M section 2 (Propositions 2.3-2.8) and section 7.1 (t = ln(d_max / d_bar))."

py_doc 6 src/physics/quadrature.py \
    "Tanh-sinh quadrature of F - 1 with ln F formed by log1p." \
    "Sources: $M Proposition 2.5 and the section 3.5 numerical note; protocol 4.2 (tau_Q)." \
    "Verified against the section 10.2 examples E1-E5 in Phase 7."

py_doc 6 src/physics/roughness_priors.py \
    "Empirical roughness: Strickler n_S = c_S D50^(1/6) and Jarrett for S >= S_J." \
    "Sources: $M sections 7.5 and 7.9, Proposition 7.5 (b_J < 5/3);" \
    "protocol 4.1 (P3-07, P3-08). Values from params priors."

# ------------------------------------------------------------------ src/models (Phase 6)
py_doc 6 src/models/__init__.py \
    "Model ladder: M-null, A_J, M0-M6 and the M5 ablations." \
    "ARCHITECTURE RULE: no batch-coupled layer (e.g. BatchNorm in training mode) upstream of Phi;" \
    "LayerNorm only ($M Proposition 3.4 scope; P3-17). Models read the frozen fold table and" \
    "never create folds ($P 3.3); every learned transform is fitted on training folds (3.4)."

py_doc 6 src/models/baselines.py \
    "Baselines M-null, A_J = Q_E / V_E, M0 (per-division power law) and M1 (Zarrabi Tier-2 XGB)." \
    "Sources: $P section 4.1 (P3-D8, O-2 (b), O-11)."

py_doc 6 src/models/backbone.py \
    "Shared MLP backbone: 3 x 128, SiLU, LayerNorm (row-wise)." \
    "Sources: $P section 4.2 (P3-17); values from params model.backbone."

py_doc 6 src/models/heads.py \
    "Output heads (w, r, nu, ell) with bounded logistic transforms and heteroscedastic outputs." \
    "Sources: $M sections 7.6 and 9.1; $P section 4.1 (routing I_blind, P3-11, O-4)."

py_doc 6 src/models/m3_physics.py \
    "M3: physics only, W regressor, training-fold median r and empirical n." \
    "Sources: $M section 7.9; $P section 4.1."

py_doc 6 src/models/m4_soft.py \
    "M4: soft Manning residual with the capped penalty schedule." \
    "Sources: $M sections 7.7 and 7.7.1; $P section 4.1 (P3-09, O-4)."

py_doc 6 src/models/m5_hard.py \
    "M5 (primary): hard hybrid through Phi with full kappa; L_phys identically zero." \
    "Sources: $M sections 7.4 and 7.10 (ablations A1-A4); $P section 4.1."

py_doc 6 src/models/m6_two_stage.py \
    "M6: two-stage model with back-calculated n (M6-E headline, M6-NN primary, M6-XGB)." \
    "Sources: $M section 7.8; $P section 4.1 (P3-10)."

py_doc 6 src/models/ensemble.py \
    "Deep ensemble (M = 10), mixture intervals and the point predictor exp(mu*)." \
    "Sources: $M sections 9.1-9.3; $P section 4.2 (P3-13, P3-14)."

py_doc 6 src/models/losses.py \
    "Data, NLL and beta-NLL losses with the sigma floor; prior and residual penalties." \
    "Sources: $M sections 7.0-7.5, 8.5 and 9.1; $P section 4.2."

# ------------------------------------------------------------------ src/evaluation (Phases 6-7)
py_doc 6 src/evaluation/__init__.py \
    "Evaluation: metrics, cluster bootstrap, dependence audit, hypothesis procedures and UQ." \
    "Sources: $P sections 3.6 and 4.3-4.7; $M section 9."

py_doc 6 src/evaluation/metrics.py \
    "Primary, secondary and descriptive metrics and the physical-consistency analogues." \
    "Sources: $P sections 4.3.1 and 4.3.2."

py_doc 6 src/evaluation/bootstrap.py \
    "Paired cluster bootstrap stratified by division; Bonferroni intervals for S2." \
    "Sources: $P sections 4.4 and 4.5 (O-7); values from params evaluation."

py_doc 6 src/evaluation/dependence_audit.py \
    "Spatial and network dependence audit (diagnostic only; it cannot change the splits)." \
    "Sources: $P section 3.6. Global and local Moran I via esda/libpysal; own NumPy variogram;" \
    "scipy.sparse.csgraph for components and nesting (P4-D7 (a)). Values from params audit."

py_doc 7 src/evaluation/hypotheses.py \
    "Pre-registered H1-H4 procedures and the outcome decision matrix." \
    "Sources: $P sections 4.4 and 4.7 (O-10); $M section 9.5 (frozen statistics)."

py_doc 7 src/evaluation/uq.py \
    "UQ diagnostics: coverage, mean interval width, CRPS and PIT." \
    "Sources: $P section 4.6; $M section 9.4."

# ------------------------------------------------------------------ src/utils (Phases 5-6)
py_doc 5 src/utils/__init__.py \
    "Shared utilities: seeding, I/O conventions and run logging." \
    "No scientific logic lives here."

py_doc 5 src/utils/seeding.py \
    "Deterministic seeding of Python, NumPy and torch from params seeds." \
    "Sources: $P sections 1.5 and 3.2."

py_doc 5 src/utils/io.py \
    "Parquet and GeoPackage I/O carrying input dvc.lock hashes in file metadata." \
    "Sources: $P section 1.2 (EPSG:5070 for distances, EPSG:4269 for storage)."

py_doc 5 src/utils/logging.py \
    "Run logging; every run records Settings.params_sha256 (ADL-052)." \
    "Imported as src.utils.logging, so it does not shadow the standard library module."

# ------------------------------------------------------------------ scripts (Phase 6+)
py_main 6 scripts/train.py \
    "Thin CLI: train one model on one family/fold from the frozen fold table." \
    "Sources: $P sections 3.3, 3.4 and 4.1-4.2."

py_main 6 scripts/evaluate.py \
    "Thin CLI: score stored predictions and run the pre-registered procedures." \
    "Sources: $P sections 4.3-4.7."

py_main 6 scripts/audit.py \
    "Thin CLI: run the spatial and network dependence audit." \
    "Sources: $P section 3.6."

# ------------------------------------------------------------------ keep files
keep tests/physics/.gitkeep
keep reports/.gitkeep
keep docs/figures/.gitkeep
keep docs/tables/.gitkeep
keep docs/review/.gitkeep
keep docker/.gitkeep
keep .github/workflows/.gitkeep

printf 'scaffold.sh: %d created, %d skipped\n' "$created" "$skipped"
