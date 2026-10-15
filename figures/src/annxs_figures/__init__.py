"""Isolated publication figure and table package for Phase 9 (P9-D3 (a), P9-D5 (a), P9-D6 (a)).

Enforces the OpenMP / no-torch / no-xgboost rule (ADL-051, ADL-086) and strict SHA-256
provenance against ``reports/results/scoring_ledger.json`` and ``.agents/.research_state.md`` §17.
"""

from __future__ import annotations

from annxs_figures.loader import (
    EXPECTED_MERKLE_ROOT,
    EXPECTED_PARAMS_SHA256,
    EXPECTED_UV_LOCK_SHA256,
    FORBIDDEN_MODULES,
    FROZEN_TABLES_SHA256,
    LEDGER_SEQUENCE,
    RESULTS_SHA256,
    ForbiddenImportBlocker,
    Outcome47,
    ProvenanceError,
    VerifiedBundle,
    assert_no_forbidden_imports,
    load_verified_bundle,
    sha256_file,
    verify_ledger_dict,
)
from annxs_figures.tables import (
    ROUNDING_RULES_TEXT,
    GeneratedTables,
    build_t1_hypotheses_and_outcome,
    build_t2_metrics_matrix,
    build_t3_s2_contrasts,
    build_t4_uq_calibration,
    build_t5_physical_consistency,
    build_t6_ablations_and_sensitivity,
    build_t7_srnd_benchmark,
    build_t8_s1_hyg_consistency,
    build_t9_frame_and_data_summary,
    format_num,
    generate_all_tables,
    metric_fmt_kind,
)

assert_no_forbidden_imports()

__all__ = [
    "EXPECTED_MERKLE_ROOT",
    "EXPECTED_PARAMS_SHA256",
    "EXPECTED_UV_LOCK_SHA256",
    "FORBIDDEN_MODULES",
    "FROZEN_TABLES_SHA256",
    "LEDGER_SEQUENCE",
    "RESULTS_SHA256",
    "ROUNDING_RULES_TEXT",
    "ForbiddenImportBlocker",
    "GeneratedTables",
    "Outcome47",
    "ProvenanceError",
    "VerifiedBundle",
    "assert_no_forbidden_imports",
    "build_t1_hypotheses_and_outcome",
    "build_t2_metrics_matrix",
    "build_t3_s2_contrasts",
    "build_t4_uq_calibration",
    "build_t5_physical_consistency",
    "build_t6_ablations_and_sensitivity",
    "build_t7_srnd_benchmark",
    "build_t8_s1_hyg_consistency",
    "build_t9_frame_and_data_summary",
    "format_num",
    "generate_all_tables",
    "load_verified_bundle",
    "metric_fmt_kind",
    "sha256_file",
    "verify_ledger_dict",
]
