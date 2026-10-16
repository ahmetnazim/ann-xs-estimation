"""Isolated publication figure and table package for Phase 9 (P9-D3 (a), P9-D5 (a), P9-D6 (a)).

Enforces the OpenMP / no-torch / no-xgboost rule (ADL-051, ADL-086) and strict SHA-256
provenance against ``reports/results/scoring_ledger.json`` and ``.agents/.research_state.md`` §17.
"""

from __future__ import annotations

from annxs_figures.fig1_domain_arch import generate_fig1
from annxs_figures.fig2_benchmark import generate_fig2
from annxs_figures.fig3_consistency_h4 import generate_fig3
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
from annxs_figures.style import (
    ASCE_DOUBLE_COL_IN,
    ASCE_ONE_HALF_COL_IN,
    ASCE_RASTER_DPI,
    ASCE_SINGLE_COL_IN,
    DIVISION_PALETTE,
    MODEL_PALETTE,
    OKABE_ITO,
    SPLIT_PALETTE,
    AnnotationRecord,
    AnnotationRegistry,
    apply_asce_style,
    check_cvd_and_grayscale_distinguishability,
    label_panel,
    save_figure_bundle,
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
    "ASCE_DOUBLE_COL_IN",
    "ASCE_ONE_HALF_COL_IN",
    "ASCE_RASTER_DPI",
    "ASCE_SINGLE_COL_IN",
    "DIVISION_PALETTE",
    "EXPECTED_MERKLE_ROOT",
    "EXPECTED_PARAMS_SHA256",
    "EXPECTED_UV_LOCK_SHA256",
    "FORBIDDEN_MODULES",
    "FROZEN_TABLES_SHA256",
    "LEDGER_SEQUENCE",
    "MODEL_PALETTE",
    "OKABE_ITO",
    "RESULTS_SHA256",
    "ROUNDING_RULES_TEXT",
    "SPLIT_PALETTE",
    "AnnotationRecord",
    "AnnotationRegistry",
    "ForbiddenImportBlocker",
    "GeneratedTables",
    "Outcome47",
    "ProvenanceError",
    "VerifiedBundle",
    "apply_asce_style",
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
    "check_cvd_and_grayscale_distinguishability",
    "format_num",
    "generate_all_tables",
    "generate_fig1",
    "generate_fig2",
    "generate_fig3",
    "label_panel",
    "load_verified_bundle",
    "metric_fmt_kind",
    "save_figure_bundle",
    "sha256_file",
    "verify_ledger_dict",
]
