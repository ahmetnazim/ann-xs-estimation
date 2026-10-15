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

assert_no_forbidden_imports()

__all__ = [
    "EXPECTED_MERKLE_ROOT",
    "EXPECTED_PARAMS_SHA256",
    "EXPECTED_UV_LOCK_SHA256",
    "FORBIDDEN_MODULES",
    "FROZEN_TABLES_SHA256",
    "LEDGER_SEQUENCE",
    "RESULTS_SHA256",
    "ForbiddenImportBlocker",
    "Outcome47",
    "ProvenanceError",
    "VerifiedBundle",
    "assert_no_forbidden_imports",
    "load_verified_bundle",
    "sha256_file",
    "verify_ledger_dict",
]
