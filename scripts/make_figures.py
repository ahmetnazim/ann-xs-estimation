"""CLI entry point for the Phase 9 publication figure and table pipeline (P9-D3 (a), P9-D8 (a)).

Verifies the scoring ledger and all ``reports/results/*`` SHA-256 digests under the OpenMP /
no-torch / no-xgboost import guard (ADL-051, ADL-086), then writes deterministic outputs under
``docs/figures/`` and ``docs/tables/``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from annxs_figures import (
    FROZEN_TABLES_SHA256,
    RESULTS_SHA256,
    AnnotationRegistry,
    ForbiddenImportBlocker,
    assert_no_forbidden_imports,
    generate_all_tables,
    generate_fig1,
    generate_fig2,
    generate_fig3,
    load_verified_bundle,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_provenance_manifest(repo_root: Path) -> dict[str, object]:
    """Verify all Phase 9 inputs and build the deterministic provenance manifest dictionary.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.

    Returns
    -------
    dict[str, object]
        Deterministic provenance record covering results, sealed predictions, and frozen tables.
    """
    with ForbiddenImportBlocker.guard():
        bundle = load_verified_bundle(repo_root)
        frozen_verified: dict[str, str] = {}
        for rel_path, expected_sha in sorted(FROZEN_TABLES_SHA256.items()):
            full_path = repo_root / rel_path
            if full_path.is_file():
                actual_sha = sha256_file(full_path)
                if actual_sha != expected_sha:
                    msg = f"Frozen table SHA-256 mismatch for {rel_path}: {actual_sha}"
                    raise RuntimeError(msg)
                frozen_verified[rel_path] = actual_sha

        sealed_verified: dict[str, str] = {}
        for rel_path, expected_sha in sorted(bundle.sealed_hashes.items()):
            full_path = repo_root / rel_path
            if full_path.is_file():
                actual_sha = sha256_file(full_path)
                if actual_sha != expected_sha:
                    msg = f"Sealed prediction SHA-256 mismatch for {rel_path}: {actual_sha}"
                    raise RuntimeError(msg)
                sealed_verified[rel_path] = actual_sha

        return {
            "audit_label": bundle.event3["audit_label"],
            "frozen_tables_sha256": frozen_verified,
            "ledger_events": [rec["event"] for rec in bundle.ledger["events"]],
            "ledger_header": bundle.ledger["header"],
            "matrix_rows": len(bundle.matrix_df),
            "outcome_4_7": {
                "claim": bundle.outcome_4_7.claim,
                "evidence_tier": bundle.outcome_4_7.evidence_tier,
                "outcome": bundle.outcome_4_7.outcome,
                "row": bundle.outcome_4_7.row,
                "source_file": bundle.outcome_4_7.source_file,
                "statement": bundle.outcome_4_7.statement,
                "venue": bundle.outcome_4_7.venue,
            },
            "results_sha256": dict(sorted(RESULTS_SHA256.items())),
            "sealed_predictions_sha256": sealed_verified,
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the Phase 9 figure and table pipeline.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status code (``0`` on success).
    """
    parser = argparse.ArgumentParser(description="Phase 9 publication figure and table pipeline.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Path to repository root directory.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify provenance and exit without writing files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root: Path = args.repo_root.resolve()

    assert_no_forbidden_imports()
    manifest = build_provenance_manifest(repo_root)
    if args.verify_only:
        return 0

    figures_dir = repo_root / "docs" / "figures"
    tables_dir = repo_root / "docs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    registry = AnnotationRegistry()
    with ForbiddenImportBlocker.guard():
        bundle = load_verified_bundle(repo_root)
        generated_tables = generate_all_tables(bundle, tables_dir)
        f1_paths = generate_fig1(bundle, repo_root, figures_dir, registry)
        f2_paths = generate_fig2(bundle, repo_root, figures_dir, registry)
        f3_paths = generate_fig3(bundle, repo_root, figures_dir, registry)

    ann_path = figures_dir / "figure_annotations.json"
    registry.write_json(ann_path)

    tables_sha256: dict[str, str] = {
        str(generated_tables.markdown_path.relative_to(repo_root)): sha256_file(
            generated_tables.markdown_path
        )
    }
    for csv_path in sorted(generated_tables.csv_paths.values()):
        tables_sha256[str(csv_path.relative_to(repo_root))] = sha256_file(csv_path)
    manifest["tables_sha256"] = dict(sorted(tables_sha256.items()))

    figures_sha256: dict[str, str] = {
        str(ann_path.relative_to(repo_root)): sha256_file(ann_path),
    }
    for path_map in (f1_paths, f2_paths, f3_paths):
        for p in sorted(path_map.values()):
            figures_sha256[str(p.relative_to(repo_root))] = sha256_file(p)
    manifest["figures_sha256"] = dict(sorted(figures_sha256.items()))

    manifest_path = figures_dir / "provenance_manifest.json"
    manifest_text = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
