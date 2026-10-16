"""Provenance, vector/DPI compliance, and determinism tests for Batch D figures (Fig. 4-5, S1-S5).

Verifies that:
1. ``fig4_ood_and_uq_calibration``, ``fig5_row4_diagnosis``, and supplementary figures
   ``figS1_event1_spatial_dependence`` through ``figS5_sind_ehydro_rivers`` are generated as
   vector PDF, vector SVG, and 600 DPI PNG.
2. Every panel label ``(a)``, ``(b)``, ..., evidence tier (`Confirmatory`, `Secondary`,
   `Descriptive`), `[0.85, 0.95]` UQ acceptance band, `98.75%` S2 Bonferroni CI level,
   Event 1 S-ID audit label, S-IND UQ note (`131 locations / 18 rivers (UQ: 33 single-COMID rows)`),
   pre-registered `not computed` explanations, and verbatim §4.7 row 4 text appear in the SVGs.
3. Every numeric callout registered in :class:`AnnotationRegistry` across Fig. 4, Fig. 5, and
   Fig. S1-S5 equals its exact source value in ``reports/results/*`` at the displayed precision.
4. Re-running Fig. 4 and Fig. 5 produces byte-identical PDF, SVG, and PNG outputs.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from annxs_figures import (
    ASCE_RASTER_DPI,
    AnnotationRegistry,
    VerifiedBundle,
    format_num,
    generate_fig4,
    generate_fig5,
    generate_supplementary_figures,
    load_verified_bundle,
    sha256_file,
)
from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    S_IND_UQ_NOTE,
    VERBATIM_ROW4_OUTCOME,
    VERBATIM_ROW4_STATEMENT,
    VERBATIM_ROW4_VENUE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bundle() -> VerifiedBundle:
    """Load the cryptographically verified Phase 8 results bundle once per module."""
    return load_verified_bundle(REPO_ROOT)


@pytest.fixture(scope="module")
def batch_d_outputs(
    bundle: VerifiedBundle,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, dict[str, Path]], AnnotationRegistry]:
    """Generate Fig. 4, Fig. 5, and Fig. S1-S5 once in a temp directory and return paths + reg."""
    out_dir = tmp_path_factory.mktemp("figures_batch_d")
    reg = AnnotationRegistry()
    f4 = generate_fig4(bundle, REPO_ROOT, out_dir, reg)
    f5 = generate_fig5(bundle, REPO_ROOT, out_dir, reg)
    supp = generate_supplementary_figures(bundle, REPO_ROOT, out_dir, reg)
    return {"fig4": f4, "fig5": f5, **supp}, reg


def _resolve_json_path(doc: dict[str, Any], dotted_key: str) -> float | int:
    cur: Any = doc
    parts = dotted_key.split(".")
    i = 0
    while i < len(parts):
        part = parts[i]
        if "[" in part and part.endswith("]"):
            k_name, idx_str = part[:-1].split("[", 1)
            cur = cur[k_name][int(idx_str)]
            i += 1
        elif part in cur:
            cur = cur[part]
            i += 1
        elif i + 1 < len(parts) and f"{part}.{parts[i + 1]}" in cur:
            cur = cur[f"{part}.{parts[i + 1]}"]
            i += 2
        else:
            raise KeyError(f"Cannot resolve {part!r} in {dotted_key!r}")
    if isinstance(cur, (int, float)):
        return cur
    raise TypeError(f"Expected numeric value at {dotted_key}, got {type(cur)}")


def test_batch_d_files_vector_and_dpi(
    batch_d_outputs: tuple[dict[str, dict[str, Path]], AnnotationRegistry],
) -> None:
    """Verify PDF, SVG, and >= 300 DPI PNG outputs exist for Fig. 4-5 and Fig. S1-S5."""
    paths_by_fig, _ = batch_d_outputs
    expected_keys = {"fig4", "fig5", "figS1", "figS2", "figS3", "figS4", "figS5"}
    assert set(paths_by_fig) == expected_keys

    for fig_key, paths in paths_by_fig.items():
        pdf_p = paths["pdf"]
        svg_p = paths["svg"]
        png_p = paths["png"]
        assert pdf_p.is_file(), f"{fig_key} PDF missing"
        assert pdf_p.stat().st_size > 1000, f"{fig_key} PDF empty"
        assert svg_p.is_file(), f"{fig_key} SVG missing"
        assert svg_p.stat().st_size > 1000, f"{fig_key} SVG empty"
        assert png_p.is_file(), f"{fig_key} PNG missing"
        assert png_p.stat().st_size > 1000, f"{fig_key} PNG empty"

        assert pdf_p.read_bytes().startswith(b"%PDF-"), f"{fig_key} invalid PDF header"
        svg_text = svg_p.read_text(encoding="utf-8")
        assert "<text " in svg_text, f"{fig_key} SVG missing vector <text> elements"

        with Image.open(png_p) as im:
            dpi_x, dpi_y = im.info.get("dpi", (0, 0))
            assert round(dpi_x) >= ASCE_RASTER_DPI
            assert round(dpi_y) >= ASCE_RASTER_DPI
            assert im.width >= 2000


def test_batch_d_panel_labels_and_verbatim_strings(
    batch_d_outputs: tuple[dict[str, dict[str, Path]], AnnotationRegistry],
) -> None:
    """Verify panel tags, evidence labels, UQ bands, S2 98.75% CIs, and §4.7 row 4 text in SVGs."""
    paths_by_fig, _ = batch_d_outputs
    svg4 = html.unescape(paths_by_fig["fig4"]["svg"].read_text(encoding="utf-8"))
    svg5 = html.unescape(paths_by_fig["fig5"]["svg"].read_text(encoding="utf-8"))
    svgs1 = html.unescape(paths_by_fig["figS1"]["svg"].read_text(encoding="utf-8"))
    svgs2 = html.unescape(paths_by_fig["figS2"]["svg"].read_text(encoding="utf-8"))
    svgs3 = html.unescape(paths_by_fig["figS3"]["svg"].read_text(encoding="utf-8"))
    svgs4 = html.unescape(paths_by_fig["figS4"]["svg"].read_text(encoding="utf-8"))
    svgs5 = html.unescape(paths_by_fig["figS5"]["svg"].read_text(encoding="utf-8"))

    # Fig. 4 checks: panels (a)-(e), [0.85, 0.95] band, H3 & H2 CIs, S-IND UQ note, far not computed
    for token in (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "Confirmatory",
        "Secondary",
        "Descriptive",
        "[0.85, 0.95]",
        S_IND_UQ_NOTE,
        EVENT1_AUDIT_LABEL,
        "fewer than 30 sites in a repeat",
        "-0.0138",
        "-0.0494",
        "0.8997",
        "0.9025",
        "0.8182",
    ):
        assert token in svg4, f"Fig. 4 SVG missing token: {token!r}"

    # Fig. 5 checks: panels (a)-(d), 98.75% S2 CIs, §5.1 elasticities, verbatim §4.7 row 4
    for token in (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "98.75%",
        "-3/10",
        "-3/5",
        "identical to primary by construction (no Phi); not trained",
        "fewer than 30 sites in a repeat",
        VERBATIM_ROW4_OUTCOME,
        VERBATIM_ROW4_STATEMENT,
        VERBATIM_ROW4_VENUE,
    ):
        assert token in svg5, f"Fig. 5 SVG missing token: {token!r}"

    # Fig. S1-S5 checks
    for svg_doc, fig_name in (
        (svgs1, "Fig. S1"),
        (svgs2, "Fig. S2"),
        (svgs3, "Fig. S3"),
        (svgs4, "Fig. S4"),
        (svgs5, "Fig. S5"),
    ):
        for p_tag in ("(a)", "(b)", "(c)", "(d)"):
            assert p_tag in svg_doc, f"{fig_name} SVG missing panel tag {p_tag}"

    assert EVENT1_AUDIT_LABEL in svgs1
    assert "consistency check, not independent validation (protocol 4.6)" in svgs3
    assert "not computed" in svgs3
    assert "[0.85, 0.95]" in svgs4
    assert S_IND_PANEL_NOTE in svgs5
    assert S_IND_UQ_NOTE in svgs5


def test_batch_d_annotation_provenance(
    bundle: VerifiedBundle,
    batch_d_outputs: tuple[dict[str, dict[str, Path]], AnnotationRegistry],
) -> None:
    """Reconcile every registered Batch D annotation against its exact source in bundle."""
    _, reg = batch_d_outputs
    assert len(reg.records) >= 100

    mat = bundle.matrix_df
    for rec in reg.records:
        assert rec.displayed_text == format_num(rec.raw_value, rec.fmt_kind)

        if rec.source_file == "reports/results/event3_matrix.parquet":
            parts = rec.source_key.split("|")
            spec, analysis, fam, tgt, strat_tok, col_or_met = parts
            if ":" in strat_tok:
                skind, sval = strat_tok.split(":", 1)
            elif "=" in strat_tok:
                skind, sval = strat_tok.split("=", 1)
            else:
                skind, sval = strat_tok, "all"
            sub = mat.loc[
                (mat["spec"] == spec)
                & (mat["analysis"] == analysis)
                & (mat["family"] == fam)
                & (mat["target"] == tgt)
                & (mat["stratum_kind"] == skind)
                & (mat["stratum"] == sval)
                & (mat["metric"] == col_or_met),
                "value",
            ]
            assert len(sub) == 1, f"Missing matrix row for {rec.source_key}"
            assert math.isclose(
                float(rec.raw_value), float(sub.iloc[0]), rel_tol=0.0, abs_tol=1e-15
            )
        elif rec.source_file == "reports/results/audit.json":
            src_val = _resolve_json_path(bundle.audit, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        elif rec.source_file == "reports/results/event3_hypotheses.json":
            src_val = _resolve_json_path(bundle.hypotheses, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        elif rec.source_file == "reports/results/event3_uq.json":
            src_val = _resolve_json_path(bundle.uq, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        elif rec.source_file == "reports/results/event3_consistency.json":
            src_val = _resolve_json_path(bundle.consistency, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        elif rec.source_file == "reports/results/event3_dependence.json":
            src_val = _resolve_json_path(bundle.dependence, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        elif rec.source_file == "reports/results/event3_descriptive.json":
            src_val = _resolve_json_path(bundle.descriptive, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        else:  # pragma: no cover
            pytest.fail(f"Unexpected source_file: {rec.source_file}")


def test_batch_d_determinism(bundle: VerifiedBundle, tmp_path: Path) -> None:
    """Verify that running generate_fig4 and generate_fig5 twice yields identical SHA-256."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    reg_a = AnnotationRegistry()
    reg_b = AnnotationRegistry()

    f4_a = generate_fig4(bundle, REPO_ROOT, dir_a, reg_a)
    f4_b = generate_fig4(bundle, REPO_ROOT, dir_b, reg_b)
    for fmt in ("pdf", "svg", "png"):
        assert sha256_file(f4_a[fmt]) == sha256_file(f4_b[fmt])

    f5_a = generate_fig5(bundle, REPO_ROOT, dir_a, reg_a)
    f5_b = generate_fig5(bundle, REPO_ROOT, dir_b, reg_b)
    for fmt in ("pdf", "svg", "png"):
        assert sha256_file(f5_a[fmt]) == sha256_file(f5_b[fmt])
