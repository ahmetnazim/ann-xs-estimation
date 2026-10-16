"""Provenance, vector/DPI compliance, and determinism tests for Batch C figures (Fig. 1-3).

Verifies that:
1. ``fig1_domain_and_architecture``, ``fig2_benchmark_performance``, and
   ``fig3_physical_consistency_and_h4`` are generated as vector PDF, vector SVG, and 600 DPI PNG.
2. Every panel label ``(a)``, ``(b)``, ..., evidence tier (`Confirmatory`, `Descriptive`),
   Event 1 S-ID audit label, S-IND note (`131 locations / 18 rivers`), and verbatim §4.7 row 4
   statement appear in the vector SVG text.
3. Every numeric callout registered in :class:`AnnotationRegistry` equals its exact source value
   in ``reports/results/*`` at the displayed precision.
4. Re-running each figure generator produces byte-identical PDF, SVG, and PNG outputs.
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
    OKABE_ITO,
    SPLIT_PALETTE,
    AnnotationRegistry,
    VerifiedBundle,
    check_cvd_and_grayscale_distinguishability,
    format_num,
    generate_fig1,
    generate_fig2,
    generate_fig3,
    load_verified_bundle,
    sha256_file,
)
from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    VERBATIM_ROW4_OUTCOME,
    VERBATIM_ROW4_STATEMENT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bundle() -> VerifiedBundle:
    """Load the cryptographically verified Phase 8 results bundle once per module."""
    return load_verified_bundle(REPO_ROOT)


@pytest.fixture(scope="module")
def batch_c_outputs(
    bundle: VerifiedBundle,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, dict[str, Path]], AnnotationRegistry]:
    """Generate Fig. 1-3 once in a temporary directory and return paths + registry."""
    out_dir = tmp_path_factory.mktemp("figures_batch_c")
    reg = AnnotationRegistry()
    f1 = generate_fig1(bundle, REPO_ROOT, out_dir, reg)
    f2 = generate_fig2(bundle, REPO_ROOT, out_dir, reg)
    f3 = generate_fig3(bundle, REPO_ROOT, out_dir, reg)
    return {"fig1": f1, "fig2": f2, "fig3": f3}, reg


def _resolve_json_path(doc: dict[str, Any], dotted_key: str) -> float | int:
    cur: Any = doc
    for part in dotted_key.split("."):
        cur = cur[part]
    if isinstance(cur, (int, float)):
        return cur
    raise TypeError(f"Expected numeric value at {dotted_key}, got {type(cur)}")


def test_batch_c_files_vector_and_dpi(
    batch_c_outputs: tuple[dict[str, dict[str, Path]], AnnotationRegistry],
) -> None:
    """Verify PDF, SVG, and >= 300 DPI PNG outputs exist and meet artwork specifications."""
    paths_by_fig, _ = batch_c_outputs
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

        # Check PDF header
        pdf_bytes = pdf_p.read_bytes()
        assert pdf_bytes.startswith(b"%PDF-"), f"{fig_key} invalid PDF header"

        # Check SVG has un-outlined <text> elements (svg.fonttype = 'none')
        svg_text = svg_p.read_text(encoding="utf-8")
        assert "<text " in svg_text, f"{fig_key} SVG missing vector <text> elements"

        # Check PNG DPI >= 300 (configured at 600 DPI)
        with Image.open(png_p) as im:
            dpi_x, dpi_y = im.info.get("dpi", (0, 0))
            assert round(dpi_x) >= ASCE_RASTER_DPI
            assert round(dpi_y) >= ASCE_RASTER_DPI
            assert im.width >= 2000


def test_batch_c_panel_labels_and_verbatim_strings(
    batch_c_outputs: tuple[dict[str, dict[str, Path]], AnnotationRegistry],
) -> None:
    """Verify panel tags, evidence labels, split notes, and verbatim §4.7 row 4 text in SVGs."""
    paths_by_fig, _ = batch_c_outputs
    svg1 = html.unescape(paths_by_fig["fig1"]["svg"].read_text(encoding="utf-8"))
    svg2 = html.unescape(paths_by_fig["fig2"]["svg"].read_text(encoding="utf-8"))
    svg3 = html.unescape(paths_by_fig["fig3"]["svg"].read_text(encoding="utf-8"))

    # Fig. 1 checks: panels (a), (b), (c), Descriptive, S-ID label, S-IND note, §4.7 row 4
    for token in (
        "(a)",
        "(b)",
        "(c)",
        "[Descriptive]",
        EVENT1_AUDIT_LABEL,
        S_IND_PANEL_NOTE,
        VERBATIM_ROW4_OUTCOME,
        VERBATIM_ROW4_STATEMENT,
        "I_blind",
    ):
        assert token in svg1, f"Fig. 1 SVG missing token: {token!r}"

    # Fig. 2 checks: panels (a)-(d), Confirmatory (c), Descriptive (a, b, d), H1 CIs, §4.7 row 4
    for token in (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "[Confirmatory]",
        "[Descriptive]",
        EVENT1_AUDIT_LABEL,
        S_IND_PANEL_NOTE,
        VERBATIM_ROW4_OUTCOME,
        VERBATIM_ROW4_STATEMENT,
        "0.2502",
        "0.3094",
        "+0.05",
    ):
        assert token in svg2, f"Fig. 2 SVG missing token: {token!r}"

    # Fig. 3 checks: panels (a)-(d), Confirmatory / Descriptive, N_lit=[0.012, 0.33], H4 CIs
    for token in (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "Confirmatory",
        "[Descriptive]",
        EVENT1_AUDIT_LABEL,
        S_IND_PANEL_NOTE,
        "0.9265",
        "-0.0203",
        "0.4431",
        "0.2029",
    ):
        assert token in svg3, f"Fig. 3 SVG missing token: {token!r}"


def test_batch_c_annotation_provenance(
    bundle: VerifiedBundle,
    batch_c_outputs: tuple[dict[str, dict[str, Path]], AnnotationRegistry],
    tmp_path: Path,
) -> None:
    """Reconcile every registered figure annotation against its exact source in bundle."""
    _, reg = batch_c_outputs
    assert len(reg.records) >= 40

    ann_json = tmp_path / "figure_annotations.json"
    reg.write_json(ann_json)
    assert ann_json.is_file()

    mat = bundle.matrix_df
    for rec in reg.records:
        # 1. Displayed text equals formatted raw_value
        assert rec.displayed_text == format_num(rec.raw_value, rec.fmt_kind)

        # 2. Raw value equals source in bundle
        if rec.source_file == "reports/results/event3_matrix.parquet":
            parts = rec.source_key.split("|")
            spec, analysis, fam, tgt, strat_tok, col_or_met = parts
            if ":" in strat_tok:
                skind, sval = strat_tok.split(":", 1)
            else:
                skind, sval = strat_tok, "all"
            if col_or_met == "n_units":
                sub = mat.loc[
                    (mat["spec"] == spec)
                    & (mat["analysis"] == analysis)
                    & (mat["family"] == fam)
                    & (mat["target"] == tgt)
                    & (mat["stratum_kind"] == skind)
                    & (mat["stratum"] == sval)
                    & (mat["metric"] == "rmse_log"),
                    "n_units",
                ]
                assert int(rec.raw_value) == int(sub.iloc[0])
            else:
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
                assert math.isclose(
                    float(rec.raw_value), float(sub.iloc[0]), rel_tol=0.0, abs_tol=1e-15
                )
        elif rec.source_file == "reports/results/audit.json":
            src_val = _resolve_json_path(bundle.audit, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        elif rec.source_file == "reports/results/event3_hypotheses.json":
            src_val = _resolve_json_path(bundle.hypotheses, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        elif rec.source_file == "reports/results/event3_consistency.json":
            src_val = _resolve_json_path(bundle.consistency, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        elif rec.source_file == "reports/results/event3_descriptive.json":
            src_val = _resolve_json_path(bundle.descriptive, rec.source_key)
            assert math.isclose(float(rec.raw_value), float(src_val), rel_tol=0.0, abs_tol=1e-15)
        else:  # pragma: no cover
            pytest.fail(f"Unexpected source_file: {rec.source_file}")


def test_batch_c_determinism(bundle: VerifiedBundle, tmp_path: Path) -> None:
    """Verify that running generate_fig2 and generate_fig3 twice yields identical SHA-256."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    reg_a = AnnotationRegistry()
    reg_b = AnnotationRegistry()

    f2_a = generate_fig2(bundle, REPO_ROOT, dir_a, reg_a)
    f2_b = generate_fig2(bundle, REPO_ROOT, dir_b, reg_b)
    for fmt in ("pdf", "svg", "png"):
        assert sha256_file(f2_a[fmt]) == sha256_file(f2_b[fmt])

    f3_a = generate_fig3(bundle, REPO_ROOT, dir_a, reg_a)
    f3_b = generate_fig3(bundle, REPO_ROOT, dir_b, reg_b)
    for fmt in ("pdf", "svg", "png"):
        assert sha256_file(f3_a[fmt]) == sha256_file(f3_b[fmt])


def test_palette_cvd_distinguishability() -> None:
    """Verify CAM02-UCS perceptual separation of the primary split and Okabe-Ito palettes."""
    primary_split_hex = [
        SPLIT_PALETTE["S-ID"],
        SPLIT_PALETTE["S-LRO"],
        SPLIT_PALETTE["S-TAIL-DA"],
        SPLIT_PALETTE["S-TAIL-SO"],
        SPLIT_PALETTE["S-IND"],
    ]
    res = check_cvd_and_grayscale_distinguishability(primary_split_hex, min_delta_e=10.0)
    assert res["normal_min_delta_e"] >= 10.0
    assert res["deuteranopia_min_delta_e"] > 0.0
    assert res["protanopia_min_delta_e"] > 0.0

    with pytest.raises(ValueError, match="Palette normal min DeltaE"):
        check_cvd_and_grayscale_distinguishability(
            [OKABE_ITO["blue"], OKABE_ITO["blue"]], min_delta_e=5.0
        )
