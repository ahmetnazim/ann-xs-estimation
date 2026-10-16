"""Step 5-7 typography, accessibility, captions, manifest, and frozen-state quality gate tests.

Verifies across the complete Phase 9 output suite (`Fig. 1`-`Fig. 5`, `Fig. S1`-`Fig. S5`,
`T1`-`T9`, `docs/figures/captions.md`, `docs/figures/figure_annotations.json`, and
`docs/figures/provenance_manifest.json`):
1. ASCE *Journal of Hydrologic Engineering* artwork compliance: TrueType/Type 42 embedded fonts in
   PDFs (`pdf.fonttype = 42`), physical column width (`504 pt = 7.0 in`), editable vector `<text>`
   in SVGs (`svg.fonttype = 'none'`), sans-serif font family, minimum font size and stroke widths,
   and `600 DPI` PNG resolution (`4200 px` width).
2. Colour-vision-deficiency (deuteranopia, protanopia, tritanopia) and ITU-R BT.709 grayscale
   luminance distinguishability across palettes and rendered PNGs.
3. Factual captions (`docs/figures/captions.md`) and full on-disk `provenance_manifest.json`
   SHA-256 reconciliation (`32` files in `docs/figures/` and `10` files in `docs/tables/`).
4. Frozen-state integrity (`scoring_ledger.json` 7-event chain, `reports/results/*` §17 digests,
   `params.yaml` and root `uv.lock` digests, and `torch`/`xgboost` absence).
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from annxs_figures import (
    ASCE_DOUBLE_COL_IN,
    ASCE_RASTER_DPI,
    EXPECTED_PARAMS_SHA256,
    EXPECTED_UV_LOCK_SHA256,
    MODEL_PALETTE,
    RESULTS_SHA256,
    SPLIT_PALETTE,
    VerifiedBundle,
    assert_no_forbidden_imports,
    check_cvd_and_grayscale_distinguishability,
    format_num,
    generate_captions,
    load_verified_bundle,
    sha256_file,
)
from annxs_figures.captions import _mat_val
from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    S_IND_UQ_NOTE,
    VERBATIM_ROW4_OUTCOME,
    VERBATIM_ROW4_STATEMENT,
    VERBATIM_ROW4_VENUE,
)
from annxs_figures.style import ASCE_ARTWORK_ACCESS_DATE, ASCE_ARTWORK_URL

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
TABLES_DIR = REPO_ROOT / "docs" / "tables"

ALL_FIGURE_STEMS: tuple[str, ...] = (
    "fig1_domain_and_architecture",
    "fig2_benchmark_performance",
    "fig3_physical_consistency_and_h4",
    "fig4_ood_and_uq_calibration",
    "fig5_row4_diagnosis",
    "figS1_event1_spatial_dependence",
    "figS2_per_division_slro",
    "figS3_secondary_analyses",
    "figS4_training_and_conformal",
    "figS5_sind_ehydro_rivers",
)


@pytest.fixture(scope="module")
def bundle() -> VerifiedBundle:
    """Load the cryptographically verified Phase 8 results bundle once per module."""
    return load_verified_bundle(REPO_ROOT)


def test_asce_vector_pdf_svg_and_png_specs() -> None:
    """Verify PDF Type 42 fonts, SVG vector text & font sizes, and PNG 600 DPI across 10 figures."""
    min_width_pt = ASCE_DOUBLE_COL_IN * 72.0  # 504.0 pt base canvas
    max_width_pt = 9.1 * 72.0  # tight bbox width ceiling
    max_height_pt = 9.0 * 72.0  # 648.0 pt (ASCE max page height)

    font_size_re = re.compile(r"font-size:\s*([0-9.]+)px")
    stroke_width_re = re.compile(r"stroke-width:\s*([0-9.]+)")
    mediabox_re = re.compile(
        rb"/MediaBox\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]"
    )

    for stem in ALL_FIGURE_STEMS:
        pdf_path = FIGURES_DIR / f"{stem}.pdf"
        svg_path = FIGURES_DIR / f"{stem}.svg"
        png_path = FIGURES_DIR / f"{stem}.png"

        assert pdf_path.is_file(), f"Missing PDF for {stem}"
        assert svg_path.is_file(), f"Missing SVG for {stem}"
        assert png_path.is_file(), f"Missing PNG for {stem}"

        # 1. PDF checks: header, TrueType/Type 42 font embedding, physical dimensions
        pdf_bytes = pdf_path.read_bytes()
        assert pdf_bytes.startswith(b"%PDF-"), f"Invalid PDF header in {stem}"
        assert b"/FontDescriptor" in pdf_bytes, f"PDF {stem} missing embedded FontDescriptor"
        assert b"/CIDFontType2" in pdf_bytes or b"/TrueType" in pdf_bytes, (
            f"PDF {stem} missing TrueType/Type 42 embedded font stream"
        )

        mb_match = mediabox_re.search(pdf_bytes)
        assert mb_match is not None, f"PDF {stem} missing /MediaBox"
        x0, y0, x1, y1 = (float(mb_match.group(i)) for i in range(1, 5))
        width_pt = x1 - x0
        height_pt = y1 - y0
        assert min_width_pt <= width_pt <= max_width_pt, (
            f"PDF {stem} width {width_pt} pt outside [{min_width_pt}, {max_width_pt}] pt"
        )
        assert 300.0 <= height_pt <= max_height_pt, (
            f"PDF {stem} height {height_pt} pt outside [300, {max_height_pt}] pt"
        )

        # 2. SVG checks: vector <text>, sans-serif font, minimum font size & stroke width
        svg_text = svg_path.read_text(encoding="utf-8")
        assert "<text " in svg_text, f"SVG {stem} missing vector <text> elements"
        assert "DejaVu Sans" in svg_text or "Arial" in svg_text, (
            f"SVG {stem} missing sans-serif font specification"
        )

        font_sizes = [float(m) for m in font_size_re.findall(svg_text)]
        assert len(font_sizes) >= 10, f"SVG {stem} has too few font-size declarations"
        assert min(font_sizes) >= 3.4, f"SVG {stem} minimum font size {min(font_sizes)} < 3.4 pt"
        assert max(font_sizes) <= 12.0, f"SVG {stem} maximum font size {max(font_sizes)} > 12.0 pt"

        stroke_widths = [float(w) for w in stroke_width_re.findall(svg_text) if float(w) > 0.0]
        assert len(stroke_widths) >= 10, f"SVG {stem} has too few positive stroke widths"
        assert min(stroke_widths) >= 0.25, (
            f"SVG {stem} minimum positive stroke width {min(stroke_widths)} < 0.25 pt"
        )

        # 3. PNG checks: 600 DPI and >= 4200 px width
        with Image.open(png_path) as im:
            dpi_x, dpi_y = im.info.get("dpi", (0.0, 0.0))
            assert round(dpi_x) == ASCE_RASTER_DPI, f"PNG {stem} dpi_x={dpi_x} != 600"
            assert round(dpi_y) == ASCE_RASTER_DPI, f"PNG {stem} dpi_y={dpi_y} != 600"
            assert im.width >= round(ASCE_DOUBLE_COL_IN * ASCE_RASTER_DPI), (
                f"PNG {stem} width {im.width} < 4200 px"
            )


def test_cvd_and_grayscale_accessibility_across_suite() -> None:
    """Verify CVD & grayscale luminance separation across palettes and all 10 rendered PNGs."""
    # 1. Palette-level verification (headline model comparison & split palettes)
    headline_models = [
        MODEL_PALETTE["M5"],
        MODEL_PALETTE["M2-XGB"],
        MODEL_PALETTE["M2-MLP"],
        MODEL_PALETTE["M4"],
        MODEL_PALETTE["M1"],
        MODEL_PALETTE["M0"],
    ]
    report = check_cvd_and_grayscale_distinguishability(headline_models, min_delta_e=10.0)
    assert report["normal_min_delta_e"] >= 10.0
    assert report["deuteranopia_min_delta_e"] >= 5.0
    assert report["protanopia_min_delta_e"] >= 2.0

    m5_vs_m2_report = check_cvd_and_grayscale_distinguishability(
        [MODEL_PALETTE["M5"], MODEL_PALETTE["M2-XGB"]], min_delta_e=25.0
    )
    assert m5_vs_m2_report["normal_min_delta_e"] >= 25.0
    assert m5_vs_m2_report["deuteranopia_min_delta_e"] >= 15.0
    assert m5_vs_m2_report["protanopia_min_delta_e"] >= 15.0

    split_report = check_cvd_and_grayscale_distinguishability(
        [
            SPLIT_PALETTE["S-ID"],
            SPLIT_PALETTE["S-LRO"],
            SPLIT_PALETTE["S-TAIL-DA"],
            SPLIT_PALETTE["S-TAIL-SO"],
            SPLIT_PALETTE["S-IND"],
        ],
        min_delta_e=10.0,
    )
    assert split_report["normal_min_delta_e"] >= 10.0

    # 2. Rendered PNG grayscale & deuteranopia contrast audit (subsampled for speed)
    deutan_mat = np.array(
        [
            [0.29275, 0.70725, 0.00000],
            [0.29275, 0.70725, 0.00000],
            [-0.02234, 0.02234, 1.00000],
        ],
        dtype=np.float64,
    )
    for stem in ALL_FIGURE_STEMS:
        png_path = FIGURES_DIR / f"{stem}.png"
        with Image.open(png_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float64)[::8, ::8, :] / 255.0
        # ITU-R BT.709 relative luminance
        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        dyn_range = float(lum.max() - lum.min())
        std_lum = float(lum.std())
        assert dyn_range >= 0.85, f"PNG {stem} grayscale dynamic range {dyn_range:.3f} < 0.85"
        assert std_lum >= 0.05, f"PNG {stem} grayscale std {std_lum:.3f} < 0.05"

        # Deuteranopia simulated luminance std
        deut_rgb = np.clip(arr @ deutan_mat.T, 0.0, 1.0)
        deut_lum = (
            0.2126 * deut_rgb[:, :, 0] + 0.7152 * deut_rgb[:, :, 1] + 0.0722 * deut_rgb[:, :, 2]
        )
        assert float(deut_lum.std()) >= 0.05, f"PNG {stem} deuteranopia std < 0.05"


def test_captions_and_provenance_manifest_integrity(
    bundle: VerifiedBundle,
    tmp_path: Path,
) -> None:
    """Verify docs/figures/captions.md, figure_annotations.json, and provenance_manifest.json."""
    # Test captions generator determinism and error branch
    cap_tmp = generate_captions(bundle, tmp_path)
    cap_disk = FIGURES_DIR / "captions.md"
    assert cap_disk.is_file(), "docs/figures/captions.md missing on disk"
    assert sha256_file(cap_tmp) == sha256_file(cap_disk), "captions.md is not byte-deterministic"

    with pytest.raises(KeyError, match="Expected 1 matrix row"):
        _mat_val(bundle, spec="NONEXISTENT", family="S-LRO")

    cap_text = cap_disk.read_text(encoding="utf-8")
    for required_token in (
        "### Fig. 1",
        "### Fig. 2",
        "### Fig. 3",
        "### Fig. 4",
        "### Fig. 5",
        "### Fig. S1",
        "### Fig. S2",
        "### Fig. S3",
        "### Fig. S4",
        "### Fig. S5",
        ASCE_ARTWORK_URL,
        ASCE_ARTWORK_ACCESS_DATE,
        VERBATIM_ROW4_OUTCOME,
        VERBATIM_ROW4_VENUE,
        VERBATIM_ROW4_STATEMENT,
        EVENT1_AUDIT_LABEL,
        S_IND_PANEL_NOTE,
        S_IND_UQ_NOTE,
        "Confirmatory",
        "Secondary",
        "Descriptive",
        "[0.85, 0.95]",
        "98.75%",
        "+0.2502",
        "+0.3094",
        "0.8997",
        "0.9025",
        "0.8182",
    ):
        assert required_token in cap_text, f"captions.md missing token: {required_token!r}"

    # Verify figure_annotations.json covers all 10 figures
    ann_path = FIGURES_DIR / "figure_annotations.json"
    assert ann_path.is_file()
    records = json.loads(ann_path.read_text(encoding="utf-8"))
    assert isinstance(records, list)
    assert len(records) >= 200
    fig_ids = {r["figure_id"] for r in records}
    assert fig_ids == {
        "fig1",
        "fig2",
        "fig3",
        "fig4",
        "fig5",
        "figS1",
        "figS2",
        "figS3",
        "figS4",
        "figS5",
    }
    for r in records:
        assert r["displayed_text"] == format_num(r["raw_value"], r["fmt_kind"])

    # Verify provenance_manifest.json matches every on-disk file in docs/figures and docs/tables
    manifest_path = FIGURES_DIR / "provenance_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    figures_sha = manifest["figures_sha256"]
    assert len(figures_sha) == 32  # 10 figures * 3 formats + figure_annotations.json + captions.md
    for rel_path, expected_sha in figures_sha.items():
        full_p = REPO_ROOT / rel_path
        assert full_p.is_file(), f"Missing manifest figure artifact: {rel_path}"
        assert sha256_file(full_p) == expected_sha, f"SHA-256 mismatch for {rel_path}"

    tables_sha = manifest["tables_sha256"]
    assert len(tables_sha) == 10  # benchmark_results.md + T1..T9.csv
    for rel_path, expected_sha in tables_sha.items():
        full_p = REPO_ROOT / rel_path
        assert full_p.is_file(), f"Missing manifest table artifact: {rel_path}"
        assert sha256_file(full_p) == expected_sha, f"SHA-256 mismatch for {rel_path}"


def test_phase9_frozen_state_and_no_forbidden_imports(bundle: VerifiedBundle) -> None:
    """Verify frozen Phase 8 artifacts, ledger chain, and absence of torch/xgboost."""
    assert_no_forbidden_imports()
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("xgboost") is None

    assert len(bundle.ledger["events"]) == 7
    for rel_path, expected_sha in RESULTS_SHA256.items():
        assert sha256_file(REPO_ROOT / rel_path) == expected_sha

    assert sha256_file(REPO_ROOT / "params.yaml") == EXPECTED_PARAMS_SHA256
    assert sha256_file(REPO_ROOT / "uv.lock") == EXPECTED_UV_LOCK_SHA256
