"""Verification suite for the static WebGIS bundle, data parity, and physical invariants.

Verifies that `webgis/` and `webgis/data/` match canonical reports (`reports/results/event3.json`,
`reports/operational/ops3_results.json`), satisfy SHA-256 manifest integrity, and enforce
continuity and Dingman-Manning physical bounds across all 3,988 CONUS gauges and 131 USACE
eHydro locations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBGIS_DIR = REPO_ROOT / "webgis"
DATA_DIR = WEBGIS_DIR / "data"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_webgis_static_files_and_manifest_sha256() -> None:
    """Verify all static WebGIS files exist and match SHA-256 digests in manifest.json."""
    for rel in ("index.html", "styles.css", "app.js", "README.md", "data/manifest.json"):
        path = WEBGIS_DIR / rel
        assert path.is_file(), f"Missing required WebGIS file: {rel}"
        assert path.stat().st_size > 0, f"Empty WebGIS file: {rel}"

    manifest: dict[str, Any] = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["package"] == "ann-xs-estimation-webgis"
    assert manifest["n_conus_sites"] == 3988
    assert manifest["n_ehydro_locations"] == 131
    assert manifest["n_physio_divisions"] == 8

    files_list: list[dict[str, Any]] = manifest["files"]
    expected_data_files = {
        "benchmarks.json",
        "physio_divisions.geojson",
        "sites_conus.json",
        "sites_ehydro.json",
    }
    observed_files = {item["file"] for item in files_list}
    assert observed_files == expected_data_files

    for item in files_list:
        fname = item["file"]
        fpath = DATA_DIR / fname
        assert fpath.is_file(), f"Missing data file {fname}"
        assert fpath.stat().st_size == item["size_bytes"], f"Byte size mismatch for {fname}"
        assert _sha256_file(fpath) == item["sha256"], f"SHA-256 mismatch for {fname}"


def test_webgis_physio_divisions_geojson_schema() -> None:
    """Verify 8 Fenneman & Johnson physiographic divisions in EPSG:4326 with bootstrap winners."""
    fc: dict[str, Any] = json.loads(
        (DATA_DIR / "physio_divisions.geojson").read_text(encoding="utf-8")
    )
    assert fc["type"] == "FeatureCollection"
    features: list[dict[str, Any]] = fc["features"]
    assert len(features) == 8

    expected_divs = {
        "APPALACHIAN HIGHLANDS",
        "ATLANTIC PLAIN",
        "INTERIOR HIGHLANDS",
        "INTERIOR PLAINS",
        "INTERMONTANE PLATEAUS",
        "LAURENTIAN UPLAND",
        "PACIFIC MOUNTAIN SYSTEM",
        "ROCKY MOUNTAIN SYSTEM",
    }
    observed_divs = set()
    for feat in features:
        props = feat["properties"]
        div_name = props["division"]
        observed_divs.add(div_name)
        assert props["n_sites"] > 0
        for crit in ("geom", "op_fp", "op_dyn"):
            assert props[f"{crit}_best"] is not None
            assert 0.05 < props[f"{crit}_best_rmse"] < 10.0
        geom = feat["geometry"]
        assert geom["type"] in {"Polygon", "MultiPolygon"}

    assert observed_divs == expected_divs


def test_webgis_benchmarks_parity_with_canonical_reports() -> None:
    """Verify benchmarks.json matches reports/results/event3.json and ops3_results.json."""
    bench: dict[str, Any] = json.loads((DATA_DIR / "benchmarks.json").read_text(encoding="utf-8"))
    ev3: dict[str, Any] = json.loads(
        (REPO_ROOT / "reports" / "results" / "event3.json").read_text(encoding="utf-8")
    )
    ops3: dict[str, Any] = json.loads(
        (REPO_ROOT / "reports" / "operational" / "ops3_results.json").read_text(encoding="utf-8")
    )

    assert bench["meta"]["run_id"] == ev3["run_id"]
    assert bench["meta"]["params_sha256"] == ev3["params_sha256"]
    assert bench["meta"]["plan_sha256"] == ev3["plan_sha256"]
    assert len(bench["metrics_matrix"]) > 100

    # Verify M5 S-ID, S-LRO, and S-IND RMSE in metrics_matrix match canonical values
    matrix_lookup = {
        (row["family"], row["model"], row["target"]): row["metrics"]
        for row in bench["metrics_matrix"]
    }
    assert math.isclose(matrix_lookup[("S-ID", "M5", "dbar")]["rmse_log"], 0.7036, abs_tol=5e-4)
    assert math.isclose(matrix_lookup[("S-LRO", "M5", "dbar")]["rmse_log"], 0.7289, abs_tol=5e-4)
    assert math.isclose(matrix_lookup[("S-IND", "M5", "A")]["rmse_log"], 0.9495, abs_tol=5e-4)

    # Verify operational CCHE1D readiness parity and OPS-1 v3 table keys
    assert bench["operational"]["ops3_diagnostics"]["cche1d_readiness"] == ops3["cche1d_readiness"]
    v2_table = bench["operational"]["ops3_diagnostics"]["v2_all_models"]["S-ID"]["table"]
    assert isinstance(v2_table, list)
    assert len(v2_table) > 20
    dyn_table = bench["operational"]["ops3_diagnostics"]["dynamic_roughness"]["table"]
    assert all("rmse_dyn_n_m" in r and "rmse_const_n_m" in r for r in dyn_table)
    oa_table = bench["operational"]["ops3_diagnostics"]["off_anchor_states"]["table"]
    assert all("rmse_ln_dbar_all" in r and "rmse_raw_dmax_high_m" in r for r in oa_table)
    assert isinstance(bench["event1_audit"]["label"], dict)
    assert "text" in bench["event1_audit"]["label"]


def test_webgis_sites_physical_invariants_and_continuity() -> None:
    """Verify continuity A = W * d_mean and Dingman-Manning bounds across CONUS and eHydro sites."""
    conus: dict[str, Any] = json.loads((DATA_DIR / "sites_conus.json").read_text(encoding="utf-8"))
    ehydro: dict[str, Any] = json.loads(
        (DATA_DIR / "sites_ehydro.json").read_text(encoding="utf-8")
    )

    conus_sites: list[dict[str, Any]] = conus["sites"]
    ehydro_sites: list[dict[str, Any]] = ehydro["locations"]
    assert conus["n_sites"] == 3988
    assert len(conus_sites) == 3988
    assert ehydro["n_locations"] == 131
    assert ehydro["n_rivers"] == 18
    assert ehydro["n_single_comid_uq"] == 33
    assert len(ehydro_sites) == 131

    headline_models: list[str] = conus["schema"]["headline_models"]
    assert len(headline_models) == 10
    assert {"M6-NN", "M6-XGB", "M-null"}.issubset(set(headline_models))
    m5_idx = headline_models.index("M5")

    # Check physical bounds and exact continuity closure on M5 across all 3,988 CONUS sites
    n_with_3dep = 0
    for site in conus_sites:
        assert -126.0 <= site["lo"] <= -65.0
        assert 24.0 <= site["la"] <= 50.0
        if site["fp"] is not None:
            n_with_3dep += 1
            assert len(site["fp"][2]) == 21

        for regime_key in ("sid", "slro"):
            preds_list = site["p"].get(regime_key, [])
            m5_entry = preds_list[m5_idx] if len(preds_list) > m5_idx else None
            if m5_entry is not None:
                w, d_mean, d_max, area, r_exp, manning_n = m5_entry[:6]
                assert w > 0.0
                assert d_mean > 0.0
                assert d_max > 0.0
                assert area >= 0.0
                # Continuity closure A = W * d_mean within JSON 2-decimal rounding tolerance
                assert math.isclose(area, w * d_mean, rel_tol=0.015, abs_tol=0.02), (
                    f"Continuity violation at {site['id']}: A={area}, W*d={w * d_mean}"
                )
                assert 0.5 <= r_exp <= 10.0, f"Dingman r out of bounds at {site['id']}: {r_exp}"
                assert 0.005 <= manning_n <= 0.60, (
                    f"Manning n out of bounds at {site['id']}: {manning_n}"
                )
                if len(m5_entry) >= 10:
                    w_lo, w_hi, d_lo, d_hi = m5_entry[6:10]
                    assert 0.0 <= w_lo <= w_hi
                    assert 0.0 <= d_lo <= d_hi
                    assert d_hi > 0.0

    assert n_with_3dep == 3350

    # Check eHydro independent sites
    n_ehydro_uq = 0
    for esite in ehydro_sites:
        assert "M5" in esite["pred"]
        m5_e = esite["pred"]["M5"]
        w, d_mean, d_max, area, r_exp = m5_e[:5]
        assert w > 0.0
        assert d_mean > 0.0
        if d_max is not None:
            assert d_max > 0.0
        assert area >= 0.0
        assert math.isclose(area, w * d_mean, rel_tol=0.015, abs_tol=0.02)
        if r_exp is not None:
            assert 0.5 <= r_exp <= 10.0
        if esite["single_comid_uq"]:
            n_ehydro_uq += 1
        if len(m5_e) >= 10 and m5_e[6] is not None:
            w_lo, w_hi, d_lo, d_hi = m5_e[6:10]
            assert 0.0 < w_lo <= w_hi
            assert 0.0 < d_lo <= d_hi

    assert n_ehydro_uq == 33


def test_webgis_dom_ids_and_accessibility_hooks() -> None:
    """Verify every DOM ID referenced in app.js exists in index.html and accessibility tags hold."""
    html = (WEBGIS_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEBGIS_DIR / "app.js").read_text(encoding="utf-8")
    css = (WEBGIS_DIR / "styles.css").read_text(encoding="utf-8")

    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_ids = set(re.findall(r'getElementById\("([^"]+)"\)', js))
    missing_ids = js_ids - html_ids
    assert not missing_ids, f"DOM IDs referenced in app.js missing from index.html: {missing_ids}"

    # Verify modern web guidance semantics and single semantic <caption> per table
    assert 'name="color-scheme"' in html
    assert '<details class="controls-details"' in html
    assert 'aria-pressed="' in html
    assert len(re.findall(r"<caption\b", html)) == len(
        re.findall(r'<table class="data-table"', html)
    )
    assert "color-scheme: light dark" in css
    assert "overscroll-behavior: contain" in css
    assert "@container (width < 520px)" in css

    # Verify manuscript cross-references and internal protocol codes are absent from UI copy
    for token in (
        "§4.7",
        "Row 4",
        "Row-4",
        "Table T2",
        "Table T7",
        "Table T10",
        "Table T11",
        "Event 1",
        "Event 2",
        "OPS-1 v1",
        "OPS-1 v2",
        "OPS-1 v3",
        "H-FP1",
        "H-V3-",
    ):
        assert token not in html, f"Manuscript/protocol reference '{token}' still in index.html"
        assert token not in js, f"Manuscript/protocol reference '{token}' still in app.js"
