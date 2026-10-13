"""Tests for the Phase 8 stage 0a/0b builders (docs/3 13.3, 13.5, 13.6): synthetic data only.

``src/data/srnd_frame.py`` (row-level S-RND frame) and ``src/data/ind_features.py`` (S-IND
predictors per member COMID, common-support rule, catchment division, E-R8 aggregation). Known
answers are computed by hand in the docstrings of the tests. Both ``run()`` entry points are
exercised on a temporary repository (``ANNXS_REPO_ROOT``) with the raw readers monkeypatched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from src.config import load_settings
from src.data import ind_features as inf
from src.data import srnd_frame as sf
from src.data.io import read_parquet, read_parquet_meta
from src.models.baselines import M1_FEATURES

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_settings().protocol
NN = tuple(PROTOCOL.transforms.features)
M1 = sf.M1_SET
ALL_PRED = [*NN, *(c for c in M1 if c not in NN)]


# ----------------------------------------------------------------------------- shared
def test_m1_set_equals_runner_m1_features() -> None:
    """``src.data`` restates the runner's M1 set (it must never import ``src.models``)."""
    assert M1 == M1_FEATURES


def test_new_data_modules_are_torch_free() -> None:
    code = (
        "import sys, src.data.srnd_frame, src.data.ind_features; "
        "print('torch' in sys.modules, 'xgboost' in sys.modules, "
        "any(m.startswith('src.models') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False False"


def test_path_helpers() -> None:
    root = Path("/x")
    assert sf.srnd_frame_path(root) == root / "srnd" / "srnd_frame.parquet"
    assert inf.ind_features_path(root) == root / "ind" / "ehydro_ind_features.parquet"


# ----------------------------------------------------------------------------- S-RND frame
def _site_table(n_sites: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n_sites):
        base: dict[str, Any] = {c: float(rng.uniform(1.0, 5.0)) for c in ALL_PRED}
        base["F_S"] = False
        for variant in ("primary", "sa3"):
            rows.append(
                {
                    "site_no": f"{i:08d}",
                    "variant": variant,
                    "comid": 100 + i,
                    "division": "A",
                    **base,
                    "W": 99.0,  # site-level targets must not leak into the row frame
                    "dbar": 99.0,
                    "F_M1_missing": False,
                }
            )
    st = pd.DataFrame(rows)
    st.loc[st["site_no"].eq("00000002"), "CAT_EVI_AMJ_2012"] = np.nan
    st.loc[st["site_no"].eq("00000002"), "F_M1_missing"] = True
    return st


def _assignment_and_init(n_sites: int = 3, per_site: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = [f"{i:08d}" for i in range(n_sites) for _ in range(per_site)]
    meas = [f" {k}" for _ in range(n_sites) for k in range(per_site)]
    rid = np.arange(len(sites), dtype=np.int64) * 10 + 7
    assign = pd.DataFrame(
        {
            "site_no": sites,
            "q_meas_no": meas,
            "row_id": rid,
            "is_test": np.arange(len(sites)) % 4 == 0,
            "seed": np.int64(42),
        }
    )
    init = pd.DataFrame(
        {
            "row_id": rid[::-1],
            "site_no": sites[::-1],
            "q_meas_no": meas[::-1],
            "W": np.linspace(5.0, 20.0, len(sites))[::-1],
            "dbar": np.linspace(0.5, 2.0, len(sites))[::-1],
            "dmax": np.linspace(1.0, 4.0, len(sites))[::-1],
            "A": np.linspace(3.0, 40.0, len(sites))[::-1],
            "F_eps": np.arange(len(sites)) % 3 == 0,
            "F_t": False,
            "F_qual": np.arange(len(sites)) % 5 == 0,
            "F_tout": False,
            "Q_obs": 1.0,
        }
    )
    return assign, init


def test_srnd_frame_join_keeps_every_row_and_row_targets() -> None:
    assign, init = _assignment_and_init()
    st = _site_table()
    frame, counts = sf.build_srnd_frame(assign, init, st, NN, M1)
    assert len(frame) == 12
    assert frame["row_id"].is_monotonic_increasing
    by_row = init.set_index("row_id")
    for col in sf.SRND_TARGETS:  # per-measurement values, not the site-level 99.0
        np.testing.assert_array_equal(frame[col], by_row.loc[frame["row_id"], col].to_numpy())
    assert "Q_obs" not in frame
    site1 = st.loc[st["site_no"].eq("00000001")].iloc[0]
    rows1 = frame.loc[frame["site_no"].eq("00000001")]
    assert rows1["damdensws"].eq(site1["damdensws"]).all()
    assert counts["rows"] == 12
    assert (counts["rows_train"], counts["rows_test"]) == (9, 3)
    assert counts["sites_in_both"] == 3
    assert counts["F_M1_missing_rows"] == {"train": 3, "test": 1}
    assert counts["flag_rows"]["F_eps"] == int(init["F_eps"].sum())
    assert counts["predictors_m1"] == list(M1)


def test_srnd_frame_refusals() -> None:
    assign, init = _assignment_and_init()
    st = _site_table()
    bad = init.copy()
    bad.loc[bad["row_id"].eq(7), "site_no"] = "00000009"
    with pytest.raises(ValueError, match="disagree"):
        sf.build_srnd_frame(assign, bad, st, NN, M1)
    with pytest.raises(ValueError, match="without a hydroswot_init row"):
        sf.build_srnd_frame(assign, init.iloc[1:], st, NN, M1)
    with pytest.raises(ValueError, match="missing from site_table"):
        sf.build_srnd_frame(assign, init, st.loc[~st["site_no"].eq("00000000")], NN, M1)
    differ = st.copy()
    differ.loc[1, "damdensws"] = -1.0
    with pytest.raises(ValueError, match="differ between variants"):
        sf.build_srnd_frame(assign, init, differ, NN, M1)
    with pytest.raises(KeyError, match="site_table lacks"):
        sf.build_srnd_frame(assign, init, st.drop(columns=["AI"]), NN, M1)
    neg = init.copy()
    neg.loc[0, "W"] = 0.0
    with pytest.raises(ValueError, match="invariant 5"):
        sf.build_srnd_frame(assign, neg, st, NN, M1)
    nn_gap = st.copy()
    nn_gap.loc[nn_gap["site_no"].eq("00000001"), "runoffws"] = np.nan
    with pytest.raises(ValueError, match="NN predictors missing"):
        sf.build_srnd_frame(assign, init, nn_gap, NN, M1)
    flag = st.copy()
    flag["F_M1_missing"] = False
    with pytest.raises(ValueError, match="F_M1_missing disagrees"):
        sf.build_srnd_frame(assign, init, flag, NN, M1)
    with pytest.raises(ValueError, match="banned"):
        sf.build_srnd_frame(assign, init, st, (*NN, "W"), M1)


# ----------------------------------------------------------------------------- S-IND members
def _ind() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loc_id": ["L2", "L1", "L3", "L4"],
            "river": ["R", "R", "S", "S"],
            "comids": ["21;22", "11", "31;32;33", "41"],
            "n_comids": [2, 1, 3, 1],
            "dedupe_status": ["kept", "kept", "kept", "dropped_comid"],
        }
    )


def test_member_table_expands_kept_locations_in_order() -> None:
    mem = inf.member_table(_ind())
    assert mem["loc_id"].tolist() == ["L1", "L2", "L2", "L3", "L3", "L3"]
    assert mem["member"].tolist() == [0, 0, 1, 0, 1, 2]
    assert mem["comid"].tolist() == [11, 21, 22, 31, 32, 33]
    assert set(mem["n_comids"]) == {1, 2, 3}


def test_member_table_refuses_targets_and_bad_rows() -> None:
    with pytest.raises(ValueError, match="must not read the S-IND targets"):
        inf.member_table(_ind().assign(W=1.0))
    with pytest.raises(KeyError, match="lacks"):
        inf.member_table(_ind().drop(columns=["river"]))
    bad = _ind()
    bad.loc[0, "n_comids"] = 3
    with pytest.raises(ValueError, match="n_comids differs"):
        inf.member_table(bad)


def _per_comid(comids: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    pc = pd.DataFrame({"comid": comids})
    for c in ALL_PRED:
        if c not in ("Q_E",):
            pc[c] = rng.uniform(1.0, 5.0, len(comids))
    pc["F_S"] = False
    pc["F_AI_touched"] = [True, False, False, False, False, False]
    return pc


def _erom(comids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comid": comids,
            "Q_E": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "Q_C": 1.0,
            "V_E": [2.0, 4.0, 5.0, np.nan, 6.0, 3.0],
        }
    )


def test_build_ind_features_support_rule_and_counts() -> None:
    """L1 (11) fine; L2 (21, 22): 22 lacks AI -> R14_nn; L3: 32 lacks an M1 predictor -> R14_m1.

    Then the R13 and R12 precedences are checked on a copy: Q_E <= 0 beats a missing slope.
    """
    mem = inf.member_table(_ind())
    comids = mem["comid"].tolist()
    pc = _per_comid(comids)
    pc.loc[pc["comid"].eq(22), "AI"] = np.nan
    pc.loc[pc["comid"].eq(32), "CAT_SANDAVE"] = np.nan
    lengths = {11: 1.0, 21: 2.0, 22: 3.0, 31: 1.0, 32: 1.0, 33: 2.0}
    div = pd.Series({11: "A", 21: pd.NA, 22: "B", 31: "A", 32: "A", 33: "A"}, dtype="string")
    log_feats = [f for f, k in PROTOCOL.transforms.features.items() if k == "log"]
    table, counts = inf.build_ind_features(mem, pc, _erom(comids), lengths, div, NN, M1, log_feats)
    assert table["exclusion"].tolist() == ["", "", "R14_nn", "", "R14_m1", ""]
    assert table["loc_in_support"].tolist() == [True, False, False, False, False, False]
    assert table.loc[0, "A_J"] == pytest.approx(5.0)
    assert np.isnan(table.loc[table["comid"].eq(31), "A_J"]).all()
    assert table["lengthkm"].tolist() == [1.0, 2.0, 3.0, 1.0, 1.0, 2.0]
    assert pd.isna(table.loc[table["comid"].eq(21), "division"]).all()
    assert table.loc[table["comid"].eq(32), "F_M1_missing"].all()
    assert not {"W", "dbar", "A"} & set(table.columns)
    assert counts["comids_by_reason"] == {"R13": 0, "R12": 0, "R14_nn": 1, "R14_m1": 1}
    assert counts["locations_excluded"] == 2
    assert counts["locations_in_support"] == 1
    assert counts["locations_in_support_single_comid"] == 1
    assert counts["rivers_in_support"] == ["R"]
    assert counts["ai_touched_fallback_in_support"] == 1
    assert counts["nn_log_nonpositive_rows_in_support"] == 0
    pc2 = _per_comid(comids)
    pc2.loc[pc2["comid"].isin([11, 21]), "S"] = np.nan
    erom2 = _erom(comids)
    erom2.loc[erom2["comid"].eq(11), "Q_E"] = 0.0
    pc2.loc[pc2["comid"].eq(31), "AI"] = -1.0
    t2, c2 = inf.build_ind_features(mem, pc2, erom2, lengths, div, NN, M1, log_feats)
    assert t2["exclusion"].tolist()[:3] == ["R13", "R12", ""]
    assert c2["locations_excluded_with_reason"]["R13"] == 1
    assert c2["nn_log_nonpositive_rows_in_support"] == 1  # comid 31: AI = -1 (log feature)
    assert c2["aj_unavailable_locations_in_support"] == 1


def test_build_ind_features_refusals() -> None:
    mem = inf.member_table(_ind())
    comids = mem["comid"].tolist()
    lengths = dict.fromkeys(comids, 1.0)
    div = pd.Series(dict.fromkeys(comids, "A"), dtype="string")
    pc = _per_comid(comids)
    with pytest.raises(ValueError, match="one row per comid"):
        inf.build_ind_features(mem, pd.concat([pc, pc]), _erom(comids), lengths, div, NN, M1)
    with pytest.raises(ValueError, match="LENGTHKM > 0"):
        inf.build_ind_features(mem, pc, _erom(comids), {**lengths, 33: 0.0}, div, NN, M1)
    with pytest.raises(ValueError, match="LENGTHKM > 0"):
        inf.build_ind_features(mem, pc, _erom(comids), {11: 1.0}, div, NN, M1)
    with pytest.raises(ValueError, match="banned"):
        inf.build_ind_features(mem, pc, _erom(comids), lengths, div, (*NN, "V_E"), M1)


# ----------------------------------------------------------------------------- E-R8 aggregation
def test_aggregate_e_r8_known_answer() -> None:
    """L_c = (1, 3): Ŵ = (1·2 + 3·6)/4 = 5, d̂ = (1·1 + 3·2)/4 = 1.75, Â = 8.75; L = 2 alone."""
    out = inf.aggregate_e_r8(
        ["b", "a", "a"],
        np.array([2.0, 1.0, 3.0]),
        np.array([4.0, 2.0, 6.0]),
        np.array([0.5, 1.0, 2.0]),
    )
    assert out["loc_id"].tolist() == ["a", "b"]
    np.testing.assert_allclose(out["W"], [5.0, 4.0])
    np.testing.assert_allclose(out["dbar"], [1.75, 0.5])
    np.testing.assert_allclose(out["A"], [8.75, 2.0])


def test_length_weighted_mean_masks_and_refusals() -> None:
    """A_J rule: a location with any missing member value is unavailable (not re-weighted)."""
    got = inf.length_weighted_mean(
        pd.Series(["x", "x", "y"]), np.array([1.0, 1.0, 2.0]), np.array([2.0, np.nan, 3.0])
    )
    assert np.isnan(got["x"])
    assert got["y"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="equal length"):
        inf.length_weighted_mean(["x"], np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="LENGTHKM"):
        inf.length_weighted_mean(["x"], np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="finite or missing"):
        inf.length_weighted_mean(["x"], np.array([1.0]), np.array([np.inf]))
    with pytest.raises(ValueError, match="w_hat"):
        inf.aggregate_e_r8(["x"], np.array([1.0]), np.array([np.nan]), np.array([1.0]))
    with pytest.raises(ValueError, match="d_hat"):
        inf.aggregate_e_r8(["x"], np.array([1.0]), np.array([1.0]), np.array([0.0]))


# ----------------------------------------------------------------------------- division
def _polygons() -> gpd.GeoDataFrame:
    """Two divisions side by side in EPSG:5070 (x < 0: WEST, x > 0: EAST) and one unnamed."""
    return gpd.GeoDataFrame(
        {"DIVISION": ["WEST", "EAST", pd.NA]},
        geometry=[box(-1e5, 0, 0, 1e5), box(0, 0, 1e5, 1e5), box(0, 2e5, 1e5, 3e5)],
        crs="EPSG:5070",
    )


def test_catchment_divisions_point_on_surface() -> None:
    """A U-shaped catchment whose centroid lies outside it still gets its own division.

    The U spans x in [1e4, 4e4]; its centroid (x = 2.5e4, y ~ 3.6e4) lies in the notch, outside
    the polygon, but the point on the surface is inside and in EAST. The second catchment is in
    WEST, the third in an unnamed polygon (missing), the fourth outside every polygon (missing).
    """
    u_shape = Polygon(
        [
            (1e4, 1e4),
            (4e4, 1e4),
            (4e4, 9e4),
            (3e4, 9e4),
            (3e4, 2e4),
            (2e4, 2e4),
            (2e4, 9e4),
            (1e4, 9e4),
        ]
    )
    cats = gpd.GeoDataFrame(
        {"comid": [1, 2, 3, 4]},
        geometry=[
            u_shape,
            box(-5e4, 1e4, -4e4, 2e4),
            box(1e4, 2.1e5, 2e4, 2.2e5),
            box(5e5, 5e5, 5.1e5, 5.1e5),
        ],
        crs="EPSG:5070",
    )
    assert not u_shape.contains(u_shape.centroid)
    got = inf.catchment_divisions(cats, _polygons())
    assert got.index.tolist() == [1, 2, 3, 4]
    assert got.loc[1] == "EAST"
    assert got.loc[2] == "WEST"
    assert pd.isna(got.loc[3])
    assert pd.isna(got.loc[4])


# ----------------------------------------------------------------------------- run() entry points
@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    (root / "data" / "processed").mkdir(parents=True)
    monkeypatch.setenv("ANNXS_REPO_ROOT", str(root))
    monkeypatch.setenv("ANNXS_DVC_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("ANNXS_RUN_ROOT", str(tmp_path / "store" / "runs"))
    return root


def test_srnd_frame_run_and_main(repo: Path) -> None:
    proc = repo / "data" / "processed"
    assign, init = _assignment_and_init()
    (proc / "splits").mkdir()
    (proc / "qc").mkdir()
    assign.to_parquet(proc / "splits" / "srnd_assignment.parquet")
    init.to_parquet(proc / "qc" / "hydroswot_init.parquet")
    st = _site_table()
    for c in [c for c in NN if c not in st]:
        st[c] = 1.0
    st.to_parquet(proc / "site_table.parquet")
    sf.main([])
    out = sf.srnd_frame_path(proc)
    assert len(read_parquet(out)) == 12
    meta = read_parquet_meta(out)
    assert meta["stage"] == "srnd_frame"
    assert set(meta["inputs"]) == {
        "splits/srnd_assignment.parquet",
        "qc/hydroswot_init.parquet",
        "site_table.parquet",
    }
    report = json.loads((repo / "reports" / "srnd_frame.json").read_text())
    assert report["rows"] == 12
    assert not {"W", "dbar", "A", "dmax"} & {k for k in report if not isinstance(report[k], int)}


def test_ind_features_run_and_main(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc = repo / "data" / "processed"
    (proc / "ind").mkdir()
    (proc / "unit_mapped").mkdir()
    ind = _ind().assign(vpus=["01", "01", "02", "03"], W=1.0, dbar=1.0, A=1.0)
    ind.to_parquet(proc / "ind" / "ehydro_ind.parquet")
    comids = [11, 21, 22, 31, 32, 33]
    _erom(comids).to_parquet(proc / "unit_mapped" / "nhd_erom.parquet")
    for name in ("nhd_vaa.parquet", "nhd_elevslope.parquet"):
        (proc / "unit_mapped" / name).write_bytes(b"x")
    for unit in ("streamcat", "usgs_attrs", "d50", "aridity", "nhdplusv21", "physiography"):
        (repo / "data" / "raw" / unit).mkdir(parents=True)
        (repo / "data" / "raw" / unit / "MANIFEST.sha256").write_text(unit)
    seen: dict[str, Any] = {}

    def fake_predictors(settings: Any, wanted: set[int]) -> tuple[pd.DataFrame, dict[str, int]]:
        seen["comids"] = wanted
        return _per_comid(sorted(wanted)), {"comids": len(wanted)}

    def fake_lengths(nhd: Path, vpus: list[str], wanted: set[int]) -> dict[int, float]:
        seen["vpus"] = vpus
        return dict.fromkeys(wanted, 1.5)

    def fake_divisions(settings: Any, vpus: list[str], wanted: set[int]) -> pd.Series:
        return pd.Series(dict.fromkeys(wanted, "A"), dtype="string")

    monkeypatch.setattr("src.data.features._per_site_predictors", fake_predictors)
    monkeypatch.setattr(inf, "_flowline_lengths", fake_lengths)
    monkeypatch.setattr(inf, "_divisions", fake_divisions)
    inf.main([])
    assert seen["comids"] == set(comids)
    assert seen["vpus"] == ["01", "02"]
    table = read_parquet(inf.ind_features_path(proc))
    assert len(table) == 6
    assert not {"W", "dbar", "A"} & set(table.columns)
    assert read_parquet_meta(inf.ind_features_path(proc))["stage"] == "ind_features"
    report = json.loads((repo / "reports" / "ind_features.json").read_text())
    assert report["locations_kept"] == 3
    assert report["locations_in_support"] == 3
    assert report["ai"] == {"comids": 6}


def test_flowline_lengths_and_divisions_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw readers are replaced; the selection by VPU and COMID is what is tested."""
    files = {"01": tmp_path / "a.dbf", "02": tmp_path / "b.dbf"}
    tables = {
        files["01"]: pd.DataFrame({"COMID": [11, 12], "LENGTHKM": [0.5, 9.0]}),
        files["02"]: pd.DataFrame({"COMID": [21], "LENGTHKM": [2.5]}),
    }
    monkeypatch.setattr("src.data.nhdplus.find_table_files", lambda unit, table: files)
    monkeypatch.setattr("src.data.readers.read_dbf", lambda path, fields: tables[path])
    got = inf._flowline_lengths(tmp_path, ["01", "02"], {11, 21})
    assert got == {11: 0.5, 21: 2.5}
    layers = {
        "01": gpd.GeoDataFrame(
            {"comid": [11, 12]},
            geometry=[box(-5e4, 1e4, -4e4, 2e4), box(0, 0, 1, 1)],
            crs="EPSG:5070",
        ),
        "02": gpd.GeoDataFrame(
            {"comid": [21]}, geometry=[box(1e4, 1e4, 2e4, 2e4)], crs="EPSG:5070"
        ),
    }
    monkeypatch.setattr(
        "src.data.nhdplus.read_layer", lambda unit, layer, vpu, crs="EPSG:5070": layers[vpu]
    )
    monkeypatch.setattr("src.data.qc.load_divisions", lambda path: _polygons())
    settings = load_settings()
    div = inf._divisions(settings, ["01", "02"], {11, 21})
    assert div.to_dict() == {11: "WEST", 21: "EAST"}
    layers["02"] = pd.concat([layers["02"], layers["02"]], ignore_index=True)
    with pytest.raises(ValueError, match="more than one catchment"):
        inf._divisions(settings, ["01", "02"], {11, 21})
