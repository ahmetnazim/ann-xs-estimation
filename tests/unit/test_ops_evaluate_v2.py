"""Unit tests for OPS-1 v2 scoring (src/operational/evaluate_v2.py). Synthetic; no network."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import BootstrapParams
from src.operational.config_v2 import load_operational_v2
from src.operational.evaluate import site_bin_table
from src.operational.evaluate_v2 import (
    SiteTask,
    all_responses,
    filter_truncated,
    load_terrain,
    score_v2,
    site_responses,
)
from src.operational.floodplain import control_stage_change, prepare_terrain

pytestmark = pytest.mark.unit

S2 = load_operational_v2()
P2, P1 = S2.params, S2.v1.params
BOOT = BootstrapParams(b=40, alpha_b=0.05, seed=3, stratify="division")


def _record(site: str, z: list[float | None], status: str = "ok", shift: float = 0.0) -> dict:
    from pyproj import Transformer

    to_g = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    n = len(z)
    xs = np.linspace(0.0, 400.0, n)
    lon, lat = to_g.transform(xs, np.zeros(n))
    return {
        "site_no": site,
        "status": status,
        "geometry": {"cx": 200.0, "cy": 0.0, "nx": 1.0, "ny": 0.0, "length": 400.0},
        "datum_shift_m": shift,
        "samples": {"lon": list(lon), "lat": list(lat), "z": z},
    }


def _valley(n: int = 101, wse: float = 100.0) -> list[float | None]:
    x = np.linspace(0.0, 400.0, n)
    z = np.where(np.abs(x - 200.0) < 15, wse - 0.2, wse + 1.0 + 0.02 * np.abs(x - 200.0))
    return [float(v) for v in z]


def test_load_terrain_exclusions(tmp_path: Path) -> None:
    recs = {
        "a": _record("a", _valley()),
        "b": _record("b", _valley(), status="excluded_datum"),
        "c": _record("c", [None] * 50 + _valley(51)),
        "d": _record("d", _valley(wse=120.0)),
        "e": _record("e", _valley(), shift=-0.3),
    }
    for k, r in recs.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(r))
    anchors = pd.DataFrame({"site_no": ["a", "b", "c", "d", "e", "f"], "w_a": 20.0, "wse_a": 100.0})
    ter, counts = load_terrain(tmp_path, anchors, P2)
    assert set(ter) == {"a", "e"}
    assert counts["excluded_datum"] == 1
    assert counts["excluded_nodata"] == 1
    assert counts["excluded_anchor_dem"] == 1
    assert counts["excluded_not_acquired"] == 1
    assert ter["e"][1] == pytest.approx(99.7)
    assert ter["a"][0].x_c == pytest.approx(200.0, abs=5.0)


def test_filter_truncated() -> None:
    t = prepare_terrain(np.array([0.0, 1.0, 2.0]), np.array([105.0, 99.0, 103.0]), 1.0, 1.0, 0.05)
    obs = pd.DataFrame({"site_no": ["a", "a", "b"], "dh_obs": [1.0, 5.0, 0.0]})
    out, counts = filter_truncated(obs, {"a": (t, 100.0)})
    assert out["dh_obs"].tolist() == [1.0]
    assert counts["rows_excluded_truncated"] == 1
    assert counts["rows_at_usable_sites"] == 2


def _task(terrain_z: float) -> SiteTask:
    xt = np.linspace(-500, 500, 501)
    ter = prepare_terrain(xt, np.full(501, terrain_z), 0.0, 50.0, 0.05)
    ln_q = np.log(np.array([0.5, 1.0, 3.0, 10.0]))
    return SiteTask(
        site_no="s",
        terrain=ter,
        wse=100.0,
        ln_q=ln_q,
        keys=(("S-ID", "M0", 0), ("S-ID", "M5", 0)),
        w0=np.array([30.0, 30.0]),
        dmax0=np.array([1.5, 2.0]),
        r=np.array([2.0, 1.5]),
        channel_points=2001,
        z_levels=400,
    )


def test_site_responses_walls_match_control() -> None:
    """With terrain far above the stages, floodplain == control (closed form)."""
    s, ctrl, fp, tr = site_responses(_task(1000.0))
    assert s == "s"
    for k, (d0, r) in enumerate([(1.5, 2.0), (2.0, 1.5)]):
        assert ctrl[k] == pytest.approx(control_stage_change(d0, r, _task(1000.0).ln_q))
        assert fp[k] == pytest.approx(ctrl[k], abs=5e-3)
    assert not tr.any()
    res = all_responses([_task(1000.0)], workers=1)
    assert np.array_equal(res[0][1], ctrl)


def _tables(seed: int, scale: dict[tuple[str, str], float]) -> dict[tuple[str, str], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sites = [f"s{i}" for i in range(30)]
    q = np.array([0.4, 1.2, 3.0, 7.0, 15.0])
    out = {}
    for key, sc in scale.items():
        rows = []
        for s in sites:
            e = sc * (1.0 + 0.1 * rng.standard_normal(q.size)) * np.log1p(q)
            rows.append(pd.DataFrame({"repeat": 0, "site_no": s, "q_ratio": q, "e_h": e}))
        resp = pd.concat(rows, ignore_index=True).assign(e_w=np.nan)
        out[key] = site_bin_table(resp, P1.metrics.flow_bin_edges, 2.0)
    return out


def test_score_v2_verdicts() -> None:
    """Floodplain halves every model's error; M5 = M2-XGB; M3 = M0 under floodplain."""
    models = ["M0", "M2-XGB", "M3", "M5"]
    base = {"M0": 1.0, "M2-XGB": 0.9, "M3": 0.6, "M5": 0.9}
    scale = {("control", m): v for m, v in base.items()}
    scale |= {("floodplain", m): 0.5 for m in models}
    tables = _tables(0, scale)
    common = {f"s{i}" for i in range(30)}
    div = {s: ("e" if int(s[1:]) % 2 else "w") for s in common}
    geo = {m: np.full(30, 0.1 * (k + 1)) for k, m in enumerate(models)}
    res = score_v2(
        "S-ID", tables, common, div, geo, models, [0], P1.metrics.flow_bin_edges, P2, BOOT
    )
    h = res["hypotheses"]
    assert h["H-FP1"]["verdict"] == "supported"
    assert h["H-FP2"]["verdict"].startswith("supported")
    assert h["H-FP3"]["verdict"] == "compensating-error diagnosis supported"
    assert set(h["H-FP4"]["spearman_rho"]) == {"control", "floodplain"}
    assert res["n_common_sites"] == 30
    tab = pd.DataFrame(res["table"])
    m0 = tab.loc[tab["model"] == "M0"]
    assert np.allclose(m0["skill_h_vs_M0"], 0.0)
    assert res["bootstrap"]["b"] == 40
    assert math.isfinite(res["ranking"]["control"][0]["rmse_h_high_m"])


def test_score_v2_falsified_paths() -> None:
    models = ["M0", "M2-XGB", "M3", "M5"]
    scale = {("control", m): 1.0 for m in models}
    scale |= {
        ("floodplain", "M0"): 1.5,
        ("floodplain", "M2-XGB"): 0.5,
        ("floodplain", "M3"): 0.3,
        ("floodplain", "M5"): 1.2,
    }
    tables = _tables(1, scale)
    common = {f"s{i}" for i in range(30)}
    div = dict.fromkeys(common, "x")
    geo = {m: np.zeros(30) + k for k, m in enumerate(models)}
    res = score_v2(
        "S-ID", tables, common, div, geo, models, [0], P1.metrics.flow_bin_edges, P2, BOOT
    )
    h = res["hypotheses"]
    assert h["H-FP1"]["verdict"] == "falsified"
    assert h["H-FP2"]["verdict"] == "falsified"
    assert h["H-FP3"]["verdict"].startswith("not supported")
