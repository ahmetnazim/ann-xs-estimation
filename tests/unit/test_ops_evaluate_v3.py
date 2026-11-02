"""Unit tests for OPS-1 v3 evaluation functions (src/operational/evaluate_v3.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import BootstrapParams
from src.operational.config_v3 import load_operational_v3
from src.operational.evaluate import anchor_geometry, site_bin_table
from src.operational.evaluate_v3 import (
    HYBRID_MODEL,
    ORACLE_MODEL,
    evaluate_cche1d_readiness,
    evaluate_dynamic_roughness_and_oracle,
    evaluate_off_anchor_states,
    load_ablation_predictions,
    score_all_models_v2,
)

pytestmark = pytest.mark.unit

S3 = load_operational_v3()
P3, P2, P1 = S3.params, S3.v2.params, S3.v1.params
BOOT = BootstrapParams(b=30, alpha_b=0.05, seed=7, stratify="division")


def test_score_all_models_v2_stratification_and_h_v3_1() -> None:
    """score_all_models_v2 stratifies by not_F_S and F_S and evaluates H-V3-1 and cliques."""
    rng = np.random.default_rng(11)
    sites = [f"s{i}" for i in range(20)]
    fs_map = {s: (i >= 15) for i, s in enumerate(sites)}  # 15 not_F_S, 5 F_S
    div_map = {s: ("d1" if i % 2 == 0 else "d2") for i, s in enumerate(sites)}
    models = ["M0", "M2-XGB", "M3", "M5"]
    q = np.array([0.3, 1.0, 3.0, 12.0])

    frames = []
    for v in ("control", "floodplain"):
        for m in models:
            rows = []
            for s in sites:
                is_fs = fs_map[s]
                if m == "M5":
                    sc = 2.5 if is_fs else 0.45
                elif m == "M2-XGB":
                    sc = 0.6
                elif m == "M3":
                    sc = 0.55
                else:
                    sc = 0.8
                if v == "control":
                    sc *= 1.3
                e = sc * (1.0 + 0.05 * rng.standard_normal(q.size))
                rows.append(pd.DataFrame({"repeat": 0, "site_no": s, "q_ratio": q, "e_h": e}))
            resp = pd.concat(rows, ignore_index=True).assign(e_w=np.nan)
            sb = site_bin_table(resp, P1.metrics.flow_bin_edges, 2.0)
            frames.append(sb.assign(variant=v, family="S-ID", model=m))

    bins_df = pd.concat(frames, ignore_index=True)
    res = score_all_models_v2(
        "S-ID", bins_df, fs_map, div_map, models, P1.metrics.flow_bin_edges, P3, BOOT
    )
    assert res["n_sites_total"] == 20
    assert res["n_sites_not_fs"] == 15
    assert res["n_sites_fs"] == 5
    h1 = res["H-V3-1"]
    assert h1["verdict"].startswith("supported")
    assert h1["delta_rmse_not_fs_m"] < 0.0
    assert h1["delta_rmse_fs_m"] >= 1.0
    assert len(res["cliques"]) > 0
    assert len(res["pairwise_contrasts"]) > 0


def test_dynamic_roughness_and_off_anchor_states(tmp_path: Path) -> None:
    """evaluate_dynamic_roughness_and_oracle and evaluate_off_anchor_states run end-to-end."""
    sites = [f"s{i}" for i in range(12)]
    prim = pd.DataFrame(
        {
            "site_no": sites,
            "W": np.full(12, 40.0),
            "dbar": np.full(12, 1.6),
            "dmax": np.full(12, 2.4),
            "A": np.full(12, 64.0),
            "V": np.full(12, 1.0),
            "F_S": [i >= 9 for i in range(12)],
            "division": ["east" if i % 2 == 0 else "west" for i in range(12)],
        }
    )
    # Synthetic observations with realistic Leopold-Maddock scaling (f_dmax ~ 0.30, gamma ~ -0.32)
    obs_rows = []
    unit_rows = []
    for i, s in enumerate(sites):
        div = "east" if i % 2 == 0 else "west"
        for k, qr in enumerate([0.4, 1.2, 3.0, 8.0]):
            lqr = float(np.log(qr))
            mid = f"{s}_m{k}"
            dmax_i = 2.4 * (qr**0.30)
            dbar_i = 1.6 * (qr**0.30)
            w_i = 40.0 * (qr**0.16)
            a_i = w_i * dbar_i
            v_i = (64.0 * qr) / a_i
            obs_rows.append(
                {
                    "site_no": s,
                    "meas_id": mid,
                    "q_m3s": 64.0 * qr,
                    "q_a_m3s": 64.0,
                    "q_ratio": qr,
                    "ln_q_ratio": lqr,
                    "dh_obs": dmax_i - 2.4,
                    "dlnw_obs": float(np.log(w_i / 40.0)),
                    "division": div,
                }
            )
            unit_rows.append(
                {
                    "site_no": s,
                    "meas_id": mid,
                    "w_m": w_i,
                    "dbar_m": dbar_i,
                    "dmax_m": dmax_i,
                    "a_m2": a_i,
                    "v_ms": v_i,
                }
            )
    obs = pd.DataFrame(obs_rows)
    unit = pd.DataFrame(unit_rows)

    def _geom(dmax_mult: float) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "repeat": 0,
                "site_no": sites,
                "mu_W": np.log(40.0),
                "mu_dbar": np.log(1.6 * dmax_mult),
                "mu_dmax": np.log(2.4 * dmax_mult),
                "mu_t": np.log(1.5),
            }
        )
        return anchor_geometry(df, None, (0.1, 100.0))

    geoms_main = {
        "M0": _geom(0.95),
        "M2-XGB": _geom(1.00),  # accurate anchor geometry
        "M3": _geom(0.44),  # severe depth shrinkage (-56%, ln bias -0.82)
        "M5": _geom(1.30),  # inflated depth
    }
    geoms_abl = {"M5_A3-SR": _geom(0.62)}

    res_dyn = evaluate_dynamic_roughness_and_oracle(
        obs,
        unit,
        prim,
        geoms_main,
        geoms_abl,
        P1,
        P3,
        (0.1, 100.0),
        quad_level=5,
        quad_trunc=4.0,
        boot=BOOT,
    )
    assert res_dyn["leopold_maddock_exponents"]["gamma_roughness_pooled"] < -0.20
    assert res_dyn["H-V3-2"]["verdict"] == "supported"
    tab_dyn = {r["model"]: r for r in res_dyn["table"]}
    assert tab_dyn[ORACLE_MODEL]["bias_const_n_m"] > 0.50
    assert abs(tab_dyn[ORACLE_MODEL]["bias_dyn_n_m"]) < 0.15

    res_off = evaluate_off_anchor_states(
        obs,
        unit,
        prim,
        geoms_main,
        P1,
        (0.1, 100.0),
        quad_level=5,
        quad_trunc=4.0,
        boot=BOOT,
    )
    assert res_off["H-V3-3"]["verdict"] == "supported"

    # Also test load_ablation_predictions and evaluate_cche1d_readiness on synthetic parquets
    for m in P1.predictions.models:
        rows_p = []
        for sp in P3.cche1d.splits:
            err = (
                0.05
                if (m == "M1" and sp.startswith("S-TAIL"))
                else (0.08 if m == "M2-XGB" else 0.2)
            )
            mu_dbar = np.log(1.6) + err
            mu_w = np.log(40.0)
            mu_a = mu_w + mu_dbar + (0.20 if m == "M2-XGB" else 0.0)
            for s in sites:
                rows_p.append(
                    {
                        "analysis": "primary",
                        "family": sp,
                        "unit": "site",
                        "ablation": "",
                        "repeat": 0,
                        "site_no": s,
                        "mu_W": mu_w,
                        "mu_dbar": mu_dbar,
                        "mu_A": mu_a,
                        "mu_dmax": mu_dbar + np.log(1.5),
                        "mu_t": np.log(1.5),
                    }
                )
        if m == "M5":
            for s in sites:
                rows_p.append(
                    {
                        "analysis": "primary",
                        "family": "S-ID",
                        "unit": "site",
                        "ablation": "M5_A3-SR",
                        "repeat": 0,
                        "site_no": s,
                        "mu_W": np.log(40.0),
                        "mu_dbar": np.log(1.6),
                        "mu_A": np.log(64.0),
                        "mu_dmax": np.log(2.4),
                        "mu_t": np.log(1.5),
                    }
                )
        pd.DataFrame(rows_p).to_parquet(tmp_path / f"{m}.parquet", index=False)

    abl_df = load_ablation_predictions(tmp_path, "M5_A3-SR", "S-ID")
    assert len(abl_df) == 12

    res_cche = evaluate_cche1d_readiness(
        tmp_path,
        prim,
        P1,
        P3,
        (0.1, 100.0),
        quad_level=5,
        quad_trunc=4.0,
        boot=BOOT,
    )
    assert res_cche["H-V3-4"]["verdict"] == "supported"
    sid_models = {r["model"]: r for r in res_cche["splits"]["S-ID"]}
    assert sid_models[HYBRID_MODEL]["selected_model"] == "M2-XGB"
    assert sid_models["M2-XGB"]["continuity_violation_cons"] == 0.0
