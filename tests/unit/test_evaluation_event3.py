"""Tests for Event 3 (docs/3 13.4, 13.13; Batch C3c). Synthetic only.

Sources: docs/3 section 13.13 (Event 3 computation rules: evaluation rows, S-ID statistics = mean
over repeats, strata and cutpoints, bootstrap designs, H1-H4, S2, far / not-nested strata, UQ,
consistency, AOA, partner tie rule, S1, S-RND weights), sections 13.5 (S-IND) and 13.6 (S-RND);
protocol 4.3-4.7; proofs 9.5 (paired and two-sample cluster bootstrap).

Expected values are planted or written out by hand: model errors are ``+/- c`` with a fixed sign
pattern, so RMSE_log = c exactly and every Delta RMSE is a difference of constants; the vector
bootstrap is compared with the Phase 7 scalar bootstrap; nested partners are compared with the
Phase 7 ``nested_status``. No real table, prediction or target is read.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.config import PARAMS_PATH, load_settings
from src.data.folds import analysis_frame
from src.data.network import NetworkInputs
from src.evaluation import bootstrap as bs
from src.evaluation import dependence_audit as da
from src.evaluation import event3
from src.evaluation import event3_sets as es
from src.evaluation import event3_stats as st
from src.evaluation import event3_workers as ew
from src.evaluation import hypotheses as hy
from synthetic_sites import site_frame

pytestmark = pytest.mark.unit

SETTINGS = load_settings()
PROTOCOL = SETTINGS.protocol
CFG = PROTOCOL.evaluation.bootstrap
N_BOOT = 25
N = 40
#: Planted |error| per model in-distribution (S-ID) and out of distribution (other families).
C_ID = {"M5": 0.10, "M2-XGB": 0.20, "M6-NN": 0.12, "M4": 0.18}
C_OOD = {"M5": 0.10, "M2-XGB": 0.30, "M6-NN": 0.12, "M4": 0.18}
C_OTHER = 0.15
Q_HALF = 0.33  # half-width of the planted stored interval (log units)
LABEL = "optimistic under spatial/network dependence"


def _sign(i: int, r: int) -> float:
    return 1.0 if (i + r) % 2 == 0 else -1.0


# ----------------------------------------------------------------------------- synthetic world
@dataclass
class World:
    site_table: pd.DataFrame
    folds: pd.DataFrame
    ehydro: pd.DataFrame
    ind: pd.DataFrame
    srnd: pd.DataFrame
    network: NetworkInputs
    pathlength: dict[int, float]
    levelpath: dict[int, int]
    nu: np.ndarray
    r: np.ndarray


def _site_table() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    base = site_frame(N, seed=7)
    base["comid"] = 1000 + np.arange(N)
    base["x5070"] = rng.uniform(0.0, 3.0e5, N)
    base["y5070"] = rng.uniform(0.0, 3.0e5, N)
    base["row_id"] = 10 * np.arange(N)
    base["Q_obs"] = base["Q_E"] * np.exp(rng.normal(0.0, 0.1, N))
    base["V"] = base["Q_obs"] / base["A"]
    base["eps_A"] = rng.normal(0.0, 0.02, N)
    return pd.concat(
        [base, base.assign(variant="sa2"), base.assign(variant="sa3")], ignore_index=True
    )


def _folds(site_nos: Sequence[str]) -> pd.DataFrame:
    rows: list[tuple[str, int, int, str, bool]] = []
    for r in range(3):
        for f in range(5):
            rows += [("S-ID", r, f, s, (i + r) % 5 == f) for i, s in enumerate(site_nos)]
    for f in range(4):
        rows += [("S-LRO", 0, f, s, i % 4 == f) for i, s in enumerate(site_nos)]
    rows += [("S-TAIL-DA", 0, 0, s, i % 5 == 1) for i, s in enumerate(site_nos)]
    rows += [("S-TAIL-SO", 0, 0, s, i % 5 == 2) for i, s in enumerate(site_nos)]
    rows += [("S-IND", 0, 0, s, False) for s in site_nos]
    out = pd.DataFrame(rows, columns=["family", "repeat", "outer_fold", "site_no", "is_test"])
    out["inner_fold"] = np.where(out["is_test"], -1, np.arange(len(out)) % 3).astype("int8")
    for m in (1, 2):
        out[f"es_member_{m}"] = ~out["is_test"] & (np.arange(len(out)) % 4 == m)
    return out


def _ind_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    loc_rows, feat_rows = [], []
    for j in range(12):
        loc = f"L{j:02d}"
        comids = [5000 + j] + ([6000 + j] if j % 4 == 0 else [])
        w, d = 50.0 + 5.0 * j, 2.0 + 0.1 * j
        loc_rows.append(
            {
                "loc_id": loc,
                "river": f"R{j % 3}",
                "comid_outlet": 5000 + j,
                "n_comids": len(comids),
                "W": w,
                "dbar": d,
                "A": w * d,
                "DA_km2": 100.0 * (j + 1),
                "dedupe_status": "kept",
            }
        )
        for k, c in enumerate(comids):
            feat_rows.append(
                {
                    "loc_id": loc,
                    "river": f"R{j % 3}",
                    "n_comids": len(comids),
                    "member": k,
                    "comid": c,
                    "division": None if j == 5 else ("A" if j % 2 else "B"),
                    "stream_order": 3.0,
                    "loc_in_support": True,
                }
            )
    loc_rows.append({**loc_rows[0], "loc_id": "L99", "dedupe_status": "duplicate"})
    return pd.DataFrame(loc_rows), pd.DataFrame(feat_rows)


def _srnd(site_table: pd.DataFrame) -> pd.DataFrame:
    prim = analysis_frame(site_table, "primary")
    rows = []
    for k in range(30):
        s = prim.iloc[k // 3]
        rows.append(
            {
                "row_id": 9000 + k,
                "site_no": s["site_no"],
                "is_test": k % 3 != 0,
                "W": float(s["W"]) * 1.1,
                "dbar": float(s["dbar"]) * 0.9,
                "A": float(s["A"]),
                "dmax": float(s["dmax"]),
                "division": s["division"],
                "stream_order": s["stream_order"],
                "DA_km2": s["DA_km2"],
            }
        )
    return pd.DataFrame(rows)


def _network(comids: Sequence[int]) -> tuple[NetworkInputs, dict[int, float], dict[int, int]]:
    edges, area = [], {}
    for j in range(len(comids) // 2):
        a, b = comids[2 * j], comids[2 * j + 1]
        p = 0.8 if j % 2 == 0 else 0.2
        edges.append((a, b))
        area[a], area[b] = 100.0, 100.0 / p
    net = NetworkInputs(
        edges=np.asarray(edges, dtype=np.int64),
        terminal_path=dict.fromkeys(comids, 7),
        drainage_km2=area,
        missing=(),
    )
    return net, {c: float(c % 13) for c in comids}, {c: c // 4 for c in comids}


def world() -> World:
    st_ = _site_table()
    prim = analysis_frame(st_, "primary")
    ehydro, ind = _ind_tables()
    net, pl, lp = _network(prim["comid"].astype(int).tolist())
    rng = np.random.default_rng(4)
    return World(
        site_table=st_,
        folds=_folds(prim["site_no"].tolist()),
        ehydro=ehydro,
        ind=ind,
        srnd=_srnd(st_),
        network=net,
        pathlength=pl,
        levelpath=lp,
        nu=np.log(rng.uniform(0.02, 0.08, N)),
        r=rng.uniform(1.0, 5.0, N),
    )


# ----------------------------------------------------------------------------- predictions
def _members(rec: dict[str, Any]) -> None:
    for k in ("W", "dbar", "t", "A"):
        mu = rec[f"mu_{k}"]
        for m, off in ((1, 0.01), (2, -0.01)):
            rec[f"m{m:02d}_mu_{k}"] = mu + off
            rec[f"m{m:02d}_sigma_{k}"] = 0.2
        rec[f"q_lo_{k}"] = mu - Q_HALF
        rec[f"q_hi_{k}"] = mu + Q_HALF


def _key(model: str, abl: str, analysis: str, family: str, unit: str) -> dict[str, Any]:
    return {
        "spec": f"{model}_{abl}" if abl else model,
        "model": model,
        "ablation": abl,
        "analysis": analysis,
        "family": family,
        "unit": unit,
    }


def _site_preds(w: World, model: str, abl: str, analysis: str) -> list[dict[str, Any]]:
    frame = analysis_frame(w.site_table, analysis).set_index("site_no")
    out = []
    for family in es.SITE_FAMILIES:
        c = (C_ID if family == "S-ID" else C_OOD).get(model, C_OTHER)
        sel = (
            w.folds["family"].eq(family) & w.folds["is_test"] & w.folds["site_no"].isin(frame.index)
        )
        for s, r, f in w.folds.loc[sel, ["site_no", "repeat", "outer_fold"]].itertuples(
            index=False
        ):
            i, row = int(s), frame.loc[s]
            e = c * _sign(i, int(r))
            rec = {**_key(model, abl, analysis, family, "site"), "repeat": int(r)}
            rec.update(outer_fold=int(f), site_no=s, comid=int(row["comid"]))
            if model == "A_J":
                rec["mu_A"] = math.nan if i % 10 == 3 else math.log(row["A"]) + e
                out.append(rec)
                continue
            for k in ("W", "dbar", "A", "dmax"):
                rec[f"mu_{k}"] = math.log(float(row[k])) + e
            rec["mu_t"] = float(row["t_obs"]) + 0.05 * _sign(i, int(r))
            rec.update(
                nu_mean=float(w.nu[i]) + 0.01 * int(r),
                r_mean=float(w.r[i]),
                nu_back=float(w.nu[i]),
                inv4_violation=i == 7,
                resid_mean=1.0e-3 * _sign(i, int(r)),
                resid_maxabs=1.0e-12,
                iters_max=4,
            )
            _members(rec)
            out.append(rec)
    return out


def _ind_preds(w: World, model: str, abl: str, analysis: str) -> list[dict[str, Any]]:
    c = C_OOD.get(model, C_OTHER)
    kept = w.ehydro.loc[w.ehydro["dedupe_status"].eq("kept")].reset_index(drop=True)
    out = []
    for j, loc in kept.iterrows():
        e = c * _sign(int(j), 0)
        base = {"repeat": 0, "outer_fold": 0, "loc_id": loc["loc_id"], "river": loc["river"]}
        base["n_comids"] = int(loc["n_comids"])
        rec = {**_key(model, abl, analysis, "S-IND", "location"), **base}
        if model == "A_J":
            rec["mu_A"] = math.nan if j == 0 else math.log(loc["A"]) + e
        else:
            for k in ("W", "dbar", "A"):
                rec[f"mu_{k}"] = math.log(float(loc[k])) + e
        out.append(rec)
        for k2, comid in enumerate(w.ind.loc[w.ind["loc_id"].eq(loc["loc_id"]), "comid"]):
            crec = {**_key(model, abl, analysis, "S-IND", "comid"), **base}
            crec.update(comid=int(comid), member=k2, lengthkm=1.0)
            for k in ("W", "dbar", "A"):
                crec[f"mu_{k}"] = math.log(float(loc[k])) + e
            crec["mu_t"] = 0.5
            _members(crec)
            out.append(crec)
    return out


def _srnd_preds(w: World, model: str) -> list[dict[str, Any]]:
    test = w.srnd.loc[w.srnd["is_test"]].reset_index(drop=True)
    out = []
    for k, row in test.iterrows():
        rec = {**_key(model, "", "primary", "S-RND", "srnd_row"), "repeat": 0, "outer_fold": 0}
        rec.update(row_id=int(row["row_id"]), site_no=row["site_no"])
        e = 0.1 * _sign(int(k), 0)
        for t in ("W", "dbar", "A", "dmax"):
            rec[f"mu_{t}"] = math.log(float(row[t])) + e
        if model == "M1" and k == 0:
            rec["mu_W"] = math.nan  # a row missing an M1 predictor
        out.append(rec)
    return out


def predictions(w: World) -> dict[str, pd.DataFrame]:
    files: dict[str, list[dict[str, Any]]] = {}
    for model in event3.LADDER:
        analyses = ("primary", *event3.SA_MODELS.get(model, ()))
        recs = []
        for analysis in analyses:
            recs += _site_preds(w, model, "", analysis) + _ind_preds(w, model, "", analysis)
        if model in ("M1", "M2-XGB"):
            recs += _srnd_preds(w, model)
        files[model] = recs
    for abl in event3.M5_ABLATIONS:
        files[f"M5_{abl}"] = _site_preds(w, "M5", abl, "primary") + _ind_preds(
            w, "M5", abl, "primary"
        )
    for model in event3.A4_MODELS:
        files[f"{model}_A4"] = _site_preds(w, model, "A4", "primary") + _ind_preds(
            w, model, "A4", "primary"
        )
    return {k: pd.DataFrame(v) for k, v in files.items()}


def audit_strata(w: World) -> pd.DataFrame:
    """Event 1 strata from the Phase 7 ``nested_status`` (the reference for nested partners)."""
    frame = analysis_frame(w.site_table, "primary")
    comids = frame["comid"].to_numpy(dtype=np.int64)
    parts = []
    for r in range(3):
        for f in range(5):
            blk = w.folds.loc[
                w.folds["family"].eq("S-ID") & w.folds["repeat"].eq(r) & w.folds["outer_fold"].eq(f)
            ].set_index("site_no")
            test = blk.reindex(frame["site_no"])["is_test"].to_numpy(dtype=bool)
            res = da.nested_status(
                comids[~test].tolist(),
                comids[test].tolist(),
                w.network.edges,
                w.network.terminal_path,
                w.network.drainage_km2,
                PROTOCOL.audit,
            )
            idx = np.flatnonzero(test)
            parts.append(
                pd.DataFrame(
                    {
                        "family": "S-ID",
                        "repeat": np.int8(r),
                        "outer_fold": np.int8(f),
                        "site_no": frame["site_no"].to_numpy()[idx],
                        "comid": comids[idx],
                        "far": idx % 3 == 0,  # site-constant (C3c-k needs rows in every repeat)
                        "psi_max": res.psi_max,
                        "nested": np.where(
                            res.strong, "strong", np.where(res.weak, "weak", "none")
                        ),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def inputs(w: World) -> event3.Event3Inputs:
    prim = analysis_frame(w.site_table, "primary")
    qc = pd.DataFrame(
        {
            "row_id": np.arange(0, 10 * N + 50, 5),
        }
    )
    qc["chan_mat_cd"] = np.where(qc["row_id"] % 20 == 0, "SAND", None)
    qc["F_eps"] = qc["row_id"] % 15 == 0
    hyg = pd.DataFrame(
        {"site_no": prim["site_no"].iloc[:12].tolist(), "manning_n": np.linspace(0.02, 0.06, 12)}
    )
    return event3.Event3Inputs(
        preds=predictions(w),
        site_table=w.site_table,
        folds=w.folds,
        ehydro=w.ehydro,
        ind_features=w.ind,
        srnd=w.srnd,
        audit_strata=audit_strata(w),
        audit={"label": {"optimistic": True, "text": LABEL}},
        mstar={"mstar": {"dbar": "M2-XGB", "A": "M2-XGB"}},
        hyg=hyg,
        hyg_counts={"rows": 12, "kept": 12},
        qc_rows=qc,
        network=w.network,
        pathlength=w.pathlength,
        levelpath=w.levelpath,
    )


def fake_log_kappa(ell: np.ndarray, wl: np.ndarray, r: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(ell) - np.asarray(wl)) + 0.1 * np.log(np.asarray(r))


def fake_aoa(w: World):
    def aoa(blocks: Sequence[tuple[str, int, int]]) -> list[dict[str, Any]]:
        out = []
        for fam, r, f in blocks:
            if fam == "S-IND":
                units = [f"{a}|{b}" for a, b in zip(w.ind["loc_id"], w.ind["comid"], strict=True)]
            else:
                sel = (
                    w.folds["family"].eq(fam)
                    & w.folds["repeat"].eq(r)
                    & w.folds["outer_fold"].eq(f)
                    & w.folds["is_test"]
                )
                units = w.folds.loc[sel, "site_no"].tolist()
            di = [0.9 if k % 4 == 0 else 0.1 for k in range(len(units))]
            out.append(
                {
                    "family": fam,
                    "repeat": r,
                    "outer_fold": f,
                    "threshold": 0.5,
                    "di": di,
                    "units": units,
                    "features": ["a", "b"],
                    "importance": [0.6, 0.4],
                    "n_train": 30,
                }
            )
        return out

    return aoa


@pytest.fixture(scope="module")
def result() -> tuple[World, event3.Event3Inputs, event3.Event3Result]:
    w = world()
    inp = inputs(w)
    res = event3.run_event3(inp, PROTOCOL, log_kappa=fake_log_kappa, aoa=fake_aoa(w), n_boot=N_BOOT)
    return w, inp, res


# ----------------------------------------------------------------------------- bootstrap
def test_bootstrap_matrix_columns_equal_the_scalar_bootstraps() -> None:
    """docs/3 13.13 e: one draw sequence for all statistics = the Phase 7 scalar bootstraps."""
    rng = np.random.default_rng(0)
    e = rng.normal(size=30)
    clusters = np.repeat(np.arange(10), 3).astype(str)
    strata = np.where(np.arange(30) < 15, "A", "B")

    def f1(idx: np.ndarray, _g: np.ndarray) -> float:
        return float(np.mean(e[idx]))

    def f2(idx: np.ndarray, _g: np.ndarray) -> float:
        return float(np.sqrt(np.mean(e[idx] ** 2)))

    mat = bs.bootstrap_matrix(
        lambda i, g: [f1(i, g), f2(i, g)], clusters, strata, CFG, n_boot=N_BOOT
    )
    assert mat.shape == (N_BOOT, 2)
    for k, f in enumerate((f1, f2)):
        ref = bs.paired_cluster_bootstrap(f, clusters, strata, CFG, n_boot=N_BOOT)
        assert np.array_equal(mat[:, k], ref)
    eb = rng.normal(size=12)
    cb = np.repeat(np.arange(4), 3).astype(str)

    def g1(ia: np.ndarray, _ga: np.ndarray, ib: np.ndarray, _gb: np.ndarray) -> float:
        return float(np.mean(e[ia]) - np.mean(eb[ib]))

    two = bs.two_sample_bootstrap_matrix(
        lambda *a: [g1(*a), 2.0 * g1(*a)], clusters, strata, cb, None, CFG, n_boot=N_BOOT
    )
    ref2 = bs.two_sample_cluster_bootstrap(g1, clusters, strata, cb, None, CFG, n_boot=N_BOOT)
    assert np.array_equal(two[:, 0], ref2)
    assert np.allclose(two[:, 1], 2.0 * ref2, rtol=0, atol=0)
    with pytest.raises(ValueError, match="n_boot"):
        bs.bootstrap_matrix(lambda i, g: [0.0], clusters, None, CFG, n_boot=0)
    with pytest.raises(ValueError, match="n_boot"):
        bs.two_sample_bootstrap_matrix(lambda *a: [0.0], clusters, None, cb, None, CFG, n_boot=0)
    with pytest.raises(ValueError, match="non-finite"):
        bs.bootstrap_matrix(lambda i, g: [math.nan], clusters, None, CFG, n_boot=2)
    sizes = iter([1, 2])
    with pytest.raises(ValueError, match="shape"):
        bs.bootstrap_matrix(lambda i, g: [0.0] * next(sizes), clusters, None, CFG, n_boot=2)


def test_run_design_point_and_degenerate_ci() -> None:
    """A constant-|e| Delta RMSE has the same value on every draw (CI = point)."""
    e1 = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    e2 = 2.0 * e1
    design = st.Design(cluster=np.array(list("aabbcc")), stratum=None, repeat=np.array([0, 1] * 3))

    def drmse(parts: Sequence[np.ndarray]) -> float:
        return float(
            np.mean(
                [math.sqrt(np.mean(e1[p] ** 2)) - math.sqrt(np.mean(e2[p] ** 2)) for p in parts]
            )
        )

    res = st.run_design(design, [st.BootStat("d", drmse, 0.95)], CFG, n_boot=N_BOOT)
    assert res["d"]["estimate"] == pytest.approx(-0.1, abs=1e-12)
    assert res["d"]["lo"] == pytest.approx(-0.1, abs=1e-12)
    assert res["d"]["hi"] == pytest.approx(-0.1, abs=1e-12)
    with pytest.raises(ValueError, match="no statistics"):
        st.run_design(design, [], CFG)


# ----------------------------------------------------------------------------- sets
def test_da_cuts_and_classes_by_hand() -> None:
    """Protocol 4.3.3: numpy.quantile (linear) cutpoints; class = searchsorted(right) + 1."""
    frame = pd.DataFrame({"DA_km2": np.arange(1.0, 11.0)})
    cuts = es.da_cuts(frame)
    assert cuts == pytest.approx((2.8, 4.6, 6.4, 8.2), abs=1e-12)  # 1 + 9 q
    assert es.da_class(np.array([1.0, 2.8, 2.9, 10.0]), cuts).tolist() == [1, 2, 2, 5]
    with pytest.raises(ValueError, match="positive"):
        es.da_cuts(pd.DataFrame({"DA_km2": [1.0, 2.0, 0.0, 3.0, 4.0]}))
    with pytest.raises(ValueError, match="non-finite"):
        es.da_class(np.array([math.nan]), cuts)


def test_site_rows_refuse_missing_duplicated_misplaced_and_unlabelled() -> None:
    """docs/3 13.13 a: every evaluation row needs exactly one prediction from its own block."""
    w = world()
    cuts = es.da_cuts(analysis_frame(w.site_table, "primary"))
    preds = pd.DataFrame(_site_preds(w, "M5", "", "primary"))
    ok = es.site_rows(preds, w.site_table, w.folds, cuts, model="M5", family="S-ID")
    assert len(ok) == 3 * N
    assert ok["unit_id"].equals(ok["site_no"])
    assert set(ok["boot_stratum"]) == {"A", "B"}
    sid = preds["family"].eq("S-ID")

    def call(p: pd.DataFrame, **kw: Any) -> pd.DataFrame:
        return es.site_rows(p, w.site_table, w.folds, cuts, model="M5", family="S-ID", **kw)

    with pytest.raises(ValueError, match="missing predictions"):
        call(preds.drop(index=preds.index[sid][0]))
    with pytest.raises(ValueError, match="duplicated"):
        call(pd.concat([preds, preds.loc[sid].iloc[:1]]))
    moved = preds.copy()
    k = moved.index[sid][0]
    moved.loc[k, "outer_fold"] = (int(moved.loc[k, "outer_fold"]) + 1) % 5
    with pytest.raises(ValueError, match="outer-test block"):
        call(moved)
    extra = preds.loc[sid].iloc[:1].assign(site_no="99999999")
    with pytest.raises(ValueError, match="outside the evaluated frame"):
        call(pd.concat([preds, extra]))
    with pytest.raises(ValueError, match="null ablation"):
        call(preds.assign(ablation=None))
    strata = audit_strata(w)[["site_no", "repeat", "outer_fold", "far", "nested"]]
    with pytest.raises(ValueError, match="stratum label"):
        call(preds, strata=strata.iloc[1:])
    with pytest.raises(ValueError, match="no S-LRO test rows"):
        es.test_assignment(analysis_frame(w.site_table, "primary"), w.folds.iloc[:0], "S-LRO")
    dup = pd.concat([w.folds, w.folds.loc[w.folds["family"].eq("S-LRO") & w.folds["is_test"]]])
    dup.loc[dup.index.duplicated(keep="first"), "outer_fold"] = 3
    with pytest.raises(ValueError, match="more than once"):
        es.test_assignment(analysis_frame(w.site_table, "primary"), dup, "S-LRO")
    short = w.folds.loc[~(w.folds["family"].eq("S-ID") & w.folds["site_no"].eq("00000003"))]
    with pytest.raises(ValueError, match="once per repeat"):
        es.test_assignment(analysis_frame(w.site_table, "primary"), short, "S-ID")


def test_ind_and_srnd_rows() -> None:
    """docs/3 13.5 / 13.6: kept locations, outlet strata, A_J masking, S-RND weights 1/|I_j|."""
    w = world()
    cuts = es.da_cuts(analysis_frame(w.site_table, "primary"))
    m5 = pd.DataFrame(_ind_preds(w, "M5", "", "primary"))
    rows, dropped = es.ind_rows(m5, w.ehydro, w.ind, cuts, model="M5")
    assert dropped == 0
    assert len(rows) == 12
    assert rows.set_index("unit_id").loc["L05", "division"] == es.NO_STRATUM
    assert set(rows["cluster"]) == {"R0", "R1", "R2"}
    aj = pd.DataFrame(_ind_preds(w, "A_J", "", "primary"))
    rows_aj, dropped_aj = es.ind_rows(aj, w.ehydro, w.ind, cuts, model="A_J")
    assert (len(rows_aj), dropped_aj) == (11, 1)
    loc = m5.loc[m5["unit"].eq("location")]
    with pytest.raises(ValueError, match="duplicated"):
        es.ind_rows(pd.concat([m5, loc.iloc[:1]]), w.ehydro, w.ind, cuts, model="M5")
    with pytest.raises(ValueError, match="not kept"):
        es.ind_rows(
            pd.concat([m5, loc.iloc[:1].assign(loc_id="L99")]), w.ehydro, w.ind, cuts, model="M5"
        )
    m1 = pd.DataFrame(_srnd_preds(w, "M1"))
    srnd, n_drop = es.srnd_rows(m1, w.srnd, cuts, model="M1")
    assert n_drop == 1
    assert len(srnd) == 19
    # site 0 has rows 9001, 9002 in the test split (k % 3 != 0): weight 1/2 each (by hand)
    assert srnd.set_index("row_id").loc[9002, "weight"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="duplicated"):
        es.srnd_rows(pd.concat([m1, m1.iloc[:1]]), w.srnd, cuts, model="M1")
    with pytest.raises(ValueError, match="one to one"):
        es.srnd_rows(m1.iloc[1:], w.srnd, cuts, model="M1")


# ----------------------------------------------------------------------------- statistics
def test_repeat_cells_average_over_repeats_by_hand() -> None:
    """docs/3 13.13 b, d: per-repeat metric, then the mean; undefined -> None; counts."""
    rows = pd.DataFrame(
        {
            "repeat": [0, 0, 1, 1],
            "unit_id": ["a", "b", "a", "b"],
            "mu_dbar": np.log([1.0, 1.0, 1.0, 1.0]) + np.array([0.1, -0.1, 0.3, -0.3]),
            "o_dbar": [1.0, 1.0, 1.0, math.nan],
        }
    )
    cells, counts = st.repeat_cells(rows, "dbar")  # type: ignore[misc]
    # repeat 0: rmse 0.1; repeat 1: one finite row, rmse 0.3 -> mean 0.2 (hand)
    assert cells["rmse_log"] == pytest.approx(0.2, abs=1e-12)
    assert counts == {"n_rows": 3, "n_units": 2, "n_excluded": 1, "n_repeats": 2}
    assert st.repeat_cells(rows, "W") is None
    t = pd.DataFrame(
        {
            "repeat": [0, 0, 0],
            "unit_id": list("abc"),
            "mu_t": [0.2, 0.4, 0.6],
            "o_t": [0.3, 0.5, 0.1],
        }
    )
    tc, _ = st.repeat_cells(t, "t")  # type: ignore[misc]
    assert set(tc) == set(st.T_METRICS)
    assert tc["bias_t"] == pytest.approx((-0.1 - 0.1 + 0.5) / 3, abs=1e-12)
    assert tc["rho_r"] == pytest.approx(spearmanr([0.2, 0.4, 0.6], [0.3, 0.5, 0.1])[0])
    dm = st.cell(np.log([1.0, 2.0]), np.log([1.0, 2.0]), "dmax")
    assert set(dm) == set(st.DMAX_METRICS)
    assert dm["mult_bias"] == pytest.approx(0.0, abs=1e-15)


# ----------------------------------------------------------------------------- nested partners
def test_nested_partner_tie_rule_by_hand() -> None:
    """docs/3 13.13 p: partner = max psi; ties -> smallest |Delta PathLength|, then lowest COMID."""
    # chain 1 -> 2 -> 3 -> 4; test site 3 (DA 100); training sites 1, 2 (DA 50), 4 (DA 200)
    net = NetworkInputs(
        edges=np.array([[1, 2], [2, 3], [3, 4]], dtype=np.int64),
        terminal_path={1: 9, 2: 9, 3: 9, 4: 9, 5: 9},
        drainage_km2={1: 50.0, 2: 50.0, 3: 100.0, 4: 200.0, 5: 10.0},
        missing=(),
    )
    reach = event3._Reach(net, [1, 2, 3, 4, 5])
    pl = {1: 10.0, 2: 8.0, 3: 6.0, 4: 4.0}
    lp = {1: 1, 2: 1, 3: 1, 4: 2}
    tab = event3.nested_partners([1, 2, 4], [3], reach, net, pl, lp, 0.5)
    row = tab.iloc[0]
    # psi: 50/100, 50/100, 100/200 = 0.5 for all; |dPL| = 4, 2, 2 -> tie of 2 and 4 -> COMID 2
    assert row["psi_max"] == 0.5
    assert row["partner"] == 2
    assert row["dpath_km"] == 2.0
    assert bool(row["same_levelpath"])
    assert row["nested"] == "strong"
    other = event3.nested_partners([4], [1], reach, net, pl, lp, 0.5)  # 1 -> 4: psi 0.25
    assert other.iloc[0]["nested"] == "weak"
    none = event3.nested_partners([1], [5], reach, net, pl, lp, 0.5)  # 5 is isolated
    assert none.iloc[0]["nested"] == "none"
    assert none.iloc[0]["partner"] is None


def test_nested_partners_match_phase7_nested_status() -> None:
    """psi_max of every S-ID test instance equals ``nested_status`` (Event 1 strata)."""
    w = world()
    strata = audit_strata(w)
    frame = analysis_frame(w.site_table, "primary")
    reach = event3._Reach(w.network, frame["comid"].astype(int).tolist())
    comids = frame["comid"].to_numpy(dtype=np.int64)
    for (r, f), ref in strata.groupby(["repeat", "outer_fold"]):
        blk = w.folds.loc[
            w.folds["family"].eq("S-ID") & w.folds["repeat"].eq(r) & w.folds["outer_fold"].eq(f)
        ].set_index("site_no")
        test = blk.reindex(frame["site_no"])["is_test"].to_numpy(dtype=bool)
        tab = event3.nested_partners(
            comids[~test].tolist(),
            comids[test].tolist(),
            reach,
            w.network,
            w.pathlength,
            w.levelpath,
            PROTOCOL.audit.nested_psi,
        )
        assert tab["psi_max"].tolist() == ref["psi_max"].tolist()
        assert tab["nested"].tolist() == ref["nested"].tolist()


# ----------------------------------------------------------------------------- end to end
def test_event3_planted_hypothesis_statistics(result: Any) -> None:
    """H1, H3 and S2 from planted constant |errors| (hand values; degenerate CIs)."""
    _, _, res = result
    h = res.documents["hypotheses"]
    assert h["audit_label"] == LABEL
    for k in ("dbar", "A"):
        for fam in ("S-LRO", "S-IND"):
            cell = h["H1"][fam][k]
            assert cell["estimate"] == pytest.approx(0.10 - 0.30, abs=1e-12)
            assert cell["lo"] == pytest.approx(-0.20, abs=1e-12)
            assert cell["hi"] == pytest.approx(-0.20, abs=1e-12)
    assert h["H1"]["S-IND"]["n_locations"] == 12
    assert h["H1"]["S-IND"]["n_clusters"] == 3
    want_h1 = hy.decide_h1(hy.CI(-0.2, -0.2), hy.CI(-0.2, -0.2))
    assert h["H1"]["decision"]["supported"] is want_h1.supported
    # H3: L = -RMSE_ID + RMSE_OOD; M5: -0.1 + 0.1 = 0; M*: -0.2 + 0.3 = 0.1 -> delta -0.1
    for name in ("H3-LRO", "H3-TAIL", "H3-TAIL-SO"):
        assert h["H3"][name]["delta"]["estimate"] == pytest.approx(-0.1, abs=1e-12)
        assert h["H3"][name]["L_Mstar"]["estimate"] == pytest.approx(0.1, abs=1e-12)
    assert h["H3"]["H3-TAIL"]["n_sites"] == 8
    s2 = h["S2"]
    for fam in ("S-ID", "S-LRO"):
        assert s2[fam]["s2_M5_vs_M6NN"]["estimate"] == pytest.approx(-0.02, abs=1e-12)
        assert s2[fam]["s2_M4_vs_M5"]["estimate"] == pytest.approx(0.08, abs=1e-12)
        assert s2[fam]["s2_M4_vs_M5"]["level"] == pytest.approx(1 - 0.05 / 4)
    kq = s2["kappa_quintiles_descriptive"]["S-ID"]
    assert sum(kq[f"Q{q}"]["n_rows"] for q in range(1, 6)) == 3 * N
    rows = h["outcome_4_7"]["rows"]
    assert set(rows) <= {1, 2, 3, 4}
    if rows:
        for k in rows:
            assert event3.OUTCOME_TEXT[k][1] in h["outcome_4_7"]["statement"]
    else:
        assert h["outcome_4_7"]["statement"] == "no row of protocol 4.7 matches"


def test_event3_h2_by_hand(result: Any) -> None:
    """H2 = mean over repeats of |B(M5, Q1)| - |B(M*, Q1)| with B = c * mean(sign) (hand)."""
    w, _, res = result
    frame = analysis_frame(w.site_table, "primary")
    cuts = np.quantile(frame["DA_km2"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    q1 = [int(s) for s, d in zip(frame["site_no"], frame["DA_km2"], strict=True) if d <= cuts[0]]
    b = [np.mean([_sign(i, r) for i in q1]) for r in range(3)]
    want = float(np.mean([abs(0.10 * x) - abs(0.20 * x) for x in b]))
    got = res.documents["hypotheses"]["H2"]["S-ID_Q1"]
    assert got["h2"]["estimate"] == pytest.approx(want, abs=1e-12)
    assert got["n_sites"] == len(q1)


def test_event3_h4_site_level_by_hand(result: Any) -> None:
    """H4 (b): rho_S(n-hat, D50) on site means of nu over repeats (= nu + 0.01 here)."""
    w, _, res = result
    frame = analysis_frame(w.site_table, "primary")
    h4 = res.documents["hypotheses"]["H4"]["S-ID"]
    want = spearmanr(np.exp(w.nu + 0.01), frame["D50_m"].to_numpy())[0]
    assert h4["rho_n_d50"] == pytest.approx(want, abs=1e-12)
    assert h4["n_sites"] == N
    assert set(h4["ci"]) == {"n_d50", "n_s", "r_robs"}
    assert h4["rmse_t"] == pytest.approx(0.05, abs=1e-12)
    caveats = res.documents["hypotheses"]["H4"]["caveats"]
    assert set(caveats["n_hat_by_chan_mat_cd"]) == {"SAND", "missing"}


def test_event3_strata_rule(result: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """docs/3 13.13 k: far / not-nested CIs need STRATUM_MIN_SITES sites in every repeat."""
    w, inp, res = result
    doc = res.documents["hypotheses"]["far_not_nested_strata"]
    assert doc["applies"] is True
    assert "not_computed" in doc["far"]["H2"]  # 14 far sites per repeat < 30
    monkeypatch.setattr(event3, "STRATUM_MIN_SITES", 2)
    res2 = event3.run_event3(
        inp, PROTOCOL, log_kappa=fake_log_kappa, aoa=fake_aoa(w), n_boot=N_BOOT
    )
    far = res2.documents["hypotheses"]["far_not_nested_strata"]["far"]
    assert far["S2"]["s2_M4_vs_M5"]["estimate"] == pytest.approx(0.08, abs=1e-12)
    assert far["S2"]["s2_M4_vs_M5"]["opposite_sign_to_full_set"] is False
    assert min(far["S2"]["sites_per_repeat"].values()) >= 2
    assert event3._opposite({"lo": -2.0, "hi": -1.0}, {"lo": 1.0, "hi": 2.0}) is True


def test_event3_uq_consistency_matrix_and_descriptives(result: Any) -> None:
    """UQ coverage of |e| = 0.1 inside +/- 0.33 is 1; matrix RMSE_log = c; S-RND rows; L8."""
    _, _, res = result
    uq = res.documents["uq"]["M5|primary"]
    dbar = uq["S-ID"]["dbar"]["overall"]
    assert dbar["coverage"] == 1.0
    assert dbar["mean_log_width"] == pytest.approx(2 * Q_HALF, abs=1e-12)
    assert sum(dbar["pit_counts"]) == dbar["n_rows"] == 3 * N
    assert set(uq["S-ID"]["dbar"]["by"]) == {"division", "da_q"}
    assert uq["S-IND"]["n_locations_single"] == 9  # 12 locations, 3 with two COMIDs
    assert uq["S-IND"]["not_computed_multi_comid"] == 3
    assert "M2-XGB" in res.documents["uq"]["models_without_distribution"]
    cons = res.documents["consistency"]
    m4 = cons["models"]["M4"]["S-ID"]["manning_residual_M4"]
    assert m4["rms"] == pytest.approx(1.0e-3, abs=1e-15)
    assert cons["models"]["M5"]["S-ID"]["solver_residual_max"] == pytest.approx(1e-12)
    assert cons["models"]["M5"]["S-ID"]["back_calc"]["n_invariant4"] == 3  # site 7, 3 repeats
    assert "predicted_closure" in cons["models"]["M2-XGB"]["S-ID"]
    mat = res.matrix

    def value(spec: str, fam: str, target: str, metric: str, kind: str = "all") -> float:
        sel = (
            mat["spec"].eq(spec)
            & mat["analysis"].eq("primary")
            & mat["family"].eq(fam)
            & mat["target"].eq(target)
            & mat["stratum_kind"].eq(kind)
            & mat["metric"].eq(metric)
        )
        assert int(sel.sum()) == 1
        return float(mat.loc[sel, "value"].iloc[0])

    assert value("M5", "S-ID", "dbar", "rmse_log") == pytest.approx(0.10, abs=1e-12)
    assert value("M5", "S-ID", "dbar", "bias_log") == pytest.approx(0.0, abs=1e-12)
    assert value("M2-XGB", "S-LRO", "A", "rmse_log") == pytest.approx(0.30, abs=1e-12)
    assert value("M2-XGB", "S-RND", "dbar", "rmse_log") == pytest.approx(0.10, abs=1e-12)
    assert set(mat.loc[mat["spec"].eq("M5") & mat["family"].eq("S-ID"), "stratum_kind"]) == set(
        st.STRATA_KINDS
    )
    desc = res.documents["descriptive"]
    assert desc["S_RND"]["rows"] == {"M1": 19, "M2-XGB": 20}
    assert desc["dropped_rows"]["A_J|primary|S-IND"] == 1
    assert desc["L8_zarrabi_product"]["status"] == "not computed"
    assert "rho_r_robs" not in desc["ablations_M5"]["M5_A2"]["H4_points_S-ID"]
    assert "H4_points_S-ID_prop_7_6_mapped" in desc["ablations_M5"]["M5_A1"]
    assert desc["S1"]["n_matched"] == 12
    assert desc["sensitivity_status"]["M2-XGB"]["role"] == "realised M*"
    dep = res.documents["dependence"]
    assert dep["nested_3_6_5"]["S-ID"]["n_test_instances"] == 3 * N
    blocks = dep["aoa_3_6_6"]["blocks"]
    assert len(blocks) == 4 + 1 + 1 + 1
    assert set(dep["aoa_3_6_6"]["metrics"]) == {"S-LRO", "S-TAIL-DA", "S-TAIL-SO", "S-IND"}
    from src.evaluation.events import json_safe

    json.dumps(json_safe(res.documents), allow_nan=False)  # strict JSON as written by the CLI


def test_event3_refuses_psi_differing_from_event1() -> None:
    """docs/3 13.13 p: the S-ID psi_max must equal Event 1 exactly."""
    w = world()
    inp = inputs(w)
    bad = inp.audit_strata.copy()
    bad.loc[0, "psi_max"] = bad.loc[0, "psi_max"] + 0.25
    with pytest.raises(ValueError, match="differs from Event 1"):
        event3.run_event3(
            event3.Event3Inputs(**{**inp.__dict__, "audit_strata": bad}),
            PROTOCOL,
            log_kappa=fake_log_kappa,
            aoa=fake_aoa(w),
            n_boot=2,
        )
    with pytest.raises(ValueError, match="differ from audit_strata"):
        event3.run_event3(
            event3.Event3Inputs(**{**inp.__dict__, "audit_strata": inp.audit_strata.iloc[1:]}),
            PROTOCOL,
            log_kappa=fake_log_kappa,
            aoa=fake_aoa(w),
            n_boot=2,
        )


# ----------------------------------------------------------------------------- workers
def _kappa_payload() -> dict[str, Any]:
    return {
        "ell": [0.1, 0.5],
        "w": [2.0, 3.0],
        "r": [1.5, 3.0],
        "level": int(PROTOCOL.solver.quad_level),
        "trunc": float(PROTOCOL.solver.quad_trunc),
    }


def test_kappa_worker_inline_and_spawned() -> None:
    """ADL-051: ln kappa in a spawned torch worker equals the direct call; failures raise."""
    import torch

    from src.physics.shape import log_kappa

    p = _kappa_payload()
    want = log_kappa(
        torch.tensor(p["ell"], dtype=torch.float64),
        torch.tensor(p["w"], dtype=torch.float64),
        torch.tensor(p["r"], dtype=torch.float64),
        level=p["level"],
        trunc=p["trunc"],
    ).tolist()
    (inline,) = ew.run_jobs(ew.kappa_job, [p], n_workers=1, inline=True)
    assert inline["k"] == pytest.approx(want, abs=1e-12)
    (spawned,) = ew.run_jobs(ew.kappa_job, [p], n_workers=1, forbid="xgboost")
    assert spawned["k"] == pytest.approx(want, abs=1e-12)
    assert ew.run_jobs(ew.kappa_job, [], n_workers=1) == []
    with pytest.raises(ew.WorkerError, match="kappa job failed: KeyError"):
        ew.run_jobs(ew.kappa_job, [{"ell": [0.1]}], n_workers=1, inline=True)
    doc = json.loads(ew.aoa_job(json.dumps({"params_path": "/nonexistent.yaml"})))
    assert doc["ok"] is False


def test_aoa_worker_known_answer(tmp_path: Path) -> None:
    """docs/3 13.9: importance, training CV DI, threshold and DI from the member boosters.

    ADL-051: xgboost never loads in the pytest process (torch is loaded there; the two libomp
    copies segfault). Boosters are trained in a torch-free child; AOA runs in spawned workers.
    """
    from src.data.io import sha256_file
    from src.models.registry import ModelSpec, fit_dir

    st_ = _site_table()
    frame = analysis_frame(st_, "primary")
    folds = _folds(frame["site_no"].tolist())
    spec = ModelSpec("M2-XGB")
    features = list(spec.features(PROTOCOL))
    site_path, fold_path = tmp_path / "site.parquet", tmp_path / "folds.parquet"
    st_.to_parquet(site_path)
    folds.to_parquet(fold_path)
    ind = frame.iloc[:6][features].copy()
    ind["loc_id"] = [f"L{k}" for k in range(6)]
    ind["comid"] = 7000 + np.arange(6)
    ind["loc_in_support"] = [True] * 5 + [False]
    ind_path = tmp_path / "ind.parquet"
    ind.to_parquet(ind_path)
    run_dir = tmp_path / "run"
    x_all = frame[features].to_numpy(dtype=np.float64)
    y_all = np.log(frame["dbar"].to_numpy(dtype=np.float64))
    np.save(tmp_path / "x.npy", x_all)
    np.save(tmp_path / "y.npy", y_all)
    jobs: list[tuple[str, int]] = []
    for family in ("S-LRO", "S-IND"):
        for m in (1, 2):
            mdir = fit_dir(run_dir, spec, "primary", family, 0, 0, "member", m)
            mdir.mkdir(parents=True)
            jobs.append((str(mdir / "booster_dbar.json"), m))
            header = {
                "spec": "M2-XGB",
                "analysis": "primary",
                "family": family,
                "repeat": 0,
                "outer_fold": 0,
                "kind": "member",
                "index": m,
                "params_sha256": SETTINGS.params_sha256,
            }
            doc = {
                "header": header,
                "features": features,
                "fit": {"dbar": {"best_iteration": 4}},
                "seed": 100 + m,
                "site_table_sha256": sha256_file(site_path),
                "fold_table_sha256": sha256_file(fold_path),
            }
            (mdir / "run.json").write_text(json.dumps(doc))
    train_boosters = (
        "import json, sys\n"
        "import numpy as np\n"
        "import xgboost as xgb\n"
        "x, y = np.load(sys.argv[1]), np.load(sys.argv[2])\n"
        "for path, seed in json.loads(sys.argv[3]):\n"
        "    params = {'max_depth': 2, 'eta': 0.5, 'seed': seed, 'nthread': 1}\n"
        "    b = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=5)\n"
        "    open(path, 'wb').write(bytes(b.save_raw(raw_format='json')))\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            train_boosters,
            str(tmp_path / "x.npy"),
            str(tmp_path / "y.npy"),
            json.dumps(jobs),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "False"
    base = {
        "params_path": str(PARAMS_PATH),
        "params_sha256": SETTINGS.params_sha256,
        "site_path": str(site_path),
        "site_sha256": sha256_file(site_path),
        "fold_path": str(fold_path),
        "fold_sha256": sha256_file(fold_path),
        "ind_path": str(ind_path),
        "ind_sha256": sha256_file(ind_path),
        "run_dir": str(run_dir),
        "members": 2,
        "repeat": 0,
        "outer_fold": 0,
    }
    lro, sind = ew.run_jobs(
        ew.aoa_job,
        [{**base, "family": "S-LRO"}, {**base, "family": "S-IND"}],
        n_workers=2,
        forbid="torch",
    )
    imp = np.asarray(lro["importance"])
    assert imp.shape == (len(features),)
    assert bool((imp >= 0).all())
    blk = folds.loc[folds["family"].eq("S-LRO") & folds["outer_fold"].eq(0)].set_index("site_no")
    test = blk.reindex(frame["site_no"])["is_test"].to_numpy(dtype=bool)
    train_x = x_all[~test]
    inner = blk.reindex(frame["site_no"])["inner_fold"].to_numpy()[~test]
    want_thr = da.aoa_threshold(da.training_cv_di(train_x, imp, inner), PROTOCOL.audit_rules)
    assert lro["threshold"] == pytest.approx(want_thr, abs=1e-12)
    want_di = da.dissimilarity_index(train_x, x_all[test], imp)
    assert lro["di"] == pytest.approx(want_di.tolist(), abs=1e-12)
    assert lro["units"] == frame["site_no"].to_numpy()[test].tolist()
    assert sind["units"] == [f"L{k}|{7000 + k}" for k in range(5)]
    bad = {**base, "family": "S-LRO", "site_sha256": "0" * 64}
    with pytest.raises(ew.WorkerError, match="site table differs"):
        ew.run_jobs(ew.aoa_job, [bad], n_workers=1, forbid="torch")
    with pytest.raises(ew.WorkerError, match="params_sha256"):
        ew.run_jobs(
            ew.aoa_job,
            [{**base, "family": "S-LRO", "params_sha256": "x"}],
            n_workers=1,
            forbid="torch",
        )
    import torch  # noqa: F401  (already loaded in this process by the kappa test and models tests)

    with pytest.raises(ew.WorkerError, match="must not hold torch"):
        ew.run_jobs(ew.aoa_job, [{**base, "family": "S-LRO"}], n_workers=1, inline=True)
