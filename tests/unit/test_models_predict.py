"""Tests for the Phase 8 prediction writer ``src/models/predict.py`` (synthetic data only).

One module fixture trains a reduced run root (tiny backbone, 6 epochs, 2-point grids, M = 2) on
one synthetic S-ID block and one S-IND block (all sites train): every baseline, M1 / M2-XGB
(S-ID, S-IND, S-RND), M6-XGB and the five NN rungs on S-ID, M2-MLP on S-IND, ``M5_A1`` and M5
under SA1 (``Q_C``). Predictions go through the production path (spawn pools; boosters in
torch-free workers). Known answers are recomputed here from the artefacts with the frozen
library functions; the in-process functions must reproduce the pool's block outputs bit for bit.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.config import Settings, load_settings
from src.data.folds import get_view
from src.data.ind_features import aggregate_e_r8
from src.data.io import read_parquet_meta, sha256_file, write_parquet
from src.evaluation.ledger import Ledger
from src.models import orchestrator as orch
from src.models import plan as pl
from src.models import predict as pr
from src.models.baselines import BoosterFit, M0PowerLaw, MNull, m1_floor, predict_booster
from src.models.checkpoint import load_model
from src.models.ensemble import ensemble_roughness, mixture_cdf
from src.models.registry import ModelSpec, read_json, write_json
from src.models.runner import log_targets
from src.physics.manning_layer import PhiConfig, phi_inverse_nu
from synthetic_sites import fold_table, site_frame

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN = {"commit": "abc123", "dirty": False}
RUN_ID = "p8-0123456789ab"
BLOCKS = [("S-ID", 0, 0), ("S-IND", 0, 0)]
LOCS = ["L0", "L0", "L1", "L2", "L2", "L2", "L3", "L4", "L4", "L5", "L5", "L6"]


# ----------------------------------------------------------------------------- tables
def _settings(root: Path) -> Settings:
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    raw["model"]["backbone"].update(hidden_layers=1, hidden_width=8)
    raw["model"]["optimizer"].update(batch_size=16, max_epochs=6, patience=2)
    raw["model"]["inner_grid"].update(weight_decay=[0.0, 1.0e-4], lr=[1.0e-3], m4_lambda_max=[10.0])
    raw["model"]["xgb"].update(
        max_depth=[2, 3], eta=[0.3], min_child_weight=[1], max_rounds=20, early_stopping_rounds=5
    )
    raw["model"]["baselines"]["m1_max_depth"] = [2]
    raw["uq"].update(warm_start_epochs=2, ensemble_m=2)
    path = root / "params.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_settings(path)


def _site_table() -> pd.DataFrame:
    frame = site_frame()
    frame["comid"] = np.arange(1000, 1000 + len(frame), dtype=np.int64)
    return frame


def _folds(sites: pd.Series) -> pd.DataFrame:
    """The synthetic S-ID block plus an S-IND block whose test part is empty (docs/3 13.5)."""
    n = len(sites)
    ind = pd.DataFrame(
        {
            "family": "S-IND",
            "repeat": np.int8(0),
            "outer_fold": np.int8(0),
            "site_no": sites.to_numpy(),
            "is_test": False,
            "inner_fold": (np.arange(n) % 3).astype("int8"),
        }
    )
    for m in range(1, 11):
        ind[f"es_member_{m}"] = np.arange(n) % 7 == m % 7
    return pd.concat([fold_table(sites), ind], ignore_index=True)


def _ind_table() -> pd.DataFrame:
    """12 member COMIDs of 7 locations; L6 is outside common support; one division missing.

    Row 5 misses an M1 predictor and row 3 has no ``V_E`` (both in L2), from ``site_frame``.
    """
    frame = site_frame(n=12, seed=23).drop(columns=["W", "dbar", "dmax", "A", "t_obs", "site_no"])
    frame.insert(0, "loc_id", LOCS)
    frame.insert(1, "river", "R")
    frame.insert(2, "n_comids", frame.groupby("loc_id")["loc_id"].transform("size"))
    frame.insert(3, "member", frame.groupby("loc_id").cumcount())
    frame.insert(4, "comid", np.arange(5000, 5012, dtype=np.int64))
    frame.insert(5, "lengthkm", np.linspace(0.5, 3.0, 12))
    frame["division"] = frame["division"].astype("string")
    frame.loc[7, "division"] = pd.NA
    frame["loc_in_support"] = frame["loc_id"] != "L6"
    return frame


def _srnd_frame(path: Path, site_sha: str) -> pd.DataFrame:
    """Three measurements per synthetic site; every 4th row (by row_id) is an S-RND test row."""
    sites = _site_table()
    rng = np.random.default_rng(5)
    rows = sites.loc[sites.index.repeat(3)].reset_index(drop=True)
    rows["row_id"] = np.arange(len(rows), dtype=np.int64)
    rows["q_meas_no"] = [f" {k}" for k in range(len(rows))]
    rows["is_test"] = rows["row_id"] % 4 == 0
    rows["seed"] = np.int64(7)
    for col in ("W", "dbar", "dmax", "A"):
        rows[col] = rows[col] * np.exp(rng.normal(0.0, 0.05, len(rows)))
    for flag in ("F_eps", "F_t", "F_qual", "F_tout"):
        rows[flag] = False
    write_parquet(
        rows,
        path,
        stage="srnd_frame",
        params_sha256="0" * 64,
        inputs={"site_table.parquet": site_sha},
        sort_by=["row_id"],
    )
    return rows


def _keep(f: pl.PlannedFit) -> bool:
    if f.stage == "train@baselines":
        return True
    if f.stage == "train@xgb":
        if f.ablation is not None or f.analysis != "primary":
            return False
        return f.model != "M6-XGB" or f.family == "S-ID"
    if f.stage == "train@nn_primary":
        return f.family == "S-ID" or f.model == "M2-MLP"
    if f.stage == "train@ablations":
        return f.family == "S-ID" and f.ablation == "A1"
    return f.family == "S-ID" and f.model == "M5" and f.analysis == "SA1"


@dataclass(frozen=True)
class Run:
    settings: Settings
    root: Path
    run_dir: Path
    ctx: pr.PredictContext
    fits: list[pl.PlannedFit]
    site: pd.DataFrame
    folds: pd.DataFrame
    ind: pd.DataFrame
    srnd: pd.DataFrame
    reports: dict[str, dict[str, Any]]

    def tables(self) -> pr.PredictTables:
        return pr.PredictTables.load(self.ctx)

    def file(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / "out" / name)

    def predict(self, stage: str, dest: Path, **kw: Any) -> dict[str, Any]:
        kw.setdefault("n_workers", 2)
        return pr.run_predictions(
            stage, self.fits, ctx=self.ctx, settings=self.settings, dest_dir=dest, **kw
        )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("predict")
    settings = _settings(root)
    site, folds, ind = _site_table(), _folds(_site_table()["site_no"]), _ind_table()
    paths = {k: root / f"{k}.parquet" for k in ("site", "folds", "ind", "srnd")}
    site.to_parquet(paths["site"])
    folds.to_parquet(paths["folds"])
    ind.to_parquet(paths["ind"])
    srnd = _srnd_frame(paths["srnd"], sha256_file(paths["site"]))
    run_dir = root / RUN_ID
    hashes = {k: sha256_file(p) for k, p in paths.items()}
    run_ctx = orch.RunContext(
        params_path=str(settings.params_path),
        site_path=str(paths["site"]),
        fold_path=str(paths["folds"]),
        run_dir=str(run_dir),
        params_sha256=settings.params_sha256,
        site_sha256=hashes["site"],
        fold_sha256=hashes["folds"],
        srnd_path=str(paths["srnd"]),
        srnd_sha256=hashes["srnd"],
    )
    ledger = root / "scoring_ledger.json"
    led = Ledger.create(
        ledger,
        run_id=RUN_ID,
        plan_sha256="1" * 64,
        params_sha256=settings.params_sha256,
        uv_lock_sha256="2" * 64,
    )
    common: dict[str, Any] = {"git": CLEAN, "params_sha256": settings.params_sha256}
    led.append("seal", inputs=[{"path": "p", "sha256": "a" * 64}], merkle_root="c" * 64, **common)
    led.append("event_1", inputs=[{"path": "p", "sha256": "a" * 64}], **common)
    led.append("checkpoint_1", note="synthetic", **common)
    fits = [f for f in pl.build_plan(settings.protocol, BLOCKS) if _keep(f)]
    for stage in pl.STAGES:
        orch.run_stage(
            stage,
            fits,
            ctx=run_ctx,
            settings=settings,
            run_id=RUN_ID,
            ledger_path=ledger,
            n_workers=2,
            inline=stage not in ("train@xgb",),
        )
    ctx = pr.PredictContext(
        params_path=str(settings.params_path),
        run_dir=str(run_dir),
        site_path=str(paths["site"]),
        fold_path=str(paths["folds"]),
        params_sha256=settings.params_sha256,
        site_sha256=hashes["site"],
        fold_sha256=hashes["folds"],
        ind_path=str(paths["ind"]),
        ind_sha256=hashes["ind"],
        srnd_path=str(paths["srnd"]),
        srnd_sha256=hashes["srnd"],
    )
    reports: dict[str, dict[str, Any]] = {}
    for stage in pr.PREDICT_STAGES:
        reports[stage] = pr.run_predictions(
            stage,
            fits,
            ctx=ctx,
            settings=settings,
            dest_dir=root / "out",
            n_workers=2,
            repo_root=root,
        )
    return Run(settings, root, run_dir, ctx, fits, site, folds, ind, srnd, reports)


def _job(run: Run, model: str, family: str = "S-ID", **kw: Any) -> pr.PredictJob:
    jobs = pr.prediction_jobs(run.fits, "predict@all")
    want = {"model": model, "family": family, "ablation": None, "analysis": "primary", **kw}
    (job,) = [j for j in jobs if all(getattr(j, k) == v for k, v in want.items())]
    return job


def _rows(run: Run, family: str, part: str = "test") -> np.ndarray:
    return get_view(run.site, run.folds, family, 0, 0, part)


def _block(run: Run, job: pr.PredictJob) -> pd.DataFrame:
    path, _ = pr.block_paths(run.ctx.stage_dir, job, "main")
    return pd.read_parquet(path)


def _site_rows(frame: pd.DataFrame, job: pr.PredictJob) -> pd.DataFrame:
    sel = frame["family"].eq(job.family) & frame["analysis"].eq(job.analysis)
    sel &= frame["unit"].eq(job.unit) & frame["spec"].eq(job.spec_name)
    return frame.loc[sel].reset_index(drop=True)


# ----------------------------------------------------------------------------- files and counts
def test_stage_files_counts_and_schema(trained: Run) -> None:
    audit = trained.reports["predict@audit"]
    assert list(audit["files"]) == ["out/audit_inputs.parquet"]
    info = audit["files"]["out/audit_inputs.parquet"]
    n_test = _rows(trained, "S-ID").size
    assert info["rows_by_family_unit"] == {"S-ID/site": 2 * n_test}
    frame = trained.file(pr.AUDIT_FILE)
    assert set(frame["model"]) == set(pr.AUDIT_MODELS)
    assert not set(pr.OBSERVED) & set(frame.columns)
    assert info["sha256"] == sha256_file(trained.root / "out" / pr.AUDIT_FILE)
    files = trained.reports["predict@all"]["files"]
    specs = {"M-null", "A_J", "M0", "M1", "M2-XGB", "M6-XGB", "M2-MLP", "M3", "M4", "M5"}
    assert set(files) == {f"out/{s}.parquet" for s in specs | {"M6-NN", "M5_A1"}}
    n_srnd, n_ind = int(trained.srnd["is_test"].sum()), int(trained.ind["loc_in_support"].sum())
    assert files["out/M1.parquet"]["rows_by_family_unit"] == {
        "S-ID/site": n_test,
        "S-IND/comid": n_ind,
        "S-IND/location": 6,
        "S-RND/srnd_row": n_srnd,
    }
    assert files["out/M5.parquet"]["rows_by_family_unit"] == {"S-ID/site": 2 * n_test}
    meta = read_parquet_meta(trained.root / "out" / "M1.parquet")
    assert meta["params_sha256"] == trained.settings.params_sha256
    assert meta["stage"] == "predict@all"
    for path in (trained.root / "out").glob("*.parquet"):
        assert not set(pr.OBSERVED) & set(pd.read_parquet(path).columns), path.name


def test_import_never_loads_torch_or_xgboost() -> None:
    code = "import sys, src.models.predict; print('torch' in sys.modules, 'xgboost' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False"


# ----------------------------------------------------------------------------- blinding
def test_block_frame_blinds_every_prediction_row(trained: Run) -> None:
    tables = trained.tables()
    assert tables.ind is not None
    assert len(tables.ind) == int(trained.ind["loc_in_support"].sum())
    sid = pr.block_frame(_job(trained, "M2-MLP"), tables)
    cols = [c for c in pr.OBSERVED if c in sid.pred.columns]
    assert cols
    assert sid.pred.iloc[sid.rows][cols].isna().all().all()
    assert sid.train.iloc[sid.outer_train][["W", "dbar"]].notna().all().all()
    assert list(sid.keys.columns) == ["site_no", "comid"]
    srnd = pr.block_frame(_job(trained, "M1", "S-RND"), tables)
    assert srnd.pred.iloc[srnd.rows][["W", "dbar", "dmax", "A"]].isna().all().all()
    assert srnd.rows.size == int(trained.srnd["is_test"].sum())
    assert list(srnd.keys.columns) == ["row_id", "site_no"]
    ind = pr.block_frame(_job(trained, "M2-MLP", "S-IND"), tables)
    assert ind.outer_train.size == len(trained.site)
    assert list(ind.keys["loc_id"]) == [x for x in LOCS if x != "L6"]
    bare = pr.PredictTables(tables.site_table, tables.folds, None, None, tables.hashes)
    with pytest.raises(ValueError, match="srnd_frame table"):
        pr.block_frame(_job(trained, "M1", "S-RND"), bare)
    with pytest.raises(ValueError, match="ind_features table"):
        pr.block_frame(_job(trained, "M1", "S-IND"), bare)
    leaky = tables.folds.copy()
    leaky.loc[leaky.index[leaky["family"].eq("S-IND")][0], "is_test"] = True
    with pytest.raises(ValueError, match="no site test rows"):
        pr.block_frame(
            _job(trained, "M1", "S-IND"),
            pr.PredictTables(tables.site_table, leaky, tables.ind, None, tables.hashes),
        )


# ----------------------------------------------------------------------------- known answers
def test_baseline_known_answers(trained: Run) -> None:
    test, train = _rows(trained, "S-ID"), _rows(trained, "S-ID", "train")
    sub, tr = trained.site.iloc[test], trained.site.iloc[train]
    y = log_targets(tr)
    mnull = MNull.fit(tr["DA_km2"].to_numpy(dtype=np.float64), y).predict(
        sub["DA_km2"].to_numpy(dtype=np.float64)
    )
    got = _site_rows(trained.file("M-null.parquet"), _job(trained, "M-null"))
    assert got["site_no"].tolist() == sub["site_no"].tolist()
    for k in ("W", "dbar", "dmax", "A"):
        np.testing.assert_array_equal(got[f"mu_{k}"].to_numpy(), mnull[k])
    np.testing.assert_array_equal(got["mu_t"], got["mu_dmax"] - got["mu_dbar"])
    min_sites = trained.settings.protocol.model.baselines.m0_min_division_sites
    m0 = M0PowerLaw.fit(
        tr["division"].to_numpy(), tr["DA_km2"].to_numpy(), y[:, 0], y[:, 1], min_sites
    )
    want, _ = m0.predict(sub["division"].to_numpy(), sub["DA_km2"].to_numpy())
    got = _site_rows(trained.file("M0.parquet"), _job(trained, "M0"))
    np.testing.assert_allclose(got["mu_W"], want["W"], rtol=0, atol=1e-12)
    assert got["m0_pooled"].all()
    aj = _site_rows(trained.file("A_J.parquet"), _job(trained, "A_J"))
    want_a = np.log(sub["Q_E"].to_numpy()) - np.log(sub["V_E"].to_numpy())
    np.testing.assert_allclose(aj["mu_A"], want_a, rtol=0, atol=1e-12)
    assert (aj["aj_available"] == np.isfinite(want_a)).all()
    ind = _site_rows(trained.file("M0.parquet"), _job(trained, "M0", "S-IND"))
    assert ind["division_missing"].sum() == 1
    assert ind.loc[ind["division_missing"], "m0_pooled"].all()


def _boosters(run: Run, job: pr.PredictJob, x: np.ndarray, t: str) -> list[np.ndarray]:
    out = []
    for m in range(1, job.members + 1):
        mdir = run.run_dir / job.block_id / f"member_{m}"
        doc = read_json(mdir / "run.json")
        bf = BoosterFit(
            (mdir / f"booster_{t}.json").read_bytes(), doc["fit"][t]["best_iteration"], float("nan")
        )
        out.append(predict_booster(bf, x))
    return out


def test_xgb_known_answers_and_m1_floor(trained: Run) -> None:
    test = _rows(trained, "S-ID")
    sub = trained.site.iloc[test]
    job = _job(trained, "M2-XGB")
    got = _site_rows(trained.file("M2-XGB.parquet"), job)
    spec_features = ModelSpec("M2-XGB", None).features(trained.settings.protocol)
    x = sub[list(spec_features)].to_numpy(dtype=np.float64)
    for t in ("W", "dbar", "dmax", "A"):
        preds = _boosters(trained, job, x, t)
        np.testing.assert_array_equal(got[f"m01_mu_{t}"], preds[0])
        np.testing.assert_allclose(got[f"mu_{t}"], np.mean(preds, axis=0), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(got["mu_t"], got["mu_dmax"] - got["mu_dbar"])
    job = _job(trained, "M1")
    got = _site_rows(trained.file("M1.parquet"), job)
    ok = ~sub["F_M1_missing"].to_numpy()
    assert (got["m1_missing"].to_numpy() == ~ok).all()
    assert got.loc[~ok, "mu_W"].isna().all()
    m1_features = list(ModelSpec("M1", None).features(trained.settings.protocol))
    x = sub[m1_features].to_numpy(dtype=np.float64)[ok]
    q = trained.settings.protocol.model.baselines.m1_floor_quantile
    for t in ("W", "dbar"):
        logs = []
        for m, pred in enumerate(_boosters(trained, job, x, t), start=1):
            rows = get_view(trained.site, trained.folds, "S-ID", 0, 0, "es_train", m)
            rows = rows[~trained.site["F_M1_missing"].to_numpy()[rows]]
            floored, _ = m1_floor(pred, trained.site.iloc[rows][t].to_numpy(), q)
            logs.append(np.log(floored))
        np.testing.assert_allclose(got.loc[ok, f"mu_{t}"], np.mean(logs, axis=0), atol=1e-12)
    np.testing.assert_array_equal(got["mu_A"], got["mu_W"] + got["mu_dbar"])
    assert (got["n_floored"] >= 0).all()


def test_m1_floor_is_applied_to_non_positive_predictions(
    trained: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.models.baselines as bl

    monkeypatch.setattr(bl, "predict_booster", lambda booster, x: np.full(len(x), -1.0))
    job = _job(trained, "M1")
    raw = pr.booster_predictions(job, trained.tables(), trained.settings, trained.run_dir)
    ok = ~raw["m1_missing"].to_numpy()
    assert (raw.loc[ok, "n_floored"] == 2 * job.members).all()  # both targets, every member
    q = trained.settings.protocol.model.baselines.m1_floor_quantile
    rows = get_view(trained.site, trained.folds, "S-ID", 0, 0, "es_train", 1)
    rows = rows[~trained.site["F_M1_missing"].to_numpy()[rows]]
    floor = float(np.quantile(trained.site.iloc[rows]["W"], q))
    np.testing.assert_allclose(raw.loc[ok, "m01_W"], np.log(floor), rtol=0, atol=1e-15)


def test_nu_back_known_answer_and_invariant_4(trained: Run) -> None:
    job = _job(trained, "M2-XGB")
    got = _site_rows(trained.file("M2-XGB.parquet"), job)
    sub = trained.site.iloc[_rows(trained, "S-ID")]
    cfg = PhiConfig.from_solver(trained.settings.protocol.solver)
    t = got["mu_t"].to_numpy()
    assert (got["inv4_violation"].to_numpy() == (t <= 0)).all()
    ok = t > 0
    r = 1.0 / np.expm1(t[ok])
    np.testing.assert_allclose(got.loc[ok, "r_hat"], r, rtol=1e-12)
    args = [
        got["mu_dbar"].to_numpy()[ok],
        got["mu_W"].to_numpy()[ok],
        r,
        np.log(sub["Q_E"].to_numpy()[ok]),
        np.log(sub["S"].to_numpy()[ok]),
    ]
    want = phi_inverse_nu(*(torch.from_numpy(np.ascontiguousarray(a)) for a in args), cfg)
    np.testing.assert_allclose(got.loc[ok, "nu_back"], want.numpy(), rtol=0, atol=1e-12)
    assert got.loc[~ok, "nu_back"].isna().all()
    m1 = _site_rows(trained.file("M1.parquet"), _job(trained, "M1"))
    assert m1["r_hat"].nunique() == 1  # r_bar, the training-fold median (13.9)
    assert not m1["inv4_violation"].any()
    srnd = trained.file("M1.parquet").query("family == 'S-RND'")
    assert "nu_back" not in srnd.dropna(axis=1, how="all").columns
    nan = pr.back_calc(
        np.array([np.nan, 1.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, -1.0]),
        np.zeros(2),
        np.full(2, -6.0),
        cfg,
    )
    assert np.isnan(nan).all()


def test_nn_members_match_a_forward_pass_with_the_real_targets(trained: Run) -> None:
    """The placeholder ``y`` is never read: a forward pass on the observed targets agrees."""
    job = _job(trained, "M2-MLP")
    got = _site_rows(trained.file("M2-MLP.parquet"), job)
    sub = trained.site.iloc[_rows(trained, "S-ID")]
    protocol = trained.settings.protocol
    features = ModelSpec("M2-MLP", None).features(protocol)
    for m in (1, 2):
        mdir = trained.run_dir / job.block_id / f"member_{m}"
        model, stats, _ = load_model(mdir, protocol, params_sha256=trained.settings.params_sha256)
        model.eval()
        y = np.column_stack([np.log(sub["W"]), np.log(sub["dbar"]), sub["t_obs"], np.log(sub["A"])])
        batch = {
            "x": torch.from_numpy(stats.transform(sub[list(features)].to_numpy(dtype=np.float64))),
            "phys": torch.from_numpy(np.column_stack([np.log(sub["Q_E"]), np.log(sub["S"])])),
            "y": torch.from_numpy(y),
            "omega": torch.full((len(sub),), 1.0 / len(sub), dtype=torch.float64),
        }
        with torch.no_grad():
            out = model(batch)
        np.testing.assert_allclose(got[f"m0{m}_mu_W"], out.mu[:, 0].numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            got[f"m0{m}_mu_dmax"], out.extras["mu_native"][:, 2].numpy(), rtol=0, atol=1e-12
        )
    mu = np.stack([got[f"m0{m}_mu_dbar"] for m in (1, 2)])
    sig = np.stack([got[f"m0{m}_sigma_dbar"] for m in (1, 2)])
    np.testing.assert_allclose(got["mu_dbar"], mu.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["var_epi_dbar"], mu.var(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["var_ale_dbar"], (sig**2).mean(axis=0), rtol=0, atol=1e-12)
    alpha = protocol.uq.alpha
    for col, p in (("q_lo_dbar", alpha / 2), ("q_hi_dbar", 1 - alpha / 2)):
        x = torch.from_numpy(got[col].to_numpy()[:, None].copy())
        cdf = mixture_cdf(x, torch.from_numpy(mu[:, :, None]), torch.from_numpy(sig[:, :, None]))
        np.testing.assert_allclose(cdf.numpy()[:, 0], p, rtol=0, atol=1e-9)
    assert {"q_lo_dmax", "q_hi_dmax", "nu_back", "r_hat"} <= set(got.columns)


def test_physics_rungs_record_roughness_and_residuals(trained: Run) -> None:
    sub = trained.site.iloc[_rows(trained, "S-ID")]
    for spec, analysis, discharge in (
        ("M5", "primary", "Q_E"),
        ("M5", "SA1", "Q_C"),
        ("M4", "primary", "Q_E"),
        ("M6-NN", "primary", "Q_E"),
    ):
        job = _job(trained, spec, analysis=analysis)
        got = _site_rows(trained.file(f"{spec}.parquet"), job)
        mu_star = torch.from_numpy(got[["mu_W", "mu_dbar", "mu_t", "mu_A"]].to_numpy().copy())
        phys = torch.from_numpy(np.column_stack([np.log(sub[discharge]), np.log(sub["S"])]))
        cfg = PhiConfig.from_solver(trained.settings.protocol.solver)
        want = ensemble_roughness(mu_star, phys, cfg)
        np.testing.assert_allclose(got["nu_star"], want.numpy(), rtol=0, atol=1e-10)
        assert np.isfinite(got[["nu_mean", "r_mean", "resid_mean", "resid_maxabs"]]).all().all()
    m5 = _site_rows(trained.file("M5.parquet"), _job(trained, "M5"))
    tol = 10 * trained.settings.protocol.solver.tau_l
    assert (m5["resid_maxabs"] <= tol).all()  # hard constraint: Phi solved to tolerance
    assert (m5["iters_max"] >= 1).all()
    m3 = _site_rows(trained.file("M3.parquet"), _job(trained, "M3"))
    np.testing.assert_allclose(m3["mu_W"], (m3["m01_mu_W"] + m3["m02_mu_W"]) / 2, atol=1e-12)
    assert m3["mu_t"].nunique() == 1  # r_bar (P6-Q4 (a))
    assert m3["r_mean"].nunique() == 1
    assert "q_lo_W" not in m3.columns  # no sigma, no interval
    a1 = _site_rows(trained.file("M5_A1.parquet"), _job(trained, "M5", ablation="A1"))
    assert np.isfinite(a1["nu_star"]).all()
    m6 = _site_rows(trained.file("M6-XGB.parquet"), _job(trained, "M6-XGB"))
    assert np.isfinite(m6[["mu_W", "mu_dbar", "nu_star", "var_epi_W"]]).all().all()


def test_ind_locations_follow_e_r8(trained: Run) -> None:
    for spec in ("M2-MLP", "M1", "M-null"):
        frame = trained.file(f"{spec}.parquet").query("family == 'S-IND'")
        comid, loc = frame[frame["unit"] == "comid"], frame[frame["unit"] == "location"]
        assert loc["loc_id"].tolist() == ["L0", "L1", "L2", "L3", "L4", "L5"]
        good = comid.groupby("loc_id")["mu_W"].transform(lambda s: s.notna().all())
        c = comid[good]
        want = aggregate_e_r8(
            c["loc_id"],
            c["lengthkm"].to_numpy(),
            np.exp(c["mu_W"].to_numpy()),
            np.exp(c["mu_dbar"].to_numpy()),
        )
        got = loc.set_index("loc_id").loc[want["loc_id"]]
        np.testing.assert_allclose(got["mu_A"], np.log(want["A"]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(got["mu_W"], np.log(want["W"]), rtol=0, atol=1e-12)
        missing = set(loc.loc[loc["mu_W"].isna(), "loc_id"])
        assert missing == ({"L2"} if spec == "M1" else set())
    aj = trained.file("A_J.parquet").query("family == 'S-IND' and unit == 'location'")
    assert set(aj.loc[aj["mu_A"].isna(), "loc_id"]) == {"L2"}  # one member without V_E


# ----------------------------------------------------------------------------- in-process == pool
def test_in_process_functions_reproduce_the_pool_outputs(trained: Run) -> None:
    tables, s, rd = trained.tables(), trained.settings, trained.run_dir
    for job in pr.prediction_jobs(trained.fits, "predict@all"):
        if job.is_baseline:
            frame = pr.predict_baseline(job, tables, s, rd)
        elif job.is_xgb:
            raw = pr.booster_predictions(job, tables, s, rd)
            path, _ = pr.block_paths(trained.ctx.stage_dir, job, "raw")
            pd.testing.assert_frame_equal(raw, pd.read_parquet(path))
            frame = pr.finish_xgb(job, raw, tables, s)
        else:
            frame = pr.predict_nn(job, tables, s, rd)
        pd.testing.assert_frame_equal(frame, _block(trained, job), check_exact=True)


def test_inline_driver_and_bit_identical_rerun(
    trained: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh stage directory and output directory give byte-identical files (inline, no pool)."""
    run_dir = tmp_path / RUN_ID
    shutil.copytree(trained.run_dir, run_dir, ignore=shutil.ignore_patterns("predictions"))
    ctx = pr.PredictContext(**{**json.loads(trained.ctx.to_json()), "run_dir": str(run_dir)})
    monkeypatch.setattr(pr, "xgb_raw", pr.booster_predictions)  # inline: no torch-free worker
    rep = pr.run_predictions(
        "predict@all",
        trained.fits,
        ctx=ctx,
        settings=trained.settings,
        dest_dir=tmp_path / "out",
        n_workers=1,
        inline=True,
        repo_root=tmp_path,
    )
    first = trained.reports["predict@all"]["files"]
    assert {k: v["sha256"] for k, v in rep["files"].items()} == {
        k: v["sha256"] for k, v in first.items()
    }
    again = pr.run_predictions(
        "predict@all",
        trained.fits,
        ctx=ctx,
        settings=trained.settings,
        dest_dir=tmp_path / "out",
        n_workers=1,
        inline=True,
        repo_root=tmp_path,
    )
    assert again["files"] == rep["files"]


def test_audit_blocks_are_reused_by_predict_all(
    trained: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch D F-D1: production order on one run dir. ``predict@audit`` runs with the reduced
    context (no S-IND/S-RND tables, as in ``run_experiment.py``); ``predict@all`` then reuses its
    blocks (resume key = the job's own tables) and gives the same files as the fixture.
    """
    run_dir = tmp_path / RUN_ID
    shutil.copytree(trained.run_dir, run_dir, ignore=shutil.ignore_patterns("predictions"))
    full = pr.PredictContext(**{**json.loads(trained.ctx.to_json()), "run_dir": str(run_dir)})
    reduced = dict.fromkeys(("ind_path", "ind_sha256", "srnd_path", "srnd_sha256"), "")
    audit = pr.PredictContext(**{**json.loads(full.to_json()), **reduced})
    monkeypatch.setattr(pr, "xgb_raw", pr.booster_predictions)  # inline: no torch-free worker
    common: dict[str, Any] = {
        "settings": trained.settings,
        "dest_dir": tmp_path / "out",
        "n_workers": 1,
        "inline": True,
        "repo_root": tmp_path,
    }
    pr.run_predictions("predict@audit", trained.fits, ctx=audit, **common)
    tables = pr.PredictTables.load(full)
    xjob = _job(trained, "M2-XGB")
    for phase in ("raw", "main"):
        res = pr.execute_job(xjob, phase, tables, trained.settings, run_dir, full.stage_dir)
        assert res["skipped"] is True, phase
    rep = pr.run_predictions("predict@all", trained.fits, ctx=full, **common)
    assert {k: v["sha256"] for k, v in rep["files"].items()} == {
        k: v["sha256"] for k, v in trained.reports["predict@all"]["files"].items()
    }
    ijob = _job(trained, "M2-XGB", "S-IND")
    with pytest.raises(RuntimeError, match=r"lacks \['ind_features'\]"):
        pr.job_tables(ijob, pr.PredictTables.load(audit))


# ----------------------------------------------------------------------------- guards
def test_refusals(trained: Run, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown prediction stage"):
        pr.prediction_jobs(trained.fits, "predict@x")
    with pytest.raises(ValueError, match="n_workers"):
        trained.predict("predict@audit", tmp_path, n_workers=0)
    with pytest.raises(ValueError, match="no blocks"):
        pr.run_predictions(
            "predict@audit",
            [],
            ctx=trained.ctx,
            settings=trained.settings,
            dest_dir=tmp_path,
            n_workers=1,
        )
    extra = pl.PlannedFit("train@xgb", "M2-XGB", None, "primary", "S-ID", 0, 0, "member", 9, None)
    with pytest.raises(RuntimeError, match="1 fits incomplete"):
        pr.run_predictions(
            "predict@audit",
            [*trained.fits, extra],
            ctx=trained.ctx,
            settings=trained.settings,
            dest_dir=tmp_path,
            n_workers=1,
        )
    with pytest.raises(RuntimeError, match="without torch"):
        pr.xgb_raw(_job(trained, "M1"), trained.tables(), trained.settings, trained.run_dir)
    with pytest.raises(RuntimeError, match="torch-free worker"):
        pr._init_predict_worker(trained.ctx.to_json(), True)
    bad = pr.PredictContext(**{**json.loads(trained.ctx.to_json()), "params_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="params_sha256"):
        pr._init_predict_worker(bad.to_json(), False)
    for key, match in (
        ("site_sha256", "plan hashes"),
        ("ind_sha256", "ind_features"),
        ("srnd_sha256", "srnd_frame differs"),
    ):
        bad = pr.PredictContext(**{**json.loads(trained.ctx.to_json()), key: "0" * 64})
        with pytest.raises(RuntimeError, match=match):
            pr.PredictTables.load(bad)
    other = tmp_path / "srnd.parquet"
    _srnd_frame(other, "f" * 64)
    bad = pr.PredictContext(**{**json.loads(trained.ctx.to_json()), "srnd_path": str(other)})
    with pytest.raises(ValueError, match="not built from this site_table"):
        pr.PredictTables.load(bad)
    tables = trained.tables()
    with pytest.raises(ValueError, match="no boosters"):
        pr.booster_predictions(_job(trained, "M5"), tables, trained.settings, trained.run_dir)
    with pytest.raises(ValueError, match="no raw phase"):
        pr.execute_job(
            _job(trained, "M5"), "raw", tables, trained.settings, trained.run_dir, tmp_path
        )
    with pytest.raises(ValueError, match="phase must be"):
        pr.execute_job(
            _job(trained, "M5"), "x", tables, trained.settings, trained.run_dir, tmp_path
        )
    with pytest.raises(FileNotFoundError, match="raw phase did not complete"):
        pr.execute_job(
            _job(trained, "M1"), "main", tables, trained.settings, trained.run_dir, tmp_path
        )
    none = pr.PredictJob("M5", None, "primary", "S-ID", 0, 0, 0)
    with pytest.raises(ValueError, match="no members planned"):
        pr.predict_nn(none, tables, trained.settings, trained.run_dir)


def test_artefact_checks_and_stale_sidecars(trained: Run, tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_ID
    shutil.copytree(trained.run_dir, run_dir)
    tables, s = trained.tables(), trained.settings
    stage_dir = run_dir / "predictions"
    job = _job(trained, "M2-MLP")
    res = pr.execute_job(job, "main", tables, s, run_dir, stage_dir)
    assert res["skipped"] is True
    member = run_dir / job.block_id / "member_1" / "run.json"
    original = read_json(member)
    write_json(member, {**original, "wall_s": -1.0})  # the artefact changed after prediction
    with pytest.raises(RuntimeError, match="stale output"):
        pr.execute_job(job, "main", tables, s, run_dir, stage_dir)
    xjob = _job(trained, "M2-XGB")
    _, raw_side = pr.block_paths(stage_dir, xjob, "raw")
    write_json(raw_side, {**read_json(raw_side), "phase": "other"})
    main_path, _ = pr.block_paths(stage_dir, xjob, "main")
    main_path.unlink()
    with pytest.raises(RuntimeError, match="names other artefacts"):
        pr.execute_job(xjob, "main", tables, s, run_dir, stage_dir)
    edits = (
        ({"header": {**original["header"], "params_sha256": "0" * 64}}, "different params"),
        ({"site_table_sha256": "0" * 64}, "different site or fold"),
        ({"header": {**original["header"], "index": 2}}, "coordinates"),
    )
    for edit, match in edits:
        write_json(member, {**original, **edit})
        with pytest.raises(ValueError, match=match):
            pr.predict_nn(job, tables, s, run_dir)
    member.unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        pr.predict_nn(job, tables, s, run_dir)
    sjob = _job(trained, "M1", "S-RND")
    srun = run_dir / sjob.block_id / "member_1" / "run.json"
    write_json(srun, {**read_json(srun), "srnd_frame_sha256": "0" * 64})
    with pytest.raises(ValueError, match="different srnd_frame"):
        pr.booster_predictions(sjob, tables, s, run_dir)
    mjob = _job(trained, "M1")
    mrun = run_dir / mjob.block_id / "member_1" / "run.json"
    doc = read_json(mrun)
    write_json(mrun, {**doc, "rows_used": {**doc["rows_used"], "train": 1}})
    with pytest.raises(ValueError, match="M1 training rows differ"):
        pr.booster_predictions(mjob, tables, s, run_dir)
    write_json(mrun, {**doc, "features": ["DA_km2"]})
    with pytest.raises(ValueError, match="booster features differ"):
        pr.booster_predictions(mjob, tables, s, run_dir)


def test_sealed_files_are_never_rewritten(trained: Run, tmp_path: Path) -> None:
    jobs = pr.prediction_jobs(trained.fits, "predict@audit")
    dest = tmp_path / "out"
    rep = pr.collate("predict@audit", jobs, trained.ctx, trained.settings, dest, repo_root=tmp_path)
    ((rel, info),) = rep["files"].items()
    assert rel == "out/audit_inputs.parquet"
    assert info["sha256"] == trained.reports["predict@audit"]["files"][rel]["sha256"]
    same = pr.collate(
        "predict@audit",
        jobs,
        trained.ctx,
        trained.settings,
        dest,
        sealed={rel: info["sha256"]},
        repo_root=tmp_path,
    )
    assert same["files"] == rep["files"]
    with pytest.raises(pr.SealedFileError, match="sealed with other content"):
        pr.collate(
            "predict@audit",
            jobs,
            trained.ctx,
            trained.settings,
            dest,
            sealed={rel: "0" * 64},
            repo_root=tmp_path,
        )
    assert sha256_file(dest / pr.AUDIT_FILE) == info["sha256"]
    assert not list(dest.glob("*.tmp"))
    gone = pr.PredictContext(**{**json.loads(trained.ctx.to_json()), "run_dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError, match="run the prediction phases first"):
        pr.collate("predict@audit", jobs, gone, trained.settings, dest)
    outside = pr._rel(Path("/elsewhere/x.parquet"), tmp_path)
    assert outside == "/elsewhere/x.parquet"


def test_worker_failure_is_reported_and_stops(
    trained: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / RUN_ID
    shutil.copytree(trained.run_dir, run_dir, ignore=shutil.ignore_patterns("predictions"))
    ctx = pr.PredictContext(**{**json.loads(trained.ctx.to_json()), "run_dir": str(run_dir)})

    def boom(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise FloatingPointError("non-finite")

    monkeypatch.setattr(pr, "xgb_raw", pr.booster_predictions)  # inline: no torch-free worker
    monkeypatch.setattr(pr, "predict_baseline", boom)
    with pytest.raises(pr.PredictFailureError) as info:
        pr.run_predictions(
            "predict@audit",
            trained.fits,
            ctx=ctx,
            settings=trained.settings,
            dest_dir=tmp_path / "out",
            n_workers=1,
            inline=True,
            logger=logging.getLogger("t"),
        )
    rep = info.value.report
    assert (rep["error"], rep["phase"]) == ("FloatingPointError", "main")
    assert rep["block_id"] == "M-null/primary/S-ID/r0/f0"
    assert "Traceback" in rep["traceback"]
    assert not (tmp_path / "out").exists()  # nothing is collated after a failure


@pytest.mark.skipif(os.name != "posix", reason="spawn pool smoke test")
def test_pool_isolation_flag(trained: Run) -> None:
    res = pr._run_phase([_job(trained, "M2-XGB")], "raw", trained.ctx, 1, False)
    assert res[0]["torch_loaded"] is False
    assert res[0]["skipped"] is True
