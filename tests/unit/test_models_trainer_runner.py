"""Tests for Batch D: seeding, metrics, registry, trainer, checkpoints, runner and the train CLI.

Synthetic data only (no site table, no fold table on disk except temporary copies). Runner tests
use a reduced ``params.yaml`` written to ``tmp_path`` (tiny backbone, few epochs, small grids);
its ``params_sha256`` differs from the frozen file and is recorded in every ``run.json``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.config import Settings, load_settings
from src.data.dataset import SiteDataset, make_loader
from src.data.normstats import NormStats
from src.evaluation.metrics import rmse_log
from src.models import registry as rg
from src.models.backbone import assert_row_wise
from src.models.checkpoint import (
    MODEL_FILE,
    NORMSTATS_FILE,
    STATE_FILE,
    build_model,
    load_model,
    load_state,
    save_model,
    save_state,
)
from src.models.m3_physics import m3_r_bar
from src.models.m4_soft import M4Soft
from src.models.m5_hard import M5Hard
from src.models.runner import FitRequest, Tables, _resolve_rows, run_fit, run_select
from src.models.trainer import (
    EarlyStopping,
    TrainConfig,
    evaluate_monitor,
    fit,
    predict,
    xi_ratio,
)
from src.physics.manning_layer import PhiConvergenceError
from src.utils.seeding import block_seed, derive_seed, fit_seed

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
PROTOCOL = SETTINGS.protocol
NN_FEATURES = tuple(PROTOCOL.transforms.features)
SHA_SITE, SHA_FOLD = "a" * 64, "b" * 64


# ----------------------------------------------------------------------------- synthetic data
def site_frame(n: int = 60, seed: int = 11) -> pd.DataFrame:
    """Primary-variant site table with every NN, M1, target and physics column."""
    rng = np.random.default_rng(seed)
    da = np.exp(rng.normal(5.0, 1.0, n))
    q = 0.02 * da**0.9 * np.exp(rng.normal(0, 0.2, n))
    s = np.where(
        np.arange(n) % 4 == 0, np.exp(rng.normal(-5.0, 0.3, n)), np.exp(rng.normal(-8, 0.8, n))
    )
    w = 4.0 * q**0.5 * np.exp(rng.normal(0, 0.2, n))
    dbar = 0.3 * q**0.35 * np.exp(rng.normal(0, 0.2, n))
    t = rng.uniform(0.2, 0.9, n)
    df = pd.DataFrame(
        {
            "site_no": [f"{i:08d}" for i in range(n)],
            "variant": "primary",
            "F_DA": False,
            "F_Qsel": False,
            "F_S": np.arange(n) % 9 == 0,
            "division": np.where(np.arange(n) % 2 == 0, "A", "B"),
            "DA_km2": da,
            "stream_order": rng.integers(1, 8, n).astype(float),
            "z_min_m": rng.normal(200, 80, n),
            "S": s,
            "Q_E": q,
            "Q_C": q * np.exp(rng.normal(0, 0.1, n)),
            "V_E": np.where(np.arange(n) % 10 == 3, np.nan, q / (w * dbar)),
            "W": w,
            "dbar": dbar,
            "t_obs": t,
            "dmax": dbar * np.exp(t),
            "A": w * dbar,
            "D50_m": np.exp(rng.normal(math.log(0.01), 1.0, n)),
            "AI": np.exp(rng.normal(0.0, 0.4, n)),
            "damdensws": np.where(np.arange(n) % 3 == 0, 0.0, rng.uniform(0.01, 0.4, n)),
        }
    )
    for name in NN_FEATURES:
        if name not in df:
            kind = PROTOCOL.transforms.features[name]
            df[name] = np.exp(rng.normal(1.0, 0.5, n)) if kind == "log" else rng.normal(10, 3, n)
    m1 = (
        "ACC_NDAMS2010",
        "CAT_POPDENS10",
        "CAT_EVI_AMJ_2012",
        "CAT_EVI_JAS_2012",
        "CAT_EVI_JFM_2012",
        "CAT_EVI_OND_2011",
        "CAT_CLAYAVE",
        "CAT_SILTAVE",
        "CAT_SANDAVE",
        "m1_urban",
        "m1_forest",
        "m1_agriculture",
    )
    for name in m1:
        df[name] = rng.uniform(0.0, 50.0, n)
    missing = np.arange(n) % 11 == 5
    df.loc[missing, "CAT_EVI_AMJ_2012"] = np.nan
    df["F_M1_missing"] = missing
    return df


def fold_table(sites: pd.Series) -> pd.DataFrame:
    """One S-ID block: every 5th site is test; inner fold = position mod 3; ES by position mod 7."""
    n = len(sites)
    is_test = np.arange(n) % 5 == 0
    out = pd.DataFrame(
        {
            "family": "S-ID",
            "repeat": np.int8(0),
            "outer_fold": np.int8(0),
            "site_no": sites.to_numpy(),
            "is_test": is_test,
            "inner_fold": np.where(is_test, -1, np.arange(n) % 3).astype("int8"),
        }
    )
    for m in range(1, 11):
        out[f"es_member_{m}"] = ~is_test & (np.arange(n) % 7 == m % 7)
    return out


def tables(frame: pd.DataFrame | None = None) -> Tables:
    frame = site_frame() if frame is None else frame
    return Tables(frame, fold_table(frame["site_no"]), SHA_SITE, SHA_FOLD)


@pytest.fixture(scope="module")
def small_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Reduced params: tiny backbone, 6 epochs, 2-point grids (validated by the same schema)."""
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    raw["model"]["backbone"].update(hidden_layers=1, hidden_width=8)
    raw["model"]["optimizer"].update(batch_size=16, max_epochs=6, patience=2)
    raw["model"]["inner_grid"].update(weight_decay=[0.0, 1.0e-4], lr=[1.0e-3], m4_lambda_max=[10.0])
    raw["model"]["xgb"].update(
        max_depth=[2, 3], eta=[0.3], min_child_weight=[1], max_rounds=20, early_stopping_rounds=5
    )
    raw["model"]["baselines"]["m1_max_depth"] = [2]
    raw["uq"]["warm_start_epochs"] = 2
    path = tmp_path_factory.mktemp("params") / "params.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_settings(path)


def dataset(
    frame: pd.DataFrame, rows: np.ndarray, stats: NormStats, n_train: int | None = None
) -> SiteDataset:
    return SiteDataset(frame, rows, stats.names, stats, n_train or len(rows))


def fit_stats(frame: pd.DataFrame, rows: np.ndarray) -> NormStats:
    x = frame.iloc[rows][list(NN_FEATURES)].to_numpy(dtype=np.float64)
    return NormStats.fit(
        x, NN_FEATURES, PROTOCOL.transforms.features, PROTOCOL.transforms.log_offset_frac
    )


# ----------------------------------------------------------------------------- seeding, metrics
def test_derive_seed_is_stable_and_key_sensitive() -> None:
    a = derive_seed(20261008, "fit", "S-ID", 0, 0, "member", 1)
    assert a == derive_seed(20261008, "fit", "S-ID", 0, 0, "member", 1)
    assert 0 <= a < 2**32
    others = {
        derive_seed(20261008, "fit", "S-ID", 0, 0, "member", 2),
        derive_seed(20261008, "fit", "S-ID", 0, 0, "inner", 1),
        derive_seed(20261009, "fit", "S-ID", 0, 0, "member", 1),
    }
    assert a not in others
    assert len(others) == 3
    code = "from src.utils.seeding import derive_seed; print(derive_seed(1, 'x', 2))"
    env = {**os.environ, "PYTHONHASHSEED": "123"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    assert int(out.stdout) == derive_seed(1, "x", 2)  # independent of hash randomization
    with pytest.raises(TypeError):
        derive_seed(1, True)
    with pytest.raises(ValueError, match="2\\^32"):
        derive_seed(1, -1)


def test_block_and_fit_seeds_follow_protocol_3_2() -> None:
    seeds = PROTOCOL.seeds
    assert [block_seed(seeds, "S-ID", r) for r in range(3)] == list(seeds.sid_repeats)
    assert block_seed(seeds, "S-LRO", 0) == seeds.global_seed
    with pytest.raises(ValueError, match="single repeat"):
        block_seed(seeds, "S-LRO", 1)
    assert fit_seed(seeds, "S-ID", 0, 0, "member", 1) != fit_seed(seeds, "S-ID", 1, 0, "member", 1)


def test_rmse_log() -> None:
    assert rmse_log(np.array([0.0, 2.0]), np.array([0.0, 0.0])) == pytest.approx(math.sqrt(2.0))
    with pytest.raises(ValueError, match="non-finite"):
        rmse_log(np.array([np.nan]), np.array([0.0]))
    with pytest.raises(ValueError, match="equal"):
        rmse_log(np.zeros(2), np.zeros(3))


# ----------------------------------------------------------------------------- registry
def test_registry_grids_specs_and_layout(tmp_path: Path) -> None:
    grid = rg.nn_grid(PROTOCOL, "M5")
    assert len(grid) == 8
    assert grid[0] == rg.NnHyper(lr=1.0e-3, weight_decay=0.0)  # P6-D9 "grid point 1"
    assert len(rg.nn_grid(PROTOCOL, "M4")) == 24
    assert len(rg.xgb_grid_for(PROTOCOL, "M1")) == 12
    spec = rg.ModelSpec("M5", "A4")
    assert "D50_m" not in spec.features(PROTOCOL)
    assert len(spec.features(PROTOCOL)) == 20
    assert spec.hyper_source == rg.ModelSpec("M5")
    assert spec.name == "M5_A4"
    for bad in (("M4", "A1"), ("M3", "A4"), ("M9", None)):
        with pytest.raises(ValueError, match=r"unknown|does not apply"):
            rg.ModelSpec(*bad)
    d = rg.fit_dir(tmp_path, spec, "primary", "S-ID", 1, 2, "inner", 0, 3)
    assert d == tmp_path / "M5_A4" / "primary" / "S-ID" / "r1" / "f2" / "inner_0" / "g03"
    with pytest.raises(ValueError, match="non-finite"):
        rg.write_json(tmp_path / "x.json", {"a": [1.0, math.nan]})
    assert not (tmp_path / "x.json").exists()


# ----------------------------------------------------------------------------- trainer
def test_early_stopping_restarts_at_the_warm_start_switch() -> None:
    es = EarlyStopping(patience=2, restart_epoch=3)
    seq = [5.0, 4.0, 4.5, 9.0, 8.0, 8.5, 8.6]  # epoch 3 value 9 would never beat 4 without reset
    results = [es.update(e, v) for e, v in enumerate(seq)]
    assert results[3] == (True, False)  # first post-restart epoch is the new best
    assert es.best_epoch == 4
    assert es.best == 8.0
    assert results[-1] == (False, True)  # two non-improvements after epoch 4 -> stop
    with pytest.raises(FloatingPointError):
        es.update(7, math.nan)


def test_train_config() -> None:
    cfg = TrainConfig.from_protocol(PROTOCOL, lr=1e-3, weight_decay=0.0, seed=1, warm_start=True)
    assert (cfg.batch_size, cfg.max_epochs, cfg.patience, cfg.restart_epoch) == (256, 500, 50, 20)
    assert (
        TrainConfig.from_protocol(
            PROTOCOL, lr=1e-3, weight_decay=0.0, seed=1, warm_start=False
        ).restart_epoch
        is None
    )
    with pytest.raises(ValueError, match="restart_epoch"):
        replace(cfg, restart_epoch=500)


def _tiny_parts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, NormStats]:
    tr, va = np.arange(0, 40), np.arange(40, 60)
    return tr, va, fit_stats(frame, tr)


def _model(
    spec: rg.ModelSpec, settings: Settings, stats: NormStats, blind: torch.Tensor, **kw: Any
) -> Any:
    torch.manual_seed(0)
    hyper = rg.nn_grid(settings.protocol, spec.model)[0]
    return build_model(spec, settings.protocol, stats=stats, blind_idx=blind, hyper=hyper, **kw)


def _cfg(settings: Settings, seed: int = 3, warm: bool = True) -> TrainConfig:
    return TrainConfig.from_protocol(
        settings.protocol, lr=1e-3, weight_decay=1e-4, seed=seed, warm_start=warm
    )


def test_fit_is_deterministic_and_restores_the_best_state(small_settings: Settings) -> None:
    frame = site_frame()
    tr, va, stats = _tiny_parts(frame)
    train, val = dataset(frame, tr, stats), dataset(frame, va, stats)

    def run(seed: int) -> tuple[dict[str, torch.Tensor], Any]:
        model = _model(rg.ModelSpec("M5"), small_settings, stats, train.x_blind_idx)
        res = fit(model, train, val, _cfg(small_settings, seed))
        return model.state_dict(), res

    s1, r1 = run(3)
    s2, r2 = run(3)
    s3, _ = run(4)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert [h.monitor for h in r1.history] == [h.monitor for h in r2.history]
    assert any(not torch.equal(s1[k], s3[k]) for k in s1)  # loader order differs
    warm = small_settings.protocol.uq.warm_start_epochs
    assert r1.best_epoch >= warm
    assert r1.best_monitor == r1.history[r1.best_epoch].monitor
    assert r1.best_monitor == min(h.monitor for h in r1.history[warm:])
    assert all(h.mean_iterations is not None for h in r1.history)


def test_m4_monitor_uses_lambda_max_and_xi_is_logged(small_settings: Settings) -> None:
    frame = site_frame()
    tr, va, stats = _tiny_parts(frame)
    train, val = dataset(frame, tr, stats), dataset(frame, va, stats)
    model = _model(rg.ModelSpec("M4"), small_settings, stats, train.x_blind_idx)
    assert isinstance(model, M4Soft)
    loader = make_loader(val, 16, train=False, seed=0)
    batch = val[list(range(len(val)))]
    with torch.no_grad():
        out = model(batch)
        at_max = float(model.objective(batch, out, 0, monitor=True).total)
        at_e0 = float(model.objective(batch, out, 0).total)
    assert evaluate_monitor(model, loader, 0) == pytest.approx(at_max, rel=1e-12)
    assert at_max > at_e0  # lambda_max = 10 > lambda_M(0) = 1
    full = train[list(range(len(train)))]
    xi = xi_ratio(model, full, 0)
    params = list(model.parameters())
    parts = model.objective(full, model(full), 0).parts
    gd = torch.autograd.grad(parts["data"], params, retain_graph=True, allow_unused=True)
    gp = torch.autograd.grad(parts["penalty"], params, allow_unused=True)

    def nrm(gs: tuple[torch.Tensor | None, ...]) -> float:
        return math.sqrt(sum(float((g**2).sum()) for g in gs if g is not None))

    assert xi == pytest.approx(nrm(gp) / nrm(gd), rel=1e-10)
    res = fit(model, train, val, _cfg(small_settings), xi=True)
    assert all(h.xi is not None and h.xi > 0 for h in res.history)
    assert all(set(h.parts) == {"data", "penalty"} for h in res.history)


def test_phi_nonconvergence_propagates(small_settings: Settings) -> None:
    frame = site_frame()
    tr, va, stats = _tiny_parts(frame)
    train, val = dataset(frame, tr, stats), dataset(frame, va, stats)
    model = _model(rg.ModelSpec("M5"), small_settings, stats, train.x_blind_idx)
    assert isinstance(model, M5Hard)
    model.cfg = model.cfg.with_flags(phi=replace(model.cfg.phi, k_max=1))
    with pytest.raises(PhiConvergenceError):
        fit(model, train, val, _cfg(small_settings))


def test_predict_matches_one_forward_pass(small_settings: Settings) -> None:
    frame = site_frame()
    _tr, va, stats = _tiny_parts(frame)
    val = dataset(frame, va, stats)
    model = _model(rg.ModelSpec("M2-MLP"), small_settings, stats, val.x_blind_idx).eval()
    out = predict(model, val, batch_size=7)
    with torch.no_grad():
        ref = model(val[list(range(len(val)))])
    assert torch.allclose(out.mu, ref.mu, rtol=0, atol=1e-14)
    assert out.sigma is not None
    assert ref.sigma is not None
    assert torch.allclose(out.sigma, ref.sigma, rtol=0, atol=1e-14)


# ----------------------------------------------------------------------------- checkpoints
@pytest.mark.parametrize("model_name", ["M2-MLP", "M3", "M4", "M5", "M6-NN"])
def test_checkpoint_round_trip_is_bit_identical(
    small_settings: Settings, tmp_path: Path, model_name: str
) -> None:
    frame = site_frame()
    _tr, va, stats = _tiny_parts(frame)
    stats = replace(
        stats,
        provenance=tuple(
            sorted(
                {
                    "analysis": "primary",
                    "family": "S-ID",
                    "fold_table_sha256": SHA_FOLD,
                    "index": "1",
                    "outer_fold": "0",
                    "part": "es_train",
                    "repeat": "0",
                    "site_table_sha256": SHA_SITE,
                }.items()
            )
        ),
    )
    val = dataset(frame, va, stats)
    spec = rg.ModelSpec(model_name)
    r_bar = 1.5 if model_name == "M3" else None
    model = _model(spec, small_settings, stats, val.x_blind_idx, r_bar=r_bar).eval()
    hyper = rg.nn_grid(small_settings.protocol, model_name)[0]
    save_model(
        tmp_path,
        model,
        spec=spec,
        stats=stats,
        blind_idx=val.x_blind_idx.tolist(),
        hyper=hyper,
        r_bar=r_bar,
        params_sha256=small_settings.params_sha256,
    )
    assert {p.name for p in tmp_path.iterdir()} == {STATE_FILE, MODEL_FILE, NORMSTATS_FILE}
    loaded, stats2, _ = load_model(
        tmp_path, small_settings.protocol, params_sha256=small_settings.params_sha256
    )
    assert stats2 == stats
    sd1, sd2 = model.state_dict(), loaded.state_dict()
    assert sd1.keys() == sd2.keys()
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    batch = val[list(range(len(val)))]
    with torch.no_grad():
        assert torch.equal(model(batch).mu, loaded(batch).mu)
    with pytest.raises(ValueError, match="params"):
        load_model(tmp_path, small_settings.protocol, params_sha256="0" * 64)


def test_param_count_and_row_wise_guard(small_settings: Settings) -> None:
    frame = site_frame()
    _, va, stats = _tiny_parts(frame)
    val = dataset(frame, va, stats)
    model = _model(rg.ModelSpec("M5"), small_settings, stats, val.x_blind_idx)
    n = sum(p.numel() for p in model.parameters())
    assert model.get_num_trainable_params() == n
    assert model.heads.get_num_trainable_params() == n
    assert model.heads.full_trunk.get_num_trainable_params() < n
    model.heads.full_trunk.net.append(torch.nn.BatchNorm1d(8, dtype=torch.float64))
    with pytest.raises(TypeError, match="batch-coupled"):
        assert_row_wise(model)


def test_state_npz_hash_is_checked(small_settings: Settings, tmp_path: Path) -> None:
    frame = site_frame()
    _, va, stats = _tiny_parts(frame)
    val = dataset(frame, va, stats)
    model = _model(rg.ModelSpec("M2-MLP"), small_settings, stats, val.x_blind_idx)
    meta = save_state(tmp_path, model)
    load_state(tmp_path, model, sha256=meta["sha256"])
    with pytest.raises(ValueError, match="SHA-256"):
        load_state(tmp_path, model, sha256="0" * 64)
    with np.load(tmp_path / STATE_FILE, allow_pickle=False) as npz:
        assert set(npz.files) == set(model.state_dict())


# ----------------------------------------------------------------------------- runner
def req(model: str, kind: str, index: int, grid: int | None = None, **kw: Any) -> FitRequest:
    return FitRequest(
        spec=rg.ModelSpec(model, kw.pop("ablation", None)),
        analysis=kw.pop("analysis", "primary"),
        family="S-ID",
        repeat=0,
        outer_fold=0,
        kind=kind,  # type: ignore[arg-type]
        index=index,
        grid=grid,
    )


def test_test_rows_are_blanked_and_cannot_move_normstats(
    small_settings: Settings, tmp_path: Path
) -> None:
    t = tables()
    rows = _resolve_rows(t, req("M5", "member", 1, 0), NN_FEATURES)
    test = np.flatnonzero(t.folds["is_test"].to_numpy())
    assert (
        rows.frame.iloc[test][[*list(NN_FEATURES), "W", "dbar", "t_obs", "Q_E", "S"]]
        .isna()
        .all()
        .all()
    )
    assert not np.intersect1d(np.concatenate([rows.train, rows.val]), test).size
    with pytest.raises(ValueError, match="non-finite"):
        SiteDataset(rows.frame, test, NN_FEATURES, fit_stats(rows.frame, rows.train), 1)
    # changing every test-row value leaves the fitted NormStats unchanged (L5)
    other = site_frame()
    numeric = [c for c in NN_FEATURES if c != "F_S"]
    other.loc[test, numeric] = other.loc[test, numeric] * 7.0
    other.loc[test, ["W", "dbar"]] = np.nan
    a = run_fit(req("M2-MLP", "inner", 0, 0), t, small_settings, tmp_path / "a")
    b = run_fit(req("M2-MLP", "inner", 0, 0), tables(other), small_settings, tmp_path / "b")
    sub = Path("M2-MLP/primary/S-ID/r0/f0/inner_0/g00") / NORMSTATS_FILE
    assert (tmp_path / "a" / sub).read_text() == (tmp_path / "b" / sub).read_text()
    assert a["score"] == b["score"]


def test_nn_inner_grid_selection_and_member(small_settings: Settings, tmp_path: Path) -> None:
    t = tables()
    for g in range(2):
        for k in range(3):
            doc = run_fit(req("M2-MLP", "inner", k, g), t, small_settings, tmp_path)
            assert doc["parts"] == {"train": "inner_train", "val": "inner_val", "index": k}
            assert set(doc["score"]) == {"W", "dbar", "t", "A"}
            assert doc["header"]["params_sha256"] == small_settings.params_sha256
    sel = run_select(rg.ModelSpec("M2-MLP"), "primary", "S-ID", 0, 0, small_settings, tmp_path)
    means = sel["mean_score"]
    assert sel["selected"] == int(np.argmin(means))
    assert sel["target"] == "dbar"
    with pytest.raises(FileExistsError):
        run_select(rg.ModelSpec("M2-MLP"), "primary", "S-ID", 0, 0, small_settings, tmp_path)
    doc = run_fit(req("M2-MLP", "member", 1), t, small_settings, tmp_path)
    assert doc["hyper_source"]["kind"] == "selection"
    assert doc["hyper_source"]["grid"] == sel["selected"]
    assert doc["score"] is None
    member = tmp_path / "M2-MLP/primary/S-ID/r0/f0/member_1"
    assert {p.name for p in member.iterdir()} == {
        STATE_FILE,
        MODEL_FILE,
        NORMSTATS_FILE,
        "run.json",
    }
    prov = NormStats.load(member / NORMSTATS_FILE).provenance_dict
    assert (prov["part"], prov["index"], prov["site_table_sha256"]) == ("es_train", "1", SHA_SITE)
    with pytest.raises(FileExistsError):
        run_fit(req("M2-MLP", "member", 1), t, small_settings, tmp_path)
    run_fit(req("M2-MLP", "member", 1), t, small_settings, tmp_path, overwrite=True)


def test_select_requires_every_inner_fit(small_settings: Settings, tmp_path: Path) -> None:
    run_fit(req("M2-MLP", "inner", 0, 0), tables(), small_settings, tmp_path)
    with pytest.raises(FileNotFoundError, match="5 inner fits missing"):
        run_select(rg.ModelSpec("M2-MLP"), "primary", "S-ID", 0, 0, small_settings, tmp_path)


def test_m5_explicit_member_records_picard_iterations(
    small_settings: Settings, tmp_path: Path
) -> None:
    doc = run_fit(req("M5", "member", 2, 0), tables(), small_settings, tmp_path)
    assert doc["hyper_source"] == {"kind": "explicit", "grid": 0}
    assert doc["extras"]["mean_picard_iterations"] > 0
    assert doc["extras"]["n_trainable_params"] > 0
    assert doc["fit"]["best_epoch"] >= small_settings.protocol.uq.warm_start_epochs
    assert doc["rows"]["test_blanked"] == 12


def test_ablation_member_reuses_the_m5_selection(small_settings: Settings, tmp_path: Path) -> None:
    sel = rg.block_dir(tmp_path, rg.ModelSpec("M5"), "primary", "S-ID", 0, 0) / "selection.json"
    rg.write_json(sel, {"kind": "nn", "selected": 1})
    doc = run_fit(req("M5", "member", 1, ablation="A3-SR"), tables(), small_settings, tmp_path)
    assert doc["hyper_source"]["model"] == "M5"
    assert doc["hyper"]["weight_decay"] == 1.0e-4
    doc = run_fit(req("M5", "member", 1, analysis="SA1"), tables(), small_settings, tmp_path)
    assert doc["hyper_source"]["path"].startswith("M5/primary/")  # Q-D3 (a)
    with pytest.raises(ValueError, match="ablations reuse"):
        req("M5", "inner", 0, 0, ablation="A1")


def test_m4_m3_m6_members(small_settings: Settings, tmp_path: Path) -> None:
    t = tables()
    m4 = run_fit(req("M4", "member", 1, 0), t, small_settings, tmp_path)
    assert all(h["xi"] > 0 for h in m4["fit"]["history"])
    assert m4["hyper"]["lambda_max"] == 10.0
    m3 = run_fit(req("M3", "member", 1, 0), t, small_settings, tmp_path)
    outer = np.flatnonzero(~t.folds["is_test"].to_numpy())
    assert m3["extras"]["r_bar"] == m3_r_bar(t.site_table["t_obs"].to_numpy()[outer])
    m3i = run_fit(req("M3", "inner", 0, 0), t, small_settings, tmp_path)
    assert m3i["extras"]["r_bar"] != m3["extras"]["r_bar"]
    m6 = run_fit(req("M6-NN", "member", 1, 0), t, small_settings, tmp_path)
    assert 0.0 <= m6["extras"]["label_nu_outside_share"] <= 1.0
    assert m6["fit"]["history"][0]["mean_iterations"] is not None


def test_m6_xgb_uses_blind_features_for_nu(small_settings: Settings, tmp_path: Path) -> None:
    t = tables()
    doc = run_fit(req("M6-XGB", "inner", 0, 1), t, small_settings, tmp_path)
    assert set(doc["score"]) == {"nu", "w", "t"}
    nu_feats = doc["fit"]["nu"]["features"]
    assert not {"Q_E", "S", "F_S"} & set(nu_feats)
    assert "Q_E" in doc["fit"]["w"]["features"]
    run_fit(req("M6-XGB", "member", 1, 0), t, small_settings, tmp_path)
    member = tmp_path / "M6-XGB/primary/S-ID/r0/f0/member_1"
    assert {p.name for p in member.iterdir()} == {
        "booster_nu.json",
        "booster_w.json",
        "booster_t.json",
        "run.json",
    }
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("m6xgb_")]


def test_runner_rejects_srnd() -> None:
    with pytest.raises(ValueError, match="S-RND"):
        FitRequest(rg.ModelSpec("M5"), "primary", "S-RND", 0, 0, "member", 1)


# ----------------------------------------------------------------------------- CLI, torch-free XGB
def test_cli_xgb_path_is_torch_free(small_settings: Settings, tmp_path: Path) -> None:
    t = tables()
    site, folds = tmp_path / "site.parquet", tmp_path / "folds.parquet"
    t.site_table.to_parquet(site)
    t.folds.to_parquet(folds)
    ck = tmp_path / "ck"
    common = [
        "--family",
        "S-ID",
        "--fold",
        "0",
        "--params",
        str(small_settings.params_path),
        "--site-table",
        str(site),
        "--fold-table",
        str(folds),
        "--checkpoint-dir",
        str(ck),
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    script = str(REPO_ROOT / "scripts" / "train.py")
    loop = (
        "import sys, runpy\n"
        "for g in range(2):\n"
        "    for k in range(3):\n"
        f"        sys.argv = ['train.py', 'fit', '--model', 'M2-XGB', *{common!r}, "
        "'--kind', 'inner', '--index', str(k), '--grid', str(g)]\n"
        f"        try:\n            runpy.run_path({script!r}, run_name='__main__')\n"
        "        except SystemExit as e:\n            assert e.code == 0\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", loop], check=True, cwd=REPO_ROOT, env=env, capture_output=True
    )
    for args in (
        ["select", "--model", "M2-XGB", *common],
        ["fit", "--model", "M2-XGB", *common, "--kind", "member", "--index", "1"],
        ["fit", "--model", "M1", *common, "--kind", "member", "--index", "1", "--grid", "0"],
    ):
        subprocess.run(
            [sys.executable, script, *args], check=True, cwd=REPO_ROOT, env=env, capture_output=True
        )
    block = ck / "M2-XGB/primary/S-ID/r0/f0"
    sel = json.loads((block / "selection.json").read_text())
    assert sel["kind"] == "xgb"
    assert set(sel["targets"]) == {"W", "dbar", "dmax", "A"}
    assert {p.name for p in (block / "member_1").iterdir()} == {
        "booster_W.json",
        "booster_dbar.json",
        "booster_dmax.json",
        "booster_A.json",
        "run.json",
    }
    m1 = json.loads((ck / "M1/primary/S-ID/r0/f0/member_1/run.json").read_text())
    dropped = m1["extras"]["m1_missing_dropped"]
    assert dropped["train"] + dropped["val"] > 0
    assert m1["features"] == list(rg.ModelSpec("M1").features(small_settings.protocol))


def test_new_modules_import_neither_torch_nor_xgboost() -> None:
    code = (
        "import sys, src.models.registry, src.models.runner, src.evaluation.metrics, "
        "src.utils.seeding, src.utils.logging; "
        "print('torch' in sys.modules, 'xgboost' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert out.stdout.strip() == "False False"
