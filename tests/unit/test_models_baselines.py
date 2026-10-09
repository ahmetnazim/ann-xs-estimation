"""Tests for src/models/baselines.py (Phase 6, Batch C): torch-free baselines and XGB workers.

Synthetic data only. In-process XGB fits use ``nthread = 1`` because this pytest process also
holds torch (ADL-051); the isolated path runs in a fresh torch-free interpreter.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings
from src.data.dataset import BLIND_EXCLUDED as DATASET_BLIND
from src.data.schemas import BANNED_PREDICTORS
from src.models import baselines as bl

pytestmark = pytest.mark.unit

PROTOCOL = load_settings().protocol
XGB_FAST = PROTOCOL.model.xgb.model_copy(update={"max_rounds": 60, "early_stopping_rounds": 10})


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- feature sets -----------------------------------------------------------------------------


def test_feature_sets() -> None:
    nn = bl.nn_features(PROTOCOL)
    assert len(nn) == 22
    assert nn == tuple(PROTOCOL.transforms.features)
    a4 = bl.a4_features(nn)
    assert len(a4) == 20
    assert "D50_m" not in a4
    assert "AI" not in a4
    with pytest.raises(ValueError, match="A4 needs"):
        bl.a4_features(a4)
    blind = bl.blind_features(nn)
    assert set(nn) - set(blind) == {"Q_E", "S", "F_S"}
    assert bl.BLIND_EXCLUDED == DATASET_BLIND
    assert len(bl.M1_FEATURES) == 19
    assert len(set(bl.M1_FEATURES)) == 19
    assert not set(bl.M1_FEATURES) & BANNED_PREDICTORS
    assert "V_E" not in bl.M1_FEATURES


def test_log_targets() -> None:
    frame = pd.DataFrame(
        {"W": [10.0, 20.0], "dbar": [1.0, 2.0], "dmax": [2.0, 3.0], "A": [10.0, 40.0]}
    )
    y = bl.log_targets(frame)
    assert np.allclose(y, np.log(frame.to_numpy()))
    with pytest.raises(ValueError, match="> 0"):
        bl.log_targets(frame.assign(W=[0.0, 1.0]))
    with pytest.raises(KeyError, match="lacks"):
        bl.log_targets(frame.drop(columns="A"))


# --- M-null -----------------------------------------------------------------------------------


def test_mnull_quintile_medians() -> None:
    da = np.arange(1.0, 101.0)
    y = np.column_stack([np.log(da), np.log(da) / 2, np.log(da) / 3, np.log(da) / 4])
    model = bl.MNull.fit(da, y)
    assert model.cuts == pytest.approx(tuple(np.quantile(da, [0.2, 0.4, 0.6, 0.8])))
    pred = model.predict(np.array([1.0, 50.0, 100.0]))
    assert set(pred) == set(bl.LOG_TARGETS)
    low = np.median(np.log(da[da < model.cuts[0]]))
    assert pred["W"][0] == pytest.approx(low)
    cut = model.cuts[1]
    on_cut = model.predict(np.array([cut]))["W"][0]
    assert on_cut == pytest.approx(model.medians[2][0])  # DA == cut -> upper class
    with pytest.raises(ValueError, match="empty"):
        bl.MNull.fit(np.ones(10), np.zeros((10, 4)))


# --- A_J --------------------------------------------------------------------------------------


def test_aj_log_area_with_missing_velocity() -> None:
    q = np.array([10.0, 20.0, 5.0])
    v = np.array([0.5, np.nan, 1.0])
    ln_a, ok = bl.aj_log_area(q, v)
    assert ok.tolist() == [True, False, True]
    assert ln_a[0] == pytest.approx(math.log(20.0))
    assert np.isnan(ln_a[1])
    with pytest.raises(ValueError, match="V_E"):
        bl.aj_log_area(q, np.array([0.5, 0.0, 1.0]))
    with pytest.raises(ValueError, match="Q_E"):
        bl.aj_log_area(np.array([0.0, 1.0, 1.0]), v)


# --- M0 ---------------------------------------------------------------------------------------


def test_m0_recovers_division_curves_and_pools() -> None:
    g = rng(1)
    n = 120
    div = np.array(["A"] * 60 + ["B"] * 40 + ["C"] * 20)
    da = np.exp(g.uniform(0, 8, n))
    coef = {"A": (1.0, 0.4, -1.0, 0.3), "B": (2.0, 0.5, -0.5, 0.2), "C": (0.0, 0.1, 0.0, 0.1)}
    ln_w = np.array([coef[d][0] + coef[d][1] * math.log(a) for d, a in zip(div, da, strict=True)])
    ln_d = np.array([coef[d][2] + coef[d][3] * math.log(a) for d, a in zip(div, da, strict=True)])
    model = bl.M0PowerLaw.fit(div, da, ln_w, ln_d, min_sites=30)
    table = dict(model.coef)
    assert set(table) == {"A", "B"}
    assert table["A"] == pytest.approx(coef["A"], abs=1e-10)
    assert table["B"] == pytest.approx(coef["B"], abs=1e-10)
    pred, pooled = model.predict(np.array(["A", "C", "Z"]), np.array([10.0, 10.0, 10.0]))
    assert pooled.tolist() == [False, True, True]
    assert pred["W"][0] == pytest.approx(1.0 + 0.4 * math.log(10.0))
    p = model.pooled
    assert pred["W"][2] == pytest.approx(p[0] + p[1] * math.log(10.0))
    assert np.allclose(pred["A"], pred["W"] + pred["dbar"])


# --- M1 floor and grids -----------------------------------------------------------------------


def test_m1_floor_counts() -> None:
    y = np.linspace(0.1, 10.0, 1000)
    pred = np.array([-1.0, 0.0, 0.05, 2.0])
    out, n = bl.m1_floor(pred, y, PROTOCOL.model.baselines.m1_floor_quantile)
    floor = float(np.quantile(y, 0.001))
    assert n == 2
    assert out.tolist() == pytest.approx([floor, floor, 0.05, 2.0])


def test_xgb_grids() -> None:
    grid = bl.xgb_grid(PROTOCOL.model.xgb)
    assert len(grid) == 12
    assert {h.max_depth for h in grid} == {4, 6, 8}
    m1 = bl.m1_grid(PROTOCOL)
    assert len(m1) == 12
    assert {h.max_depth for h in m1} == set(PROTOCOL.model.baselines.m1_max_depth)


# --- XGB fit, predict, isolation --------------------------------------------------------------


def toy(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    g = rng(seed)
    x = g.normal(size=(n, 4))
    y = 2.0 * x[:, 0] - x[:, 1] ** 2 + 0.1 * g.normal(size=n)
    return x, y


def test_fit_booster_in_process_and_isolated_agree(tmp_path: Path) -> None:
    x_tr, y_tr = toy(300, 0)
    x_va, y_va = toy(100, 1)
    x_te, _ = toy(50, 2)
    hyper = bl.XgbHyper(max_depth=4, eta=0.1, min_child_weight=1.0)
    fit = bl.fit_booster(x_tr, y_tr, x_va, y_va, hyper, XGB_FAST, seed=7, nthread=1)
    assert 0 <= fit.best_iteration < XGB_FAST.max_rounds
    assert math.isfinite(fit.best_score)
    pred = bl.predict_booster(fit, x_te)
    assert pred.shape == (50,)
    assert pred.dtype == np.float64
    iso, preds = bl.fit_booster_isolated(
        tmp_path / "w", x_tr, y_tr, x_va, y_va, hyper, XGB_FAST, seed=7, predict={"te": x_te}
    )
    assert iso.best_iteration == fit.best_iteration
    assert np.array_equal(preds["te"], pred)
    assert np.array_equal(bl.predict_booster(iso, x_te), pred)
    assert (tmp_path / "w" / "booster.json").read_bytes()[:1] == b"{"
    assert not list((tmp_path / "w").glob("*.pkl"))


def test_fit_booster_rejects_bad_input(tmp_path: Path) -> None:
    x, y = toy(20, 3)
    hyper = bl.XgbHyper(max_depth=2, eta=0.1, min_child_weight=1.0)
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bl.fit_booster(bad, y, x, y, hyper, XGB_FAST, seed=0)
    with pytest.raises(ValueError, match="float64"):
        bl.fit_booster(x.astype(np.float32), y, x, y, hyper, XGB_FAST, seed=0)
    with pytest.raises(ValueError, match="non-finite"):
        bl.fit_booster_isolated(tmp_path, bad, y, x, y, hyper, XGB_FAST, seed=0)
