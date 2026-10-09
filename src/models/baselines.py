"""Baselines M-null, A_J = Q_E / V_E, M0 (per-division power law), M1 and M2-XGB; XGB workers.

Sources: protocol section 4.1 (ladder rows M-null, A_J, M0, M1, M2-XGB; P3-D8, O-2 (b),
O-11; M0 pooled national curve for a held-out division or one with fewer than 30 training sites;
M1 = Zarrabi Tier-2 XGB on their target scale, negative predictions floored at the 0.1th
training percentile with counts reported), section 4.2 (XGB grid ``max_depth`` x ``eta`` x
``min_child_weight``; subsample 0.8; colsample 0.8; at most 2000 rounds; early stopping 100),
section 3.4 (XGB uses raw feature values; nothing is fitted outside the training part);
ADL-051 (OpenMP rule: XGB fits run in processes that never import torch; ``nthread = 1`` where a
process is shared); P6-D7 (a) (this module is torch-free; xgboost is imported only inside the
functions that fit or predict, so importing this module never loads it); P6-D8 and Q-C2 (b)
(M1: raw metres; ``max_depth`` grid ``model.baselines.m1_max_depth``, the shared values);
P6-Q4 (a) (one booster per target on ``es_train(1)`` with early stopping on ``es_val(1)``).

Every predictor returns natural-log predictions of its targets, keyed by
``W``, ``dbar``, ``dmax``, ``A``. Missing inputs raise, except ``V_E`` for A_J, whose 669 missing
rows (ADL-066) are returned as unavailable with a mask, never imputed.

XGB process isolation: :func:`fit_booster_isolated` writes ``.npy`` arrays
(``allow_pickle=False``) and a JSON spec to a work directory and runs
``python -m src.models.baselines --xgb-fit SPEC`` in a fresh interpreter; the booster comes back
as XGBoost JSON. No pickle is used anywhere.

Owning phase: 6.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.config import ProtocolParams, XgbParams

__all__ = [
    "A4_DROPPED",
    "BLIND_EXCLUDED",
    "LOG_TARGETS",
    "M1_FEATURES",
    "M1_TARGETS",
    "BoosterFit",
    "M0PowerLaw",
    "MNull",
    "XgbHyper",
    "a4_features",
    "aj_log_area",
    "blind_features",
    "fit_booster",
    "fit_booster_isolated",
    "log_targets",
    "m1_floor",
    "m1_grid",
    "main",
    "nn_features",
    "predict_booster",
    "xgb_grid",
]

#: Zarrabi Tier-2 ("Model 2") predictors in site-table names (protocol 4.1, O-2 (b); text check
#: of the notebooks at ``a30f1f5``: 19 features, measured discharge excluded).
M1_FEATURES: Final[tuple[str, ...]] = (
    "Q_E",
    "DA_km2",
    "S",
    "stream_order",
    "z_min_m",
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
    "D50_m",
    "AI",
)
#: M1 targets (protocol 4.1: W and dbar; ``A = W dbar``).
M1_TARGETS: Final[tuple[str, ...]] = ("W", "dbar")
#: Log targets of M-null and M2-XGB (protocol 4.1).
LOG_TARGETS: Final[tuple[str, ...]] = ("W", "dbar", "dmax", "A")
#: Ablation A4 drops D50 and AI (protocol 4.1).
A4_DROPPED: Final[tuple[str, ...]] = ("D50_m", "AI")
#: I_blind exclusions (protocol 4.1, P3-11); equal to ``src.data.dataset.BLIND_EXCLUDED``
#: (tested), restated here because that module imports torch.
BLIND_EXCLUDED: Final[frozenset[str]] = frozenset({"Q_E", "S", "F_S", "V_E"})

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- feature sets
def nn_features(protocol: ProtocolParams) -> tuple[str, ...]:
    """Return the NN feature set in ``params.transforms.features`` order (M2, M4-M6; P5-D5)."""
    return tuple(protocol.transforms.features)


def a4_features(features: Sequence[str]) -> tuple[str, ...]:
    """Ablation A4: ``features`` without D50 and AI (protocol 4.1); both must be present."""
    missing = [f for f in A4_DROPPED if f not in features]
    if missing:
        raise ValueError(f"A4 needs {missing} in the feature set")
    return tuple(f for f in features if f not in A4_DROPPED)


def blind_features(features: Sequence[str]) -> tuple[str, ...]:
    """``I_blind``: ``features`` without ``Q_E``, ``S``, ``F_S``, ``V_E`` (protocol 4.1)."""
    return tuple(f for f in features if f not in BLIND_EXCLUDED)


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"frame lacks {missing}")
    x = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(x).all()):
        raise ValueError("non-finite inputs (no imputation, protocol 2.7)")
    return x


def log_targets(frame: pd.DataFrame) -> np.ndarray:
    """``(n, 4)`` ``[ln W, ln dbar, ln d_max, ln A]`` from the site table (SI units)."""
    x = _matrix(frame, ("W", "dbar", "dmax", "A"))
    if not bool((x > 0).all()):
        raise ValueError("targets must be > 0 (invariant 5)")
    return np.log(x)


# ----------------------------------------------------------------------------- M-null
@dataclass(frozen=True)
class MNull:
    """Training-fold median of each log target per drainage-area quintile (protocol 4.1).

    ``cuts`` are the 20/40/60/80 % training quantiles of DA (``numpy.quantile``, linear); a site
    with ``DA == cut`` falls in the upper class. ``medians`` is ``(5, 4)`` over ``LOG_TARGETS``.
    """

    cuts: tuple[float, ...]
    medians: tuple[tuple[float, ...], ...]

    @classmethod
    def fit(cls, da: np.ndarray, y_log: np.ndarray) -> MNull:
        """Fit on training rows; raises if a quintile class is empty (heavy DA ties)."""
        da, y_log = np.asarray(da, dtype=np.float64), np.asarray(y_log, dtype=np.float64)
        if da.ndim != 1 or y_log.shape != (da.size, 4) or da.size < 5:
            raise ValueError("need da (n >= 5,) and y_log (n, 4)")
        if not (bool(np.isfinite(da).all()) and bool(np.isfinite(y_log).all())):
            raise ValueError("non-finite training values")
        cuts = np.quantile(da, [0.2, 0.4, 0.6, 0.8])
        cls_idx = np.searchsorted(cuts, da, side="right")
        medians = []
        for k in range(5):
            rows = y_log[cls_idx == k]
            if rows.shape[0] == 0:
                raise ValueError(f"empty DA quintile class {k}")
            medians.append(tuple(float(v) for v in np.median(rows, axis=0)))
        return cls(cuts=tuple(float(c) for c in cuts), medians=tuple(medians))

    def predict(self, da: np.ndarray) -> dict[str, np.ndarray]:
        """Log predictions keyed by ``LOG_TARGETS``."""
        da = np.asarray(da, dtype=np.float64)
        if not bool(np.isfinite(da).all()):
            raise ValueError("non-finite DA")
        cls_idx = np.searchsorted(np.asarray(self.cuts), da, side="right")
        table = np.asarray(self.medians)[cls_idx]
        return {name: table[:, j] for j, name in enumerate(LOG_TARGETS)}


# ----------------------------------------------------------------------------- A_J
def aj_log_area(q_e: np.ndarray, v_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``ln A_J = ln Q_E - ln V_E`` (protocol 4.1; training-free) and the availability mask.

    Rows with missing ``V_E`` (NaN, ADL-066) are unavailable and returned as NaN with
    ``available = False``; they are reported, never imputed. A non-positive ``Q_E`` or ``V_E``
    raises.
    """
    q_e, v_e = np.asarray(q_e, dtype=np.float64), np.asarray(v_e, dtype=np.float64)
    if q_e.shape != v_e.shape:
        raise ValueError("q_e and v_e must have the same shape")
    if not (bool(np.isfinite(q_e).all()) and bool((q_e > 0).all())):
        raise ValueError("Q_E must be finite and > 0 (R13)")
    available = ~np.isnan(v_e)
    if not (bool(np.isfinite(v_e[available]).all()) and bool((v_e[available] > 0).all())):
        raise ValueError("V_E must be > 0 where present")
    out = np.full_like(q_e, np.nan)
    out[available] = np.log(q_e[available]) - np.log(v_e[available])
    return out, available


# ----------------------------------------------------------------------------- M0
def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


@dataclass(frozen=True)
class M0PowerLaw:
    """Bieger-form ``ln y = a_g + b_g ln DA`` refit per division ``g`` (protocol 4.1, O-11).

    ``coef`` maps a division to ``(a_W, b_W, a_dbar, b_dbar)``; divisions with fewer than
    ``min_sites`` training sites are absent and use ``pooled`` (the national training-fold curve),
    as do divisions unseen in training (S-LRO). ``ln A = ln W + ln dbar``.
    """

    coef: tuple[tuple[str, tuple[float, float, float, float]], ...]
    pooled: tuple[float, float, float, float]
    min_sites: int

    @classmethod
    def fit(
        cls,
        division: np.ndarray,
        da: np.ndarray,
        ln_w: np.ndarray,
        ln_dbar: np.ndarray,
        min_sites: int,
    ) -> M0PowerLaw:
        """Ordinary least squares per division and pooled, on the training rows only."""
        div = np.asarray(division).astype(str)
        x = np.log(np.asarray(da, dtype=np.float64))
        yw, yd = np.asarray(ln_w, dtype=np.float64), np.asarray(ln_dbar, dtype=np.float64)
        if not (div.shape == x.shape == yw.shape == yd.shape) or x.ndim != 1:
            raise ValueError("inputs must be 1-D of equal length")
        if not all(bool(np.isfinite(v).all()) for v in (x, yw, yd)):
            raise ValueError("non-finite training values (DA must be > 0)")
        pooled = (*_ols(x, yw), *_ols(x, yd))
        coef = []
        for g in sorted(set(div.tolist())):
            m = div == g
            if int(m.sum()) >= min_sites:
                coef.append((g, (*_ols(x[m], yw[m]), *_ols(x[m], yd[m]))))
        return cls(coef=tuple(coef), pooled=pooled, min_sites=min_sites)

    def predict(
        self, division: np.ndarray, da: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Log predictions for ``W``, ``dbar``, ``A`` and the pooled-fallback mask."""
        div = np.asarray(division).astype(str)
        x = np.log(np.asarray(da, dtype=np.float64))
        if not bool(np.isfinite(x).all()):
            raise ValueError("DA must be finite and > 0")
        table = dict(self.coef)
        rows = np.asarray([table.get(g, self.pooled) for g in div.tolist()]).reshape(-1, 4)
        pooled = np.asarray([g not in table for g in div.tolist()], dtype=bool)
        ln_w = rows[:, 0] + rows[:, 1] * x
        ln_d = rows[:, 2] + rows[:, 3] * x
        return {"W": ln_w, "dbar": ln_d, "A": ln_w + ln_d}, pooled


# ----------------------------------------------------------------------------- M1 floor
def m1_floor(
    pred_raw: np.ndarray, y_train_raw: np.ndarray, quantile: float
) -> tuple[np.ndarray, int]:
    """Floor non-positive raw-metre M1 predictions at the training ``quantile`` (protocol 4.1).

    Pre-registered: negative predictions are replaced by the 0.1th training percentile
    (``model.baselines.m1_floor_quantile``); a prediction of exactly 0 is treated the same way,
    because its log is undefined. Returns ``(floored, count)``; the count is reported.
    """
    pred = np.asarray(pred_raw, dtype=np.float64)
    y = np.asarray(y_train_raw, dtype=np.float64)
    if not (bool(np.isfinite(pred).all()) and bool(np.isfinite(y).all())):
        raise ValueError("non-finite predictions or training targets")
    floor = float(np.quantile(y, quantile))
    if not floor > 0:
        raise ValueError("the training floor must be > 0")
    bad = pred <= 0
    return np.where(bad, floor, pred), int(bad.sum())


# ----------------------------------------------------------------------------- XGB
@dataclass(frozen=True)
class XgbHyper:
    """One grid point of protocol 4.2."""

    max_depth: int
    eta: float
    min_child_weight: float


def xgb_grid(xgb: XgbParams, max_depth: Sequence[int] | None = None) -> list[XgbHyper]:
    """``max_depth x eta x min_child_weight`` (protocol 4.2); ``max_depth`` override for M1."""
    depths = tuple(xgb.max_depth if max_depth is None else max_depth)
    return [
        XgbHyper(max_depth=d, eta=e, min_child_weight=c)
        for d, e, c in product(depths, xgb.eta, xgb.min_child_weight)
    ]


def m1_grid(protocol: ProtocolParams) -> list[XgbHyper]:
    """M1 grid: ``model.baselines.m1_max_depth`` x the shared ``eta`` and ``min_child_weight``."""
    return xgb_grid(protocol.model.xgb, protocol.model.baselines.m1_max_depth)


def _booster_params(hyper: XgbHyper, xgb: XgbParams, seed: int, nthread: int) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": hyper.max_depth,
        "eta": hyper.eta,
        "min_child_weight": hyper.min_child_weight,
        "subsample": xgb.subsample,
        "colsample_bytree": xgb.colsample,
        "seed": seed,
        "nthread": nthread,
    }


@dataclass(frozen=True)
class BoosterFit:
    """A fitted booster as XGBoost JSON with its early-stopping record."""

    model_json: bytes
    best_iteration: int
    best_score: float


def _check_xy(x: np.ndarray, y: np.ndarray, name: str) -> None:
    if x.dtype != np.float64 or y.dtype != np.float64 or x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError(f"{name}: need float64 x (n, F) and y (n,)")
    if not (bool(np.isfinite(x).all()) and bool(np.isfinite(y).all())):
        raise ValueError(f"{name}: non-finite values")


def fit_booster(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    hyper: XgbHyper,
    xgb: XgbParams,
    *,
    seed: int,
    nthread: int = 1,
) -> BoosterFit:
    """Fit one booster with early stopping on ``(x_va, y_va)`` (protocol 4.2; P6-Q4 (a)).

    Intended for torch-free processes (ADL-051). Everything runs inside
    ``xgboost.config_context(nthread=nthread)``: in a process that also holds torch, per-object
    ``nthread = 1`` alone still segfaults (exit 139, reproduced in Batch C), the global setting
    does not.
    """
    import xgboost  # lazy: importing this module must not load xgboost (P6-D7)

    _check_xy(x_tr, y_tr, "train")
    _check_xy(x_va, y_va, "val")
    with xgboost.config_context(nthread=nthread):
        d_tr = xgboost.DMatrix(x_tr, label=y_tr, nthread=nthread)
        d_va = xgboost.DMatrix(x_va, label=y_va, nthread=nthread)
        booster = xgboost.train(
            _booster_params(hyper, xgb, seed, nthread),
            d_tr,
            num_boost_round=xgb.max_rounds,
            evals=[(d_va, "val")],
            early_stopping_rounds=xgb.early_stopping_rounds,
            verbose_eval=False,
        )
        model_json = bytes(booster.save_raw(raw_format="json"))
    return BoosterFit(
        model_json=model_json,
        best_iteration=int(booster.best_iteration),
        best_score=float(booster.best_score),
    )


def predict_booster(fit: BoosterFit, x: np.ndarray, *, nthread: int = 1) -> np.ndarray:
    """Predict with the trees up to ``best_iteration`` (inclusive); same thread rule as fitting."""
    import xgboost  # lazy, see fit_booster

    if x.dtype != np.float64 or x.ndim != 2 or not bool(np.isfinite(x).all()):
        raise ValueError("x must be finite float64 (n, F)")
    with xgboost.config_context(nthread=nthread):
        booster = xgboost.Booster(params={"nthread": nthread})
        booster.load_model(bytearray(fit.model_json))
        out = booster.predict(
            xgboost.DMatrix(x, nthread=nthread),
            iteration_range=(0, fit.best_iteration + 1),
            validate_features=False,
        )
    return np.asarray(out, dtype=np.float64)


def _save(path: Path, a: np.ndarray) -> str:
    np.save(path, np.ascontiguousarray(a, dtype=np.float64), allow_pickle=False)
    return path.name


def _load(path: Path) -> np.ndarray:
    out: np.ndarray = np.load(path, allow_pickle=False)
    return out


def fit_booster_isolated(
    work_dir: Path,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    hyper: XgbHyper,
    xgb: XgbParams,
    *,
    seed: int,
    predict: Mapping[str, np.ndarray] | None = None,
    nthread: int = 1,
    timeout_s: float | None = None,
) -> tuple[BoosterFit, dict[str, np.ndarray]]:
    """Fit (and optionally predict) in a fresh torch-free interpreter (ADL-051, P6-D7).

    For callers that hold torch (M6-XGB). Arrays travel as ``.npy`` (``allow_pickle=False``),
    the booster as XGBoost JSON; a non-zero worker exit raises with its stderr.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    for name, x, y in (("train", x_tr, y_tr), ("val", x_va, y_va)):
        _check_xy(x, y, name)
    spec: dict[str, Any] = {
        "x_train": _save(work_dir / "x_train.npy", x_tr),
        "y_train": _save(work_dir / "y_train.npy", y_tr),
        "x_val": _save(work_dir / "x_val.npy", x_va),
        "y_val": _save(work_dir / "y_val.npy", y_va),
        "predict": {k: _save(work_dir / f"x_pred_{k}.npy", v) for k, v in (predict or {}).items()},
        "hyper": asdict(hyper),
        "xgb": xgb.model_dump(),
        "seed": seed,
        "nthread": nthread,
    }
    spec_path = work_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=1, sort_keys=True))
    proc = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-m", "src.models.baselines", "--xgb-fit", str(spec_path)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"XGB worker failed ({proc.returncode}): {proc.stderr.strip()}")
    result = json.loads((work_dir / "result.json").read_text())
    fit = BoosterFit(
        model_json=(work_dir / "booster.json").read_bytes(),
        best_iteration=int(result["best_iteration"]),
        best_score=float(result["best_score"]),
    )
    preds = {k: _load(work_dir / f"pred_{k}.npy") for k in spec["predict"]}
    return fit, preds


def _run_worker(spec_path: Path) -> None:
    """Worker body: fit, write ``booster.json``, ``result.json`` and ``pred_<k>.npy``."""
    if "torch" in sys.modules:  # pragma: no cover - guarded by the import tests
        raise RuntimeError("the XGB worker must not import torch (ADL-051)")
    spec = json.loads(spec_path.read_text())
    root = spec_path.parent
    xgb = XgbParams.model_validate(spec["xgb"])
    fit = fit_booster(
        _load(root / spec["x_train"]),
        _load(root / spec["y_train"]),
        _load(root / spec["x_val"]),
        _load(root / spec["y_val"]),
        XgbHyper(**spec["hyper"]),
        xgb,
        seed=int(spec["seed"]),
        nthread=int(spec["nthread"]),
    )
    (root / "booster.json").write_bytes(fit.model_json)
    for key, name in spec["predict"].items():
        pred = predict_booster(fit, _load(root / name), nthread=int(spec["nthread"]))
        _save(root / f"pred_{key}.npy", pred)
    result = {"best_iteration": fit.best_iteration, "best_score": fit.best_score}
    if not math.isfinite(fit.best_score):
        raise RuntimeError("non-finite early-stopping score")
    (root / "result.json").write_text(json.dumps(result, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m src.models.baselines --xgb-fit SPEC`` (torch-free worker)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--xgb-fit", type=Path, required=True, help="JSON spec written by the parent"
    )
    args = parser.parse_args(argv)
    _run_worker(args.xgb_fit)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through fit_booster_isolated
    raise SystemExit(main())
