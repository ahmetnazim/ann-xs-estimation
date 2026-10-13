"""Phase 8 prediction writer: OOF, S-IND and S-RND predictions from the trained run root.

Sources: docs/3 section 13.3 (``predict@audit``: S-ID OOF predictions of M-null and M2-XGB into
``data/processed/predictions/audit_inputs.parquet``; ``predict@all``: every OOF and S-IND
prediction into ``data/processed/predictions/<model>.parquet``; XGB point prediction = exp of the
mean member log prediction, M1 = the log of the floored raw-metre predictions; XGB models report
no predictive distribution), section 13.4 (a seal lists the prediction files; a sealed file is
never overwritten; no metric before its event), section 13.5 (S-IND: every member COMID is
predicted by the S-IND block and aggregated as the targets were built, E-R8), section 13.6 (S-RND
predictions of M1 and M2-XGB on the S-RND test rows), section 13.9 (``nu_back`` from
``(W, dbar, r, Q_E, S)`` by Proposition 6.2, with ``r`` from ``dmax/dbar`` for M2 and ``r_bar`` for
M0 and M1; rows with ``dmax <= dbar`` are invariant-4 violations), protocol 4.1 (M1 floor at the
0.1th training percentile; ``r_bar`` = training-fold median as for M3), protocol 4.2 (mixture,
point ``exp(mu*)``, ``alpha = 0.10``), proofs 9.2-9.3 (moments, exact quantiles) and
Proposition 9.2 (``n* = Phi^-1(mu*)``).

Blinding: the observed columns :data:`OBSERVED` of every prediction row are overwritten with NaN
before any model code sees the frame, so an accidental read raises downstream. Only training-part
targets are read (the M1 floor and ``r_bar``). No metric, residual or target statistic is
computed here.

Process isolation (ADL-051): XGB boosters are evaluated in torch-free workers (phase ``raw``);
NN members, baselines, the M6-XGB stage 3 and ``nu_back`` run in torch workers (phase ``main``).
Workers exchange JSON strings only; block outputs are Parquet under ``<run_dir>/predictions``,
each with a JSON sidecar naming the artefacts it was computed from (resume key).
Importing this module never loads torch or xgboost. Owning phase: 8.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar, cast

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.folds import ANALYSES, analysis_frame, get_view
from src.data.ind_features import aggregate_e_r8, length_weighted_mean
from src.data.io import read_parquet_meta, sha256_file, write_parquet
from src.models.registry import ModelSpec, fit_dir, read_json, write_json
from src.models.runner import BASELINE_FILE, BASELINES, baseline_dir

if TYPE_CHECKING:
    from torch import Tensor

    from src.models.heads import LadderNet, ModelOutput
    from src.models.plan import PlannedFit
    from src.physics.manning_layer import PhiConfig

__all__ = [
    "AUDIT_FILE",
    "AUDIT_MODELS",
    "OBSERVED",
    "PREDICT_STAGES",
    "BlockFrame",
    "PredictContext",
    "PredictFailureError",
    "PredictJob",
    "PredictTables",
    "SealedFileError",
    "back_calc",
    "block_frame",
    "execute_job",
    "prediction_jobs",
    "predictions_dir",
    "run_predictions",
]

PREDICT_STAGES: Final[tuple[str, ...]] = ("predict@audit", "predict@all")
#: docs/3 13.3 row 3: the Event 1 inputs (S-ID OOF of M-null and M2-XGB).
AUDIT_MODELS: Final[tuple[str, ...]] = ("M-null", "M2-XGB")
AUDIT_FILE: Final[str] = "audit_inputs.parquet"
#: Observed quantities: blanked on every prediction row (never read with a prediction here).
OBSERVED: Final[tuple[str, ...]] = (
    "W",
    "dbar",
    "dmax",
    "A",
    "V",
    "t_obs",
    "eps_A",
    "Q_obs",
    "Q_meas",
)
#: Canonical ladder columns ``[ln W, ln dbar, t, ln A]`` (ADL-077).
CANONICAL: Final[tuple[str, ...]] = ("W", "dbar", "t", "A")
XGB_TARGETS: Final[Mapping[str, tuple[str, ...]]] = {
    "M1": ("W", "dbar"),
    "M2-XGB": ("W", "dbar", "dmax", "A"),
    "M6-XGB": ("nu", "w", "t"),
}
#: Models without an n head get the Manning back-calculation (docs/3 13.9, protocol 4.3.2).
BACK_CALC_MODELS: Final[frozenset[str]] = frozenset({"M0", "M1", "M2-MLP", "M2-XGB"})
#: Models with Phi or a Manning residual: n* and the solver/penalty residual are recorded.
PHI_MODELS: Final[frozenset[str]] = frozenset({"M3", "M4", "M5", "M6-NN"})
UNIT_KEYS: Final[Mapping[str, tuple[str, ...]]] = {
    "site": ("site_no", "comid"),
    "comid": ("loc_id", "river", "n_comids", "member", "comid", "lengthkm"),
    "srnd_row": ("row_id", "site_no"),
}
BLOCK_COLUMNS: Final[tuple[str, ...]] = (
    "spec",
    "model",
    "ablation",
    "analysis",
    "family",
    "repeat",
    "outer_fold",
    "unit",
)
_RUN: Final[str] = "run.json"
_P = ParamSpec("_P")
_T = TypeVar("_T")
_JobKey = tuple[str, str | None, str, str, int, int]


class PredictFailureError(RuntimeError):
    """A prediction job failed; the remaining jobs drained (same rule as docs/3 13.7)."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        """Keep the JSON failure report (block, phase, error, traceback)."""
        super().__init__(
            f"prediction {report['block_id']} ({report['phase']}) failed: "
            f"{report['error']}: {report['message']}"
        )
        self.report = dict(report)


class SealedFileError(RuntimeError):
    """A sealed prediction file would change (docs/3 13.4: never overwritten)."""


def predictions_dir(processed_dir: Path) -> Path:
    """``data/processed/predictions`` (docs/3 13.3)."""
    return processed_dir / "predictions"


# ----------------------------------------------------------------------------- jobs
@dataclass(frozen=True)
class PredictJob:
    """All members (or the baseline fit) of one model on one block."""

    model: str
    ablation: str | None
    analysis: str
    family: str
    repeat: int
    fold: int
    members: int

    @property
    def spec_name(self) -> str:
        """``M5`` or ``M5_A1`` (P6-D6 layout)."""
        return self.model if self.ablation is None else f"{self.model}_{self.ablation}"

    @property
    def block_id(self) -> str:
        """``<model>/<analysis>/<family>/r<rep>/f<fold>`` relative to the run root."""
        return f"{self.spec_name}/{self.analysis}/{self.family}/r{self.repeat}/f{self.fold}"

    @property
    def unit(self) -> str:
        """``comid`` (S-IND), ``srnd_row`` (S-RND) or ``site``."""
        return {"S-IND": "comid", "S-RND": "srnd_row"}.get(self.family, "site")

    @property
    def is_baseline(self) -> bool:
        """M-null, A_J and M0 (closed form; ``params.json``)."""
        return self.model in BASELINES

    @property
    def is_xgb(self) -> bool:
        """M1, M2-XGB and M6-XGB: boosters evaluated in a torch-free worker first."""
        return self.model in XGB_TARGETS

    def to_json(self) -> dict[str, Any]:
        """Plain dict (worker transport)."""
        return asdict(self)

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> PredictJob:
        """Inverse of :meth:`to_json`."""
        return cls(**doc)


def prediction_jobs(fits: Sequence[PlannedFit], stage: str) -> list[PredictJob]:
    """One job per block with members or a baseline fit, in plan order (docs/3 13.3)."""
    if stage not in PREDICT_STAGES:
        raise ValueError(f"unknown prediction stage {stage!r}; expected {PREDICT_STAGES}")
    counts: dict[_JobKey, int] = {}
    for f in fits:
        if f.kind not in ("member", "outer"):
            continue
        key: _JobKey = (f.model, f.ablation, f.analysis, f.family, f.repeat, f.fold)
        counts[key] = counts.get(key, 0) + int(f.kind == "member")
    jobs = [PredictJob(*key, members=n) for key, n in counts.items()]
    if stage == "predict@audit":
        jobs = [
            j
            for j in jobs
            if j.model in AUDIT_MODELS
            and j.ablation is None
            and j.analysis == "primary"
            and j.family == "S-ID"
        ]
    return jobs


def required_fits(fits: Sequence[PlannedFit], jobs: Sequence[PredictJob]) -> list[PlannedFit]:
    """Return the member and baseline fits the jobs read (all must be complete)."""
    keys = {(j.model, j.ablation, j.analysis, j.family, j.repeat, j.fold) for j in jobs}
    return [
        f
        for f in fits
        if f.kind in ("member", "outer")
        and (f.model, f.ablation, f.analysis, f.family, f.repeat, f.fold) in keys
    ]


# ----------------------------------------------------------------------------- tables
@dataclass(frozen=True)
class PredictContext:
    """Everything a worker needs, as plain strings (JSON transport, never pickled objects)."""

    params_path: str
    run_dir: str
    site_path: str
    fold_path: str
    params_sha256: str
    site_sha256: str
    fold_sha256: str
    ind_path: str = ""
    ind_sha256: str = ""
    srnd_path: str = ""
    srnd_sha256: str = ""
    log_level: str = "INFO"

    def to_json(self) -> str:
        """JSON string for the worker initializer."""
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> PredictContext:
        """Inverse of :meth:`to_json`."""
        return cls(**json.loads(text))

    @property
    def stage_dir(self) -> Path:
        """``<run_dir>/predictions`` (block outputs and sidecars)."""
        return Path(self.run_dir) / "predictions"


@dataclass(frozen=True)
class PredictTables:
    """The frozen tables read at prediction time, verified against the plan hashes."""

    site_table: pd.DataFrame
    folds: pd.DataFrame
    ind: pd.DataFrame | None
    srnd: pd.DataFrame | None
    hashes: dict[str, str]

    @classmethod
    def load(cls, ctx: PredictContext) -> PredictTables:
        """Read and hash the tables; S-IND keeps the rows in common support only."""
        site_sha = sha256_file(Path(ctx.site_path))
        hashes = {"site_table": site_sha, "fold_table": sha256_file(Path(ctx.fold_path))}
        if (hashes["site_table"], hashes["fold_table"]) != (ctx.site_sha256, ctx.fold_sha256):
            raise RuntimeError("prediction tables differ from the plan hashes")
        ind = srnd = None
        if ctx.ind_path:
            hashes["ind_features"] = sha256_file(Path(ctx.ind_path))
            if hashes["ind_features"] != ctx.ind_sha256:
                raise RuntimeError("ind_features differs from the plan hash")
            full = pd.read_parquet(ctx.ind_path)
            ind = full.loc[full["loc_in_support"].astype(bool)].reset_index(drop=True)
        if ctx.srnd_path:
            meta = read_parquet_meta(Path(ctx.srnd_path))
            if meta.get("inputs", {}).get("site_table.parquet") != site_sha:
                raise ValueError("srnd_frame was not built from this site_table")
            hashes["srnd_frame"] = sha256_file(Path(ctx.srnd_path))
            if hashes["srnd_frame"] != ctx.srnd_sha256:
                raise RuntimeError("srnd_frame differs from the plan hash")
            srnd = pd.read_parquet(ctx.srnd_path)
        return cls(
            site_table=pd.read_parquet(ctx.site_path),
            folds=pd.read_parquet(ctx.fold_path),
            ind=ind,
            srnd=srnd,
            hashes=hashes,
        )


@dataclass(frozen=True)
class BlockFrame:
    """Training frame and prediction rows of one block.

    ``train`` holds the block's analysis rows (S-RND: the S-RND frame) with the observed columns
    of the outer test rows blanked; ``outer_train`` indexes it. ``pred`` holds the prediction rows
    ``rows`` (the outer test sites, the supported S-IND member COMIDs or the S-RND test rows) with
    their observed columns blanked; ``keys`` are their key columns.
    """

    train: pd.DataFrame
    outer_train: np.ndarray
    pred: pd.DataFrame
    rows: np.ndarray
    keys: pd.DataFrame


def _blank(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    cols = [c for c in OBSERVED if c in out.columns]
    out[cols] = out[cols].astype(np.float64)
    out.loc[out.index[rows], cols] = np.nan
    return out


def _keys(frame: pd.DataFrame, rows: np.ndarray, unit: str) -> pd.DataFrame:
    cols = [c for c in UNIT_KEYS[unit] if c in frame.columns]
    return frame.iloc[rows][cols].reset_index(drop=True)


def block_frame(job: PredictJob, tables: PredictTables) -> BlockFrame:
    """Rows of one block; the observed columns of every prediction row are NaN (blinding)."""
    if job.family == "S-RND":
        if tables.srnd is None:
            raise ValueError("S-RND predictions need the srnd_frame table (docs/3 13.6)")
        test = np.flatnonzero(tables.srnd["is_test"].astype(bool).to_numpy())
        train = np.flatnonzero(~tables.srnd["is_test"].astype(bool).to_numpy())
        frame = _blank(tables.srnd, test)
        return BlockFrame(frame, train, frame, test, _keys(frame, test, "srnd_row"))
    frame = analysis_frame(tables.site_table, job.analysis)
    view = (frame, tables.folds, job.family, job.repeat, job.fold)
    test, train = get_view(*view, "test"), get_view(*view, "train")
    frame = _blank(frame, test)
    if job.family == "S-IND":
        if test.size:
            raise ValueError("the S-IND block has no site test rows (docs/3 13.5)")
        if tables.ind is None:
            raise ValueError("S-IND predictions need the ind_features table (docs/3 13.5)")
        ind = _blank(tables.ind, np.arange(len(tables.ind)))
        rows = np.arange(len(ind))
        return BlockFrame(frame, train, ind, rows, _keys(ind, rows, "comid"))
    return BlockFrame(frame, train, frame, test, _keys(frame, test, "site"))


def _base(job: PredictJob, bf: BlockFrame) -> pd.DataFrame:
    out = bf.keys.copy()
    meta: dict[str, Any] = {
        "spec": job.spec_name,
        "model": job.model,
        "ablation": job.ablation or "",
        "analysis": job.analysis,
        "family": job.family,
        "repeat": job.repeat,
        "outer_fold": job.fold,
        "unit": job.unit,
    }
    for i, (k, v) in enumerate(meta.items()):
        out.insert(i, k, v)
    return out


# ----------------------------------------------------------------------------- artefacts
def _member_dir(run_dir: Path, job: PredictJob, m: int) -> Path:
    spec = ModelSpec(job.model, job.ablation)
    return fit_dir(run_dir, spec, job.analysis, job.family, job.repeat, job.fold, "member", m)


def _artefact_dirs(run_dir: Path, job: PredictJob) -> list[Path]:
    if job.is_baseline:
        return [baseline_dir(run_dir, job.model, job.family, job.repeat, job.fold)]
    if job.members < 1:
        raise ValueError(f"{job.block_id}: no members planned")
    return [_member_dir(run_dir, job, m) for m in range(1, job.members + 1)]


def _checked_run(
    directory: Path, job: PredictJob, index: int, settings: Settings, tables: PredictTables
) -> dict[str, Any]:
    path = directory / _RUN
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; the fit is incomplete")
    doc = read_json(path)
    head = doc["header"]
    if head["params_sha256"] != settings.params_sha256:
        raise ValueError(f"{path}: trained under different params")
    if (doc["site_table_sha256"], doc["fold_table_sha256"]) != (
        tables.hashes["site_table"],
        tables.hashes["fold_table"],
    ):
        raise ValueError(f"{path}: trained on different site or fold tables")
    if job.family == "S-RND" and doc.get("srnd_frame_sha256") != tables.hashes.get("srnd_frame"):
        raise ValueError(f"{path}: trained on a different srnd_frame")
    kind = "outer" if job.is_baseline else "member"
    got = tuple(head.get(k) for k in ("spec", "analysis", "family", "repeat", "outer_fold"))
    want = (job.spec_name, job.analysis, job.family, job.repeat, job.fold)
    if got != want or (head.get("kind"), head.get("index")) != (kind, index):
        raise ValueError(f"{path}: coordinates do not match {job.block_id}")
    return doc


def _sidecar(
    job: PredictJob, phase: str, run_dir: Path, settings: Settings, tables: PredictTables
) -> dict[str, Any]:
    """Resume key: the job, params, table hashes and the SHA-256 of each artefact ``run.json``."""
    arts = {
        (d / _RUN).relative_to(run_dir).as_posix(): sha256_file(d / _RUN)
        for d in _artefact_dirs(run_dir, job)
        if (d / _RUN).is_file()
    }
    return {
        "job": job.to_json(),
        "phase": phase,
        "params_sha256": settings.params_sha256,
        "tables": dict(sorted(tables.hashes.items())),
        "artefacts": arts,
    }


def block_paths(stage_dir: Path, job: PredictJob, phase: str) -> tuple[Path, Path]:
    """``<stage_dir>/<block_id>.<phase>.parquet`` and its JSON sidecar."""
    base = f"{job.block_id}.{phase}"
    return stage_dir / f"{base}.parquet", stage_dir / f"{base}.json"


# ----------------------------------------------------------------------------- XGB (torch-free)
def _m1_train_rows(job: PredictJob, bf: BlockFrame, tables: PredictTables, m: int) -> np.ndarray:
    if job.family == "S-RND":
        rows = bf.outer_train
    else:
        view = (bf.train, tables.folds, job.family, job.repeat, job.fold)
        rows = get_view(*view, "es_train", m)
    missing = bf.train["F_M1_missing"].astype(bool).to_numpy()
    return np.asarray(rows[~missing[rows]], dtype=np.int64)


def xgb_raw(
    job: PredictJob, tables: PredictTables, settings: Settings, run_dir: Path
) -> pd.DataFrame:
    """:func:`booster_predictions`, refused in a process holding torch (ADL-051)."""
    if "torch" in sys.modules:
        raise RuntimeError("boosters are evaluated in a process without torch (ADL-051)")
    return booster_predictions(job, tables, settings, run_dir)


def booster_predictions(
    job: PredictJob, tables: PredictTables, settings: Settings, run_dir: Path
) -> pd.DataFrame:
    """Raw member predictions of the boosters (``m<mm>_<target>``) on the prediction rows.

    M1: log of the floored raw-metre prediction (floor from member ``m``'s own training rows,
    protocol 4.1; per-row ``n_floored``); M2-XGB: log predictions; M6-XGB: the stage-2 regressors
    ``(nu, w, t)`` (stage 3 runs in a torch worker). Production calls go through :func:`xgb_raw`.
    """
    from src.models.baselines import BoosterFit, m1_floor, predict_booster

    if not job.is_xgb:
        raise ValueError(f"{job.model} has no boosters")
    protocol = settings.protocol
    features = ModelSpec(job.model, job.ablation).features(protocol)
    bf = block_frame(job, tables)
    sub = bf.pred.iloc[bf.rows]
    n = len(sub)
    out = _base(job, bf)
    ok = np.ones(n, dtype=bool)
    if job.model == "M1":  # R14 "M1 only": rows missing an M1 predictor get no M1 prediction
        ok = ~sub["F_M1_missing"].astype(bool).to_numpy()
        out["m1_missing"] = ~ok
    n_floored = np.zeros(n, dtype=np.int64)
    for m, mdir in enumerate(_artefact_dirs(run_dir, job), start=1):
        doc = _checked_run(mdir, job, m, settings, tables)
        y_rows = None
        if job.model == "M1":
            y_rows = _m1_train_rows(job, bf, tables, m)
            used = doc["rows"]["train"] if job.family == "S-RND" else doc["rows_used"]["train"]
            if int(used) != y_rows.size:
                raise ValueError(f"{mdir}: M1 training rows differ from the fit ({used})")
        for t in XGB_TARGETS[job.model]:
            fit = doc["fit"][t]
            cols = list(fit.get("features", doc.get("features", features)))
            if job.model != "M6-XGB" and cols != list(features):
                raise ValueError(f"{mdir}: booster features differ from {job.spec_name}")
            x = sub[cols].to_numpy(dtype=np.float64)
            if not bool(np.isfinite(x[ok]).all()):
                raise ValueError(f"{job.block_id}: non-finite predictors (no imputation, 2.7)")
            booster = BoosterFit(
                model_json=(mdir / f"booster_{t}.json").read_bytes(),
                best_iteration=int(fit["best_iteration"]),
                best_score=math.nan,
            )
            pred = predict_booster(booster, x[ok])
            vals = np.full(n, np.nan)
            if y_rows is not None:
                y_tr = bf.train.iloc[y_rows][t].to_numpy(dtype=np.float64)
                q = protocol.model.baselines.m1_floor_quantile
                floored, _ = m1_floor(pred, y_tr, q)
                n_floored[ok] += pred <= 0
                vals[ok] = np.log(floored)
            else:
                vals[ok] = pred
            out[f"m{m:02d}_{t}"] = vals
    if job.model == "M1":
        out["n_floored"] = n_floored
    return out


# ----------------------------------------------------------------------------- torch helpers
def _phys(sub: pd.DataFrame, discharge: str) -> np.ndarray:
    phys = np.column_stack([np.log(sub[discharge]), np.log(sub["S"])]).astype(np.float64)
    if not bool(np.isfinite(phys).all()):
        raise ValueError(f"non-finite ln {discharge} or ln S on prediction rows")
    return phys


def back_calc(
    mu_w: np.ndarray,
    mu_dbar: np.ndarray,
    r_hat: np.ndarray,
    q_e: np.ndarray,
    s: np.ndarray,
    cfg: PhiConfig,
) -> np.ndarray:
    """``nu_back = Phi^-1(ln dbar; ln W, r, ln Q_E, ln S)`` (Proposition 6.2) where defined.

    Rows with a non-finite input or ``r <= 0`` get NaN (they are counted by the caller).
    """
    import torch

    from src.physics.manning_layer import phi_inverse_nu

    arrays = [np.asarray(a, dtype=np.float64) for a in (mu_w, mu_dbar, r_hat, q_e, s)]
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays]) & (arrays[2] > 0)
    out = np.full(arrays[0].shape, np.nan)
    if valid.any():
        w, ell, r, q, sl = (torch.from_numpy(a[valid].copy()) for a in arrays)
        out[valid] = phi_inverse_nu(ell, w, r, q, sl, cfg).numpy()
    return out


def _add_back_calc(out: pd.DataFrame, job: PredictJob, bf: BlockFrame, settings: Settings) -> None:
    from src.models.m3_physics import m3_r_bar
    from src.physics.manning_layer import PhiConfig

    sub = bf.pred.iloc[bf.rows]
    mu_w, mu_d = out["mu_W"].to_numpy(), out["mu_dbar"].to_numpy()
    if job.model in ("M0", "M1"):  # r_bar: training-fold median of r_obs, as for M3 (13.9)
        r_bar = m3_r_bar(bf.train.iloc[bf.outer_train]["t_obs"].to_numpy(dtype=np.float64))
        r_hat = np.full(len(out), r_bar)
        inv4 = np.zeros(len(out), dtype=bool)
    else:  # M2: r = dbar / (dmax - dbar) = 1 / (exp(t) - 1); t <= 0 violates invariant 4
        t = out["mu_dmax"].to_numpy() - mu_d
        inv4 = np.isfinite(t) & (t <= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_hat = np.where(t > 0, 1.0 / np.expm1(np.where(t > 0, t, 1.0)), np.nan)
    cfg = PhiConfig.from_solver(settings.protocol.solver)
    q_e = np.log(sub["Q_E"].to_numpy(dtype=np.float64))
    s = np.log(sub["S"].to_numpy(dtype=np.float64))
    out["r_hat"] = r_hat
    out["inv4_violation"] = inv4
    out["nu_back"] = back_calc(mu_w, mu_d, r_hat, q_e, s, cfg)


def _batch(x: Tensor, phys: Tensor) -> dict[str, Tensor]:
    """Batch contract (ADL-077); ``y`` is a placeholder: no rung's forward reads it."""
    import torch

    n = int(x.shape[0])
    return {
        "x": x,
        "phys": phys,
        "y": torch.zeros((n, 4), dtype=torch.float64),
        "omega": torch.full((n,), 1.0 / n, dtype=torch.float64),
    }


def _forward(model: LadderNet, x: Tensor, phys: Tensor, batch_size: int) -> ModelOutput:
    import torch

    from src.models.heads import ModelOutput

    model.eval()
    mus: list[Tensor] = []
    sigmas: list[Tensor] = []
    extras: dict[str, list[Tensor]] = {}
    with torch.no_grad():
        for lo in range(0, int(x.shape[0]), batch_size):
            out = model(_batch(x[lo : lo + batch_size], phys[lo : lo + batch_size]))
            b = int(out.mu.shape[0])
            mus.append(out.mu)
            if out.sigma is not None:
                sigmas.append(out.sigma)
            for k, v in out.extras.items():
                if v.dim() >= 1 and int(v.shape[0]) == b:
                    extras.setdefault(k, []).append(v)
    return ModelOutput(
        mu=torch.cat(mus),
        sigma=torch.cat(sigmas) if sigmas else None,
        extras={k: torch.cat(v) for k, v in extras.items()},
    )


def _mixture_columns(
    out: pd.DataFrame,
    names: Sequence[str],
    mu: Tensor,
    sigma: Tensor | None,
    alpha: float,
    *,
    members: bool = True,
) -> None:
    """Member columns, ``var_ale``/``var_epi`` and the exact central interval (proofs 9.2-9.3)."""
    from src.models.ensemble import mixture_moments, mixture_quantile

    mix = mixture_moments(mu, sigma)
    lo = hi = None
    if sigma is not None:
        lo = mixture_quantile(alpha / 2.0, mu, sigma)
        hi = mixture_quantile(1.0 - alpha / 2.0, mu, sigma)
    for j, k in enumerate(names):
        if members:
            for m in range(int(mu.shape[0])):
                out[f"m{m + 1:02d}_mu_{k}"] = mu[m, :, j].numpy()
                if sigma is not None:
                    out[f"m{m + 1:02d}_sigma_{k}"] = sigma[m, :, j].numpy()
        out[f"var_ale_{k}"] = mix.var_ale[:, j].numpy()
        out[f"var_epi_{k}"] = mix.var_epi[:, j].numpy()
        if lo is not None and hi is not None:
            out[f"q_lo_{k}"] = lo[:, j].numpy()
            out[f"q_hi_{k}"] = hi[:, j].numpy()


def _physics_columns(
    out: pd.DataFrame,
    mu_star: Tensor,
    outs: Sequence[ModelOutput],
    mus: Tensor,
    phys: Tensor,
    cfg: PhiConfig,
    r_const: float | None,
) -> None:
    """``nu_mean``, ``r_mean``, ``n*`` (Prop 9.2), Manning residual summary, Picard maximum."""
    import torch

    from src.models.ensemble import ensemble_roughness
    from src.physics.manning_layer import manning_residual

    nu = torch.stack([o.extras["nu"] for o in outs])
    r = (
        torch.full_like(nu, float(r_const))
        if r_const is not None
        else torch.stack([o.extras["r"] for o in outs])
    )
    resid = []
    for m, o in enumerate(outs):
        if "R_M" in o.extras:  # M4: the explicit penalty residual (proofs 7.7)
            resid.append(o.extras["R_M"])
        else:  # hard models: the solver residual of eq. 3.1 at the member's own outputs
            args = (mus[m, :, 1], mus[m, :, 0], r[m], nu[m], phys[:, 0], phys[:, 1])
            resid.append(manning_residual(*args, cfg))
    res = torch.stack(resid)
    out["nu_mean"] = nu.mean(dim=0).numpy()
    out["r_mean"] = r.mean(dim=0).numpy()
    out["nu_star"] = ensemble_roughness(mu_star, phys, cfg).numpy()
    out["resid_mean"] = res.mean(dim=0).numpy()
    out["resid_maxabs"] = res.abs().max(dim=0).values.numpy()
    if all("iterations" in o.extras for o in outs):
        its = torch.stack([o.extras["iterations"].to(torch.int64) for o in outs])
        out["iters_max"] = its.max(dim=0).values.numpy()


def predict_nn(
    job: PredictJob, tables: PredictTables, settings: Settings, run_dir: Path
) -> pd.DataFrame:
    """Ensemble predictions of one NN block (members, mixture, point, physics diagnostics).

    ``mu*`` is the mean of the member means (P3-14), except M3: Phi at the mean member ``w``
    (P6-Q4 (a)). The central ``1 - alpha`` interval uses the exact mixture quantiles; rungs
    without sigma (M3, M6-NN) have none. M2-MLP's ``dmax`` uses its native head.
    """
    import torch

    from src.models.checkpoint import load_model
    from src.models.ensemble import stack_members

    protocol = settings.protocol
    features = ModelSpec(job.model, job.ablation).features(protocol)
    bf = block_frame(job, tables)
    sub = bf.pred.iloc[bf.rows]
    if len(sub) == 0:
        raise ValueError(f"{job.block_id}: no prediction rows")
    raw_x = sub[list(features)].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(raw_x).all()):
        raise ValueError(f"{job.block_id}: non-finite predictors (no imputation, 2.7)")
    phys = torch.from_numpy(_phys(sub, ANALYSES[job.analysis].discharge))
    outs: list[ModelOutput] = []
    models: list[LadderNet] = []
    xs: list[Tensor] = []
    r_bars: set[float] = set()
    for m, mdir in enumerate(_artefact_dirs(run_dir, job), start=1):
        _checked_run(mdir, job, m, settings, tables)
        model, stats, doc = load_model(mdir, protocol, params_sha256=settings.params_sha256)
        if tuple(stats.names) != tuple(features):
            raise ValueError(f"{mdir}: feature set differs from {job.spec_name}")
        x = torch.from_numpy(stats.transform(raw_x))
        outs.append(_forward(model, x, phys, protocol.model.optimizer.batch_size))
        models.append(model)
        xs.append(x)
        if doc.get("r_bar") is not None:
            r_bars.add(float(doc["r_bar"]))
    mu, sigma = stack_members(outs)
    mu_star = mu.mean(dim=0)
    if job.model == "M3":
        if len(r_bars) != 1:
            raise ValueError(f"{job.block_id}: M3 members disagree on r_bar")
        mu_star = models[0].physics(mu[:, :, 0].mean(dim=0), _batch(xs[0], phys)).mu  # type: ignore[operator]
    out = _base(job, bf)
    for j, k in enumerate(CANONICAL):
        out[f"mu_{k}"] = mu_star[:, j].numpy()
    out["mu_dmax"] = (mu_star[:, 1] + mu_star[:, 2]).numpy()
    alpha = protocol.uq.alpha
    _mixture_columns(out, CANONICAL, mu, sigma, alpha)
    if job.model == "M2-MLP":  # native ln dmax head (independent; closure not enforced)
        mu_n, sig_n = stack_members(outs, native=True)
        sig_dmax = None if sig_n is None else sig_n[:, :, 2:3]
        _mixture_columns(out, ("dmax",), mu_n[:, :, 2:3], sig_dmax, alpha)
        out["mu_dmax"] = mu_n[:, :, 2].mean(dim=0).numpy()
        _add_back_calc(out, job, bf, settings)
    if job.model in PHI_MODELS:
        cfg: PhiConfig = cast("Any", models[0]).cfg.phi  # the member's own Phi (kappa follows A1)
        r_const = next(iter(r_bars)) if job.model == "M3" else None
        _physics_columns(out, mu_star, outs, mu, phys, cfg, r_const)
    return out


def finish_xgb(
    job: PredictJob, raw: pd.DataFrame, tables: PredictTables, settings: Settings
) -> pd.DataFrame:
    """Point predictions from the raw member predictions; M6-XGB stage 3; ``nu_back``."""
    m_count = job.members
    keys = [*BLOCK_COLUMNS, *UNIT_KEYS[job.unit], "n_floored", "m1_missing"]
    out = raw[[c for c in keys if c in raw.columns]].copy()
    bf = block_frame(job, tables)
    if job.model in ("M1", "M2-XGB"):
        for t in XGB_TARGETS[job.model]:
            cols = [f"m{m:02d}_{t}" for m in range(1, m_count + 1)]
            for c in cols:
                out[c.replace(f"_{t}", f"_mu_{t}")] = raw[c].to_numpy()
            out[f"mu_{t}"] = raw[cols].to_numpy().mean(axis=1)
        if job.model == "M1":
            out["mu_A"] = out["mu_W"] + out["mu_dbar"]
        else:
            out["mu_t"] = out["mu_dmax"] - out["mu_dbar"]
        if job.unit != "srnd_row":
            _add_back_calc(out, job, bf, settings)
        return out
    import torch

    from src.models.m6_two_stage import M6Config, m6_stage3

    cfg = M6Config.from_protocol(settings.protocol)
    sub = bf.pred.iloc[bf.rows]
    phys = torch.from_numpy(_phys(sub, ANALYSES[job.analysis].discharge))
    outs: list[ModelOutput] = []
    for m in range(1, m_count + 1):
        g = {
            t: torch.from_numpy(raw[f"m{m:02d}_{t}"].to_numpy(dtype=np.float64, copy=True))
            for t in ("nu", "w", "t")
        }
        outs.append(m6_stage3(g["w"], g["t"], g["nu"], phys, cfg))
    mu = torch.stack([o.mu for o in outs])
    mu_star = mu.mean(dim=0)
    for j, k in enumerate(CANONICAL):
        out[f"mu_{k}"] = mu_star[:, j].numpy()
    out["mu_dmax"] = (mu_star[:, 1] + mu_star[:, 2]).numpy()
    _mixture_columns(out, CANONICAL, mu, None, settings.protocol.uq.alpha)
    _physics_columns(out, mu_star, outs, mu, phys, cfg.phi, None)
    return out


def _quad(values: Sequence[float]) -> tuple[float, float, float, float]:
    """``(a_W, b_W, a_dbar, b_dbar)`` from ``params.json`` (exactly four numbers)."""
    a_w, b_w, a_d, b_d = (float(v) for v in values)
    return (a_w, b_w, a_d, b_d)


def predict_baseline(
    job: PredictJob, tables: PredictTables, settings: Settings, run_dir: Path
) -> pd.DataFrame:
    """M-null, A_J and M0 from their ``params.json`` (fitted on the outer training set)."""
    from src.models.baselines import M0PowerLaw, MNull, aj_log_area

    (bdir,) = _artefact_dirs(run_dir, job)
    _checked_run(bdir, job, 0, settings, tables)
    params = read_json(bdir / BASELINE_FILE)
    bf = block_frame(job, tables)
    sub = bf.pred.iloc[bf.rows]
    out = _base(job, bf)
    if job.model == "M-null":
        mnull = MNull(
            cuts=tuple(float(c) for c in params["cuts"]),
            medians=tuple(tuple(float(v) for v in r) for r in params["medians"]),
        )
        pred = mnull.predict(sub["DA_km2"].to_numpy(dtype=np.float64))
        for k in ("W", "dbar", "dmax", "A"):
            out[f"mu_{k}"] = pred[k]
        out["mu_t"] = out["mu_dmax"] - out["mu_dbar"]
    elif job.model == "A_J":
        ln_a, available = aj_log_area(
            sub["Q_E"].to_numpy(dtype=np.float64), sub["V_E"].to_numpy(dtype=np.float64)
        )
        out["mu_A"] = ln_a
        out["aj_available"] = available
    else:
        m0 = M0PowerLaw(
            coef=tuple(sorted((str(g), _quad(c)) for g, c in params["coef"].items())),
            pooled=_quad(params["pooled"]),
            min_sites=int(params["min_sites"]),
        )
        division = sub["division"].astype("string")
        pred, pooled = m0.predict(
            division.fillna("").to_numpy(dtype=str), sub["DA_km2"].to_numpy(dtype=np.float64)
        )
        for k in ("W", "dbar", "A"):
            out[f"mu_{k}"] = pred[k]
        out["m0_pooled"] = pooled
        out["division_missing"] = division.isna().to_numpy()
        _add_back_calc(out, job, bf, settings)
    return out


# ----------------------------------------------------------------------------- job execution
def _write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def execute_job(
    job: PredictJob,
    phase: str,
    tables: PredictTables,
    settings: Settings,
    run_dir: Path,
    stage_dir: Path,
) -> dict[str, Any]:
    """Run one job phase unless its output exists with a matching sidecar (resume)."""
    if phase not in ("raw", "main"):
        raise ValueError("phase must be 'raw' or 'main'")
    if phase == "raw" and not job.is_xgb:
        raise ValueError(f"{job.model} has no raw phase")
    path, side_path = block_paths(stage_dir, job, phase)
    side = _sidecar(job, phase, run_dir, settings, tables)
    if path.is_file() and side_path.is_file():
        if read_json(side_path) != side:
            raise RuntimeError(f"{side_path} names other artefacts; remove the stale output")
        return {"block_id": job.block_id, "phase": phase, "skipped": True}
    if phase == "raw":
        frame = xgb_raw(job, tables, settings, run_dir)
    elif job.is_baseline:
        frame = predict_baseline(job, tables, settings, run_dir)
    elif job.is_xgb:
        raw_path, raw_side = block_paths(stage_dir, job, "raw")
        if not (raw_path.is_file() and raw_side.is_file()):
            raise FileNotFoundError(f"{raw_path} missing; the raw phase did not complete")
        if read_json(raw_side) != _sidecar(job, "raw", run_dir, settings, tables):
            raise RuntimeError(f"{raw_side} names other artefacts")
        frame = finish_xgb(job, pd.read_parquet(raw_path), tables, settings)
    else:
        frame = predict_nn(job, tables, settings, run_dir)
    _write_atomic_parquet(frame, path)
    write_json(side_path, side)
    return {"block_id": job.block_id, "phase": phase, "skipped": False, "rows": len(frame)}


_PWORKER: dict[str, Any] = {}


def _init_predict_worker(ctx_json: str, torch_free: bool) -> None:
    """Pool initializer: load and verify settings and tables once per worker process."""
    ctx = PredictContext.from_json(ctx_json)
    if torch_free and "torch" in sys.modules:
        raise RuntimeError("torch-free worker started with torch loaded (ADL-051)")
    settings = load_settings(ctx.params_path)
    if settings.params_sha256 != ctx.params_sha256:
        raise RuntimeError("worker params_sha256 differs from the plan")
    _PWORKER.clear()
    _PWORKER.update(settings=settings, tables=PredictTables.load(ctx), ctx=ctx)


def _run_job(job_json: str, phase: str) -> str:
    """Worker entry point: JSON in, JSON out; exceptions are reported, never pickled."""
    job = PredictJob.from_json(json.loads(job_json))
    out: dict[str, Any] = {"block_id": job.block_id, "phase": phase, "ok": True}
    try:
        ctx: PredictContext = _PWORKER["ctx"]
        out.update(
            execute_job(
                job,
                phase,
                _PWORKER["tables"],
                _PWORKER["settings"],
                Path(ctx.run_dir),
                ctx.stage_dir,
            )
        )
    except BaseException as exc:  # reported to the scheduler, which stops
        out.update(
            ok=False, error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
        )
    out["torch_loaded"] = "torch" in sys.modules
    return json.dumps(out)


class _Inline(Executor):
    """Synchronous executor for tests (no isolation; the torch-free check is skipped)."""

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        future: Future[_T] = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def _run_phase(
    jobs: Sequence[PredictJob], phase: str, ctx: PredictContext, n_workers: int, inline: bool
) -> list[dict[str, Any]]:
    from src.models.orchestrator import thread_env

    if not jobs:
        return []
    torch_free = phase == "raw"
    if inline:
        _init_predict_worker(ctx.to_json(), False)
        executor: Executor = _Inline()
    else:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=get_context("spawn"),
            initializer=_init_predict_worker,
            initargs=(ctx.to_json(), torch_free),
        )
    results: list[dict[str, Any]] = []
    with thread_env():
        try:
            futures = [executor.submit(_run_job, json.dumps(j.to_json()), phase) for j in jobs]
            wait(futures)
            results = [json.loads(f.result()) for f in futures]
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    for res in results:
        if res["ok"] and torch_free and res["torch_loaded"] and not inline:
            res.update(
                ok=False, error="IsolationError", message="torch loaded in a torch-free worker"
            )
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise PredictFailureError(failed[0])
    return results


# ----------------------------------------------------------------------------- collation
def _ind_locations(frame: pd.DataFrame) -> pd.DataFrame:
    """E-R8 location rows from the per-COMID predictions of one S-IND block (docs/3 13.5).

    A location is missing when any member COMID has no prediction (M1 rows missing a predictor,
    A_J rows without ``V_E``), as for the targets (E-R8); the others use :func:`aggregate_e_r8`.
    """
    groups = frame.drop_duplicates("loc_id").sort_values("loc_id", kind="mergesort")
    out = groups[[*BLOCK_COLUMNS, "loc_id", "river", "n_comids"]].reset_index(drop=True)
    out["unit"] = "location"
    ids = out["loc_id"].astype(str)
    loc = frame["loc_id"].astype(str)
    length = frame["lengthkm"].to_numpy(dtype=np.float64)
    if "mu_W" in frame and "mu_dbar" in frame:
        w, d = np.exp(frame["mu_W"].to_numpy()), np.exp(frame["mu_dbar"].to_numpy())
        bad = set(loc[~(np.isfinite(w) & np.isfinite(d))])
        keep = ~loc.isin(bad).to_numpy()
        agg = aggregate_e_r8(loc[keep], length[keep], w[keep], d[keep]).set_index("loc_id")
        agg = agg.reindex(ids)
        if set(agg.index[agg["W"].isna()]) != bad:
            raise RuntimeError("location set mismatch")
        for k in ("W", "dbar", "A"):
            out[f"mu_{k}"] = np.log(agg[k].to_numpy())
    else:  # A_J: Â is the length-weighted mean of A_J,c (missing if any member is missing)
        a = length_weighted_mean(loc, length, np.exp(frame["mu_A"].to_numpy()))
        out["mu_A"] = np.log(a.reindex(ids).to_numpy())
    return out


def _digest(obj: object) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_checked(
    frame: pd.DataFrame,
    dest: Path,
    sealed: Mapping[str, str],
    rel: str,
    *,
    stage: str,
    settings: Settings,
    inputs: Mapping[str, str],
) -> str:
    """Write ``frame`` unless ``dest`` is sealed with other bytes; return the file SHA-256."""
    tmp = dest.with_suffix(".parquet.tmp")
    write_parquet(frame, tmp, stage=stage, params_sha256=settings.params_sha256, inputs=inputs)
    new = sha256_file(tmp)
    if rel in sealed and sealed[rel] != new:
        tmp.unlink()
        raise SealedFileError(f"{rel} is sealed with other content (docs/3 13.4)")
    os.replace(tmp, dest)
    return new


def collate(
    stage: str,
    jobs: Sequence[PredictJob],
    ctx: PredictContext,
    settings: Settings,
    dest_dir: Path,
    *,
    sealed: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Concatenate the block outputs into the stage files; returns a counts-only report."""
    sealed = sealed or {}
    root = repo_root or settings.runtime.repo_root
    frames: dict[str, list[pd.DataFrame]] = {}
    arts: dict[str, Any] = {}
    for job in jobs:
        path, side = block_paths(ctx.stage_dir, job, "main")
        if not (path.is_file() and side.is_file()):
            raise FileNotFoundError(f"{path} missing; run the prediction phases first")
        block = pd.read_parquet(path)
        arts[job.block_id] = read_json(side)["artefacts"]
        frames.setdefault(job.spec_name, []).append(block)
        if job.family == "S-IND":
            frames[job.spec_name].append(_ind_locations(block))
    inputs = {
        "tables": _digest(_table_hashes(ctx)),
        "artefacts": _digest(arts),
    }
    dest_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}

    def emit(name: str, frame: pd.DataFrame) -> None:
        dest = dest_dir / name
        rel = _rel(dest, root)
        sha = _write_checked(
            frame, dest, sealed, rel, stage=stage, settings=settings, inputs=inputs
        )
        counts = frame.groupby(["family", "unit"]).size().reset_index(name="n")
        by = zip(counts["family"], counts["unit"], counts["n"], strict=True)
        files[rel] = {
            "sha256": sha,
            "rows": len(frame),
            "rows_by_family_unit": {f"{f}/{u}": int(c) for f, u, c in by},
        }

    if stage == "predict@audit":
        cols = [*BLOCK_COLUMNS, "site_no", "comid", "mu_W", "mu_dbar", "mu_dmax", "mu_A", "mu_t"]
        audit = pd.concat([f for parts in frames.values() for f in parts], ignore_index=True)
        emit(AUDIT_FILE, audit[[c for c in cols if c in audit.columns]])
    else:
        for spec, parts in frames.items():
            emit(f"{spec}.parquet", pd.concat(parts, ignore_index=True))
    return {
        "stage": stage,
        "params_sha256": settings.params_sha256,
        "inputs": inputs,
        "files": files,
    }


def _table_hashes(ctx: PredictContext) -> dict[str, str]:
    out = {"site_table": ctx.site_sha256, "fold_table": ctx.fold_sha256}
    if ctx.ind_sha256:
        out["ind_features"] = ctx.ind_sha256
    if ctx.srnd_sha256:
        out["srnd_frame"] = ctx.srnd_sha256
    return out


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# ----------------------------------------------------------------------------- driver
def run_predictions(
    stage: str,
    fits: Sequence[PlannedFit],
    *,
    ctx: PredictContext,
    settings: Settings,
    dest_dir: Path,
    n_workers: int,
    sealed: Mapping[str, str] | None = None,
    inline: bool = False,
    logger: logging.Logger | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Both phases, then the stage files and their counts-only report (no metric)."""
    log = logger or logging.getLogger("annxs")
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    jobs = prediction_jobs(fits, stage)
    if not jobs:
        raise ValueError(f"{stage}: the plan has no blocks to predict")
    run_dir = Path(ctx.run_dir)
    missing = [
        f.fit_id for f in required_fits(fits, jobs) if not f.completion_path(run_dir).is_file()
    ]
    if missing:
        raise RuntimeError(f"{stage}: {len(missing)} fits incomplete, e.g. {missing[:3]}")
    raw = _run_phase([j for j in jobs if j.is_xgb], "raw", ctx, n_workers, inline)
    main = _run_phase(jobs, "main", ctx, n_workers, inline)
    log.info(
        "%s: %d raw and %d main block outputs (%d reused)",
        stage,
        len(raw),
        len(main),
        sum(r.get("skipped", False) for r in raw + main),
    )
    report = collate(stage, jobs, ctx, settings, dest_dir, sealed=sealed, repo_root=repo_root)
    report["blocks"] = len(jobs)
    return report
