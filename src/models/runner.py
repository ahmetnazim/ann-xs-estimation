"""One fit (inner or member) of one model on one fold block, and the inner selection.

Sources: protocol section 3.3 (no model code creates folds; every view comes from
``fold_assignment.parquet``), section 3.4 (split-then-scale; nested selection by the inner-fold
mean log-RMSE of ``dbar`` for NN models or of the model's own targets for XGB; member ``m``
trains on the outer training set minus its grouped 10 % ES split; the outer test fold is touched
once, for scoring, which is **not** done here), section 4.1 (ladder rows; M1 rows missing an M1
predictor are excluded, R14 "M1 only", protocol 2.3), section 4.2 (training settings) and
section 4.8 (ablations reuse the M5 hyperparameters); P6-D1 (a) (no outer-test row is predicted
or scored), P6-D5 (a), P6-D6 (a), P6-D7 (a), P6-Q4 (a), Q-C4 (M6-XGB ``nu`` booster on
I_blind), Batch D Q-D1 to Q-D5.

Leakage guards (L5, P6-D1): before any model code sees the analysis frame, every feature,
target and physics column of the outer **test** rows is overwritten with NaN, so an accidental
read of a test row raises (``SiteDataset``, ``NormStats`` and the XGB inputs reject non-finite
values). Every training and validation part is checked to be disjoint from the test rows.

Importing this module never loads torch or xgboost (ADL-051): the NN and M6-XGB paths import
torch inside their functions; the M1 / M2-XGB path refuses to run in a process that holds torch.
Owning phase: 6.
"""

from __future__ import annotations

import logging
import math
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.config import ProtocolParams, Settings
from src.data.folds import ANALYSES, analysis_frame, fit_provenance, get_view
from src.data.io import sha256_file
from src.data.normstats import NormStats
from src.evaluation.metrics import rmse_log
from src.models.baselines import XgbHyper, blind_features
from src.models.registry import (
    WARM_START_MODELS,
    FitKind,
    ModelSpec,
    block_dir,
    fit_dir,
    nn_grid,
    read_json,
    write_json,
    xgb_grid_for,
)
from src.utils.logging import run_header
from src.utils.seeding import fit_seed, seed_python_numpy

__all__ = [
    "SELECTION_FILE",
    "FitRequest",
    "Tables",
    "run_fit",
    "run_select",
    "selection_analysis",
]

SELECTION_FILE: Final[str] = "selection.json"
_RUN: Final[str] = "run.json"
_NN_TARGETS: Final[tuple[str, ...]] = ("W", "dbar", "t", "A")
_BLANKED: Final[tuple[str, ...]] = ("W", "dbar", "dmax", "A", "t_obs", "Q_E", "Q_C", "S", "V_E")
_UNSUPPORTED_FAMILIES: Final[frozenset[str]] = frozenset({"S-RND"})


@dataclass(frozen=True)
class FitRequest:
    """Coordinates of one fit.

    ``grid`` is required for inner fits; for members it overrides the selection (recorded as
    ``hyper_source = explicit``; used by the P6-D9 pilot).
    """

    spec: ModelSpec
    analysis: str
    family: str
    repeat: int
    outer_fold: int
    kind: FitKind
    index: int
    grid: int | None = None

    def __post_init__(self) -> None:
        """Validate the analysis, family, kind and the grid argument."""
        if self.analysis not in ANALYSES:
            raise ValueError(f"unknown analysis {self.analysis!r}")
        if self.family in _UNSUPPORTED_FAMILIES:
            raise ValueError("S-RND uses its own row-level table; not trained in Phase 6")
        if self.kind == "inner" and self.grid is None:
            raise ValueError("inner fits need --grid")
        if self.kind == "inner" and self.spec.ablation is not None:
            raise ValueError("ablations reuse the base model's selection (protocol 4.8)")
        if self.kind == "member" and self.index < 1:
            raise ValueError("members are numbered 1..M")


@dataclass(frozen=True)
class Tables:
    """The frozen site and fold tables and their SHA-256 (``folds.fit_provenance``)."""

    site_table: pd.DataFrame
    folds: pd.DataFrame
    site_sha256: str
    fold_sha256: str

    @classmethod
    def load(cls, site_path: Path, fold_path: Path) -> Tables:
        """Read both parquet files and hash their bytes (``src.data.io.sha256_file``)."""
        return cls(
            site_table=pd.read_parquet(site_path),
            folds=pd.read_parquet(fold_path),
            site_sha256=sha256_file(site_path),
            fold_sha256=sha256_file(fold_path),
        )


def selection_analysis(analysis: str) -> str:
    """Analysis whose inner selection the members use (Q-D3 (a)): always ``primary``."""
    if analysis not in ANALYSES:
        raise ValueError(f"unknown analysis {analysis!r}")
    return "primary"


@dataclass(frozen=True)
class _Rows:
    frame: pd.DataFrame
    train: np.ndarray
    val: np.ndarray
    outer_train: np.ndarray
    n_test: int
    train_part: str
    val_part: str


def _resolve_rows(tables: Tables, req: FitRequest, features: tuple[str, ...]) -> _Rows:
    frame = analysis_frame(tables.site_table, req.analysis)
    view = (frame, tables.folds, req.family, req.repeat, req.outer_fold)
    test = get_view(*view, "test")
    outer = get_view(*view, "train")
    tr_part, va_part = (
        ("inner_train", "inner_val") if req.kind == "inner" else ("es_train", "es_val")
    )
    tr = get_view(*view, tr_part, req.index)
    va = get_view(*view, va_part, req.index)
    for name, rows in ((tr_part, tr), (va_part, va)):
        if rows.size == 0:
            raise ValueError(f"empty {name} part")
        if np.intersect1d(rows, test).size or np.setdiff1d(rows, outer).size:
            raise RuntimeError(f"{name} overlaps the outer test fold (L5)")
    if np.intersect1d(tr, va).size:
        raise RuntimeError(f"{tr_part} and {va_part} overlap")
    cols = sorted({*features, *(c for c in _BLANKED if c in frame.columns)})
    frame = frame.copy()
    frame[cols] = frame[cols].astype(np.float64)
    frame.loc[frame.index[test], cols] = np.nan  # test rows unreadable from here on (P6-D1)
    return _Rows(frame, tr, va, outer, int(test.size), tr_part, va_part)


def _ids(req: FitRequest) -> dict[str, Any]:
    out = asdict(req)
    out["spec"] = req.spec.name
    return out


def _selection(root: Path, req: FitRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    src_spec = req.spec.hyper_source
    analysis = selection_analysis(req.analysis)
    path = (
        block_dir(root, src_spec, analysis, req.family, req.repeat, req.outer_fold) / SELECTION_FILE
    )
    if not path.is_file():
        raise FileNotFoundError(f"no inner selection at {path}; run `train.py select` first")
    doc = read_json(path)
    return doc, {"kind": "selection", "path": str(path.relative_to(root)), "model": src_spec.name}


def _check_overwrite(directory: Path, overwrite: bool) -> None:
    if (directory / _RUN).exists():
        if not overwrite:
            raise FileExistsError(f"{directory / _RUN} exists (use --overwrite)")
        shutil.rmtree(directory)


def _y_phys(frame: pd.DataFrame, rows: np.ndarray, discharge: str) -> tuple[np.ndarray, np.ndarray]:
    sub = frame.iloc[rows]
    y = np.column_stack(
        [np.log(sub["W"]), np.log(sub["dbar"]), sub["t_obs"], np.log(sub["A"])]
    ).astype(np.float64)
    phys = np.column_stack([np.log(sub[discharge]), np.log(sub["S"])]).astype(np.float64)
    return y, phys


# ----------------------------------------------------------------------------- NN
def _fit_nn(
    req: FitRequest,
    rows: _Rows,
    tables: Tables,
    settings: Settings,
    root: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    from src.data.dataset import SiteDataset
    from src.models.checkpoint import NORMSTATS_FILE, build_model, save_model
    from src.models.m3_physics import m3_r_bar
    from src.models.m6_two_stage import M6Config, m6_labels
    from src.models.trainer import TrainConfig, fit, predict
    from src.utils.seeding import seed_torch

    protocol = settings.protocol
    spec = req.spec
    features = spec.features(protocol)
    grid = nn_grid(protocol, spec.model)
    if req.grid is not None:
        hyper = grid[req.grid]
        source: dict[str, Any] = {
            "kind": "explicit" if req.kind == "member" else "grid",
            "grid": req.grid,
        }
    else:
        doc, source = _selection(root, req)
        if doc["kind"] != "nn":
            raise ValueError("selection.json is not an NN selection")
        hyper = grid[int(doc["selected"])]
        source["grid"] = int(doc["selected"])
    frame = rows.frame
    x_tr = frame.iloc[rows.train][list(features)].to_numpy(dtype=np.float64)
    prov = fit_provenance(
        req.analysis,
        req.family,
        req.repeat,
        req.outer_fold,
        rows.train_part,
        req.index,
        site_table_sha256=tables.site_sha256,
        fold_table_sha256=tables.fold_sha256,
    )
    stats = NormStats.fit(
        x_tr, features, protocol.transforms.features, protocol.transforms.log_offset_frac, prov
    )
    discharge = ANALYSES[req.analysis].discharge
    seed = fit_seed(protocol.seeds, req.family, req.repeat, req.outer_fold, req.kind, req.index)
    seed_python_numpy(seed)
    train_ds = SiteDataset(frame, rows.train, features, stats, len(rows.train), discharge)
    val_ds = SiteDataset(frame, rows.val, features, stats, len(rows.val), discharge)
    extras: dict[str, Any] = {}
    r_bar = None
    if spec.model == "M3":  # Q-D4 (a): inner fits on inner_train(k), members on the outer train
        r_rows = rows.train if req.kind == "inner" else rows.outer_train
        r_bar = m3_r_bar(frame.iloc[r_rows]["t_obs"].to_numpy(dtype=np.float64))
        extras["r_bar"] = r_bar
    seed_torch(seed, threads=settings.runtime.torch_threads_per_worker)
    model = build_model(
        spec, protocol, stats=stats, blind_idx=train_ds.x_blind_idx, hyper=hyper, r_bar=r_bar
    )
    extras["n_trainable_params"] = model.get_num_trainable_params()
    cfg = TrainConfig.from_protocol(
        protocol,
        lr=hyper.lr,
        weight_decay=hyper.weight_decay,
        seed=seed,
        warm_start=spec.model in WARM_START_MODELS,
    )
    logger.info("fit %s %s seed %d hyper %s", spec.name, out_dir, seed, hyper)
    result = fit(model, train_ds, val_ds, cfg, xi=spec.model == "M4", logger=logger)
    if spec.model == "M6-NN":
        labels = m6_labels(train_ds.y, train_ds.phys, M6Config.from_protocol(protocol))
        extras["label_nu_outside_share"] = float(labels.nu_outside.double().mean())
        extras["label_r_clipped_share"] = float(labels.r_clipped.double().mean())
    iters = [r.mean_iterations for r in result.history if r.mean_iterations is not None]
    if iters:
        extras["mean_picard_iterations"] = float(np.mean(iters))
        extras["max_picard_iterations"] = max(
            int(r.max_iterations) for r in result.history if r.max_iterations is not None
        )
    score: dict[str, float] | None = None
    if req.kind == "inner":  # selection statistic on inner_val(k) (protocol 3.4 item 2)
        out = predict(model, val_ds, protocol.model.optimizer.batch_size)
        y = val_ds.y
        score = {
            name: rmse_log(out.mu[:, j].numpy(), y[:, j].numpy())
            for j, name in enumerate(_NN_TARGETS)
        }
        stats.save(out_dir / NORMSTATS_FILE)  # no weights for inner fits (Q-D2 (a))
    else:
        save_model(
            out_dir,
            model,
            spec=spec,
            stats=stats,
            blind_idx=train_ds.x_blind_idx.tolist(),
            hyper=hyper,
            r_bar=r_bar,
            params_sha256=settings.params_sha256,
        )
    return {
        "seed": seed,
        "hyper": asdict(hyper),
        "hyper_source": source,
        "fit": result.to_json(),
        "score": score,
        "extras": extras,
    }


# ----------------------------------------------------------------------------- XGB
def _xgb_hypers(
    req: FitRequest, protocol: ProtocolParams, root: Path, targets: tuple[str, ...]
) -> tuple[dict[str, XgbHyper], dict[str, Any]]:
    grid = xgb_grid_for(protocol, req.spec.model)
    if req.grid is not None:
        kind = "explicit" if req.kind == "member" else "grid"
        return dict.fromkeys(targets, grid[req.grid]), {"kind": kind, "grid": req.grid}
    doc, source = _selection(root, req)
    if doc["kind"] != "xgb" or sorted(doc["targets"]) != sorted(targets):
        raise ValueError("selection.json does not match this XGB model")
    chosen = {t: int(doc["targets"][t]["selected"]) for t in targets}
    source["grid"] = chosen
    return {t: grid[g] for t, g in chosen.items()}, source


def _fit_xgb_plain(
    req: FitRequest, rows: _Rows, settings: Settings, root: Path, out_dir: Path
) -> dict[str, Any]:
    """M1 and M2-XGB in this (torch-free) process (ADL-051, P6-D7)."""
    if "torch" in sys.modules:
        raise RuntimeError("M1 / M2-XGB must run in a process without torch (ADL-051)")
    from src.models.baselines import fit_booster, m1_floor, predict_booster

    protocol = settings.protocol
    spec = req.spec
    features = spec.features(protocol)
    frame = rows.frame
    tr, va = rows.train, rows.val
    extras: dict[str, Any] = {}
    if spec.model == "M1":  # R14 "M1 only" (protocol 2.3): rows missing an M1 predictor drop out
        missing = frame["F_M1_missing"].astype(bool).to_numpy()
        extras["m1_missing_dropped"] = {
            "train": int(missing[tr].sum()),
            "val": int(missing[va].sum()),
        }
        tr, va = tr[~missing[tr]], va[~missing[va]]
        targets: tuple[str, ...] = ("W", "dbar")
    else:
        targets = ("W", "dbar", "dmax", "A")

    def target(rows_: np.ndarray, name: str) -> np.ndarray:
        raw = frame.iloc[rows_][name].to_numpy(dtype=np.float64)
        return raw if spec.model == "M1" else np.log(raw)

    hypers, source = _xgb_hypers(req, protocol, root, targets)
    seed = fit_seed(protocol.seeds, req.family, req.repeat, req.outer_fold, req.kind, req.index)
    seed_python_numpy(seed)
    x_tr = frame.iloc[tr][list(features)].to_numpy(dtype=np.float64)
    x_va = frame.iloc[va][list(features)].to_numpy(dtype=np.float64)
    boosters: dict[str, Any] = {}
    score: dict[str, float] = {}
    for name in targets:
        y_tr, y_va = target(tr, name), target(va, name)
        bf = fit_booster(x_tr, y_tr, x_va, y_va, hypers[name], protocol.model.xgb, seed=seed)
        boosters[name] = {
            "hyper": asdict(hypers[name]),
            "best_iteration": bf.best_iteration,
            "best_score": bf.best_score,
        }
        if req.kind == "inner":
            pred = predict_booster(bf, x_va)
            if spec.model == "M1":  # floor at the training percentile (protocol 4.1, 11.1)
                q = protocol.model.baselines.m1_floor_quantile
                pred, n_floor = m1_floor(pred, y_tr, q)
                boosters[name]["n_floored"] = n_floor
                score[name] = rmse_log(np.log(pred), np.log(y_va))
            else:
                score[name] = rmse_log(pred, y_va)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"booster_{name}.json").write_bytes(bf.model_json)
    return {
        "seed": seed,
        "hyper_source": source,
        "features": list(features),
        "rows_used": {"train": int(tr.size), "val": int(va.size)},
        "fit": boosters,
        "score": score if req.kind == "inner" else None,
        "extras": extras,
    }


def _fit_m6_xgb(
    req: FitRequest, rows: _Rows, settings: Settings, root: Path, out_dir: Path
) -> dict[str, Any]:
    """M6-XGB: Prop. 6.2 labels here (torch), boosters in a torch-free worker (P6-D7, Q-C4)."""
    import torch

    from src.models.baselines import fit_booster_isolated
    from src.models.m6_two_stage import M6Config, m6_labels

    protocol = settings.protocol
    features = req.spec.features(protocol)
    blind = blind_features(features)
    frame = rows.frame
    discharge = ANALYSES[req.analysis].discharge
    cfg = M6Config.from_protocol(protocol)
    labels = {}
    for part, part_rows in (("train", rows.train), ("val", rows.val)):
        y, phys = _y_phys(frame, part_rows, discharge)
        lab = m6_labels(torch.from_numpy(y), torch.from_numpy(phys), cfg)
        labels[part] = lab
    extras = {
        "label_nu_outside_share": float(labels["train"].nu_outside.double().mean()),
        "label_r_clipped_share": float(labels["train"].r_clipped.double().mean()),
    }
    targets = ("nu", "w", "t")
    cols = {"nu": blind, "w": features, "t": features}  # Q-C4: nu booster on I_blind
    hypers, source = _xgb_hypers(req, protocol, root, targets)
    seed = fit_seed(protocol.seeds, req.family, req.repeat, req.outer_fold, req.kind, req.index)
    seed_python_numpy(seed)
    boosters: dict[str, Any] = {}
    score: dict[str, float] = {}
    work_root = Path(tempfile.mkdtemp(prefix="m6xgb_", dir=root))
    try:
        for name in targets:
            x_tr = frame.iloc[rows.train][list(cols[name])].to_numpy(dtype=np.float64)
            x_va = frame.iloc[rows.val][list(cols[name])].to_numpy(dtype=np.float64)
            y_tr = getattr(labels["train"], name).numpy().astype(np.float64)
            y_va = getattr(labels["val"], name).numpy().astype(np.float64)
            bf, preds = fit_booster_isolated(
                work_root / name,
                x_tr,
                y_tr,
                x_va,
                y_va,
                hypers[name],
                protocol.model.xgb,
                seed=seed,
                predict={"val": x_va} if req.kind == "inner" else None,
            )
            boosters[name] = {
                "hyper": asdict(hypers[name]),
                "best_iteration": bf.best_iteration,
                "best_score": bf.best_score,
                "features": list(cols[name]),
            }
            if req.kind == "inner":
                score[name] = math.sqrt(float(np.mean((preds["val"] - y_va) ** 2)))
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"booster_{name}.json").write_bytes(bf.model_json)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return {
        "seed": seed,
        "hyper_source": source,
        "fit": boosters,
        "score": score if req.kind == "inner" else None,
        "extras": extras,
    }


# ----------------------------------------------------------------------------- entry points
def run_fit(
    req: FitRequest,
    tables: Tables,
    settings: Settings,
    root: Path,
    *,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run one fit and write its artefacts under ``root`` (P6-D6 layout); return ``run.json``.

    ``run.json`` is written last and marks completion; an existing one is refused unless
    ``overwrite``. No outer-test row is read, predicted or scored (P6-D1).
    """
    log = logger or logging.getLogger("annxs")
    start = time.perf_counter()
    spec = req.spec
    out_dir = fit_dir(
        root,
        spec,
        req.analysis,
        req.family,
        req.repeat,
        req.outer_fold,
        req.kind,
        req.index,
        req.grid,
    )
    _check_overwrite(out_dir, overwrite)
    rows = _resolve_rows(tables, req, spec.features(settings.protocol))
    if spec.is_nn:
        body = _fit_nn(req, rows, tables, settings, root, out_dir, log)
    elif spec.model == "M6-XGB":
        body = _fit_m6_xgb(req, rows, settings, root, out_dir)
    else:
        body = _fit_xgb_plain(req, rows, settings, root, out_dir)
    doc = {
        "header": run_header(settings.params_sha256, settings.runtime.repo_root, **_ids(req)),
        "site_table_sha256": tables.site_sha256,
        "fold_table_sha256": tables.fold_sha256,
        "parts": {"train": rows.train_part, "val": rows.val_part, "index": req.index},
        "rows": {
            "train": int(rows.train.size),
            "val": int(rows.val.size),
            "outer_train": int(rows.outer_train.size),
            "test_blanked": rows.n_test,
        },
        **body,
        "wall_s": time.perf_counter() - start,
    }
    write_json(out_dir / _RUN, doc)
    return doc


def run_select(
    spec: ModelSpec,
    analysis: str,
    family: str,
    repeat: int,
    outer_fold: int,
    settings: Settings,
    root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Inner selection (protocol 3.4 item 2): argmin of the inner-fold mean score per grid point.

    NN: ``dbar`` log-RMSE; XGB: each target's own score (P6-Q4). Ties go to the lower grid index.
    Every inner ``run.json`` of the grid x ``inner_k`` folds must exist and share the params and
    table hashes. Writes ``selection.json`` in the block directory.
    """
    if spec.ablation is not None:
        raise ValueError("ablations reuse the base model's selection (protocol 4.8)")
    if analysis != selection_analysis(analysis):
        raise ValueError("selection runs on the primary analysis only (Q-D3 (a))")
    protocol = settings.protocol
    k_inner = protocol.splits.inner_k
    n_grid = len(
        nn_grid(protocol, spec.model) if spec.is_nn else xgb_grid_for(protocol, spec.model)
    )
    base = block_dir(root, spec, analysis, family, repeat, outer_fold)
    out = base / SELECTION_FILE
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (use --overwrite)")
    runs: dict[tuple[int, int], dict[str, Any]] = {}
    missing = []
    for g in range(n_grid):
        for k in range(k_inner):
            path = fit_dir(root, spec, analysis, family, repeat, outer_fold, "inner", k, g) / _RUN
            if path.is_file():
                runs[(g, k)] = read_json(path)
            else:
                missing.append(str(path.relative_to(root)))
    if missing:
        raise FileNotFoundError(f"{len(missing)} inner fits missing, e.g. {missing[:3]}")
    hashes = {
        (r["header"]["params_sha256"], r["site_table_sha256"], r["fold_table_sha256"])
        for r in runs.values()
    }
    if len(hashes) != 1:
        raise ValueError("inner fits disagree on params or table hashes")
    params_sha, site_sha, fold_sha = hashes.pop()
    if params_sha != settings.params_sha256:
        raise ValueError("inner fits were run under different params")
    doc: dict[str, Any] = {
        "model": spec.name,
        "analysis": analysis,
        "family": family,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "params_sha256": params_sha,
        "site_table_sha256": site_sha,
        "fold_table_sha256": fold_sha,
        "inner_k": k_inner,
    }

    def choose(target: str) -> dict[str, Any]:
        scores = [
            [float(runs[(g, k)]["score"][target]) for k in range(k_inner)] for g in range(n_grid)
        ]
        means = [float(np.mean(s)) for s in scores]
        best = int(np.argmin(means))  # first minimum: ties -> lower grid index
        return {"scores": scores, "mean_score": means, "selected": best}

    if spec.is_nn:
        grid = nn_grid(protocol, spec.model)
        res = choose("dbar")
        doc.update({"kind": "nn", "target": "dbar", **res, "hyper": asdict(grid[res["selected"]])})
    else:
        xgrid = xgb_grid_for(protocol, spec.model)
        targets = sorted(runs[(0, 0)]["score"])
        per = {}
        for t in targets:
            res = choose(t)
            per[t] = {**res, "hyper": asdict(xgrid[res["selected"]])}
        doc.update({"kind": "xgb", "targets": per})
    write_json(out, doc)
    return doc
