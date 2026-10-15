"""Event 3 isolated workers: AOA importance (xgboost) and ln kappa (torch) in spawned processes.

Sources: docs/3 section 13.9 (AOA importance: M2-XGB permutation importance of the d-bar booster;
for each of the 10 members, on its ``es_val`` rows, 10 permutations per feature with
``default_rng(SeedSequence([member_seed, feature_index]))``; mean increase in the d-bar log-RMSE,
averaged over members, negatives set to 0; the cross-validated training DI uses the block's frozen
``inner_fold`` column), protocol 3.6.6 (DI, AOA threshold = upper whisker of the training CV DI)
and 4.5 (aspect-ratio strata: quintiles of kappa-hat); proofs Proposition 7.6 (A1 map
``nu = nu' + (2/3) ln kappa``); docs/3 section 13.13 items j, i and n.

Process isolation (ADL-051): boosters are evaluated in a spawned process without torch, and
``ln kappa`` (the torch quadrature of :func:`src.physics.shape.log_kappa`) in a spawned process
without xgboost; the Event 3 process itself imports neither. Workers exchange JSON strings only.
This module imports only the standard library at import time. Owning phase: 8.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, wait
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar

if TYPE_CHECKING:  # import-time stdlib only (module docstring)
    import numpy as np
    from numpy.typing import NDArray

__all__ = [
    "AOA_FAMILIES",
    "WorkerError",
    "aoa_job",
    "kappa_job",
    "run_jobs",
]

#: docs/3 3.6.6: AOA is reported for S-LRO, S-TAIL and S-IND.
AOA_FAMILIES: Final[tuple[str, ...]] = ("S-LRO", "S-TAIL-DA", "S-TAIL-SO", "S-IND")
_P = ParamSpec("_P")
_T = TypeVar("_T")


class WorkerError(RuntimeError):
    """A worker job failed (JSON report: job, error, traceback)."""


def _wrap(kind: str, payload: str, body: Callable[[dict[str, Any]], dict[str, Any]]) -> str:
    out: dict[str, Any] = {"kind": kind, "ok": True}
    try:
        out["result"] = body(json.loads(payload))
    except BaseException as exc:  # reported to the parent, which stops
        out.update(ok=False, error=type(exc).__name__, message=str(exc))
        out["traceback"] = traceback.format_exc()
    out["torch_loaded"] = "torch" in sys.modules
    out["xgboost_loaded"] = "xgboost" in sys.modules
    return json.dumps(out)


# ----------------------------------------------------------------------------- AOA (xgboost)
def _aoa(p: dict[str, Any]) -> dict[str, Any]:
    if "torch" in sys.modules:
        raise RuntimeError("the AOA worker must not hold torch (ADL-051)")
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from src.config import load_settings
    from src.data.folds import analysis_frame, get_view
    from src.data.io import sha256_file
    from src.evaluation import dependence_audit as da
    from src.models.baselines import BoosterFit, predict_booster
    from src.models.registry import ModelSpec, fit_dir, read_json

    settings = load_settings(Path(p["params_path"]))
    if settings.params_sha256 != p["params_sha256"]:
        raise RuntimeError("worker params_sha256 differs from the event")
    for key in ("site", "fold", "ind"):
        if sha256_file(Path(p[f"{key}_path"])) != p[f"{key}_sha256"]:
            raise RuntimeError(f"{key} table differs from the event inputs")
    protocol = settings.protocol
    site_table = pd.read_parquet(p["site_path"])
    folds = pd.read_parquet(p["fold_path"])
    frame = analysis_frame(site_table, "primary")
    spec = ModelSpec("M2-XGB")
    features = list(spec.features(protocol))
    family, rep, fold = str(p["family"]), int(p["repeat"]), int(p["outer_fold"])
    run_dir = Path(p["run_dir"])
    members: list[da.ImportanceMember] = []
    for m in range(1, int(p["members"]) + 1):
        mdir = fit_dir(run_dir, spec, "primary", family, rep, fold, "member", m)
        doc = read_json(mdir / "run.json")
        head = doc["header"]
        want = ("M2-XGB", "primary", family, rep, fold, "member", m)
        got = tuple(head.get(k) for k in ("spec", "analysis", "family", "repeat", "outer_fold"))
        if (*got, head.get("kind"), head.get("index")) != want:
            raise ValueError(f"{mdir}: coordinates differ")
        if head["params_sha256"] != settings.params_sha256:
            raise ValueError(f"{mdir}: trained under different params")
        if (
            doc["site_table_sha256"] != p["site_sha256"]
            or doc["fold_table_sha256"] != p["fold_sha256"]
        ):
            raise ValueError(f"{mdir}: trained on different tables")
        if list(doc["features"]) != features:
            raise ValueError(f"{mdir}: feature set differs")
        rows = get_view(frame, folds, family, rep, fold, "es_val", m)
        x = frame.iloc[rows][features].to_numpy(dtype=np.float64)
        y = np.log(frame.iloc[rows]["dbar"].to_numpy(dtype=np.float64))
        booster = BoosterFit(
            model_json=(mdir / "booster_dbar.json").read_bytes(),
            best_iteration=int(doc["fit"]["dbar"]["best_iteration"]),
            best_score=math.nan,
        )

        def predict(xx: NDArray[np.float64], b: BoosterFit = booster) -> NDArray[np.float64]:
            return np.asarray(predict_booster(b, xx), dtype=np.float64)

        members.append(da.ImportanceMember(predict=predict, x=x, y_log=y, seed=int(doc["seed"])))
    importance = da.permutation_importance(members)
    train = get_view(frame, folds, family, rep, fold, "train")
    train_x = frame.iloc[train][features].to_numpy(dtype=np.float64)
    block = folds.loc[
        folds["family"].eq(family) & folds["repeat"].eq(rep) & folds["outer_fold"].eq(fold)
    ]
    inner = block.loc[~block["is_test"].astype(bool)].set_index("site_no")["inner_fold"]
    inner_codes = inner.reindex(frame["site_no"].iloc[train]).to_numpy()
    if pd.isna(inner_codes).any():
        raise ValueError(f"{family} r{rep} f{fold}: training sites without inner_fold")
    cv_di = da.training_cv_di(train_x, importance, inner_codes.astype(np.int64))
    threshold = da.aoa_threshold(cv_di, protocol.audit_rules)
    if family == "S-IND":
        ind = pd.read_parquet(p["ind_path"])
        ind = ind.loc[ind["loc_in_support"].astype(bool)].reset_index(drop=True)
        test_x = ind[features].to_numpy(dtype=np.float64)
        units = [f"{a}|{b}" for a, b in zip(ind["loc_id"].astype(str), ind["comid"], strict=True)]
    else:
        test = get_view(frame, folds, family, rep, fold, "test")
        test_x = frame.iloc[test][features].to_numpy(dtype=np.float64)
        units = frame["site_no"].iloc[test].astype(str).tolist()
    di = da.dissimilarity_index(train_x, test_x, importance)
    return {
        "family": family,
        "repeat": rep,
        "outer_fold": fold,
        "features": features,
        "importance": importance.tolist(),
        "threshold": threshold,
        "n_train": int(train.size),
        "units": units,
        "di": di.tolist(),
    }


def aoa_job(payload: str) -> str:
    """Worker entry point for one AOA block (JSON in, JSON out)."""
    return _wrap("aoa", payload, _aoa)


# ----------------------------------------------------------------------------- kappa (torch)
def _kappa(p: dict[str, Any]) -> dict[str, Any]:
    if "xgboost" in sys.modules:
        raise RuntimeError("the kappa worker must not hold xgboost (ADL-051)")
    import torch

    from src.physics.shape import log_kappa

    torch.set_num_threads(1)
    ell = torch.tensor(p["ell"], dtype=torch.float64)
    w = torch.tensor(p["w"], dtype=torch.float64)
    r = torch.tensor(p["r"], dtype=torch.float64)
    with torch.no_grad():
        k = log_kappa(ell, w, r, level=int(p["level"]), trunc=float(p["trunc"]))
    return {"k": k.tolist()}


def kappa_job(payload: str) -> str:
    """Worker entry point: ``ln kappa(e^(ell - w), r)`` per row (JSON in, JSON out)."""
    return _wrap("kappa", payload, _kappa)


# ----------------------------------------------------------------------------- executor
class _Inline(Executor):
    """Synchronous executor for tests (no process isolation)."""

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        future: Future[_T] = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def run_jobs(
    fn: Callable[[str], str],
    payloads: Sequence[dict[str, Any]],
    *,
    n_workers: int,
    inline: bool = False,
    forbid: str | None = None,
) -> list[dict[str, Any]]:
    """Run ``fn`` on every payload in spawned processes; return the results in order.

    ``forbid`` names a module (``torch`` or ``xgboost``) that must not be loaded in the worker;
    any failure or isolation breach raises :class:`WorkerError` after all jobs drained.
    """
    if not payloads:
        return []
    executor: Executor = (
        _Inline()
        if inline
        else ProcessPoolExecutor(max_workers=max(1, n_workers), mp_context=get_context("spawn"))
    )
    try:
        futures = [executor.submit(fn, json.dumps(p, sort_keys=True)) for p in payloads]
        wait(futures)
        docs = [json.loads(f.result()) for f in futures]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    for d in docs:
        if not d["ok"]:
            raise WorkerError(f"{d['kind']} job failed: {d['error']}: {d['message']}")
        if forbid is not None and not inline and d[f"{forbid}_loaded"]:
            raise WorkerError(f"{d['kind']} worker loaded {forbid} (ADL-051)")
    return [dict(d["result"]) for d in docs]
