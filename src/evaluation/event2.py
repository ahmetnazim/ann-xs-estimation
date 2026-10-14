"""Event 2: M* selection on S-ID OOF predictions (docs/3 13.4; protocol 4.4).

Sources: docs/3 section 13.4 (Event 2 = ``scripts/evaluate.py select-mstar``: M* per
``k in {dbar, A}`` by protocol 4.4; candidates M1, M2-MLP, M2-XGB; S-ID OOF; mean over the three
repeats; M1's A is ``W-hat * dbar-hat``; an exact tie raises, ADL-106), protocol 4.3.1
(``RMSE_log = [mean_j e_j^2]^(1/2)``, one row per site, uniform site weights) and 4.4 (M* fixed
before any other split is scored). Selection by :func:`src.evaluation.hypotheses.select_mstar`.

Rows used: ``family == S-ID``, ``analysis == primary``, no ablation, ``unit == site``; each
(candidate, site, repeat) exactly once, from the block where the site is an outer-test site
(:func:`src.evaluation.event1.sid_test_assignment`). M1 has no A head, so its ``ln A`` is
recomputed here as ``mu_W + mu_dbar`` (C3b-m). Torch-free and xgboost-free. Owning phase: 8.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import pandas as pd

from src.config import ProtocolParams
from src.evaluation.event1 import FAMILY, audit_frame, primary_sid_rows, sid_test_assignment
from src.evaluation.hypotheses import select_mstar
from src.evaluation.metrics import rmse_log

__all__ = ["MSTAR_TARGETS", "run_event2", "sid_rmse_by_repeat"]

#: docs/3 13.4: M* is chosen per target k in {dbar, A}.
MSTAR_TARGETS: Final[tuple[str, ...]] = ("dbar", "A")


def sid_rmse_by_repeat(
    preds: pd.DataFrame, frame: pd.DataFrame, folds: pd.DataFrame, model: str
) -> dict[str, dict[int, float]]:
    """``{k: {repeat: RMSE_log}}`` of one candidate's S-ID OOF predictions (protocol 4.3.1)."""
    test = sid_test_assignment(frame, folds)
    sel = preds["model"].eq(model) & primary_sid_rows(preds)
    rows = preds.loc[sel].astype({"repeat": int, "outer_fold": int}).copy()
    if rows.duplicated(["site_no", "repeat"]).any():
        raise ValueError(f"{model}: duplicated S-ID OOF predictions")
    if model == "M1":
        rows["mu_A"] = rows["mu_W"] + rows["mu_dbar"]
    merged = test.merge(
        rows, on=["site_no", "repeat"], how="left", suffixes=("", "_pred"), indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{model}: missing S-ID OOF predictions")
    if not merged["outer_fold"].eq(merged["outer_fold_pred"]).all():
        raise ValueError(f"{model}: a prediction is not from the site's outer-test block")
    if len(rows) != len(merged):
        raise ValueError(f"{model}: predictions for sites outside the evaluated frame")
    obs = frame.set_index("site_no")
    out: dict[str, dict[int, float]] = {}
    for k in MSTAR_TARGETS:
        o = obs.loc[merged["site_no"], k].to_numpy(dtype=np.float64)
        if not (o > 0).all():
            raise ValueError(f"{k}: observed values must be positive")
        mu = merged[f"mu_{k}"].to_numpy(dtype=np.float64)
        per: dict[int, float] = {}
        for r in sorted(merged["repeat"].unique().tolist()):
            m = merged["repeat"].to_numpy() == r
            per[int(r)] = rmse_log(mu[m], np.log(o[m]))
        out[k] = per
    return out


def run_event2(
    pred_frames: Mapping[str, pd.DataFrame],
    site_table: pd.DataFrame,
    folds: pd.DataFrame,
    protocol: ProtocolParams,
) -> dict[str, Any]:
    """Build the ``mstar.json`` document: per-repeat and mean RMSE_log and M* per target."""
    cands = tuple(protocol.evaluation.mstar_candidates)
    missing = [m for m in cands if m not in pred_frames]
    if missing:
        raise ValueError(f"prediction frames missing for {missing}")
    frame = audit_frame(site_table)
    per_model = {m: sid_rmse_by_repeat(pred_frames[m], frame, folds, m) for m in cands}
    means = {
        m: {k: float(np.mean(list(v[k].values()))) for k in MSTAR_TARGETS}
        for m, v in per_model.items()
    }
    mstar = {
        k: select_mstar({m: means[m][k] for m in cands}, protocol.evaluation) for k in MSTAR_TARGETS
    }
    return {
        "schema_version": 1,
        "event": "event_2",
        "family": FAMILY,
        "candidates": list(cands),
        "targets": list(MSTAR_TARGETS),
        "n_sites": len(frame),
        "rmse_log_by_repeat": {
            m: {k: {str(r): x for r, x in v[k].items()} for k in MSTAR_TARGETS}
            for m, v in per_model.items()
        },
        "rmse_log_mean": means,
        "mstar": mstar,
        "m1_area": "ln A = mu_W + mu_dbar (A = W-hat * dbar-hat)",
    }
