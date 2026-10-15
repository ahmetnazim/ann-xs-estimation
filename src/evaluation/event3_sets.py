"""Event 3 evaluation sets: prediction rows aligned with observations and strata (docs/3 13.13).

Sources: docs/3 section 13.13 (Event 3 computation rules, Batch C3c) items a (evaluation rows), c
(strata) and the S-IND / S-RND rules of sections 13.5 and 13.6; protocol 4.3.3 (strata: drainage-
area quintiles with cutpoints from post-QC ``TotDASqKm``, stream order, division; S-ID far and
not-nested strata of 3.6) and 4.1 (the M-null quintile convention reused for the cutpoints).

One evaluation set is a frame with one row per (unit, repeat), where the unit is an outer-test site
(site families), an eHydro location (S-IND) or an S-RND test measurement. Columns:

- keys: ``unit_id``, ``cluster`` (bootstrap cluster: site, or river for S-IND), ``boot_stratum``
  (division for site clusters, ``"all"`` for S-IND), ``repeat``, ``outer_fold``, ``weight``
  (S-RND ``1/|I_j|``, otherwise 1);
- strata: ``division``, ``da_q`` (1-5), ``stream_order``, and ``far`` / ``nested`` where known;
- observations: ``o_W``, ``o_dbar``, ``o_dmax``, ``o_A`` (raw, positive), ``o_t`` (``t_obs``),
  ``o_V``, ``ln_QE``, ``ln_Qobs`` and the site attributes ``D50_m``, ``S``, ``F_S``, ``row_id``;
- every prediction column of the model's rows (``mu_<k>``, member columns, diagnostics).

A missing, duplicated or misplaced prediction raises (docs/3 13.13 a). Torch-free and
xgboost-free. Owning phase: 8.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data.folds import analysis_frame

__all__ = [
    "LOG_TARGETS",
    "NO_STRATUM",
    "SITE_FAMILIES",
    "TARGETS",
    "da_class",
    "da_cuts",
    "ind_rows",
    "model_rows",
    "site_rows",
    "srnd_rows",
    "test_assignment",
]

#: Site families with outer-test rows (protocol 3.3).
SITE_FAMILIES: Final[tuple[str, ...]] = ("S-ID", "S-LRO", "S-TAIL-DA", "S-TAIL-SO")
#: Targets in log space (observed column -> ``o_<k>``); ``t`` is already a log ratio.
LOG_TARGETS: Final[tuple[str, ...]] = ("W", "dbar", "A", "dmax")
TARGETS: Final[tuple[str, ...]] = ("dbar", "W", "A", "dmax", "t")
#: Label of a missing division (a stratum of its own; docs/3 13.13 c, e).
NO_STRATUM: Final[str] = "none"
_SITE_OBS: Final[Mapping[str, str]] = {
    "W": "o_W",
    "dbar": "o_dbar",
    "dmax": "o_dmax",
    "A": "o_A",
    "t_obs": "o_t",
    "V": "o_V",
}
_SITE_ATTRS: Final[tuple[str, ...]] = ("D50_m", "S", "F_S", "row_id", "comid")
_BLOCK: Final[tuple[str, ...]] = (
    "spec",
    "model",
    "ablation",
    "analysis",
    "family",
    "repeat",
    "outer_fold",
    "unit",
)


def da_cuts(primary_frame: pd.DataFrame) -> tuple[float, float, float, float]:
    """Quintile cutpoints of post-QC ``DA_km2`` (``numpy.quantile``, linear; protocol 4.3.3)."""
    da = primary_frame["DA_km2"].to_numpy(dtype=np.float64)
    if da.size < 5 or not bool(np.isfinite(da).all()) or not bool((da > 0).all()):
        raise ValueError("DA_km2 must be finite and positive on at least five sites")
    c = np.quantile(da, [0.2, 0.4, 0.6, 0.8])
    return (float(c[0]), float(c[1]), float(c[2]), float(c[3]))


def da_class(da: NDArray[np.float64], cuts: Sequence[float]) -> NDArray[np.int64]:
    """Quintile class 1-5: ``searchsorted(cuts, DA, side="right") + 1`` (the M-null convention)."""
    a = np.asarray(da, dtype=np.float64)
    if not bool(np.isfinite(a).all()):
        raise ValueError("non-finite drainage area")
    out: NDArray[np.int64] = np.searchsorted(np.asarray(cuts), a, side="right").astype(np.int64)
    return out + 1


def _labels(values: pd.Series) -> pd.Series:
    out: pd.Series = values.astype("string").fillna(NO_STRATUM).astype(str)
    return out


def test_assignment(frame: pd.DataFrame, folds: pd.DataFrame, family: str) -> pd.DataFrame:
    """``(site_no, repeat, outer_fold)`` of every frame site's outer-test block in ``family``.

    Raises unless every site of ``frame`` is an outer-test site exactly once per repeat of the
    family (S-ID: three repeats; the other site families: one).
    """
    sel = folds["family"].eq(family) & folds["is_test"].astype(bool)
    test = folds.loc[sel, ["site_no", "repeat", "outer_fold"]].astype(
        {"repeat": int, "outer_fold": int}
    )
    if test.empty:
        raise ValueError(f"the fold table has no {family} test rows")
    repeats = sorted(test["repeat"].unique().tolist())
    test = test.loc[test["site_no"].isin(frame["site_no"])]
    if test.duplicated(["site_no", "repeat"]).any():
        raise ValueError(f"a {family} site is an outer-test site more than once in a repeat")
    if family == "S-ID":
        want = len(frame) * len(repeats)
        if len(test) != want:
            raise ValueError("every site must be an S-ID outer-test site once per repeat")
    return test.reset_index(drop=True)


def _check_ablation(preds: pd.DataFrame) -> None:
    if preds["ablation"].isna().any():
        raise ValueError("prediction rows with a null ablation; the writer stores '' (F-C3b-1)")


def model_rows(
    preds: pd.DataFrame, *, model: str, ablation: str, analysis: str, family: str, unit: str
) -> pd.DataFrame:
    """Select the rows of one (model, ablation, analysis, family, unit) in a prediction file."""
    _check_ablation(preds)
    sel = (
        preds["model"].eq(model)
        & preds["ablation"].eq(ablation)
        & preds["analysis"].eq(analysis)
        & preds["family"].eq(family)
        & preds["unit"].eq(unit)
    )
    out = preds.loc[sel].astype({"repeat": int, "outer_fold": int}).reset_index(drop=True)
    return out


def _site_obs(frame: pd.DataFrame, cuts: Sequence[float]) -> pd.DataFrame:
    obs = pd.DataFrame({"site_no": frame["site_no"].astype(str)})
    for col, name in _SITE_OBS.items():
        obs[name] = frame[col].to_numpy(dtype=np.float64)
    obs["ln_QE"] = np.log(frame["Q_E"].to_numpy(dtype=np.float64))
    obs["ln_Qobs"] = np.log(frame["Q_obs"].to_numpy(dtype=np.float64))
    for col in _SITE_ATTRS:
        obs[col] = frame[col].to_numpy()
    obs["division"] = _labels(frame["division"]).to_numpy()
    obs["stream_order"] = _labels(frame["stream_order"].astype("Int64")).to_numpy()
    obs["da_q"] = da_class(frame["DA_km2"].to_numpy(dtype=np.float64), cuts)
    return obs


def site_rows(
    preds: pd.DataFrame,
    site_table: pd.DataFrame,
    folds: pd.DataFrame,
    cuts: Sequence[float],
    *,
    model: str,
    family: str,
    ablation: str = "",
    analysis: str = "primary",
    strata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the evaluation rows of one model on one site family (docs/3 13.13 a and c).

    ``strata`` (optional) holds ``site_no, repeat, outer_fold`` plus extra label columns (Event 1
    ``far``/``nested`` for S-ID; the S-LRO nested class); every evaluated row must match one.
    """
    frame = analysis_frame(site_table, analysis)
    test = test_assignment(frame, folds, family)
    test["site_no"] = test["site_no"].astype(str)
    rows = model_rows(
        preds, model=model, ablation=ablation, analysis=analysis, family=family, unit="site"
    )
    rows["site_no"] = rows["site_no"].astype(str)
    tag = f"{model}{'_' + ablation if ablation else ''}|{analysis}|{family}"
    if rows.duplicated(["site_no", "repeat"]).any():
        raise ValueError(f"{tag}: duplicated predictions")
    merged = test.merge(
        rows, on=["site_no", "repeat"], how="left", suffixes=("", "_pred"), indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{tag}: missing predictions")
    if not merged["outer_fold"].eq(merged["outer_fold_pred"]).all():
        raise ValueError(f"{tag}: a prediction is not from the site's outer-test block")
    if len(rows) != len(merged):
        raise ValueError(f"{tag}: predictions for sites outside the evaluated frame")
    merged = merged.drop(columns=["_merge", "outer_fold_pred"])
    drop = [c for c in ("comid",) if c in merged.columns]
    out = merged.drop(columns=drop).merge(_site_obs(frame, cuts), on="site_no", how="left")
    out["unit_id"] = out["site_no"]
    out["cluster"] = out["site_no"]
    out["boot_stratum"] = out["division"]
    out["weight"] = 1.0
    if strata is not None:
        lab = strata.astype({"repeat": int, "outer_fold": int}).copy()
        lab["site_no"] = lab["site_no"].astype(str)
        extra = [c for c in lab.columns if c not in ("site_no", "repeat", "outer_fold")]
        out = out.merge(
            lab[["site_no", "repeat", "outer_fold", *extra]],
            on=["site_no", "repeat", "outer_fold"],
            how="left",
            indicator=True,
        )
        if not out["_merge"].eq("both").all():
            raise ValueError(f"{tag}: evaluation rows without a stratum label")
        out = out.drop(columns="_merge")
    return out.sort_values(["repeat", "unit_id"], kind="mergesort").reset_index(drop=True)


def ind_rows(
    preds: pd.DataFrame,
    ehydro: pd.DataFrame,
    features: pd.DataFrame,
    cuts: Sequence[float],
    *,
    model: str,
    ablation: str = "",
    analysis: str = "primary",
) -> tuple[pd.DataFrame, int]:
    """S-IND location rows of one model and the count of locations without a prediction.

    Observations are the kept ``ehydro_ind`` locations (docs/3 13.5). Division and stream order
    come from the outlet COMID's ``ind_features`` row (docs/3 13.13 c). Locations whose point
    prediction is non-finite for every available target are dropped and counted (A_J masking).
    """
    rows = model_rows(
        preds,
        model=model,
        ablation=ablation,
        analysis=analysis,
        family="S-IND",
        unit="location",
    )
    rows["loc_id"] = rows["loc_id"].astype(str)
    if rows.duplicated("loc_id").any():
        raise ValueError(f"{model}: duplicated S-IND location predictions")
    kept = ehydro.loc[ehydro["dedupe_status"].eq("kept")].copy()
    kept["loc_id"] = kept["loc_id"].astype(str)
    obs = pd.DataFrame(
        {
            "loc_id": kept["loc_id"],
            "o_W": kept["W"].to_numpy(dtype=np.float64),
            "o_dbar": kept["dbar"].to_numpy(dtype=np.float64),
            "o_A": kept["A"].to_numpy(dtype=np.float64),
            "da_q": da_class(kept["DA_km2"].to_numpy(dtype=np.float64), cuts),
            "comid_outlet": kept["comid_outlet"].astype(np.int64).to_numpy(),
        }
    )
    feat = features[["comid", "division", "stream_order"]].drop_duplicates("comid")
    feat = feat.rename(columns={"comid": "comid_outlet"})
    feat["comid_outlet"] = feat["comid_outlet"].astype(np.int64)
    obs = obs.merge(feat, on="comid_outlet", how="left")
    obs["division"] = _labels(obs["division"])
    obs["stream_order"] = _labels(obs["stream_order"].astype("Int64"))
    merged = rows.merge(obs, on="loc_id", how="left", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{model}: S-IND predictions for locations that are not kept")
    merged = merged.drop(columns="_merge")
    mu = [c for c in ("mu_W", "mu_dbar", "mu_A") if c in merged.columns]
    finite = np.isfinite(merged[mu].to_numpy(dtype=np.float64)).any(axis=1)
    out = merged.loc[finite].copy()
    out["unit_id"] = out["loc_id"]
    out["cluster"] = out["river"].astype(str)
    out["boot_stratum"] = "all"
    out["weight"] = 1.0
    out["repeat"] = 0
    out = out.sort_values("unit_id", kind="mergesort").reset_index(drop=True)
    return out, int((~finite).sum())


def srnd_rows(
    preds: pd.DataFrame, srnd: pd.DataFrame, cuts: Sequence[float], *, model: str
) -> tuple[pd.DataFrame, int]:
    """S-RND test rows of M1 or M2-XGB, weighted ``1/|I_j|`` (docs/3 13.6, proofs 9.5).

    ``|I_j|`` is the number of S-RND test rows of site ``j``. Rows without a prediction (M1 rows
    missing a predictor) are dropped and counted; the weights are computed before dropping.
    """
    rows = model_rows(
        preds, model=model, ablation="", analysis="primary", family="S-RND", unit="srnd_row"
    )
    test = srnd.loc[srnd["is_test"].astype(bool)].copy()
    counts = test.groupby("site_no")["row_id"].transform("size").to_numpy(dtype=np.float64)
    obs = pd.DataFrame(
        {
            "row_id": test["row_id"].to_numpy(),
            "site_no": test["site_no"].astype(str).to_numpy(),
            "o_W": test["W"].to_numpy(dtype=np.float64),
            "o_dbar": test["dbar"].to_numpy(dtype=np.float64),
            "o_A": test["A"].to_numpy(dtype=np.float64),
            "division": _labels(test["division"]).to_numpy(),
            "stream_order": _labels(test["stream_order"].astype("Int64")).to_numpy(),
            "da_q": da_class(test["DA_km2"].to_numpy(dtype=np.float64), cuts),
            "weight": 1.0 / counts,
        }
    )
    if rows["row_id"].duplicated().any():
        raise ValueError(f"{model}: duplicated S-RND predictions")
    keep = [c for c in rows.columns if c not in ("site_no",)]
    merged = obs.merge(rows[keep], on="row_id", how="left", indicator=True)
    if not merged["_merge"].eq("both").all() or len(rows) != len(obs):
        raise ValueError(f"{model}: S-RND predictions do not match the test rows one to one")
    merged = merged.drop(columns="_merge")
    finite = np.isfinite(merged[["mu_W", "mu_dbar"]].to_numpy(dtype=np.float64)).all(axis=1)
    out = merged.loc[finite].copy()
    out["unit_id"] = out["row_id"].astype(str)
    out["cluster"] = out["site_no"]
    out["boot_stratum"] = out["division"]
    out["repeat"] = 0
    return out.reset_index(drop=True), int((~finite).sum())
