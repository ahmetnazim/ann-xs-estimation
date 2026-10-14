"""Event 1: the spatial and network dependence audit on S-ID (docs/3 13.4; protocol 3.6).

Sources: docs/3 section 13.4 (Event 1 = ``scripts/audit.py``, protocol 3.6 on S-ID only: 3.6.1 (a)
site targets and (b) M-null and M2-XGB residuals averaged over the three repeats per site; global
Moran's I (W1, W2) and the correlogram, LISA counts, the variogram fit and selection, rho-hat and
rho*; d~_NN, the nested-basin shares, D, N and the 3.6.7 label; nothing involving an NN), section
13.9 (W2 with co-located sites; nested-basin BFS on the raw ``nhdplusv21`` unit), protocol 3.6.2-
3.6.5 and 3.6.7 (definitions and thresholds from ``params.yaml`` -> ``audit``, ``audit_rules``).
The statistics are those of :mod:`src.evaluation.dependence_audit` (Phase 7, ADL-106, ADL-117);
this module only assembles them.

Computation rules fixed before Event 1 (C3b decisions; docs/3 13.12):

- Series (C3b-a). Site targets ``lnW``, ``lndbar`` (natural logs) and ``t`` (``t_obs``), and the
  OOF residuals ``e = mu_k - ln o_k`` (``t``: ``mu_t - t_obs``) of M-null and M2-XGB for
  ``k`` in ``W, dbar, dmax, A, t``. Every series gets Moran's I with W1 and W2, the correlogram,
  LISA counts by F&J division and the variogram fit and selection. D and rho* use only
  ``M2-XGB:dbar``.
- Residual averaging (C3b-b). Every post-QC site is an outer-test site exactly once per S-ID
  repeat; its residual is the mean over the repeats of its OOF residual. A missing, duplicated or
  misplaced prediction raises.
- d~_NN and the nested share (C3b-c). Pooled over every (repeat, outer fold, test site) of the
  15 S-ID blocks; per-block values are reported alongside.
- Permutation seed (C3b-d). ``seeds.global`` for every Moran, correlogram and LISA test.
- Strata (C3b-f). Per S-ID (repeat, outer fold, test site): nearest-training distance, the far
  flag (distance > rho*), ``psi_max`` and the nested class, returned as a frame for Event 3.

Coordinates are EPSG:5070 metres / 1000 (km). Torch-free and xgboost-free. Owning phase: 8.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from libpysal.weights import W
from numpy.typing import NDArray
from scipy import sparse

from src.config import ProtocolParams
from src.data.folds import analysis_frame, get_view
from src.data.network import NetworkInputs
from src.evaluation import dependence_audit as da

__all__ = [
    "AUDIT_MODELS",
    "FAMILY",
    "LABEL_SERIES",
    "RESIDUAL_TARGETS",
    "SITE_TARGETS",
    "Event1Result",
    "audit_frame",
    "oof_residuals",
    "primary_sid_rows",
    "run_event1",
    "series_stats",
    "sid_blocks",
    "sid_test_assignment",
]

FloatArray = NDArray[np.float64]
FAMILY: Final[str] = "S-ID"
#: docs/3 13.4 / protocol 3.6.1 (b): the primary audit models (no NN).
AUDIT_MODELS: Final[tuple[str, ...]] = ("M-null", "M2-XGB")
#: Protocol 3.6.1 (a): site targets (series name -> observed column; t is already a log ratio).
SITE_TARGETS: Final[Mapping[str, str]] = {"lnW": "W", "lndbar": "dbar", "t": "t_obs"}
#: C3b-a: residual targets (prediction column ``mu_<k>`` -> observed column).
RESIDUAL_TARGETS: Final[Mapping[str, str]] = {
    "W": "W",
    "dbar": "dbar",
    "dmax": "dmax",
    "A": "A",
    "t": "t_obs",
}
_LOG_OBS: Final[frozenset[str]] = frozenset({"W", "dbar", "dmax", "A"})
#: Protocol 3.6.4 / 3.6.7: rho* and condition D use the M2-XGB d-bar residuals.
LABEL_SERIES: Final[str] = "M2-XGB:dbar"
#: ``src.models.predict._base`` writes ``ablation = job.ablation or ""`` (F-C3b-1).
NO_ABLATION: Final[str] = ""
_FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "site_no",
    "comid",
    "division",
    "x5070",
    "y5070",
    "W",
    "dbar",
    "dmax",
    "A",
    "t_obs",
)


@dataclass(frozen=True)
class Event1Result:
    """The ``audit.json`` document and the S-ID strata frame (Event 1 outputs)."""

    document: dict[str, Any]
    strata: pd.DataFrame


def audit_frame(site_table: pd.DataFrame) -> pd.DataFrame:
    """Primary analysis frame (post-QC sites, sorted by ``site_no``) with ``x_km``/``y_km``."""
    frame = analysis_frame(site_table, "primary")
    missing = [c for c in _FRAME_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"site table lacks {missing}")
    out = frame[list(_FRAME_COLUMNS)].copy()
    out["x_km"] = out["x5070"].to_numpy(dtype=np.float64) / 1000.0
    out["y_km"] = out["y5070"].to_numpy(dtype=np.float64) / 1000.0
    if not np.isfinite(out[["x_km", "y_km"]].to_numpy()).all():
        raise ValueError("every audited site needs finite EPSG:5070 coordinates")
    return out


def _observed(frame: pd.DataFrame, column: str) -> FloatArray:
    raw = frame[column].to_numpy(dtype=np.float64)
    if column in _LOG_OBS:
        if not (raw > 0).all():
            raise ValueError(f"{column}: observed values must be positive")
        raw = np.log(raw)
    if not np.isfinite(raw).all():
        raise ValueError(f"{column}: non-finite observed values")
    out: FloatArray = raw
    return out


def sid_blocks(folds: pd.DataFrame) -> list[tuple[int, int]]:
    """Sorted ``(repeat, outer_fold)`` of the S-ID family."""
    sid = folds.loc[folds["family"].eq(FAMILY), ["repeat", "outer_fold"]].drop_duplicates()
    blocks = sorted((int(r), int(f)) for r, f in sid.itertuples(index=False))
    if not blocks:
        raise ValueError("the fold table has no S-ID block")
    return blocks


def sid_test_assignment(frame: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """``(site_no, repeat, outer_fold)`` of every audited site's S-ID outer-test block.

    Raises unless every site of ``frame`` is an outer-test site exactly once per S-ID repeat.
    """
    sid = folds.loc[folds["family"].eq(FAMILY) & folds["is_test"].astype(bool)]
    test = sid[["site_no", "repeat", "outer_fold"]].astype({"repeat": int, "outer_fold": int})
    repeats = sorted(test["repeat"].unique().tolist())
    test = test.loc[test["site_no"].isin(frame["site_no"])]
    counts = test.groupby(["site_no", "repeat"]).size()
    if (counts != 1).any():
        raise ValueError("an S-ID site is an outer-test site more than once in a repeat")
    full = pd.MultiIndex.from_product([frame["site_no"], repeats], names=["site_no", "repeat"])
    if len(counts) != len(full):
        raise ValueError("every audited site must be an outer-test site once per S-ID repeat")
    return test


def primary_sid_rows(preds: pd.DataFrame) -> pd.Series:
    """Mask of S-ID, primary-analysis, no-ablation site rows of a prediction file.

    The writer stores "no ablation" as ``""`` (``src.models.predict._base``); a null ablation is
    not that contract and raises instead of being silently dropped (F-C3b-1).
    """
    if preds["ablation"].isna().any():
        raise ValueError("prediction rows with a null ablation; the writer stores '' (F-C3b-1)")
    mask: pd.Series = (
        preds["family"].eq(FAMILY)
        & preds["analysis"].eq("primary")
        & preds["ablation"].eq(NO_ABLATION)
        & preds["unit"].eq("site")
    )
    return mask


def oof_residuals(
    preds: pd.DataFrame,
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    models: Sequence[str] = AUDIT_MODELS,
) -> pd.DataFrame:
    """Per-site OOF residuals averaged over the S-ID repeats; columns ``<model>:<target>``.

    The rows used are ``family == S-ID``, ``analysis == primary``, no ablation, ``unit == site``.
    Each (model, site, repeat) must appear exactly once, in the block where the site is a test
    site (protocol 3.6.1 (b); C3b-b).
    """
    test = sid_test_assignment(frame, folds)
    n_rep = test["repeat"].nunique()
    rows = preds.loc[primary_sid_rows(preds)]
    obs = {k: _observed(frame, c) for k, c in RESIDUAL_TARGETS.items()}
    index = pd.Index(frame["site_no"], name="site_no")
    out = pd.DataFrame(index=index)
    for model in models:
        pm = rows.loc[rows["model"].eq(model)].astype({"repeat": int, "outer_fold": int})
        if pm.duplicated(["site_no", "repeat"]).any():
            raise ValueError(f"{model}: duplicated S-ID OOF predictions")
        merged = test.merge(
            pm, on=["site_no", "repeat"], how="left", suffixes=("", "_pred"), indicator=True
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{model}: missing S-ID OOF predictions")
        if not merged["outer_fold"].eq(merged["outer_fold_pred"]).all():
            raise ValueError(f"{model}: a prediction is not from the site's outer-test block")
        if len(pm) != len(merged):
            raise ValueError(f"{model}: predictions for sites outside the audited frame")
        pos = index.get_indexer(merged["site_no"])
        for k in RESIDUAL_TARGETS:
            mu = merged[f"mu_{k}"].to_numpy(dtype=np.float64)
            e = mu - obs[k][pos]
            if not np.isfinite(e).all():
                raise ValueError(f"{model}: non-finite {k} residuals")
            total = np.bincount(pos, weights=e, minlength=len(index))
            out[f"{model}:{k}"] = total / n_rep
    return out


def _moran(m: da.MoranResult) -> dict[str, float]:
    return {
        "I": m.i,
        "expected": m.expected,
        "z_perm": m.z_perm,
        "pseudo_p": m.pseudo_p,
        "S0": m.s0,
    }


def series_stats(
    z: FloatArray,
    coords_km: FloatArray,
    w1: W,
    w2: sparse.csr_array,
    divisions: NDArray[np.str_],
    protocol: ProtocolParams,
) -> tuple[dict[str, Any], da.VgmSelection, da.MoranResult]:
    """Moran (W1, W2), correlogram, LISA counts by division and the variogram (3.6.2-3.6.4)."""
    cfg, rules = protocol.audit, protocol.audit_rules
    seed = protocol.seeds.global_seed
    m1 = da.moran_global(z, w1.sparse, cfg.permutations, seed)
    m2 = da.moran_global(z, w2, cfg.permutations, seed)
    corr = da.correlogram(z, coords_km, cfg, seed)
    lis = da.lisa(z, w1, cfg, seed)
    lisa_counts: dict[str, dict[str, int]] = {}
    for div in sorted(set(divisions.tolist())):
        here = divisions == div
        counts = {
            q: int(np.sum(here & lis.significant & (lis.quadrant == q)))
            for q in da.QUADRANTS.values()
        }
        counts["not_significant"] = int(np.sum(here & ~lis.significant))
        lisa_counts[div] = counts
    vg = da.empirical_variogram(z, coords_km, cfg)
    fits = da.fit_variogram(vg, rules)
    sel = da.select_variogram(fits, rules)
    doc = {
        "moran_W1": _moran(m1),
        "moran_W2": _moran(m2),
        "correlogram": [
            {"lower_km": lo, "upper_km": hi, **(_moran(r) if r is not None else {"I": None})}
            for lo, hi, r in corr
        ],
        "lisa_significant_by_division": lisa_counts,
        "variogram": {
            "h_km": vg.h.tolist(),
            "gamma": vg.gamma.tolist(),
            "n_pairs": vg.n_pairs.tolist(),
            "fits": {
                n: {"c0": f.c0, "c": f.c, "a_km": f.a, "wsse": f.wsse} for n, f in fits.items()
            },
            "selected": sel.best.name,
            "unstructured": sel.unstructured,
            "rho_hat_km": sel.rho_hat,
            "nugget_sill": sel.nugget_sill,
        },
    }
    return doc, sel, m1


def _geometry(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    coords: FloatArray,
    network: NetworkInputs,
    rho_star: float,
    protocol: ProtocolParams,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    comids = frame["comid"].to_numpy(dtype=np.int64)
    parts: list[pd.DataFrame] = []
    per_block: list[dict[str, Any]] = []
    for r, f in sid_blocks(folds):
        test = get_view(frame, folds, FAMILY, r, f, "test")
        train = get_view(frame, folds, FAMILY, r, f, "train")
        nn = da.nearest_distance(coords[test], coords[train])
        nested = da.nested_status(
            comids[train].tolist(),
            comids[test].tolist(),
            network.edges,
            network.terminal_path,
            network.drainage_km2,
            protocol.audit,
        )
        cls = np.where(nested.strong, "strong", np.where(nested.weak, "weak", "none"))
        far = nn > rho_star
        parts.append(
            pd.DataFrame(
                {
                    "family": FAMILY,
                    "repeat": np.int8(r),
                    "outer_fold": np.int8(f),
                    "site_no": frame["site_no"].to_numpy()[test],
                    "comid": comids[test],
                    "nn_km": nn,
                    "far": far,
                    "psi_max": nested.psi_max,
                    "nested": cls,
                }
            )
        )
        per_block.append(
            {
                "repeat": r,
                "outer_fold": f,
                "n_test": int(test.size),
                "d_nn_median_km": float(np.median(nn)),
                "share_far": float(np.mean(far)),
                "share_strong": nested.share_strong,
                "share_weak": float(np.mean(nested.weak)),
            }
        )
    return pd.concat(parts, ignore_index=True), per_block


def run_event1(
    site_table: pd.DataFrame,
    folds: pd.DataFrame,
    preds: pd.DataFrame,
    network: NetworkInputs,
    protocol: ProtocolParams,
) -> Event1Result:
    """Assemble Event 1 (docs/3 13.4) from the frozen tables and the sealed audit inputs."""
    if network.missing:
        raise ValueError(f"{len(network.missing)} site COMIDs are not in the NHDPlus VAA")
    frame = audit_frame(site_table)
    coords = frame[["x_km", "y_km"]].to_numpy(dtype=np.float64)
    divisions = frame["division"].astype(str).to_numpy(dtype=np.str_)
    cfg = protocol.audit
    w1 = da.knn_w(coords, cfg.knn_k)
    w2, isolates = da.idw_weights(coords, cfg.idw_band_km)
    series: dict[str, FloatArray] = {
        name: _observed(frame, col) for name, col in SITE_TARGETS.items()
    }
    resid = oof_residuals(preds, frame, folds)
    series.update({c: resid[c].to_numpy(dtype=np.float64) for c in resid.columns})
    stats: dict[str, Any] = {}
    label_sel: da.VgmSelection | None = None
    label_moran: da.MoranResult | None = None
    for name, z in series.items():
        doc, sel, m1 = series_stats(z, coords, w1, w2, divisions, protocol)
        stats[name] = doc
        if name == LABEL_SERIES:
            label_sel, label_moran = sel, m1
    if label_sel is None or label_moran is None:
        raise ValueError(f"{LABEL_SERIES} residuals are missing")
    rho_hat = label_sel.rho_hat
    rho_star = da.dependence_radius(rho_hat, cfg)
    strata, per_block = _geometry(frame, folds, coords, network, rho_star, protocol)
    d_nn = float(np.median(strata["nn_km"].to_numpy()))
    share_strong = float(strata["nested"].eq("strong").mean())
    label = da.audit_label(label_moran, rho_hat, d_nn, share_strong, cfg, protocol.audit_rules)
    d_moran = bool(
        label_moran.pseudo_p < protocol.audit_rules.moran_pseudo_p
        and label_moran.i >= cfg.moran_material
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "event": "event_1",
        "family": FAMILY,
        "models": list(AUDIT_MODELS),
        "n_sites": len(frame),
        "repeats": sorted({r for r, _ in sid_blocks(folds)}),
        "n_blocks": len(per_block),
        "seed": protocol.seeds.global_seed,
        "weights": {
            "W1": {"k": cfg.knn_k},
            "W2": {
                "band_km": cfg.idw_band_km,
                "isolates": int(isolates.size),
                "colocated_pairs": da.colocated_pairs(coords),
            },
        },
        "series": stats,
        "label_series": LABEL_SERIES,
        "rho_hat_km": rho_hat,
        "rho_star_km": rho_star,
        "d_nn_median_km": d_nn,
        "nested": {
            "psi_strong": cfg.nested_psi,
            "share_strong": share_strong,
            "share_weak": float(strata["nested"].eq("weak").mean()),
            "n_test_instances": len(strata),
        },
        "far": {"share_far": float(strata["far"].mean())},
        "per_block": per_block,
        "label": {
            "D": label.d,
            "D_moran": d_moran,
            "D_range": bool(rho_hat > d_nn),
            "N": label.n,
            "optimistic": label.optimistic,
            "text": (
                "optimistic under spatial/network dependence"
                if label.optimistic
                else "S-ID reported as is; strata in the supplement"
            ),
        },
    }
    return Event1Result(document=document, strata=strata)
