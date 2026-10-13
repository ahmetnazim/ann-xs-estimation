"""Stage srnd_frame (docs/3 13.3 stage 0b): the row-level S-RND frame (docs/3 13.6).

Sources: docs/3 section 13.6 (frame = ``srnd_assignment``, 38,693 rows in U, joined to
``hydroswot_init`` on ``row_id`` for the per-measurement W, d̄ and A in SI, and to the
COMID-level predictors of ``site_table`` on ``site_no``; models M1 and M2-XGB only), P5-D7
(the frozen row-level 75:25 split over every ``hydroswot_init`` row of the sites in U),
section 2.7 (no imputation), section 3.3 (no split is created here) and section 13.11 (frozen
artefacts are read, never regenerated).

Implementation definitions (Batch C2, flagged for the PI):

* ``dmax`` is joined as well, because M2-XGB has a ``dmax`` booster (``LOG_TARGETS``).
* Every ``srnd_assignment`` row is kept. The R8-R10 flags (``F_eps``, ``F_t``, ``F_qual``,
  ``F_tout``) are carried as columns, not filtered: the frozen split was drawn over all rows.
* The predictors are the union of the NN set (``params.transforms.features``, 22) and the M1 set
  (19), plus ``F_M1_missing`` (rows of sites missing an M1 predictor drop out of M1 only, as in
  the runner) and ``comid``/``division`` for traceability. They are COMID-level and must be equal
  across the ``site_table`` variants of a site (checked).

``reports/srnd_frame.json`` holds counts only (no target statistic). Torch-free and
xgboost-free (ADL-051). Owning phase: 8.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.io import read_parquet, sha256_file, write_json, write_parquet
from src.data.schemas import SrndAssignment, assert_no_banned

__all__ = [
    "M1_SET",
    "SRND_TARGETS",
    "build_srnd_frame",
    "main",
    "srnd_frame_path",
]

LOG = logging.getLogger(__name__)

#: Per-measurement targets joined from ``hydroswot_init`` (SI; docs/3 13.6).
SRND_TARGETS: Final[tuple[str, ...]] = ("W", "dbar", "dmax", "A")
#: QC flags carried (not filtered) from ``hydroswot_init`` (protocol 2.4).
_FLAGS: Final[tuple[str, ...]] = ("F_eps", "F_t", "F_qual", "F_tout")
#: The M1 (Zarrabi Tier-2) predictors in site-table names and runner order; equal to
#: ``src.models.baselines.M1_FEATURES`` (tested; restated so ``src.data`` never imports models).
M1_SET: Final[tuple[str, ...]] = (
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


def srnd_frame_path(processed_dir: Path) -> Path:
    """``data/processed/srnd/srnd_frame.parquet`` (docs/3 13.3 stage 0b)."""
    return processed_dir / "srnd" / "srnd_frame.parquet"


def _site_predictors(site_table: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    cols = ["site_no", "comid", "division", *predictors, "F_M1_missing"]
    missing = [c for c in cols if c not in site_table.columns]
    if missing:
        raise KeyError(f"site_table lacks {missing}")
    per_site = site_table[cols].drop_duplicates()
    if per_site["site_no"].duplicated().any():
        raise ValueError("site_table: COMID-level predictors differ between variants of a site")
    return per_site.reset_index(drop=True)


def build_srnd_frame(
    assignment: pd.DataFrame,
    init: pd.DataFrame,
    site_table: pd.DataFrame,
    nn_predictors: Sequence[str],
    m1_predictors: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the frozen S-RND split to its targets and the site predictors (docs/3 13.6).

    Parameters
    ----------
    assignment
        ``splits/srnd_assignment.parquet`` (:class:`SrndAssignment`).
    init
        ``qc/hydroswot_init.parquet``.
    site_table
        ``site_table.parquet`` (long; one row per site and variant).
    nn_predictors, m1_predictors
        The NN set (22) and the M1 set (19); NN predictors must be finite on every row
        (R14 was applied by the features stage), M1 predictors may be missing
        (``F_M1_missing``).

    Returns
    -------
    tuple
        The frame (sorted by ``row_id``) and a counts-only report.
    """
    predictors = [*nn_predictors, *(c for c in m1_predictors if c not in set(nn_predictors))]
    assert_no_banned(predictors)
    assignment = SrndAssignment.validate(assignment)
    targets = init[["row_id", "site_no", "q_meas_no", *SRND_TARGETS, *_FLAGS]]
    joined = assignment.merge(
        targets, on="row_id", how="left", suffixes=("", "_init"), validate="one_to_one"
    )
    if joined["site_no_init"].isna().any():
        raise ValueError("srnd_assignment rows without a hydroswot_init row")
    same = joined["site_no"].eq(joined["site_no_init"]) & joined["q_meas_no"].eq(
        joined["q_meas_no_init"]
    )
    if not bool(same.all()):
        raise ValueError("srnd_assignment and hydroswot_init disagree on site_no or q_meas_no")
    joined = joined.drop(columns=["site_no_init", "q_meas_no_init"])
    sites = _site_predictors(site_table, predictors)
    frame = joined.merge(sites, on="site_no", how="left", validate="many_to_one")
    if frame["comid"].isna().any():
        raise ValueError("srnd_assignment sites missing from site_table (U)")
    y = frame[list(SRND_TARGETS)].to_numpy(dtype=np.float64)
    if not (bool(np.isfinite(y).all()) and bool((y > 0).all())):
        raise ValueError("S-RND targets must be finite and > 0 (invariant 5)")
    x_nn = frame[list(nn_predictors)].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(x_nn).all()):
        raise ValueError("NN predictors missing on S-RND rows (no imputation, protocol 2.7)")
    m1_missing = frame[list(m1_predictors)].isna().any(axis=1).to_numpy()
    if not np.array_equal(m1_missing, frame["F_M1_missing"].astype(bool).to_numpy()):
        raise ValueError("F_M1_missing disagrees with the M1 predictors")
    frame = frame.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    test = frame["is_test"].astype(bool)
    counts: dict[str, Any] = {
        "rows": len(frame),
        "rows_train": int((~test).sum()),
        "rows_test": int(test.sum()),
        "sites": int(frame["site_no"].nunique()),
        "sites_train": int(frame.loc[~test, "site_no"].nunique()),
        "sites_test": int(frame.loc[test, "site_no"].nunique()),
        "sites_in_both": len(set(frame.loc[test, "site_no"]) & set(frame.loc[~test, "site_no"])),
        "F_M1_missing_rows": {
            "train": int(frame.loc[~test, "F_M1_missing"].sum()),
            "test": int(frame.loc[test, "F_M1_missing"].sum()),
        },
        "flag_rows": {f: int(frame[f].sum()) for f in _FLAGS},
        "predictors_nn": list(nn_predictors),
        "predictors_m1": list(m1_predictors),
    }
    return frame, counts


def run(settings: Settings) -> dict[str, Any]:
    """Build ``srnd_frame.parquet`` and ``reports/srnd_frame.json``."""
    rt = settings.runtime
    p_assign = rt.processed_dir / "splits" / "srnd_assignment.parquet"
    p_init = rt.processed_dir / "qc" / "hydroswot_init.parquet"
    p_site = rt.processed_dir / "site_table.parquet"
    nn = list(settings.protocol.transforms.features)
    m1 = list(M1_SET)
    frame, counts = build_srnd_frame(
        read_parquet(p_assign), read_parquet(p_init), read_parquet(p_site), nn, m1
    )
    out = srnd_frame_path(rt.processed_dir)
    write_parquet(
        frame,
        out,
        stage="srnd_frame",
        params_sha256=settings.params_sha256,
        inputs={
            "splits/srnd_assignment.parquet": sha256_file(p_assign),
            "qc/hydroswot_init.parquet": sha256_file(p_init),
            "site_table.parquet": sha256_file(p_site),
        },
        sort_by=["row_id"],
    )
    report = {"params_sha256": settings.params_sha256, **counts}
    write_json(rt.reports_dir / "srnd_frame.json", report)
    LOG.info("srnd_frame: %d rows at %d sites", counts["rows"], counts["sites"])
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.srnd_frame`` (the dvc.yaml ``srnd_frame`` stage)."""
    argparse.ArgumentParser(description="Row-level S-RND frame (docs/3 13.6).").parse_args(argv)
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    run(settings)


if __name__ == "__main__":
    main()
