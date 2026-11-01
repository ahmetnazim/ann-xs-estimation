"""Datum-free observed stage and width responses at the S-ID gauges (OPS-1, docs/4 section 3).

For every primary site the *anchor* is the HYDROSWOT row selected by ``condition_extract``
(``site_table.parquet`` ``row_id``). Every other measurement ``i`` of the same site gives

* ``q_ratio = Q_i / Q_a`` (``q_va``, m3/s),
* ``dh_obs = 0.3048 (stage_i - stage_a)`` (``stage_va``, gauge height; datum cancels),
* ``dlnw_obs = ln(W_i / W_a)`` (``stream_wdth_va``; missing when either width is missing).

Exclusions (section 3.2, pre-registered in ``params_operational.yaml``; every count reported):
site-level ``ad_site_cd`` in the excluded set (anchor row's code) or missing anchor stage;
row-level duplicate ``(site_no, q_meas_no)`` (lowest ``row_id`` kept, as QC R11), channel number
not equal to the whole-channel value, non-positive or missing flow, missing stage, the anchor row
itself, ``|t_i - t_a|`` beyond the window, ``|dh_obs|`` beyond the plausibility bound.

``stage_va`` and ``ad_site_cd`` are not carried by the v1.0.0 unit-mapped table, so they are read
from the delivered text file with the v1.0.0 reader and sentinel rules; ``row_id`` is the 0-based
row of that file in both tables (``unit_map.map_hydroswot``). Torch-free.

Owning phase: OPS-1 (post-release extension).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.config import FT_TO_M
from src.data.readers import cast_float, mask_sentinels, read_hydroswot_text
from src.data.schemas import SENTINELS
from src.operational.config import ObservationParams

__all__ = [
    "DAYS_PER_YEAR",
    "OBS_COLUMNS",
    "build_observations",
    "read_stage_fields",
]

DAYS_PER_YEAR: Final[float] = 365.25
OBS_COLUMNS: Final[tuple[str, ...]] = (
    "site_no",
    "row_id",
    "division",
    "q_ratio",
    "ln_q_ratio",
    "dh_obs",
    "dlnw_obs",
    "dt_years",
)
_UNIT_COLS: Final[tuple[str, ...]] = (
    "row_id",
    "site_no",
    "q_meas_no",
    "q_meas_dt",
    "q_meas_chan_nu",
    "Q_obs",
    "W",
)
_ANCHOR_COLS: Final[tuple[str, ...]] = ("site_no", "row_id", "division")


def read_stage_fields(path: Path) -> pd.DataFrame:
    """Return ``row_id``, ``stage_ft`` (sentinels masked) and ``ad_site_cd`` from the raw file."""
    raw = read_hydroswot_text(path)
    stage, _ = mask_sentinels(cast_float(raw["stage_va"]).values, SENTINELS)
    code = raw["ad_site_cd"].replace("", pd.NA).astype("string")
    return pd.DataFrame(
        {
            "row_id": np.arange(len(raw), dtype=np.int64),
            "stage_ft": stage.to_numpy(dtype=np.float64, na_value=np.nan),
            "ad_site_cd": code.to_numpy(),
        }
    )


def _count(counts: dict[str, int], key: str, mask: pd.Series) -> None:
    counts[key] = int(mask.sum())


def build_observations(
    unit_mapped: pd.DataFrame,
    stage: pd.DataFrame,
    anchors: pd.DataFrame,
    cfg: ObservationParams,
    *,
    window_years: float | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the observed responses (one row per non-anchor measurement kept).

    Parameters
    ----------
    unit_mapped
        ``data/processed/unit_mapped/hydroswot.parquet`` (needs ``_UNIT_COLS``).
    stage
        Output of :func:`read_stage_fields`.
    anchors
        Primary rows of ``site_table.parquet`` (``site_no``, ``row_id``, ``division``).
    cfg
        Pre-registered exclusions.
    window_years
        ``|t_i - t_a|`` limit in years; ``None`` disables the window (sensitivity run).

    Returns
    -------
    tuple
        The observation frame (:data:`OBS_COLUMNS`) sorted by ``site_no, row_id`` and the
        exclusion counts.
    """
    missing = [c for c in _UNIT_COLS if c not in unit_mapped.columns]
    missing += [c for c in _ANCHOR_COLS if c not in anchors.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    if anchors["site_no"].duplicated().any():
        raise ValueError("anchors must hold one row per site")
    counts: dict[str, int] = {}
    # validate="1:1" raises pandas MergeError (a ValueError) on a duplicated row_id
    rows = unit_mapped[list(_UNIT_COLS)].merge(stage, on="row_id", how="left", validate="1:1")

    anc = anchors[list(_ANCHOR_COLS)].rename(columns={"row_id": "anchor_row_id"})
    anc = anc.merge(
        rows[["row_id", "site_no", "q_meas_dt", "Q_obs", "W", "stage_ft", "ad_site_cd"]].rename(
            columns={
                "row_id": "anchor_row_id",
                "site_no": "_anchor_site",
                "q_meas_dt": "t_a",
                "Q_obs": "q_a",
                "W": "w_a",
                "stage_ft": "stage_a",
                "ad_site_cd": "code_a",
            }
        ),
        on="anchor_row_id",
        how="left",
        validate="1:1",
    )
    if not (anc["_anchor_site"] == anc["site_no"]).all():
        raise ValueError("anchor row_id does not belong to its site")
    counts["sites_anchor"] = len(anc)
    excluded_code = anc["code_a"].isin(list(cfg.exclude_ad_site_cd)).fillna(False).astype(bool)
    _count(counts, "sites_excluded_ad_site_cd", excluded_code)
    no_stage = ~excluded_code & ~np.isfinite(anc["stage_a"].to_numpy(dtype=np.float64))
    _count(counts, "sites_excluded_anchor_stage_missing", no_stage)
    bad_q = ~excluded_code & ~no_stage & ~(anc["q_a"] > 0)
    _count(counts, "sites_excluded_anchor_flow_invalid", bad_q)
    anc = anc.loc[~(excluded_code | no_stage | bad_q)]
    counts["sites_eligible"] = len(anc)

    obs = rows.merge(anc.drop(columns=["_anchor_site", "code_a"]), on="site_no", how="inner")
    counts["rows_at_eligible_sites"] = len(obs)
    obs = obs.sort_values(["site_no", "row_id"], kind="mergesort")
    dup = obs.duplicated(subset=["site_no", "q_meas_no"], keep="first")
    _count(counts, "rows_excluded_duplicate_measurement", dup)
    obs = obs.loc[~dup]
    chan = obs["q_meas_chan_nu"].astype("Int64")
    partial = ~chan.eq(cfg.channel_number).fillna(False).astype(bool)
    _count(counts, "rows_excluded_channel", partial)
    obs = obs.loc[~partial]
    is_anchor = obs["row_id"] == obs["anchor_row_id"]
    _count(counts, "rows_anchor", is_anchor)
    obs = obs.loc[~is_anchor]
    bad = ~(obs["Q_obs"] > 0) | ~np.isfinite(obs["stage_ft"].to_numpy(dtype=np.float64))
    _count(counts, "rows_excluded_flow_or_stage_missing", bad)
    obs = obs.loc[~bad]
    dt_years = (obs["q_meas_dt"] - obs["t_a"]).abs().dt.total_seconds() / (86400.0 * DAYS_PER_YEAR)
    if window_years is None:
        outside = pd.Series(False, index=obs.index)
    else:
        outside = ~(dt_years <= window_years)
    _count(counts, "rows_excluded_time_window", outside)
    obs = obs.loc[~outside]
    dt_years = dt_years.loc[obs.index]
    dh = FT_TO_M * (obs["stage_ft"] - obs["stage_a"])
    implausible = ~(dh.abs() <= cfg.max_abs_dh_m)
    _count(counts, "rows_excluded_implausible_dh", implausible)
    keep = ~implausible
    obs, dh, dt_years = obs.loc[keep], dh.loc[keep], dt_years.loc[keep]

    q_ratio = obs["Q_obs"].to_numpy(dtype=np.float64) / obs["q_a"].to_numpy(dtype=np.float64)
    w_i = obs["W"].to_numpy(dtype=np.float64, na_value=np.nan)
    w_a = obs["w_a"].to_numpy(dtype=np.float64, na_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlnw = np.where((w_i > 0) & (w_a > 0), np.log(w_i / w_a), np.nan)
    out = pd.DataFrame(
        {
            "site_no": obs["site_no"].astype(str).to_numpy(),
            "row_id": obs["row_id"].to_numpy(dtype=np.int64),
            "division": obs["division"].astype(str).to_numpy(),
            "q_ratio": q_ratio,
            "ln_q_ratio": np.log(q_ratio),
            "dh_obs": dh.to_numpy(dtype=np.float64),
            "dlnw_obs": dlnw,
            "dt_years": dt_years.to_numpy(dtype=np.float64),
        }
    )
    counts["rows_kept"] = len(out)
    counts["sites_with_rows"] = int(out["site_no"].nunique())
    counts["rows_with_width"] = int(np.isfinite(dlnw).sum())
    return out.reset_index(drop=True), counts
