"""Stage condition_extract: one mean-flow row per linked site (protocol section 2.6).

Sources: protocol section 2.6 (arg min over the site's ``hydroswot_init`` rows of
``|ln(Q_obs / Q_E)|``; ties to the most recent ``q_meas_dt``; ``F_Qsel`` if the ratio exceeds
``ln condition.qsel_ratio_flag``; one row per site), section 2.4 (R13: ``Q_E <= 0`` or missing
puts the site outside the domain), section 2.1 (``Q_E`` = EROM ``Q0001E`` in m3/s).

Selection variants (P5-D6: flagged rows and the SA alternatives are kept; groups and folds are
built once over the union of sites; primary analyses filter by flags). ``site_core`` is a long
table keyed by (``site_no``, ``variant``):

* ``primary``: arg min over the site's rows with ``F_eps`` and ``F_t`` false (R8 and R9 rows are
  "dropped from primary", protocol 2.4), criterion ``|ln(Q_obs / Q_E)|``.
* ``sa2``: the same rows, criterion ``|ln(Q_meas / Q_E)|`` (``meas_q_va``, SA2).
* ``sa3``: all of the site's ``hydroswot_init`` rows, criterion ``|ln(Q_obs / Q_E)|`` (SA3).

A site appears in a variant only if it has a selectable row there. Ties in the criterion go to
the latest ``q_meas_dt``, then to the lowest ``row_id``, so the result does not depend on row
order. ``F_Qsel`` uses the variant's own discharge.

Owning phase: 5.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.data.io import read_parquet, sha256_file, write_json, write_parquet

__all__ = ["SITE_CORE_COLUMNS", "VARIANTS", "extract_conditions", "main", "select_rows"]

LOG = logging.getLogger(__name__)

VARIANTS: Final[tuple[str, ...]] = ("primary", "sa2", "sa3")
SITE_CORE_COLUMNS: Final[tuple[str, ...]] = (
    "site_no", "variant", "comid", "link_method", "F_snap", "F_DA", "division", "lat", "lon",
    "x5070", "y5070", "row_id", "q_meas_dt", "W", "dbar", "dmax", "A", "V", "t_obs", "eps_A",
    "Q_obs", "Q_meas", "Q_E", "Q_C", "V_E", "DA_site_km2", "F_t", "F_qual", "F_eps", "F_tout",
    "F_Qsel", "qsel_log_ratio", "n_rows_site",
)  # fmt: skip


def select_rows(rows: pd.DataFrame, q_col: str) -> pd.Series:
    """Return the selected ``row_id`` per ``site_no``: arg min ``|ln(q / Q_E)|``.

    Ties go to the latest ``q_meas_dt``, then to the lowest ``row_id``. Rows with a
    non-positive or missing ``q`` cannot be selected.
    """
    crit = np.abs(np.log(rows[q_col] / rows["Q_E"]))
    cand = rows.assign(_crit=crit).loc[np.isfinite(crit)]
    cand = cand.sort_values(
        ["site_no", "_crit", "q_meas_dt", "row_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    first = cand.drop_duplicates("site_no")
    return pd.Series(first["row_id"].to_numpy(), index=first["site_no"].to_numpy(), name=q_col)


def extract_conditions(
    init: pd.DataFrame, links: pd.DataFrame, erom: pd.DataFrame, qsel_ratio_flag: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build ``site_core``: one row per (linked site inside the domain, selection variant).

    Parameters
    ----------
    init
        ``qc/hydroswot_init.parquet``.
    links
        ``link/site_comid.parquet``.
    erom
        ``unit_mapped/nhd_erom.parquet`` (``comid``, ``Q_E``, ``Q_C``, ``V_E``).
    qsel_ratio_flag
        ``condition.qsel_ratio_flag`` (``F_Qsel`` if ``|ln(Q_obs / Q_E)| > ln`` of it).
    """
    link_cols = ["site_no", "comid", "link_method", "F_snap", "F_DA", "x5070", "y5070"]
    sites = links[link_cols].merge(erom, on="comid", how="left")
    r13 = ~(sites["Q_E"] > 0)
    sites = sites.loc[~r13]
    rows = init.merge(sites, on="site_no", how="inner")
    eligible = rows.loc[~rows["F_eps"] & ~rows["F_t"]]
    spec = {"primary": (eligible, "Q_obs"), "sa2": (eligible, "Q_meas"), "sa3": (rows, "Q_obs")}
    n_rows = rows.groupby("site_no").size()
    parts: list[pd.DataFrame] = []
    for variant, (pool, q_col) in spec.items():
        chosen = rows.loc[rows["row_id"].isin(select_rows(pool, q_col).to_numpy())].copy()
        chosen["variant"] = variant
        chosen["qsel_log_ratio"] = np.log(chosen[q_col] / chosen["Q_E"])
        chosen["F_Qsel"] = chosen["qsel_log_ratio"].abs() > math.log(qsel_ratio_flag)
        chosen["n_rows_site"] = n_rows.reindex(chosen["site_no"]).astype("int64").to_numpy()
        parts.append(chosen)
    out = pd.concat(parts, ignore_index=True)[list(SITE_CORE_COLUMNS)]
    out["variant"] = pd.Categorical(out["variant"], categories=list(VARIANTS)).astype(str)
    out = out.sort_values(["site_no", "variant"], kind="mergesort").reset_index(drop=True)
    if out.duplicated(["site_no", "variant"]).any():
        raise AssertionError("site_core must have one row per (site_no, variant)")
    by_variant = out.groupby("variant")
    primary = out.loc[out["variant"].eq("primary")].set_index("site_no")
    sa3 = out.loc[out["variant"].eq("sa3")].set_index("site_no")
    counts = {
        "sites_linked": len(links),
        "R13_QE_le_0_or_missing": int(r13.sum()),
        "sites_union": int(out["site_no"].nunique()),
        "sites_per_variant": {k: int(v) for k, v in by_variant.size().items()},
        "F_Qsel_per_variant": {k: int(v) for k, v in by_variant["F_Qsel"].sum().items()},
        "F_DA_primary": int(primary["F_DA"].sum()),
        "sa3_row_differs_from_primary": int(
            (sa3["row_id"].reindex(primary.index) != primary["row_id"]).sum()
        ),
        "sa3_selected_F_eps": int(sa3["F_eps"].sum()),
    }
    return out, counts


def run(settings: Settings) -> None:
    """Read QC rows, links and EROM; write ``site_core.parquet``."""
    rt = settings.runtime
    p_init = rt.processed_dir / "qc" / "hydroswot_init.parquet"
    p_link = rt.processed_dir / "link" / "site_comid.parquet"
    p_erom = rt.processed_dir / "unit_mapped" / "nhd_erom.parquet"
    out, counts = extract_conditions(
        read_parquet(p_init),
        read_parquet(p_link),
        read_parquet(p_erom),
        settings.protocol.condition.qsel_ratio_flag,
    )
    LOG.info("site_core: %d sites; %s", len(out), counts)
    write_parquet(
        out,
        rt.processed_dir / "site_core.parquet",
        stage="condition_extract",
        params_sha256=settings.params_sha256,
        inputs={
            "qc/hydroswot_init.parquet": sha256_file(p_init),
            "link/site_comid.parquet": sha256_file(p_link),
            "unit_mapped/nhd_erom.parquet": sha256_file(p_erom),
        },
        sort_by=["site_no", "variant"],
    )
    write_json(
        rt.reports_dir / "condition_counts.json",
        {"params_sha256": settings.params_sha256, **counts},
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.condition_extract`` (the dvc.yaml stage)."""
    argparse.ArgumentParser(description="Mean-flow condition (protocol 2.6).").parse_args(argv)
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s", settings.params_sha256)
    run(settings)


if __name__ == "__main__":
    main()
