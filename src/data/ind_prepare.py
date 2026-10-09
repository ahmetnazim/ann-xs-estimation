"""Stage ind_prepare: eHydro independent set, deduplicated against training reaches.

Sources: protocol sections 1.1 (ehydro, O-5 (c)), 3.2 (S-IND), 3.5 (L8) and 11.3 (E-R9; P6-D10).
Reads params ``splits.ind_dedupe_m`` and ``ehydro``.

L8 rule (implemented in :func:`dedupe_l8`, docs/3 11.3 E-R9): an eHydro location is dropped if any
of its member COMIDs is a training COMID, or if any member flowline lies within
``splits.ind_dedupe_m`` (EPSG:5070) of a training site. S-IND trains on **all** post-QC sites, so
every site in U is a training site.

Status (P6-D10): the format-independent rules live in :mod:`src.data.ehydro`. :func:`main`
raises until the eHydro surveys are acquired and the survey reader (E-R5 (i)) is added in
P6-D10 step 5.

Owning phase: 5 (L8); 6 (P6-D10).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["dedupe_l8", "main"]


def dedupe_l8(
    locations: pd.DataFrame, train_sites: pd.DataFrame, radius_m: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the L8 overlap rule (docs/3 3.5 L8, 11.3 E-R9).

    Parameters
    ----------
    locations
        One row per (eHydro location, member COMID): ``loc_id``, ``comid``, ``geometry`` (the
        member flowline as a shapely geometry, EPSG:5070).
    train_sites
        ``site_no``, ``comid``, ``x5070``, ``y5070`` of every training site.
    radius_m
        ``splits.ind_dedupe_m``.

    Returns
    -------
    tuple
        Kept rows of ``locations`` and the counts (dropped by COMID, by distance, kept).
    """
    import shapely

    train_comids = set(train_sites["comid"].astype("int64"))
    by_comid = locations.groupby("loc_id")["comid"].agg(
        lambda s: bool(set(s.astype("int64")) & train_comids)
    )
    tree = shapely.STRtree(shapely.points(train_sites["x5070"], train_sites["y5070"]))
    hit, _ = tree.query(
        np.asarray(locations["geometry"].to_numpy(), dtype=object),
        predicate="dwithin",
        distance=radius_m,
    )
    near_ids = set(locations["loc_id"].to_numpy()[np.unique(hit)])
    near = pd.Series(by_comid.index.isin(list(near_ids)), index=by_comid.index)
    drop = by_comid | near
    kept = locations.loc[~locations["loc_id"].isin(drop[drop].index)].reset_index(drop=True)
    counts = {
        "locations": len(by_comid),
        "dropped_comid": int(by_comid.sum()),
        "dropped_distance_only": int((near & ~by_comid).sum()),
        "kept": int((~drop).sum()),
    }
    return kept, counts


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.ind_prepare``; pending until P6-D10 step 5."""
    argparse.ArgumentParser(description="eHydro independent set (L8).").parse_args(argv)
    raise NotImplementedError(
        "ind_prepare is pending (P6-D10 step 5): eHydro surveys not acquired; survey reader "
        "(docs/3 11.3 E-R5 (i)) not added yet"
    )


if __name__ == "__main__":
    main()
