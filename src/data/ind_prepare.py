"""Stage ind_prepare: eHydro independent set, deduplicated against training reaches.

Sources: protocol sections 1.1 (ehydro, O-5 (c)), 3.2 (S-IND) and 3.5 (L8). Reads params
``splits.ind_dedupe_m``.

L8 rule (implemented in :func:`dedupe_l8`): an eHydro location is dropped if any of its member
COMIDs is a training COMID, or if it lies within ``splits.ind_dedupe_m`` (EPSG:5070) of a
training site. S-IND trains on **all** post-QC sites, so every site in U is a training site.

Status (P5-D8): the eHydro set (T5) is not acquired yet and the author table is pending, so
:func:`main` raises. The gate may pass with this stage pending only under an explicit PI waiver
(Phase 6 carry-over; S-IND is needed only at evaluation).

Owning phase: 5.
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
    """Apply the L8 overlap rule.

    Parameters
    ----------
    locations
        One row per (eHydro location, member COMID): ``loc_id``, ``comid``, ``x5070``,
        ``y5070`` (the location point, repeated on each member row).
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
    points = locations.drop_duplicates("loc_id").set_index("loc_id")[["x5070", "y5070"]]
    tree = shapely.STRtree(shapely.points(train_sites["x5070"], train_sites["y5070"]))
    hit, _ = tree.query(
        shapely.points(points["x5070"], points["y5070"]), predicate="dwithin", distance=radius_m
    )
    near = pd.Series(False, index=points.index)
    near.iloc[np.unique(hit)] = True
    drop = by_comid.reindex(points.index) | near
    kept = locations.loc[~locations["loc_id"].isin(drop[drop].index)].reset_index(drop=True)
    counts = {
        "locations": len(points),
        "dropped_comid": int(by_comid.sum()),
        "dropped_distance_only": int((near & ~by_comid.reindex(points.index)).sum()),
        "kept": int((~drop).sum()),
    }
    return kept, counts


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.ind_prepare``; pending until the eHydro set is acquired."""
    argparse.ArgumentParser(description="eHydro independent set (L8).").parse_args(argv)
    raise NotImplementedError(
        "ind_prepare is pending (P5-D8): the eHydro set (T5) is not acquired; gate waiver needed"
    )


if __name__ == "__main__":
    main()
