"""Nested-basin network inputs from the raw NHDPlusV2.1 unit (docs/3 3.6.5, 13.9).

Moved out of :mod:`src.data.nhdplus` (ADL-121): ``nhdplus.py`` is a declared dependency of the
frozen ``validate_raw``, ``unit_map``, ``link`` and ``features`` stages, so Phase 8 additions live
here and leave its bytes, and ``dvc.lock``, unchanged. Behaviour is identical to the Batch C1 code.

Owning phase: 8.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import nhdplus
from src.data.readers import RawSchemaError

__all__ = ["NetworkInputs", "network_inputs", "read_network"]


@dataclass(frozen=True)
class NetworkInputs:
    """Inputs of :func:`src.evaluation.dependence_audit.nested_status` (docs/3 13.9).

    ``edges`` holds every PlusFlow ``(FromComID, ToComID)`` row with both ends non-zero
    (divergences included); ``terminal_path`` and ``drainage_km2`` (VAA ``TerminalPa`` and
    ``TotDASqKm``) cover the requested COMIDs only.
    """

    edges: Any  # numpy int64 array (E, 2)
    terminal_path: dict[int, int]
    drainage_km2: dict[int, float]
    missing: tuple[int, ...]  # requested COMIDs absent from the VAA


def network_inputs(
    vaa: pd.DataFrame, plusflow: pd.DataFrame, comids: Iterable[int]
) -> NetworkInputs:
    """Build the nested-basin inputs from VAA and PlusFlow frames (protocol spelling).

    Upstream reachability is evaluated by ``nested_status`` on the reversed graph; here the edges
    are only filtered (``0`` = no node). A COMID listed twice in the VAA raises.
    """
    import numpy as np

    if vaa["ComID"].duplicated().any():
        raise RawSchemaError("PlusFlowlineVAA: ComID not unique")
    flow = plusflow[["FromComID", "ToComID"]].to_numpy(dtype=np.int64)
    edges = flow[(flow[:, 0] != 0) & (flow[:, 1] != 0)]
    wanted = sorted({int(c) for c in comids})
    v = vaa.set_index("ComID")
    have = [c for c in wanted if c in v.index]
    sub = v.loc[have]
    keys = [int(c) for c in have]
    terminal = sub["TerminalPa"].to_numpy(dtype=np.int64)
    area = sub["TotDASqKm"].to_numpy(dtype=np.float64)
    return NetworkInputs(
        edges=edges,
        terminal_path={c: int(t) for c, t in zip(keys, terminal, strict=True)},
        drainage_km2={c: float(a) for c, a in zip(keys, area, strict=True)},
        missing=tuple(c for c in wanted if c not in v.index),
    )


def read_network(unit_dir: Path, comids: Iterable[int]) -> NetworkInputs:
    """Read VAA and PlusFlow over all VPUs from the raw unit and call :func:`network_inputs`."""
    vaa, _ = nhdplus.read_nhd_table(unit_dir, "vaa")
    plusflow, _ = nhdplus.read_nhd_table(unit_dir, "plusflow")
    return network_inputs(vaa, plusflow, comids)
