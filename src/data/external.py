"""Readers for the external comparison products: HyG (S1) and the Zarrabi product (L8).

Sources: docs/3 section 13.9 (S1: HyG ``Manning_n`` at matching gages, ``Gage_ID`` = ``site_no``,
finite and positive), section 13.5 (Zarrabi product ``Bankfull_Meanflow_CONUS.txt`` fields
``COMID``, ``mf_width``, ``mf_depth``; the unit is asserted from the Zenodo record
doi:10.5281/zenodo.13883263 before Event 3, or STOP), protocol 4.6 (S1) and 3.5 (L8).
Raw units ``data/raw/hyg`` (Zenodo record 7868764) and ``data/raw/zarrabi_product``.

Implementation definitions (Phase 8 Batch C; PI confirmation requested):

* **HyG gage IDs.** ``HyG_20230426.csv`` stores ``Gage_ID`` as a number, so leading zeros are
  lost (9,079 of 11,513 IDs have 7 digits). An all-digit ID shorter than ``HYG_ID_WIDTH`` (8, the
  shortest USGS station number) is left-padded with zeros; longer IDs are used as delivered.
  Non-digit IDs and IDs that occur more than once after padding are dropped and counted.
* **Zarrabi unit.** The Zenodo record (checked 2026-10-12) states no unit for ``mf_width`` or
  ``mf_depth``. :data:`ZARRABI_RECORD_UNIT` is therefore ``None`` and :func:`read_zarrabi_product`
  refuses (:class:`UnitNotAssertedError`) until a unit is recorded with its source.

These readers return values only; no comparison with a prediction happens here. Torch-free and
xgboost-free. Owning phase: 8.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "FT_TO_M",
    "HYG_ID_WIDTH",
    "ZARRABI_FIELDS",
    "ZARRABI_RECORD_UNIT",
    "UnitNotAssertedError",
    "read_hyg_manning",
    "read_zarrabi_product",
]

#: Shortest USGS station number (8 digits); shorter all-digit HyG IDs are left-padded.
HYG_ID_WIDTH: Final[int] = 8
#: docs/3 13.5: fields of ``Bankfull_Meanflow_CONUS.txt`` used for L8.
ZARRABI_FIELDS: Final[tuple[str, ...]] = ("COMID", "mf_width", "mf_depth")
#: Unit of ``mf_width``/``mf_depth`` as stated by the Zenodo record; ``None``: not stated there.
ZARRABI_RECORD_UNIT: Final[str | None] = None
FT_TO_M: Final[float] = 0.3048
_UNITS: Final[dict[str, float]] = {"m": 1.0, "ft": FT_TO_M}


class UnitNotAssertedError(RuntimeError):
    """docs/3 13.5: the Zarrabi product unit was not asserted from the Zenodo record (STOP)."""


def read_hyg_manning(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return ``site_no, manning_n`` from HyG and the counts of every row dropped (S1).

    ``Gage_ID`` is read as text; IDs are padded to :data:`HYG_ID_WIDTH` digits (module notes).
    Rows with a non-finite or non-positive ``Manning_n`` are dropped.
    """
    raw = pd.read_csv(path, usecols=["Gage_ID", "Manning_n"], dtype={"Gage_ID": str})
    ids = raw["Gage_ID"].astype("string").str.strip()
    digits = ids.str.fullmatch(r"\d+").fillna(False).astype(bool)
    short = digits & (ids.str.len() < HYG_ID_WIDTH)
    site = ids.where(~short, ids.str.zfill(HYG_ID_WIDTH))
    n = pd.to_numeric(raw["Manning_n"], errors="coerce").astype("float64")
    frame = pd.DataFrame({"site_no": site, "manning_n": n})[digits.to_numpy()]
    dup = frame["site_no"].duplicated(keep=False)
    frame = frame[~dup.to_numpy()]
    valid = np.isfinite(frame["manning_n"].to_numpy()) & (frame["manning_n"].to_numpy() > 0)
    out = frame[valid].astype({"site_no": str}).sort_values("site_no", kind="mergesort")
    counts = {
        "rows": len(raw),
        "non_digit_id": int((~digits).sum()),
        "padded_id": int(short.sum()),
        "duplicated_id_rows": int(dup.sum()),
        "manning_n_missing_or_nonpositive": int((~valid).sum()),
        "kept": len(out),
    }
    return out.reset_index(drop=True), counts


def read_zarrabi_product(
    path: Path, comids: Iterable[int], unit: str | None = ZARRABI_RECORD_UNIT
) -> pd.DataFrame:
    """Return ``comid, mf_width_m, mf_depth_m`` for ``comids`` (docs/3 13.5; L8, descriptive).

    Raises
    ------
    UnitNotAssertedError
        If ``unit`` is ``None`` (the Zenodo record states no unit; docs/3 13.5 requires a STOP).
    ValueError
        On an unknown unit or a requested COMID that occurs more than once.
    """
    if unit is None:
        raise UnitNotAssertedError(
            "docs/3 13.5: the unit of mf_width/mf_depth is not asserted from the Zenodo record "
            "(doi:10.5281/zenodo.13883263); STOP before Event 3 until the PI records it"
        )
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {sorted(_UNITS)}")
    wanted = {int(c) for c in comids}
    raw = pd.read_csv(path, usecols=list(ZARRABI_FIELDS), dtype={"COMID": "int64"})
    sub = raw[raw["COMID"].isin(wanted).to_numpy()]
    if sub["COMID"].duplicated().any():
        raise ValueError("Zarrabi product: a requested COMID occurs more than once")
    factor = _UNITS[unit]
    out = pd.DataFrame(
        {
            "comid": sub["COMID"].to_numpy(dtype=np.int64),
            "mf_width_m": sub["mf_width"].to_numpy(dtype=np.float64) * factor,
            "mf_depth_m": sub["mf_depth"].to_numpy(dtype=np.float64) * factor,
        }
    )
    return out.sort_values("comid", kind="mergesort").reset_index(drop=True)
