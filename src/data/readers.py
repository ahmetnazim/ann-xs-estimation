"""Readers for raw units as delivered, with lossless casts and sentinel masking (V2, V4).

Sources: protocol section 2.3 (raw schemas; rules V2 lossless cast, V4 sentinels become missing
before any value check), section 2.7 (missing data are never imputed). Field names are those of
the delivered files; DBF field-name case differs between NHDPlusV2.1 VPUs, so DBF columns are
resolved case-insensitively and the delivered spelling is reported.

Every reader returns raw values; nothing here converts units (unit_map, protocol 2.1) or
drops rows (qc, protocol 2.4). Torch-free (ADL-051).

Owning phase: 5.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

__all__ = [
    "HYDROSWOT_DT_FORMAT",
    "HYDROSWOT_FILE",
    "CastResult",
    "RawSchemaError",
    "cast_datetime",
    "cast_float",
    "cast_int",
    "mask_sentinels",
    "read_attr_table",
    "read_dbf",
    "read_hydroswot_text",
    "resolve_columns",
]

#: Delivered HYDROSWOT measurement file (ScienceBase item 57435ae5e4b07e28b660af55).
HYDROSWOT_FILE: Final[str] = "SWOT_ADCP_Dataset.txt"
#: ``q_meas_dt`` format as delivered (every non-empty value parses with it; Batch C profile).
HYDROSWOT_DT_FORMAT: Final[str] = "%m/%d/%Y %H:%M:%S"


class RawSchemaError(ValueError):
    """A raw file violates V2 (required field missing or a lossy cast)."""


@dataclass(frozen=True)
class CastResult:
    """Outcome of a lossless cast of one text column.

    Attributes
    ----------
    values
        Cast column; empty strings are missing.
    n_missing
        Number of empty (missing) entries.
    n_bad
        Number of non-empty entries that do not cast losslessly (a V2 failure if > 0).
    """

    values: pd.Series
    n_missing: int
    n_bad: int


def _empty(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = raw.astype("string").str.strip()
    return text, text.isna() | text.eq("")


def cast_float(raw: pd.Series) -> CastResult:
    """Cast text to ``float64``; empty is missing, anything else must parse."""
    text, empty = _empty(raw)
    values = pd.to_numeric(text.mask(empty), errors="coerce").astype("float64")
    bad = ~empty & values.isna()
    return CastResult(values, int(empty.sum()), int(bad.sum()))


def cast_int(
    raw: pd.Series, dtype: Literal["Int8", "Int16", "Int32", "Int64"] = "Int64"
) -> CastResult:
    """Cast text to a nullable integer dtype; ``"2.00"`` is accepted, ``"2.5"`` is lossy."""
    as_float = cast_float(raw)
    values = as_float.values
    fractional = values.notna() & (values != np.floor(values))
    n_bad = as_float.n_bad + int(fractional.sum())
    return CastResult(values.mask(fractional).astype(dtype), as_float.n_missing, n_bad)


def cast_datetime(raw: pd.Series, fmt: str) -> CastResult:
    """Cast text to ``datetime64[ns]`` with an exact format; empty is missing."""
    text, empty = _empty(raw)
    values = pd.to_datetime(text.mask(empty), format=fmt, errors="coerce")
    bad = ~empty & values.isna()
    return CastResult(pd.Series(values, index=raw.index), int(empty.sum()), int(bad.sum()))


def mask_sentinels(values: pd.Series, sentinels: Iterable[float]) -> tuple[pd.Series, int]:
    """Return ``values`` with every sentinel code set to missing, and the count masked (V4)."""
    hit = values.isin(list(sentinels))
    return values.mask(hit), int(hit.sum())


def read_hydroswot_text(path: Path) -> pd.DataFrame:
    """Read ``SWOT_ADCP_Dataset.txt`` as text: tab-delimited, quoted header, latin-1.

    Every column is returned as stripped ``str`` (empty string = missing) so that the casts
    above can count lossy values (V2). ``q_meas_no`` carries leading spaces as delivered.
    """
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="latin-1",
        quotechar='"',
    )
    stripped: pd.DataFrame = frame.apply(lambda col: col.str.strip())
    return stripped


def resolve_columns(available: Sequence[str], required: Iterable[str]) -> dict[str, str]:
    """Map each required field to its delivered spelling, ignoring case.

    Raises
    ------
    RawSchemaError
        If a required field is absent or matches more than one delivered column.
    """
    by_upper: dict[str, list[str]] = {}
    for name in available:
        by_upper.setdefault(name.upper(), []).append(name)
    out: dict[str, str] = {}
    missing: list[str] = []
    for field in required:
        hits = by_upper.get(field.upper(), [])
        if len(hits) == 1:
            out[field] = hits[0]
        elif not hits:
            missing.append(field)
        else:
            raise RawSchemaError(f"field {field!r} is ambiguous: {hits}")
    if missing:
        raise RawSchemaError(f"required fields missing: {missing}")
    return out


def read_dbf(path: Path, fields: Iterable[str]) -> pd.DataFrame:
    """Read attribute columns of a DBF (no geometry), renamed to the requested spelling.

    Parameters
    ----------
    path
        ``.dbf`` file (NHDPlusV2.1 tables, D50 table).
    fields
        Required fields; matched case-insensitively (:func:`resolve_columns`).
    """
    import pyogrio  # geospatial I/O is imported where used (cold start of the other stages)

    info = pyogrio.read_info(path)
    mapping = resolve_columns([str(name) for name in info["fields"]], fields)
    frame = pyogrio.read_dataframe(path, columns=list(mapping.values()), read_geometry=False)
    return pd.DataFrame(frame).rename(columns={v: k for k, v in mapping.items()})[list(mapping)]


def read_attr_table(
    path: Path, columns: Sequence[str], key: str = "COMID"
) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Read a comma-delimited attribute table: ``key`` as int64, ``columns`` as float64.

    Values that do not parse raise :class:`RawSchemaError` (V2). Returns the frame and the
    mapping ``requested -> delivered`` column names.
    """
    header = pd.read_csv(path, nrows=0).columns
    mapping = resolve_columns([str(c) for c in header], [key, *columns])
    dtypes = {mapping[key]: "int64", **{mapping[c]: "float64" for c in columns}}
    try:
        frame = pd.read_csv(path, usecols=list(dtypes), dtype=dtypes)
    except ValueError as exc:
        raise RawSchemaError(f"{path.name}: lossy cast ({exc})") from exc
    frame = frame.rename(columns={v: k for k, v in mapping.items()})[[key, *columns]]
    return frame, mapping
