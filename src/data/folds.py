"""Fold views: row indices of the analysis frame per outer/inner/ES part (protocol 3.3-3.4).

Sources: protocol section 3.3 ("no model code may create folds; all code reads this table"),
section 3.4 (nested selection; member-specific ES split), section 4.9 and P5-D6 (analyses filter
the union site table by variant and flags; folds are shared), sections 2.5 and 2.6 (primary set
excludes ``F_DA``; ``F_Qsel`` and ``F_S`` are SA4/SA5 drops).

This module **never creates folds**. It reads ``site_table.parquet`` and
``fold_assignment.parquet`` and returns NumPy index arrays into an analysis frame. Torch-free and
xgboost-free (OpenMP rule, ADL-051).

Owning phase: 5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

__all__ = ["ANALYSES", "PARTS", "Analysis", "analysis_frame", "fit_provenance", "get_view"]


@dataclass(frozen=True)
class Analysis:
    """Row selection of one analysis: a ``site_core`` variant and flags that drop sites."""

    variant: str
    drop_flags: tuple[str, ...]
    discharge: str = "Q_E"


#: Primary and SA1-SA6 (protocol 4.9). SA1 keeps the primary rows and swaps Phi's discharge.
ANALYSES: Final[Mapping[str, Analysis]] = {
    "primary": Analysis("primary", ("F_DA",)),
    "SA1": Analysis("primary", ("F_DA",), discharge="Q_C"),
    "SA2": Analysis("sa2", ("F_DA",)),
    "SA3": Analysis("sa3", ("F_DA",)),
    "SA4": Analysis("primary", ("F_DA", "F_Qsel")),
    "SA5": Analysis("primary", ("F_DA", "F_S")),
    "SA6": Analysis("primary", ()),
}
PARTS: Final[tuple[str, ...]] = (
    "train",
    "test",
    "inner_train",
    "inner_val",
    "es_train",
    "es_val",
)


def analysis_frame(site_table: pd.DataFrame, analysis: str) -> pd.DataFrame:
    """Return the analysis rows (one per site, sorted by ``site_no``, fresh 0..n-1 index)."""
    spec = ANALYSES[analysis]
    rows = site_table.loc[site_table["variant"].eq(spec.variant)]
    for flag in spec.drop_flags:
        rows = rows.loc[~rows[flag].astype(bool)]
    if rows["site_no"].duplicated().any():
        raise ValueError("analysis frame must have one row per site")
    return rows.sort_values("site_no", kind="mergesort").reset_index(drop=True)


def get_view(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    family: str,
    repeat: int,
    outer_fold: int,
    part: str,
    index: int | None = None,
) -> np.ndarray:
    """Return sorted row positions of ``frame`` in one part of one outer block.

    Parameters
    ----------
    frame
        Output of :func:`analysis_frame`.
    folds
        ``fold_assignment.parquet``.
    family, repeat, outer_fold
        Block coordinates.
    part
        ``train`` / ``test`` (outer); ``inner_train`` / ``inner_val`` (grouped inner fold
        ``index`` in 0..inner_k-1, inside the outer training set); ``es_train`` / ``es_val``
        (ensemble member ``index`` in 1..M: outer training set minus / only its ES split).
    index
        Inner fold or ensemble member; required for the inner and ES parts.
    """
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}; expected one of {PARTS}")
    block = folds.loc[
        folds["family"].eq(family) & folds["repeat"].eq(repeat) & folds["outer_fold"].eq(outer_fold)
    ]
    if block.empty:
        raise KeyError(f"no fold block ({family}, {repeat}, {outer_fold})")
    block = block.set_index("site_no").reindex(frame["site_no"])
    if block["is_test"].isna().any():
        raise ValueError("analysis frame has sites outside the fold table")
    test = block["is_test"].to_numpy(dtype=bool)
    if part == "train":
        mask = ~test
    elif part == "test":
        mask = test
    else:
        if index is None:
            raise ValueError(f"part {part!r} needs an index")
        if part.startswith("inner"):
            inner = block["inner_fold"].to_numpy(dtype=np.int64)
            if index not in set(inner[~test].tolist()):
                raise KeyError(f"inner fold {index} not in block")
            mask = ~test & ((inner == index) if part == "inner_val" else (inner != index))
        else:
            col = f"es_member_{index}"
            if col not in block.columns:
                raise KeyError(f"no early-stopping column {col}")
            es = block[col].to_numpy(dtype=bool)
            mask = ~test & (es if part == "es_val" else ~es)
    return np.flatnonzero(mask).astype(np.int64)


def fit_provenance(
    analysis: str,
    family: str,
    repeat: int,
    outer_fold: int,
    part: str,
    index: int | None,
    *,
    site_table_sha256: str,
    fold_table_sha256: str,
) -> dict[str, str]:
    """Identity of one fitting part, stored in the ``NormStats`` JSON (L5 audit trail).

    The two hashes are the SHA-256 of ``site_table.parquet`` and ``fold_assignment.parquet``
    (``src.data.io.sha256_file``). ``index`` is the inner fold or ensemble member (``""`` when
    ``None``). ``NormStats`` additionally rejects non-training parts.
    """
    if analysis not in ANALYSES:
        raise ValueError(f"unknown analysis {analysis!r}")
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}; expected one of {PARTS}")
    for name, digest in (("site_table", site_table_sha256), ("fold_table", fold_table_sha256)):
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError(f"{name}_sha256 must be 64 lower-case hex characters")
    return {
        "analysis": analysis,
        "family": family,
        "fold_table_sha256": fold_table_sha256,
        "index": "" if index is None else str(index),
        "outer_fold": str(outer_fold),
        "part": part,
        "repeat": str(repeat),
        "site_table_sha256": site_table_sha256,
    }
