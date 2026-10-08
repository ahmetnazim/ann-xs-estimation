"""Shared, deterministic I/O for the ETL stages: Parquet, JSON reports and provenance.

Sources: protocol section 1.2 (processed data are Parquet; every processed file records the
hashes of its inputs), section 1.5 (reports are DVC metrics, committed to git); PHASE_05 hard
rule on determinism (sorted keys, no wall-clock metadata).

Rules implemented here:

* Parquet is written with pyarrow, zstd compression and **no** timestamps. The key-value
  metadata ``annxs`` holds the stage name, ``params_sha256`` and the SHA-256 of every input
  file or raw-unit manifest, so a processed file names its exact inputs.
* JSON reports use sorted keys, two-space indentation, a trailing newline and no timestamps, so
  that a re-run with identical inputs gives byte-identical reports (they are committed).
* Torch-free (OpenMP rule, ADL-051).

Owning phase: 5.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "META_KEY",
    "read_parquet",
    "read_parquet_meta",
    "sha256_file",
    "unit_manifest_sha256",
    "write_json",
    "write_parquet",
]

META_KEY: Final[bytes] = b"annxs"
_CHUNK: Final[int] = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def unit_manifest_sha256(unit_dir: Path) -> str:
    """Return the SHA-256 of a raw unit's ``MANIFEST.sha256`` (its content fingerprint)."""
    return sha256_file(unit_dir / "MANIFEST.sha256")


def _jsonable(value: object) -> object:
    """Convert NumPy scalars and non-finite floats into strict JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic JSON report (sorted keys, no timestamps, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def write_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    stage: str,
    params_sha256: str,
    inputs: Mapping[str, str],
    sort_by: list[str] | None = None,
) -> None:
    """Write ``frame`` as Parquet with provenance metadata and a deterministic row order.

    Parameters
    ----------
    frame
        Table to write. The index is dropped.
    path
        Output file; parent directories are created.
    stage
        Name of the dvc.yaml stage writing the file.
    params_sha256
        SHA-256 of the exact ``params.yaml`` bytes (``Settings.params_sha256``).
    inputs
        Mapping ``input name -> SHA-256`` of the files or raw-unit manifests read.
    sort_by
        Key columns; rows are sorted by them (stable) before writing.
    """
    if sort_by:
        frame = frame.sort_values(sort_by, kind="mergesort")
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[META_KEY] = json.dumps(
        {"stage": stage, "params_sha256": params_sha256, "inputs": dict(sorted(inputs.items()))},
        sort_keys=True,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path, compression="zstd")


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file written by :func:`write_parquet`."""
    frame: pd.DataFrame = pq.read_table(path).to_pandas()
    return frame


def read_parquet_meta(path: Path) -> dict[str, Any]:
    """Return the ``annxs`` provenance metadata of a Parquet file."""
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(META_KEY)
    if raw is None:
        raise ValueError(f"{path} has no annxs provenance metadata")
    loaded: dict[str, Any] = json.loads(raw)
    return loaded
