"""Unit tests for src/data/io.py and src/data/readers.py (Phase 5, Batch C). Synthetic only.

Sources: docs/3 1.2 (Parquet artefacts), 2.1 (unit map), 2.3 (raw schemas),
2.7 (missing values are never imputed); ADL-051.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import io as dio
from src.data import readers as rd
from src.data.schemas import SENTINELS

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------------- io
def test_parquet_roundtrip_metadata_and_sort(tmp_path: Path) -> None:
    frame = pd.DataFrame({"k": [3, 1, 2], "v": [0.3, 0.1, 0.2]})
    path = tmp_path / "x.parquet"
    dio.write_parquet(
        frame, path, stage="s", params_sha256="a" * 64, inputs={"b": "2", "a": "1"}, sort_by=["k"]
    )
    back = dio.read_parquet(path)
    assert back["k"].tolist() == [1, 2, 3]
    meta = dio.read_parquet_meta(path)
    assert meta == {"stage": "s", "params_sha256": "a" * 64, "inputs": {"a": "1", "b": "2"}}


def test_parquet_is_byte_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame({"k": np.arange(100), "v": np.linspace(0, 1, 100)})
    for name in ("a.parquet", "b.parquet"):
        dio.write_parquet(frame, tmp_path / name, stage="s", params_sha256="0" * 64, inputs={})
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


def test_read_parquet_meta_requires_provenance(tmp_path: Path) -> None:
    path = tmp_path / "plain.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="provenance"):
        dio.read_parquet_meta(path)


def test_write_json_sorted_nan_to_null(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    dio.write_json(path, {"b": np.float64("nan"), "a": np.int64(2), "c": [math.inf, 1.5]})
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert list(json.loads(text)) == ["a", "b", "c"]
    assert json.loads(text) == {"a": 2, "b": None, "c": [None, 1.5]}


# ----------------------------------------------------------------------------- casts
def test_cast_float_counts_missing_and_lossy() -> None:
    res = rd.cast_float(pd.Series(["1.5", "", " 2 ", "abc", "-9999"]))
    assert res.n_missing == 1
    assert res.n_bad == 1
    assert res.values.iloc[2] == 2.0
    assert res.values.iloc[4] == -9999.0  # sentinels are masked separately (V4)


def test_cast_int_accepts_integral_decimals_only() -> None:
    res = rd.cast_int(pd.Series(["0.00", "2.00", "2.5", ""]), "Int16")
    assert res.n_bad == 1
    assert res.n_missing == 1
    assert str(res.values.dtype) == "Int16"
    assert res.values.iloc[1] == 2


def test_cast_datetime_exact_format() -> None:
    res = rd.cast_datetime(
        pd.Series(["01/02/2003 04:05:06", "2003-01-02", ""]), rd.HYDROSWOT_DT_FORMAT
    )
    assert res.n_bad == 1
    assert res.n_missing == 1
    assert res.values.iloc[0] == pd.Timestamp("2003-01-02 04:05:06")


def test_mask_sentinels_before_any_value_rule() -> None:
    values, n = rd.mask_sentinels(pd.Series([-9999.0, -9998.0, -1.0, 3.0]), SENTINELS)
    assert n == 2
    assert values.isna().tolist() == [True, True, False, False]


def test_resolve_columns_case_insensitive() -> None:
    mapping = rd.resolve_columns(["ComID", "SLOPE", "TerminalPa"], ["COMID", "Slope", "TerminalPa"])
    assert mapping == {"COMID": "ComID", "Slope": "SLOPE", "TerminalPa": "TerminalPa"}
    with pytest.raises(rd.RawSchemaError, match="missing"):
        rd.resolve_columns(["a"], ["b"])
    with pytest.raises(rd.RawSchemaError, match="ambiguous"):
        rd.resolve_columns(["a", "A"], ["a"])


def test_read_hydroswot_text_strips_and_keeps_text(tmp_path: Path) -> None:
    path = tmp_path / "h.txt"
    path.write_text('"site_no"\t"q_meas_no"\n01234567\t   12\n', encoding="latin-1")
    frame = rd.read_hydroswot_text(path)
    assert frame.loc[0, "site_no"] == "01234567"
    assert frame.loc[0, "q_meas_no"] == "12"


def test_read_attr_table_lossy_cast_raises(tmp_path: Path) -> None:
    ok = tmp_path / "ok.txt"
    ok.write_text("COMID,CAT_X\r\n1,0.5\r\n2,-9999\r\n", encoding="utf-8")
    frame, mapping = rd.read_attr_table(ok, ["CAT_X"])
    assert frame["COMID"].dtype == "int64"
    assert mapping == {"COMID": "COMID", "CAT_X": "CAT_X"}
    bad = tmp_path / "bad.txt"
    bad.write_text("COMID,CAT_X\n1,abc\n", encoding="utf-8")
    with pytest.raises(rd.RawSchemaError, match="lossy"):
        rd.read_attr_table(bad, ["CAT_X"])
