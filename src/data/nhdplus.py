"""NHDPlusV2.1 tables: location per VPU, delivered field names, readers and checks V2-V5.

Sources: protocol section 1.1 (NHDPlusV2.1 components per VPU, GageLoc national), section 2.3
(table contracts: PlusFlowlineVAA, EROM_MA0001, ElevSlope, GageLoc, PlusFlow), section 2.1
(units as delivered); NHDPlusV2 User Guide (table definitions).

Delivered layout (``acquire`` extracts each archive into a folder named after it)::

    <archive stem>/NHDPlus<DA>/NHDPlus<VPU>/<component>/<table>

with 21 VPUs (01, 02, 03N, 03S, 03W, 04-09, 10L, 10U, 11-18) and the national GageLoc under
``NHDPlusNationalData/``. Field-name case differs between tables (``ComID`` in the VAA and EROM,
``COMID`` in ElevSlope) and the DBF 10-character limit truncates ``TerminalPath`` to
``TerminalPa``; fields are therefore matched case-insensitively (:func:`readers.read_dbf`) and
returned under the protocol spelling listed here.

Owning phase: 5.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from src.data.readers import RawSchemaError, mask_sentinels, read_dbf
from src.data.schemas import SENTINELS

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

    from src.config import Settings
    from src.data.validate_raw import UnitReport

__all__ = [
    "NHD_TABLES",
    "VPUS",
    "NetworkInputs",
    "NhdTable",
    "find_table_files",
    "network_inputs",
    "read_nhd_table",
    "read_network",
    "validate_nhdplus",
]

#: The 21 NHDPlusV2.1 vector processing units of CONUS (protocol 1.1).
VPUS: Final[tuple[str, ...]] = (
    "01", "02", "03N", "03S", "03W", "04", "05", "06", "07", "08", "09",
    "10L", "10U", "11", "12", "13", "14", "15", "16", "17", "18",
)  # fmt: skip

_VPU_RE: Final[re.Pattern[str]] = re.compile(r"/NHDPlus(\d\d[A-Z]?)/")

#: Codes of flowlines outside the routed network as delivered (VAA ``Divergence`` = 9 together
#: with ``StreamOrde`` = -9; 23,585 flowlines, Batch C profile). They are not sentinels of a
#: value: unit_map sets ``StreamOrde`` to missing for them and keeps the row.
NON_NETWORK_DIVERGENCE: Final[int] = 9
NON_NETWORK_ORDER: Final[int] = -9


@dataclass(frozen=True)
class NhdTable:
    """One NHDPlusV2.1 attribute table.

    Attributes
    ----------
    filename
        DBF file name (matched case-insensitively).
    key
        Join key fields (protocol spelling).
    ints
        Integer fields (cast to int64), key included.
    floats
        Float fields (cast to float64; V4 sentinels masked).
    national
        ``True`` for a single national table (GageLoc), ``False`` for one table per VPU.
    strings
        String fields.
    """

    filename: str
    key: tuple[str, ...]
    ints: tuple[str, ...]
    floats: tuple[str, ...]
    national: bool = False
    strings: tuple[str, ...] = ()

    @property
    def fields(self) -> tuple[str, ...]:
        """All required fields in protocol spelling."""
        return (*self.ints, *self.floats, *self.strings)


#: Protocol 2.3 contracts. ``TerminalPa`` is the delivered (truncated) name of TerminalPath.
NHD_TABLES: Final[Mapping[str, NhdTable]] = {
    "vaa": NhdTable(
        "PlusFlowlineVAA.dbf",
        key=("ComID",),
        ints=(
            "ComID",
            "HydroSeq",
            "UpHydroSeq",
            "DnHydroSeq",
            "LevelPathI",
            "TerminalPa",
            "StartFlag",
            "TerminalFl",
            "Divergence",
            "StreamOrde",
        ),
        floats=("PathLength", "ArbolateSu", "TotDASqKm"),
    ),
    "erom": NhdTable(
        "EROM_MA0001.dbf", key=("ComID",), ints=("ComID",), floats=("Q0001E", "Q0001C", "V0001E")
    ),
    "elevslope": NhdTable(
        "elevslope.dbf", key=("ComID",), ints=("ComID",), floats=("Slope", "MinElevSmo")
    ),
    "plusflow": NhdTable(
        "PlusFlow.dbf", key=("FromComID", "ToComID"), ints=("FromComID", "ToComID"), floats=()
    ),
    "gageloc": NhdTable(
        "GageLoc.dbf",
        key=("Source_Fea",),
        ints=("FLComID",),
        floats=(),
        national=True,
        strings=("Source_Fea",),
    ),
}


def find_table_files(unit_dir: Path, table: NhdTable) -> dict[str, Path]:
    """Return ``{vpu: path}`` for a per-VPU table, or ``{"national": path}`` for GageLoc.

    Raises
    ------
    RawSchemaError
        If a VPU is missing or a table occurs twice for one VPU (V2).
    """
    hits = sorted(p for p in unit_dir.rglob("*") if p.name.lower() == table.filename.lower())
    if table.national:
        if len(hits) != 1:
            raise RawSchemaError(f"{table.filename}: expected one national file, got {len(hits)}")
        return {"national": hits[0]}
    found: dict[str, Path] = {}
    for path in hits:
        match = _VPU_RE.search(path.as_posix())
        if match is None:
            raise RawSchemaError(f"{path}: no NHDPlus<VPU> folder in the path")
        if match.group(1) in found:
            raise RawSchemaError(f"{table.filename}: VPU {match.group(1)} delivered twice")
        found[match.group(1)] = path
    missing = sorted(set(VPUS) - set(found))
    if missing:
        raise RawSchemaError(f"{table.filename}: VPUs missing {missing}")
    return {vpu: found[vpu] for vpu in VPUS}


def read_nhd_table(unit_dir: Path, name: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read one table over all VPUs with lossless casts and V4 sentinel masking.

    Returns the concatenated frame (``vpu`` column added; rows in VPU order, then file order)
    and the number of sentinel values masked per float field.

    Raises
    ------
    RawSchemaError
        On a missing field or a lossy cast (V2).
    """
    table = NHD_TABLES[name]
    parts: list[pd.DataFrame] = []
    for vpu, path in find_table_files(unit_dir, table).items():
        frame = read_dbf(path, table.fields)
        for col in table.ints:
            values = pd.to_numeric(frame[col], errors="raise")
            if values.isna().any() or (values != values.round()).any():
                raise RawSchemaError(f"{path.name} ({vpu}): {col} is not a complete integer")
            frame[col] = values.astype("int64")
        for col in table.floats:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")
        for col in table.strings:
            frame[col] = frame[col].astype("string").str.strip()
        frame.insert(0, "vpu", vpu)
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    masked: dict[str, int] = {}
    for col in table.floats:
        out[col], masked[col] = mask_sentinels(out[col], SENTINELS)
    return out, masked


def validate_nhdplus(unit_dir: Path, settings: Settings, report: UnitReport) -> None:
    """V2-V5 for the NHDPlusV2.1 tables used downstream (protocol 2.3)."""
    s_floor = settings.protocol.domain.s_floor
    frames: dict[str, pd.DataFrame] = {}
    info: dict[str, Any] = {}
    for name, table in NHD_TABLES.items():
        try:
            frame, masked = read_nhd_table(unit_dir, name)
        except (RawSchemaError, ValueError, TypeError) as exc:
            report.fail("V2", f"{name}: {exc}")
            continue
        frames[name] = frame
        entry: dict[str, Any] = {"rows": len(frame), "v4_sentinels": masked}
        dup = frame.duplicated(list(table.key), keep=False)
        if name in ("vaa", "erom", "elevslope") and dup.any():
            report.fail("V3", f"{name}: {table.key} not unique ({int(dup.sum())} rows)")
        entry["rows_in_duplicated_keys"] = int(dup.sum())
        entry["v5_missing"] = {c: int(frame[c].isna().sum()) for c in table.floats}
        info[name] = entry
    vaa_ids = set(frames["vaa"]["ComID"]) if "vaa" in frames else set()
    if "vaa" in frames:
        vaa = frames["vaa"]
        non_network = vaa["Divergence"].eq(NON_NETWORK_DIVERGENCE) & vaa["StreamOrde"].eq(
            NON_NETWORK_ORDER
        )
        info["vaa"]["v5_out_of_rule"] = {
            "non_network_div9_order-9": int(non_network.sum()),
            "divergence_not_0_1_2": int((~vaa["Divergence"].isin([0, 1, 2]) & ~non_network).sum()),
            "streamorde_le_0": int((vaa["StreamOrde"].le(0) & ~non_network).sum()),
            "startflag_not_0_1": int((~vaa["StartFlag"].isin([0, 1])).sum()),
            "terminalfl_not_0_1": int((~vaa["TerminalFl"].isin([0, 1])).sum()),
            "totdasqkm_le_0": int(vaa["TotDASqKm"].le(0).sum()),
        }
        info["vaa"]["rows_per_vpu"] = {k: int(v) for k, v in vaa["vpu"].value_counts().items()}
    if "erom" in frames:
        erom = frames["erom"]
        info["erom"]["v5_out_of_rule"] = {
            "q0001e_le_0_or_missing": int((erom["Q0001E"].le(0) | erom["Q0001E"].isna()).sum()),
            "v0001e_le_0_or_missing": int((erom["V0001E"].le(0) | erom["V0001E"].isna()).sum()),
            "comid_not_in_vaa": int((~erom["ComID"].isin(vaa_ids)).sum()),
        }
    if "elevslope" in frames:
        slope = frames["elevslope"]["Slope"]
        info["elevslope"]["v5_out_of_rule"] = {
            "slope_below_floor": int(slope.lt(s_floor).sum()),
            "vaa_comid_without_slope_row": len(vaa_ids - set(frames["elevslope"]["ComID"])),
        }
    if "gageloc" in frames:
        gage = frames["gageloc"]
        info["gageloc"]["v5_out_of_rule"] = {
            "flcomid_not_in_vaa": int((~gage["FLComID"].isin(vaa_ids)).sum()),
            "source_fea_not_digits": int((~gage["Source_Fea"].str.fullmatch(r"\d+")).sum()),
        }
    if "plusflow" in frames:
        flow = frames["plusflow"]
        info["plusflow"]["v5_out_of_rule"] = {
            "from_zero": int(flow["FromComID"].eq(0).sum()),
            "to_zero": int(flow["ToComID"].eq(0).sum()),
            "from_nonzero_not_in_vaa": int(
                (flow["FromComID"].ne(0) & ~flow["FromComID"].isin(vaa_ids)).sum()
            ),
            "to_nonzero_not_in_vaa": int(
                (flow["ToComID"].ne(0) & ~flow["ToComID"].isin(vaa_ids)).sum()
            ),
        }
    report.info["tables"] = info


# ----------------------------------------------------------------------------- network (3.6.5)
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


def network_inputs(vaa: pd.DataFrame, plusflow: pd.DataFrame, comids: Iterable[int]) -> NetworkInputs:
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
    vaa, _ = read_nhd_table(unit_dir, "vaa")
    plusflow, _ = read_nhd_table(unit_dir, "plusflow")
    return network_inputs(vaa, plusflow, comids)


# ----------------------------------------------------------------------------- geometry layers
#: Equal-area CRS for every distance and area computation (protocol 2.5: EPSG:5070).
CRS_ALBERS: Final[str] = "EPSG:5070"
_LAYERS: Final[Mapping[str, tuple[str, str]]] = {
    "flowline": ("NHDFlowline.shp", "COMID"),
    "catchment": ("Catchment.shp", "FEATUREID"),
}


def read_layer(unit_dir: Path, layer: str, vpu: str, crs: str = CRS_ALBERS) -> GeoDataFrame:
    """Read one VPU's flowlines or catchments as ``GeoDataFrame[comid, geometry]`` in ``crs``.

    Flowlines are delivered as measured 3-D lines; M and Z are dropped (2-D geometry only).
    The key field is matched case-insensitively (``COMID`` or ``ComID`` by VPU).
    """
    import warnings

    import pyogrio
    import shapely

    from src.data.readers import resolve_columns

    filename, key = _LAYERS[layer]
    table = NhdTable(filename.replace(".shp", ".dbf"), key=(key,), ints=(key,), floats=())
    path = find_table_files(unit_dir, table)[vpu].with_suffix(".shp")
    with warnings.catch_warnings():
        # GDAL reports that measured (M) geometries are read without M; expected and harmless.
        warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")
        fields = [str(f) for f in pyogrio.read_info(path)["fields"]]
        column = resolve_columns(fields, [key])[key]
        gdf = pyogrio.read_dataframe(path, columns=[column])
    gdf = gdf.rename(columns={column: "comid"})
    gdf["comid"] = gdf["comid"].astype("int64")
    gdf["geometry"] = shapely.force_2d(gdf.geometry.to_numpy())
    out: GeoDataFrame = gdf.to_crs(crs)
    return out
