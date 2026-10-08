"""Pandera DataFrame models and raw-field contracts for the protocol schemas (P4-D6 (a)).

Sources: protocol section 2.3 (hydroswot_raw, raw auxiliary units, site_table, ehydro_ind),
section 3.3 (fold_assignment) and section 2.4 (QC flags). Field names are those **asserted on the
delivered files** (validate_raw, V2). Where the delivered name differs from the protocol or the
publisher's field list, the delivered name is used *explicitly* and the deviation is recorded by
an ADL amendment (protocol 2.3: "never silently remapped"):

* HYDROSWOT ``xsec_area_va`` (protocol and ``FieldDefinitions.txt``: ``xsecond_area_va``),
  likewise ``q2_xsec_area_X_mean_vel_va`` and ``q4_xsec_area_D_stream_width``.

Conventions: missing numeric values are ``NaN`` in ``float64`` columns; missing codes are
``<NA>`` in string columns; integer columns that may be missing use pandas nullable ``Int``
dtypes. Sentinels have already been turned into missing values (V4) in every model below.

Owning phase: 5 (this batch: HYDROSWOT path; site_table, fold_assignment and ehydro_ind are
added with the stages that write them).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series

__all__ = [
    "BANNED_PREDICTORS",
    "HYDROSWOT_CODE_FIELDS",
    "HYDROSWOT_FLAG_FIELDS",
    "HYDROSWOT_NUMERIC_FIELDS",
    "HYDROSWOT_REQUIRED",
    "SENTINELS",
    "assert_no_banned",
    "HydroswotInit",
    "HydroswotSI",
    "NhdElevSlopeSI",
    "NhdEromSI",
    "NhdGageLocSI",
    "NhdPlusFlowSI",
    "NhdVaaSI",
]

#: Numeric sentinel codes that become missing before any value check or unit map (V4; 2.7).
SENTINELS: Final[tuple[float, ...]] = (-9999.0, -9998.0)

#: HYDROSWOT numeric fields as delivered (US customary units; converted by unit_map, 2.1).
HYDROSWOT_NUMERIC_FIELDS: Final[tuple[str, ...]] = (
    "dec_lat_va",
    "dec_long_va",
    "drain_area_va",
    "q_meas_chan_nu",
    "q_va",
    "meas_q_va",
    "q_adj_va",
    "stream_wdth_va",
    "xsec_area_va",  # delivered name; protocol/FieldDefinitions: xsecond_area_va (ADL)
    "mean_depth_va",
    "max_depth_va",
    "mean_vel_va",
    "max_vel_va",
)
#: HYDROSWOT code fields (strings; quality flags and diagnostic-only descriptors).
HYDROSWOT_CODE_FIELDS: Final[tuple[str, ...]] = (
    "coord_datum_cd",
    "site_tp_cd",
    "q_meas_td",
    "q_meas_qual_cd",
    "q_meas_used_fg",
    "q_meas_type_cd",
    "data_aging_cd",
    "chan_mat_cd",  # diagnostic only (H4 (d)); never a predictor (L4)
    "chan_stability_cd",  # diagnostic only
    "base_flw_cd",  # diagnostic only
)
#: Built-in HYDROSWOT checks (0 = pass, -1 = fail, empty = not computable).
HYDROSWOT_FLAG_FIELDS: Final[tuple[str, ...]] = (
    "q1_Outside5Percent",
    "q2_q_va_OutsideBounds",
    "q2_meas_q_va_OutsideBounds",
    "q4_OutsideBounds",
    "q5_Mean_GT_Max",
    "q6_mean_vel_GT_max_vel",
)
#: Every field validate_raw requires in SWOT_ADCP_Dataset.txt (V2).
HYDROSWOT_REQUIRED: Final[tuple[str, ...]] = (
    "site_no",
    "q_meas_no",
    "q_meas_dt",
    *HYDROSWOT_NUMERIC_FIELDS,
    *HYDROSWOT_CODE_FIELDS,
    *HYDROSWOT_FLAG_FIELDS,
)

#: L4 (protocol 3.5): columns that are never predictors. Measured discharge, the site-reported
#: drainage area, HYDROSWOT velocity and channel descriptors, the geometry targets and their
#: transforms, and ``V_E``/``A_J`` (used only inside the Jarrett relation, ADL-005).
BANNED_PREDICTORS: Final[frozenset[str]] = frozenset(
    {
        "Q_obs", "Q_meas", "Q_adj", "DA_site_km2", "V", "Vmax", "chan_mat_cd",
        "chan_stability_cd", "base_flw_cd", "W", "A", "dbar", "dmax", "t_obs", "eps_A", "V_E",
        "A_J", "q_va", "meas_q_va", "q_adj_va", "drain_area_va", "mean_vel_va", "max_vel_va",
        "stream_wdth_va", "xsec_area_va", "mean_depth_va", "max_depth_va",
    }
)  # fmt: skip


def assert_no_banned(columns: Iterable[str]) -> None:
    """Raise ``ValueError`` if any column is a banned predictor (L4)."""
    bad = sorted(set(columns) & BANNED_PREDICTORS)
    if bad:
        raise ValueError(f"banned predictors (L4): {bad}")


_SITE_NO = r"^\d{8,15}$"


class HydroswotSI(pa.DataFrameModel):
    """``unit_mapped/hydroswot.parquet``: every delivered row, sentinels missing, SI units (2.1).

    No row is removed here; QC (2.4) happens in the next stage.
    """

    row_id: Series[int] = pa.Field(ge=0, unique=True, description="0-based row in the file")
    site_no: Series[str] = pa.Field(str_matches=_SITE_NO)
    q_meas_no: Series[str] = pa.Field(str_length={"min_value": 1})
    q_meas_dt: Series[pd.Timestamp] = pa.Field(description="local time as delivered")
    site_tp_cd: Series[str] = pa.Field(nullable=True)
    coord_datum_cd: Series[str] = pa.Field(nullable=True)
    lat: Series[float] = pa.Field(ge=-90, le=90, nullable=True, description="NAD83, EPSG:4269")
    lon: Series[float] = pa.Field(ge=-180, le=180, nullable=True, description="NAD83")
    DA_site_km2: Series[float] = pa.Field(
        nullable=True, description="drain_area_va in km2; link QC only, never a predictor"
    )
    q_meas_chan_nu: Series[pd.Int16Dtype] = pa.Field(nullable=True)
    Q_obs: Series[float] = pa.Field(nullable=True, description="q_va, m3/s; never a predictor")
    Q_meas: Series[float] = pa.Field(nullable=True, description="meas_q_va, m3/s")
    Q_adj: Series[float] = pa.Field(nullable=True, description="q_adj_va, m3/s")
    W: Series[float] = pa.Field(nullable=True, description="m")
    A: Series[float] = pa.Field(nullable=True, description="m2")
    dbar: Series[float] = pa.Field(nullable=True, description="m")
    dmax: Series[float] = pa.Field(nullable=True, description="m")
    V: Series[float] = pa.Field(nullable=True, description="m/s")
    Vmax: Series[float] = pa.Field(nullable=True, description="m/s; QC only")
    q_meas_td: Series[str] = pa.Field(nullable=True)
    q_meas_qual_cd: Series[str] = pa.Field(nullable=True)
    q_meas_used_fg: Series[str] = pa.Field(nullable=True)
    q_meas_type_cd: Series[str] = pa.Field(nullable=True)
    data_aging_cd: Series[str] = pa.Field(nullable=True)
    chan_mat_cd: Series[str] = pa.Field(nullable=True)
    chan_stability_cd: Series[str] = pa.Field(nullable=True)
    base_flw_cd: Series[str] = pa.Field(nullable=True)
    # Delivered HYDROSWOT flag names (mixed case is the publisher's, kept verbatim).
    q1_Outside5Percent: Series[pd.Int8Dtype] = pa.Field(isin=[0, -1], nullable=True)  # noqa: N815
    q2_q_va_OutsideBounds: Series[pd.Int8Dtype] = pa.Field(  # noqa: N815
        isin=[0, -1], nullable=True
    )
    q2_meas_q_va_OutsideBounds: Series[pd.Int8Dtype] = pa.Field(  # noqa: N815
        isin=[0, -1], nullable=True
    )
    q4_OutsideBounds: Series[pd.Int8Dtype] = pa.Field(isin=[0, -1], nullable=True)  # noqa: N815
    q5_Mean_GT_Max: Series[pd.Int8Dtype] = pa.Field(isin=[0, -1], nullable=True)  # noqa: N815
    q6_mean_vel_GT_max_vel: Series[pd.Int8Dtype] = pa.Field(  # noqa: N815
        isin=[0, -1], nullable=True
    )

    class Config:
        """Closed schema: no extra columns, no silent coercion."""

        strict = True
        coerce = False


class HydroswotInit(HydroswotSI):
    """``qc/hydroswot_init.parquet``: rows surviving R1-R7 and R11, R8-R10 flags kept (2.4).

    Primary analyses drop ``F_eps`` and ``F_t`` rows; SA3 keeps them (P5-D6).
    """

    division: Series[str] = pa.Field(description="F&J division containing the site (R2)")
    W: Series[float] = pa.Field(gt=0.0)
    A: Series[float] = pa.Field(gt=0.0)
    dbar: Series[float] = pa.Field(gt=0.0)
    dmax: Series[float] = pa.Field(gt=0.0)
    V: Series[float] = pa.Field(gt=0.0)
    Q_obs: Series[float] = pa.Field(gt=0.0)
    eps_A: Series[float] = pa.Field(description="ln(A / (W dbar)), R8")  # noqa: N815 (protocol symbol)
    t_obs: Series[float] = pa.Field(ge=0, description="ln(dmax / dbar)")
    F_eps: Series[bool] = pa.Field(description="R8: |eps_A| > ln(1 + epsA_tol)")
    F_t: Series[bool] = pa.Field(description="R9: dmax == dbar")
    F_qual: Series[bool] = pa.Field(description="R10: POOR, not used, or not approved")
    F_tout: Series[bool] = pa.Field(description="t_obs outside the shape-target band (2.4)")

    @pa.dataframe_check
    @classmethod
    def unique_measurement(cls, df: pd.DataFrame) -> bool:
        """R11: (site_no, q_meas_no) is unique after QC."""
        return not bool(df.duplicated(["site_no", "q_meas_no"]).any())


# ----------------------------------------------------------------------------- NHDPlusV2.1 (SI)
class _Closed(pa.DataFrameModel):
    class Config:
        """Closed schema: no extra columns, no silent coercion."""

        strict = True
        coerce = False


class NhdVaaSI(_Closed):
    """``unit_mapped/nhd_vaa.parquet``: PlusFlowlineVAA, all VPUs (2.3; units km, km2)."""

    comid: Series[int] = pa.Field(unique=True)
    vpu: Series[str]
    hydroseq: Series[int]
    up_hydroseq: Series[int]
    dn_hydroseq: Series[int]
    levelpath: Series[int]
    terminalpath: Series[int]
    startflag: Series[int] = pa.Field(isin=[0, 1])
    terminalfl: Series[int] = pa.Field(isin=[0, 1])
    divergence: Series[int] = pa.Field(isin=[0, 1, 2, 9])
    non_network: Series[bool] = pa.Field(description="Divergence 9 and StreamOrde -9")
    stream_order: Series[pd.Int8Dtype] = pa.Field(ge=1, nullable=True)
    pathlength_km: Series[float] = pa.Field(nullable=True)
    arbolate_km: Series[float] = pa.Field(nullable=True)
    DA_km2: Series[float] = pa.Field(ge=0, description="TotDASqKm")


class NhdEromSI(_Closed):
    """``unit_mapped/nhd_erom.parquet``: EROM mean annual flow (Q in m3/s, V in m/s)."""

    comid: Series[int] = pa.Field(unique=True)
    Q_E: Series[float] = pa.Field(nullable=True, description="Q0001E x 0.028316846592")
    Q_C: Series[float] = pa.Field(nullable=True, description="Q0001C x 0.028316846592 (SA1)")
    V_E: Series[float] = pa.Field(nullable=True, description="V0001E x 0.3048; A_J only")


class NhdElevSlopeSI(_Closed):
    """``unit_mapped/nhd_elevslope.parquet``: smoothed slope and minimum elevation."""

    comid: Series[int] = pa.Field(unique=True)
    S: Series[float] = pa.Field(nullable=True, description="Slope (m/m); -9998 -> missing (R12)")
    z_min_m: Series[float] = pa.Field(nullable=True, description="MinElevSmo (cm) x 0.01")
    F_S: Series[bool] = pa.Field(description="S equals the data slope floor (P3-03)")


class NhdGageLocSI(_Closed):
    """``unit_mapped/nhd_gageloc.parquet``: gage-to-flowline events (national)."""

    site_no: Series[str] = pa.Field(str_matches=_SITE_NO, unique=True)
    comid: Series[int]


class NhdPlusFlowSI(_Closed):
    """``unit_mapped/nhd_plusflow.parquet``: flow-table edges; 0 marks a network end."""

    from_comid: Series[int]
    to_comid: Series[int]
