"""Figure 1: Study domain, M5 physics-informed architecture, and multi-regime validation design.

Implements ``Fig. 1`` from ``.agents/prompts/PHASE_09.md`` §1 (P9-D4 (a)):
- Panel (a) [Descriptive]: CONUS map of 3,988 HYDROSWOT sites coloured by Fenneman & Johnson
  physiographic division, the 8 S-LRO held-out divisions, and the 131 eHydro S-IND locations
   across 18 rivers (via NHDPlusV2.1 outlet COMID geometry in EPSG:5070).
- Panel (b) [Descriptive]: M5 schematic: shared backbone MLP (3x128, SiLU, LayerNorm) ->
  W, r, n heads (n-head under I_blind: no Q_E, S, F_S, V_E) -> differentiable Phi layer
  (Manning + continuity, Picard solve <= tau_l, full kappa) -> dbar, dmax, A = W*dbar, V = Q_E/A.
- Panel (c) [Descriptive]: Validation design (S-ID 3x5 grouped CV with Event 1 dependence
  audit, S-LRO 8 divisions, S-TAIL-DA, S-TAIL-SO, S-IND 131 locations / 18 rivers, S-RND
  secondary) and the verbatim §4.7 row 4 decision outcome.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import shapely
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    S_IND_UQ_NOTE,
    VerifiedBundle,
)
from annxs_figures.style import (
    ASCE_DOUBLE_COL_IN,
    DIVISION_PALETTE,
    DIVISION_SHORT,
    OKABE_ITO,
    AnnotationRegistry,
    apply_asce_style,
    label_panel,
    save_figure_bundle,
)
from annxs_figures.tables import TIER_DESCRIPTIVE

CRS_ALBERS: Final[str] = "EPSG:5070"
_VPU_RE: Final[re.Pattern[str]] = re.compile(r"/NHDPlus(\d\d[A-Z]?)/")


def _load_physio_divisions(repo_root: Path) -> gpd.GeoDataFrame:
    """Load and dissolve F&J physiographic division polygons in EPSG:5070."""
    physio_dir = repo_root / "data/raw/physiography"
    shp = next(iter(sorted(physio_dir.rglob("physio.shp"))))
    gdf = gpd.read_file(shp)[["DIVISION", "geometry"]]
    gdf["DIVISION"] = gdf["DIVISION"].astype("string").str.strip().replace("", pd.NA)
    gdf["geometry"] = shapely.make_valid(gdf.geometry.to_numpy())
    gdf = gdf.dropna(subset=["DIVISION"]).copy()
    gdf["DIVISION"] = gdf["DIVISION"].astype(str)
    gdf_5070 = gdf.to_crs(CRS_ALBERS)
    dissolved: gpd.GeoDataFrame = gdf_5070.dissolve(by="DIVISION").reset_index()
    return dissolved


def _load_sind_outlet_points(repo_root: Path, ehydro_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve EPSG:5070 centroids of the 131 kept eHydro S-IND outlet COMIDs from NHDPlusV2.1."""
    kept = ehydro_df.loc[ehydro_df["dedupe_status"].eq("kept")].copy()
    nhd_dir = repo_root / "data/raw/nhdplusv21"
    shp_by_vpu: dict[str, Path] = {}
    for p in sorted(nhd_dir.rglob("*")):
        if p.name.lower() == "nhdflowline.shp":
            m = _VPU_RE.search(p.as_posix())
            if m is not None:
                shp_by_vpu[m.group(1)] = p

    # Map each outlet COMID to candidate VPUs listed on its row
    vpu_to_comids: dict[str, set[int]] = {}
    for _, r in kept.iterrows():
        c_out = int(r["comid_outlet"])
        for vpu in str(r["vpus"]).split(";"):
            vpu_to_comids.setdefault(vpu.strip(), set()).add(c_out)

    coords_by_comid: dict[int, tuple[float, float]] = {}
    for vpu, comid_set in sorted(vpu_to_comids.items()):
        shp_path = shp_by_vpu[vpu]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types")
            fields = [str(f) for f in pyogrio.read_info(shp_path)["fields"]]
            comid_col = next(f for f in fields if f.upper() == "COMID")
            id_list = ",".join(str(c) for c in sorted(comid_set))
            sub_gdf = pyogrio.read_dataframe(
                shp_path,
                columns=[comid_col],
                where=f"{comid_col} IN ({id_list})",
            )
        if sub_gdf.empty:
            continue
        sub_gdf = sub_gdf.rename(columns={comid_col: "comid"})
        sub_gdf["comid"] = sub_gdf["comid"].astype("int64")
        sub_gdf["geometry"] = shapely.force_2d(sub_gdf.geometry.to_numpy())
        sub_5070 = sub_gdf.to_crs(CRS_ALBERS)
        centroids = shapely.centroid(sub_5070.geometry.to_numpy())
        xs = shapely.get_x(centroids)
        ys = shapely.get_y(centroids)
        for cid, x_val, y_val in zip(sub_5070["comid"].to_numpy(), xs, ys, strict=True):
            coords_by_comid[int(cid)] = (float(x_val), float(y_val))

    out_rows: list[dict[str, Any]] = []
    for _, r in kept.iterrows():
        c_out = int(r["comid_outlet"])
        if c_out not in coords_by_comid:
            raise ValueError(f"Outlet COMID {c_out} for S-IND {r['loc_id']} not found in NHDPlus")
        x5070, y5070 = coords_by_comid[c_out]
        out_rows.append(
            {
                "loc_id": str(r["loc_id"]),
                "river": str(r["river"]),
                "comid_outlet": c_out,
                "x5070": x5070,
                "y5070": y5070,
            }
        )
    return pd.DataFrame(out_rows)


def _draw_panel_a_map(
    ax: Axes,
    bundle: VerifiedBundle,
    repo_root: Path,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (a): CONUS study domain, 8 F&J S-LRO divisions, and 131 S-IND locations."""
    physio = _load_physio_divisions(repo_root)
    site_table = bundle.load_frozen_table("data/processed/site_table.parquet")
    ehydro_ind = bundle.load_frozen_table("data/processed/ind/ehydro_ind.parquet")
    primary_sites = site_table.loc[
        site_table["variant"].eq("primary") & ~site_table["F_DA"].astype(bool)
    ].copy()
    sind_pts = _load_sind_outlet_points(repo_root, ehydro_ind)

    # Get exact per-division site counts from event3_matrix.parquet
    mat = bundle.matrix_df
    div_rows = mat.loc[
        (mat["spec"] == "M5")
        & (mat["analysis"] == "primary")
        & (mat["family"] == "S-LRO")
        & (mat["target"] == "dbar")
        & (mat["stratum_kind"] == "division")
        & (mat["metric"] == "rmse_log")
    ].sort_values("stratum")
    div_counts: dict[str, int] = {}
    for _, r in div_rows.iterrows():
        d_name = str(r["stratum"])
        cnt = int(r["n_units"])
        div_counts[d_name] = cnt
        registry.register(
            figure_id="fig1",
            panel_id="a",
            label=f"division_sites:{d_name}",
            raw_value=cnt,
            fmt_kind="int",
            source_file="reports/results/event3_matrix.parquet",
            source_key=f"M5|primary|S-LRO|dbar|division:{d_name}|n_units",
            evidence_tier=TIER_DESCRIPTIVE,
        )

    n_total_sites = int(bundle.audit["n_sites"])
    n_sites_str = registry.register(
        figure_id="fig1",
        panel_id="a",
        label="n_sites_total",
        raw_value=n_total_sites,
        fmt_kind="int",
        source_file="reports/results/audit.json",
        source_key="n_sites",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    n_sind_locs = int(bundle.hypotheses["H1"]["S-IND"]["n_locations"])
    n_sind_rivers = int(bundle.hypotheses["H1"]["S-IND"]["n_clusters"])
    n_sind_str = registry.register(
        figure_id="fig1",
        panel_id="a",
        label="n_sind_locations",
        raw_value=n_sind_locs,
        fmt_kind="int",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H1.S-IND.n_locations",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    n_rivers_str = registry.register(
        figure_id="fig1",
        panel_id="a",
        label="n_sind_rivers",
        raw_value=n_sind_rivers,
        fmt_kind="int",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H1.S-IND.n_clusters",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    # Plot background F&J division polygons
    for _, prow in physio.iterrows():
        div_name = str(prow["DIVISION"])
        col = DIVISION_PALETTE.get(div_name, "#DDDDDD")
        gpd.GeoSeries([prow["geometry"]], crs=CRS_ALBERS).plot(
            ax=ax,
            facecolor=col,
            edgecolor="#444444",
            linewidth=0.35,
            alpha=0.22,
            zorder=1,
        )

    # Plot 3,988 HYDROSWOT sites by division
    legend_handles: list[Line2D] = []
    for div_name, col in DIVISION_PALETTE.items():
        sub = primary_sites.loc[primary_sites["division"].eq(div_name)]
        cnt = div_counts.get(div_name, len(sub))
        ax.scatter(
            sub["x5070"].to_numpy(dtype=np.float64),
            sub["y5070"].to_numpy(dtype=np.float64),
            s=3.5,
            c=col,
            alpha=0.80,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )
        short_lbl = f"{DIVISION_SHORT[div_name]} (n={cnt})"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=col,
                markeredgecolor="#222222",
                markeredgewidth=0.3,
                markersize=4.5,
                label=short_lbl,
            )
        )

    # Plot 131 eHydro S-IND locations
    ax.scatter(
        sind_pts["x5070"].to_numpy(dtype=np.float64),
        sind_pts["y5070"].to_numpy(dtype=np.float64),
        s=22.0,
        marker="^",
        c="#000000",
        edgecolors="#FFFFFF",
        linewidths=0.45,
        zorder=5,
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#000000",
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.4,
            markersize=5.5,
            label=f"S-IND eHydro ({n_sind_str} locs / {n_rivers_str} rivers)",
        )
    )

    ax.set_xlim(-2_500_000.0, 2_400_000.0)
    ax.set_ylim(-550_000.0, 3_280_000.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        framealpha=0.94,
        fontsize=5.8,
        handletextpad=0.3,
        columnspacing=0.8,
        borderpad=0.30,
    )

    label_panel(
        ax,
        "a",
        f"CONUS Study Domain: n = {n_sites_str} HYDROSWOT Sites (8 F&J S-LRO Divisions) & S-IND",
        TIER_DESCRIPTIVE,
        subtitle=f"S-IND: {S_IND_PANEL_NOTE} (EPSG:5070 Albers Equal Area)",
    )


def _draw_box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "#F8F9FA",
    ec: str = "#333333",
    fontsize: float = 5.7,
    fontweight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.75,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color="#111111",
        zorder=3,
    )


def _draw_arrow(
    ax: Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = "#333333",
    linestyle: str = "-",
) -> None:
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>,head_length=3.5,head_width=2.2",
        color=color,
        linewidth=0.8,
        linestyle=linestyle,
        zorder=4,
    )
    ax.add_patch(arr)


def _draw_panel_b_architecture(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (b): M5 physics-informed neural network schematic with I_blind and Phi."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    m5_cons = bundle.consistency["models"]["M5"]["S-ID"]
    solver_res_str = registry.register(
        figure_id="fig1",
        panel_id="b",
        label="M5_S-ID_solver_residual_max",
        raw_value=float(m5_cons["solver_residual_max"]),
        fmt_kind="sci2",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M5.S-ID.solver_residual_max",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    picard_iters_str = registry.register(
        figure_id="fig1",
        panel_id="b",
        label="M5_S-ID_picard_iterations_max",
        raw_value=int(m5_cons["picard_iterations_max"]),
        fmt_kind="int",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M5.S-ID.picard_iterations_max",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    # Column 1: Inputs & I_blind
    _draw_box(
        ax,
        0.01,
        0.52,
        0.20,
        0.41,
        (
            "Catchment & Reach\nPredictors x_i\n"
            "(DA, SO, z_min, AI,\nD50, StreamCat)\n[Split-then-Scale]"
        ),
        fc="#E8F0FE",
        ec=OKABE_ITO["blue"],
    )
    _draw_box(
        ax,
        0.01,
        0.05,
        0.20,
        0.41,
        (
            "Physics Drivers\nQ_E (EROM), S, F_S\n---\n"
            "I_blind Firewall:\nBlocked from n-head\n(no Q_E, S, F_S, V_E)"
        ),
        fc="#FCE8E6",
        ec=OKABE_ITO["vermilion"],
    )

    # Column 2: Shared Backbone & Latent Heads
    _draw_box(
        ax,
        0.26,
        0.52,
        0.20,
        0.41,
        (
            "Shared MLP Backbone\n3 x 128 SiLU\nLayerNorm\n"
            "M = 10 Ensemble\n(Bagging + Distinct\nInit & ES Split)"
        ),
        fc="#F1F3F4",
        ec="#444444",
    )
    _draw_box(
        ax,
        0.26,
        0.05,
        0.20,
        0.41,
        (
            "I_blind n-Head\nLinear(d_blind -> 2)\n"
            "z_n -> Bounded Sigmoid\nn_hat in [0.005, 0.60]\n(nu_hat = ln n_hat)"
        ),
        fc="#FEF7E0",
        ec=OKABE_ITO["orange"],
    )

    # Column 3: Latent Geometry Heads + Differentiable Phi Layer
    _draw_box(
        ax,
        0.51,
        0.55,
        0.22,
        0.38,
        (
            "Geometry & Shape Heads\nW_hat = exp(mu_W)\n"
            "t_hat = ln(d_max/dbar)\nr_hat = 1/(exp(t_hat)-1)\nr_hat in [0.50, 10.0]"
        ),
        fc="#E6F4EA",
        ec=OKABE_ITO["bluish_green"],
    )
    _draw_box(
        ax,
        0.51,
        0.05,
        0.22,
        0.44,
        (
            "Differentiable Layer Phi\nManning + Continuity:\n"
            "Q_E = A_hat * V_hat\nFull Dingman kappa(eta, r_hat)\n"
            f"Picard/Newton <= 1e-10\n(max res={solver_res_str}, {picard_iters_str} iter)"
        ),
        fc="#FCE8E6",
        ec=OKABE_ITO["vermilion"],
    )

    # Column 4: Hydraulically Consistent Outputs
    _draw_box(
        ax,
        0.78,
        0.07,
        0.21,
        0.85,
        (
            "Consistent Outputs\n---\ndbar_hat = Phi(W, r, nu; Q_E, S)\n"
            "W_hat = exp(mu_W)\nA_hat = W_hat * dbar_hat\n(eps_A = 0 exact)\n"
            "d_max = dbar_hat * exp(t_hat)\n(d_max > dbar_hat exact)\n"
            "V_hat = Q_E / A_hat\n90% UQ Intervals"
        ),
        fc="#F3E8FD",
        ec=OKABE_ITO["reddish_purple"],
    )

    # Arrows
    _draw_arrow(ax, 0.21, 0.72, 0.26, 0.72, color=OKABE_ITO["blue"])
    _draw_arrow(ax, 0.21, 0.58, 0.26, 0.28, color=OKABE_ITO["blue"])
    _draw_arrow(ax, 0.21, 0.22, 0.51, 0.22, color=OKABE_ITO["vermilion"], linestyle="--")
    _draw_arrow(ax, 0.46, 0.74, 0.51, 0.74, color="#333333")
    _draw_arrow(ax, 0.46, 0.26, 0.51, 0.26, color=OKABE_ITO["orange"])
    _draw_arrow(ax, 0.62, 0.55, 0.62, 0.49, color=OKABE_ITO["bluish_green"])
    _draw_arrow(ax, 0.73, 0.74, 0.78, 0.66, color=OKABE_ITO["bluish_green"])
    _draw_arrow(ax, 0.73, 0.28, 0.78, 0.36, color=OKABE_ITO["vermilion"])

    label_panel(
        ax,
        "b",
        "M5 Hard-Constrained Physics-Informed Architecture (I_blind + Differentiable Layer Phi)",
        TIER_DESCRIPTIVE,
        subtitle="docs/3_experimental_protocol.md §4.1-§4.2; docs/2_mathematical_proofs.md §5-§7",
    )


def _draw_panel_c_validation(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (c): Multi-regime validation design and verbatim §4.7 row 4 outcome."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    audit = bundle.audit
    rho_str = registry.register(
        figure_id="fig1",
        panel_id="c",
        label="audit_rho_hat_km",
        raw_value=float(audit["rho_hat_km"]),
        fmt_kind="dec2",
        source_file="reports/results/audit.json",
        source_key="rho_hat_km",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    dnn_str = registry.register(
        figure_id="fig1",
        panel_id="c",
        label="audit_d_nn_median_km",
        raw_value=float(audit["d_nn_median_km"]),
        fmt_kind="dec2",
        source_file="reports/results/audit.json",
        source_key="d_nn_median_km",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    strong_str = registry.register(
        figure_id="fig1",
        panel_id="c",
        label="audit_share_strong",
        raw_value=float(audit["nested"]["share_strong"]),
        fmt_kind="dec4",
        source_file="reports/results/audit.json",
        source_key="nested.share_strong",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    weak_str = registry.register(
        figure_id="fig1",
        panel_id="c",
        label="audit_share_weak",
        raw_value=float(audit["nested"]["share_weak"]),
        fmt_kind="dec4",
        source_file="reports/results/audit.json",
        source_key="nested.share_weak",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    mat = bundle.matrix_df
    tail_da_n = int(
        mat.loc[
            (mat["spec"] == "M5")
            & (mat["analysis"] == "primary")
            & (mat["family"] == "S-TAIL-DA")
            & (mat["target"] == "dbar")
            & (mat["stratum_kind"] == "all")
            & (mat["metric"] == "rmse_log"),
            "n_units",
        ].iloc[0]
    )
    tail_so_n = int(
        mat.loc[
            (mat["spec"] == "M5")
            & (mat["analysis"] == "primary")
            & (mat["family"] == "S-TAIL-SO")
            & (mat["target"] == "dbar")
            & (mat["stratum_kind"] == "all")
            & (mat["metric"] == "rmse_log"),
            "n_units",
        ].iloc[0]
    )
    srnd_rows_n = int(bundle.descriptive["S_RND"]["rows"]["M2-XGB"])

    tail_da_str = registry.register(
        figure_id="fig1",
        panel_id="c",
        label="S-TAIL-DA_n_units",
        raw_value=tail_da_n,
        fmt_kind="int",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M5|primary|S-TAIL-DA|dbar|all|n_units",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    tail_so_str = registry.register(
        figure_id="fig1",
        panel_id="c",
        label="S-TAIL-SO_n_units",
        raw_value=tail_so_n,
        fmt_kind="int",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M5|primary|S-TAIL-SO|dbar|all|n_units",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    srnd_str = registry.register(
        figure_id="fig1",
        panel_id="c",
        label="S-RND_n_rows",
        raw_value=srnd_rows_n,
        fmt_kind="int",
        source_file="reports/results/event3_descriptive.json",
        source_key="S_RND.rows.M2-XGB",
        evidence_tier="Secondary",
    )

    # Top row of 3 boxes: S-ID, S-LRO, S-TAIL + S-IND + S-RND
    _draw_box(
        ax,
        0.01,
        0.46,
        0.31,
        0.47,
        f"S-ID (3x5 Grouped CV; n = 3,988 sites)\n"
        f"Event 1 Audit: d_nn,med = {dnn_str} km\n"
        f"rho_hat = rho* = {rho_str} km\n"
        f"Nested: strong={strong_str}, weak={weak_str}\n"
        f'Label: "{EVENT1_AUDIT_LABEL}"',
        fc="#E8F0FE",
        ec=OKABE_ITO["blue"],
        fontsize=5.8,
    )
    _draw_box(
        ax,
        0.345,
        0.46,
        0.31,
        0.47,
        "S-LRO (Leave-One-Region-Out OOD)\n"
        "8 Held-Out F&J Divisions (n = 3,988)\n"
        "Primary Spatial OOD Benchmark (H1, H3)\n"
        f"S-TAIL-DA: n = {tail_da_str} upper-DA sites\n"
        f"S-TAIL-SO: n = {tail_so_str} high-SO sites",
        fc="#FCE8E6",
        ec=OKABE_ITO["vermilion"],
        fontsize=5.8,
    )
    _draw_box(
        ax,
        0.68,
        0.46,
        0.31,
        0.47,
        f"S-IND (Independent USACE eHydro)\n"
        f"{S_IND_PANEL_NOTE}\n"
        f"({S_IND_UQ_NOTE})\n"
        "---\n"
        f"S-RND [Secondary]: n = {srnd_str} rows\n"
        "(1/|I_j| weights across 3,988 sites)",
        fc="#FEF7E0",
        ec=OKABE_ITO["orange"],
        fontsize=5.8,
    )

    # Bottom banner: Verbatim §4.7 row 4 decision matrix outcome
    o47 = bundle.outcome_4_7
    _draw_box(
        ax,
        0.01,
        0.03,
        0.98,
        0.36,
        (
            "Pre-Registered Decision Matrix Outcome "
            f"(docs/3_experimental_protocol.md §4.7 row {o47.row}) [Confirmatory]:\n"
            f'Outcome: "{o47.outcome}"  |  Venue: "{o47.venue}"\n'
            f"{o47.statement}"
        ),
        fc="#FFF8F6",
        ec="#A50E0E",
        fontsize=5.2,
        fontweight="bold",
    )

    label_panel(
        ax,
        "c",
        "Validation Design (S-ID, S-LRO, S-TAIL, S-IND, S-RND) & Pre-Registered §4.7 Row 4 Outcome",
        TIER_DESCRIPTIVE,
    )


def generate_fig1(
    bundle: VerifiedBundle,
    repo_root: Path,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Build and save Fig. 1 (Study domain, M5 architecture, validation design)."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 6.8))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.62, 0.96, 0.94],
        hspace=0.32,
        left=0.03,
        right=0.98,
        top=0.94,
        bottom=0.02,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    _draw_panel_a_map(ax_a, bundle, repo_root, registry)
    _draw_panel_b_architecture(ax_b, bundle, registry)
    _draw_panel_c_validation(ax_c, bundle, registry)

    return save_figure_bundle(fig, figures_dir, "fig1_domain_and_architecture")
