"""Supplementary Figures S1-S5 generator for Phase 9 (P9-D4 (a), P9-D5 (a), P9-D6 (a)).

Generates:
- ``figS1_event1_spatial_dependence.{pdf,svg,png}``: Event 1 spatial/network dependence audit
  (per-block d_nn & nesting shares, empirical variograms, Moran's I correlograms, LISA clusters)
  alongside Event 3 M5 residual dependence.
- ``figS2_per_division_slro.{pdf,svg,png}``: CONUS S-LRO spatial log-depth error maps for M5
  and M2-XGB, plus stratified S-LRO RMSE_log(dbar) by F&J division and Strahler stream order.
- ``figS3_secondary_analyses.{pdf,svg,png}``: Pre-registered sensitivity analyses (SA1-SA6),
  secondary S-RND ADCP measurement benchmark, S1 HyG consistency check (protocol 4.6), and
  pre-registered 'not computed' / dropped-row audit card.
- ``figS4_training_and_conformal.{pdf,svg,png}``: Training/validation loss curves, M4 soft-penalty
  curriculum schedule Xi_e, stopping epoch & Picard summary, and multi-target 90 % UQ coverage.
- ``figS5_sind_ehydro_rivers.{pdf,svg,png}``: S-IND USACE eHydro (131 locations / 18 rivers)
  predicted vs observed dbar & A, per-river single- vs multi-COMID breakdown (33 vs 98),
  full model ladder on S-IND, and 33-location 90 % prediction intervals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    S_IND_UQ_NOTE,
    VerifiedBundle,
)
from annxs_figures.style import (
    ASCE_DOUBLE_COL_IN,
    MODEL_PALETTE,
    OKABE_ITO,
    AnnotationRegistry,
    apply_asce_style,
    label_panel,
    save_figure_bundle,
)
from annxs_figures.tables import (
    TIER_CONFIRMATORY,
    TIER_DESCRIPTIVE,
    TIER_SECONDARY,
)


def _matrix_val(
    df: pd.DataFrame,
    *,
    model: str,
    family: str,
    target: str,
    metric: str,
    ablation: str = "",
    analysis: str = "primary",
    stratum_kind: str = "all",
    stratum: str = "all",
) -> float:
    spec = f"{model}_{ablation}" if ablation else model
    sub = df.loc[
        (df["spec"] == spec)
        & (df["analysis"] == analysis)
        & (df["family"] == family)
        & (df["target"] == target)
        & (df["stratum_kind"] == stratum_kind)
        & (df["stratum"] == stratum)
        & (df["metric"] == metric)
    ]
    if len(sub) != 1:
        raise KeyError(
            f"Expected 1 row for {spec}|{analysis}|{family}|"
            f"{target}|{stratum_kind}:{stratum}|{metric}, found {len(sub)}"
        )
    return float(sub["value"].iloc[0])


def _generate_fig_s1(
    bundle: VerifiedBundle,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Render Fig. S1: Event 1 spatial/network dependence audit & Event 3 M5 residuals."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 5.5))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.96,
        top=0.89,
        bottom=0.10,
        hspace=0.44,
        wspace=0.35,
    )

    audit = bundle.audit
    dep_m5 = bundle.dependence["M5_residual_structure_3_6_1_c"]["series"]

    # ------------------------------------------------------------------
    # Panel (a): Per-block d_nn median and strong/weak nesting shares
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    blocks = audit["per_block"]
    block_ids = np.arange(len(blocks))
    block_labels = [f"R{b['repeat']}F{b['outer_fold']}" for b in blocks]
    dnn_med = [float(b["d_nn_median_km"]) for b in blocks]
    sh_strong = [100.0 * float(b["share_strong"]) for b in blocks]
    sh_weak = [100.0 * float(b["share_weak"]) for b in blocks]

    ax_a_twin = ax_a.twinx()
    ax_a.bar(
        block_ids - 0.16,
        sh_strong,
        width=0.32,
        color=OKABE_ITO["vermilion"],
        alpha=0.80,
        label=r"Strong nesting ($\Psi \geq 0.5$)",
    )
    ax_a.bar(
        block_ids + 0.16,
        sh_weak,
        width=0.32,
        color=OKABE_ITO["orange"],
        alpha=0.80,
        label=r"Weak nesting ($0 < \Psi < 0.5$)",
    )
    ax_a_twin.plot(
        block_ids,
        dnn_med,
        marker="o",
        markersize=3.6,
        linewidth=1.1,
        color=OKABE_ITO["blue"],
        label=r"Median $d_{\mathrm{nn}}$ (km)",
    )

    glob_dnn = float(audit["d_nn_median_km"])
    glob_rho_hat = float(audit["rho_hat_km"])
    glob_rho_star = float(audit["rho_star_km"])
    glob_strong = float(audit["nested"]["share_strong"])
    glob_weak = float(audit["nested"]["share_weak"])
    glob_far = float(audit["far"]["share_far"])

    dnn_txt = registry.register(
        figure_id="figS1",
        panel_id="a",
        label="global_d_nn_median_km",
        raw_value=glob_dnn,
        fmt_kind="dec2",
        source_file="reports/results/audit.json",
        source_key="d_nn_median_km",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    rho_txt = registry.register(
        figure_id="figS1",
        panel_id="a",
        label="global_rho_hat_km",
        raw_value=glob_rho_hat,
        fmt_kind="dec2",
        source_file="reports/results/audit.json",
        source_key="rho_hat_km",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    strong_txt = registry.register(
        figure_id="figS1",
        panel_id="a",
        label="nested_share_strong",
        raw_value=glob_strong,
        fmt_kind="dec4",
        source_file="reports/results/audit.json",
        source_key="nested.share_strong",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    far_txt = registry.register(
        figure_id="figS1",
        panel_id="a",
        label="far_share_far",
        raw_value=glob_far,
        fmt_kind="dec4",
        source_file="reports/results/audit.json",
        source_key="far.share_far",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax_a.set_xticks(block_ids)
    ax_a.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=5.8)
    ax_a.set_ylabel("Nesting share (%)", fontsize=7.0)
    ax_a_twin.set_ylabel(r"Median $d_{\mathrm{nn}}$ (km)", fontsize=7.0, color=OKABE_ITO["blue"])
    ax_a_twin.tick_params(axis="y", labelsize=6.2, colors=OKABE_ITO["blue"])
    ax_a_twin.spines["right"].set_visible(True)
    ax_a.set_ylim(0.0, 68.0)
    ax_a_twin.set_ylim(10.0, 22.0)
    label_panel(
        ax_a,
        "a",
        r"Per-Block S-ID Nesting & $d_{\mathrm{nn}}$",
        TIER_DESCRIPTIVE,
    )
    ax_a.text(
        0.02,
        0.96,
        f"Global med $d_{{\\mathrm{{nn}}}}={dnn_txt}$ km | "
        f"$\\hat{{\\rho}}={rho_txt}$ km ($\\rho^*={glob_rho_star:.2f}$ km)\n"
        f"Strong={strong_txt}, Weak={glob_weak:.4f}, Far={far_txt}",
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=5.2,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax_a.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.78),
        fontsize=5.0,
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
    )

    # ------------------------------------------------------------------
    # Panel (b): Empirical semivariograms gamma_hat(h) for dbar residuals
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    series_specs = [
        (
            "M-null:dbar",
            audit["series"]["M-null:dbar"],
            "reports/results/audit.json",
            "series.M-null:dbar",
            MODEL_PALETTE["M-null"],
            "s",
            "M-null (Event 1)",
        ),
        (
            "M2-XGB:dbar",
            audit["series"]["M2-XGB:dbar"],
            "reports/results/audit.json",
            "series.M2-XGB:dbar",
            MODEL_PALETTE["M2-XGB"],
            "^",
            "M2-XGB (Event 1)",
        ),
        (
            "M5:dbar",
            dep_m5["M5:dbar"],
            "reports/results/event3_dependence.json",
            "M5_residual_structure_3_6_1_c.series.M5:dbar",
            MODEL_PALETTE["M5"],
            "o",
            "M5 (Event 3)",
        ),
    ]
    for s_key, s_doc, src_file, src_prefix, color, marker, label in series_specs:
        var_doc = s_doc["variogram"]
        h_mid = [float(x) for x in var_doc["h_km"]]
        gammas = [float(g) for g in var_doc["gamma"]]
        r_km = float(var_doc["rho_hat_km"])
        r_txt = registry.register(
            figure_id="figS1",
            panel_id="b",
            label=f"{s_key}_variogram_rho_hat_km",
            raw_value=r_km,
            fmt_kind="dec2",
            source_file=src_file,
            source_key=f"{src_prefix}.variogram.rho_hat_km",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        ax_b.plot(
            h_mid,
            gammas,
            marker=marker,
            markersize=3.4,
            linewidth=1.1,
            color=color,
            label=f"{label} ($\\hat{{\\rho}}={r_txt}$ km)",
        )

    ax_b.set_xlabel("Lag distance $h$ (km)", fontsize=7.0)
    ax_b.set_ylabel(r"Empirical semivariance $\hat{\gamma}(h)$ ($\bar{d}$)", fontsize=7.0)
    ax_b.set_ylim(0.16, 0.54)
    label_panel(
        ax_b,
        "b",
        r"OOF $\ln \bar{d}$ Residual Semivariograms",
        TIER_DESCRIPTIVE,
    )
    ax_b.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.62),
        fontsize=5.3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
    )

    # ------------------------------------------------------------------
    # Panel (c): Moran's I distance correlogram for dbar residuals
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.axhline(0.0, color="#0F172A", linewidth=0.8, linestyle=":")
    for s_key, s_doc, src_file, src_prefix, color, marker, label in series_specs:
        c_bands = s_doc["correlogram"]
        h_mid = [0.5 * (float(b["lower_km"]) + float(b["upper_km"])) for b in c_bands]
        m_i = [float(b["I"]) for b in c_bands]
        i_w1 = float(s_doc["moran_W1"]["I"])
        i_w2 = float(s_doc["moran_W2"]["I"])
        w1_txt = registry.register(
            figure_id="figS1",
            panel_id="c",
            label=f"{s_key}_moran_W1_I",
            raw_value=i_w1,
            fmt_kind="dec4",
            source_file=src_file,
            source_key=f"{src_prefix}.moran_W1.I",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        w2_txt = registry.register(
            figure_id="figS1",
            panel_id="c",
            label=f"{s_key}_moran_W2_I",
            raw_value=i_w2,
            fmt_kind="dec4",
            source_file=src_file,
            source_key=f"{src_prefix}.moran_W2.I",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        ax_c.plot(
            h_mid,
            m_i,
            marker=marker,
            markersize=3.8,
            linewidth=1.2,
            color=color,
            label=f"{label} ($I(W_1)={w1_txt}, I(W_2)={w2_txt}$)",
        )

    ax_c.set_xlabel("Distance band midpoint $h$ (km)", fontsize=7.0)
    ax_c.set_ylabel(r"Moran's $I(h)$ ($\bar{d}$ residuals)", fontsize=7.0)
    label_panel(
        ax_c,
        "c",
        r"OOF $\ln \bar{d}$ Spatial Correlograms",
        TIER_DESCRIPTIVE,
    )
    ax_c.legend(loc="upper right", fontsize=5.4)

    # ------------------------------------------------------------------
    # Panel (d): LISA significant cluster counts by F&J division
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    div_order = [
        ("INTERIOR PLAINS", "Int. Plains"),
        ("ATLANTIC PLAIN", "Atl. Plain"),
        ("APPALACHIAN HIGHLANDS", "Appalachian"),
        ("INTERMONTANE PLATEAUS", "Intermontane"),
        ("ROCKY MOUNTAIN SYSTEM", "Rocky Mtn."),
        ("PACIFIC MOUNTAIN SYSTEM", "Pacific Mtn."),
        ("INTERIOR HIGHLANDS", "Int. High."),
        ("LAURENTIAN UPLAND", "Laurentian"),
    ]
    x_d = np.arange(len(div_order))
    width_d = 0.25
    for idx, (s_key, s_doc, src_file, src_prefix, color, _, label) in enumerate(series_specs):
        lisa_div = s_doc["lisa_significant_by_division"]
        totals: list[int] = []
        for raw_div, _ in div_order:
            d_counts = lisa_div[raw_div]
            tot = (
                int(d_counts["HH"])
                + int(d_counts["LL"])
                + int(d_counts["HL"])
                + int(d_counts["LH"])
            )
            totals.append(tot)
        ip_hh = int(lisa_div["INTERIOR PLAINS"]["HH"])
        registry.register(
            figure_id="figS1",
            panel_id="d",
            label=f"{s_key}_lisa_INTERIOR_PLAINS_HH",
            raw_value=ip_hh,
            fmt_kind="int",
            source_file=src_file,
            source_key=f"{src_prefix}.lisa_significant_by_division.INTERIOR PLAINS.HH",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        tot_sum = int(sum(totals))
        offset = (idx - 1.0) * width_d
        ax_d.bar(
            x_d + offset,
            totals,
            width=width_d * 0.92,
            color=color,
            edgecolor="#1E293B",
            linewidth=0.45,
            label=f"{label.split(' ')[0]} (total={tot_sum})",
        )

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([short for _, short in div_order], rotation=28, ha="right", fontsize=6.0)
    ax_d.set_ylabel("Significant LISA sites ($p < 0.05$)", fontsize=7.0)
    label_panel(
        ax_d,
        "d",
        "Local Moran's I (LISA) by Division",
        TIER_DESCRIPTIVE,
    )
    ax_d.legend(loc="upper right", fontsize=5.8)

    fig.text(
        0.50,
        0.96,
        f"Event 1 Spatial/Network Dependence Audit Verdict: S-ID is '{EVENT1_AUDIT_LABEL}' "
        f"(triggers H2/S2 not-nested & far sensitivity strata)",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color="#1E293B",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FEF3C7", "edgecolor": "#F59E0B"},
    )

    return save_figure_bundle(fig, figures_dir, "figS1_event1_spatial_dependence")


def _generate_fig_s2(
    bundle: VerifiedBundle,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Render Fig. S2: Per-division & stream-order S-LRO errors and CONUS spatial error maps."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 5.6))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.95,
        top=0.92,
        bottom=0.10,
        hspace=0.42,
        wspace=0.44,
    )

    site_df = bundle.load_frozen_table("data/processed/site_table.parquet")
    primary_sites = site_df.loc[
        site_df["variant"].eq("primary") & ~site_df["F_DA"].astype(bool),
        ["site_no", "x5070", "y5070", "dbar"],
    ].copy()
    primary_sites["site_no"] = primary_sites["site_no"].astype(str)
    primary_sites["x_km"] = primary_sites["x5070"].to_numpy(dtype=np.float64) / 1000.0
    primary_sites["y_km"] = primary_sites["y5070"].to_numpy(dtype=np.float64) / 1000.0

    m5_preds = bundle.load_sealed_prediction("M5.parquet")
    m2_preds = bundle.load_sealed_prediction("M2-XGB.parquet")

    def _slro_site_errors(preds: pd.DataFrame) -> pd.DataFrame:
        sub = preds.loc[
            (preds["family"] == "S-LRO")
            & (preds["ablation"] == "")
            & (preds["analysis"] == "primary")
            & (preds["unit"] == "site")
        ][["site_no", "mu_dbar"]].copy()
        sub["site_no"] = sub["site_no"].astype(str)
        merged = primary_sites.merge(sub, on="site_no", how="inner")
        merged["err_log"] = merged["mu_dbar"].to_numpy(dtype=np.float64) - np.log(
            merged["dbar"].to_numpy(dtype=np.float64)
        )
        return merged

    m5_map = _slro_site_errors(m5_preds)
    m2_map = _slro_site_errors(m2_preds)
    mat = bundle.matrix_df

    # ------------------------------------------------------------------
    # Panel (a) & Panel (b): CONUS spatial error maps on S-LRO
    # ------------------------------------------------------------------
    for model_key, map_df, panel_id, gs_col in (
        ("M5", m5_map, "a", 0),
        ("M2-XGB", m2_map, "b", 1),
    ):
        ax = fig.add_subplot(gs[0, gs_col])
        sc = ax.scatter(
            map_df["x_km"],
            map_df["y_km"],
            c=map_df["err_log"],
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            s=2.2,
            alpha=0.78,
            linewidths=0.0,
            rasterized=True,
        )
        rmse_val = _matrix_val(
            mat, model=model_key, family="S-LRO", target="dbar", metric="rmse_log"
        )
        bias_val = _matrix_val(
            mat, model=model_key, family="S-LRO", target="dbar", metric="bias_log"
        )
        rmse_txt = registry.register(
            figure_id="figS2",
            panel_id=panel_id,
            label=f"{model_key}_S-LRO_dbar_rmse_log",
            raw_value=rmse_val,
            fmt_kind="dec4",
            source_file="reports/results/event3_matrix.parquet",
            source_key=f"{model_key}|primary|S-LRO|dbar|all|rmse_log",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        bias_txt = registry.register(
            figure_id="figS2",
            panel_id=panel_id,
            label=f"{model_key}_S-LRO_dbar_bias_log",
            raw_value=bias_val,
            fmt_kind="dec4",
            source_file="reports/results/event3_matrix.parquet",
            source_key=f"{model_key}|primary|S-LRO|dbar|all|bias_log",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
        cbar.ax.tick_params(labelsize=5.8)
        if gs_col == 1:
            cbar.set_label(r"$\ln(\hat{\bar{d}}/\bar{d})$", fontsize=6.2)
        ax.set_xlabel("Albers Easting $x$ (km)", fontsize=6.8)
        ax.set_ylabel("Albers Northing $y$ (km)", fontsize=6.8)
        label_panel(
            ax,
            panel_id,
            f"{model_key} S-LRO Residuals (n={len(map_df)})",
            TIER_DESCRIPTIVE,
        )
        ax.text(
            0.02,
            0.04,
            f"$\\mathrm{{RMSE}}_{{\\log}}={rmse_txt}$ | $B_{{\\log}}={bias_txt}$",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=5.8,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFFFFF", "edgecolor": "#CBD5E1"},
        )

    # ------------------------------------------------------------------
    # Panel (c): S-LRO RMSE_log(dbar) by F&J physiographic division
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    div_order = [
        ("INTERIOR PLAINS", "Int. Plains"),
        ("ATLANTIC PLAIN", "Atl. Plain"),
        ("APPALACHIAN HIGHLANDS", "Appalachian"),
        ("INTERMONTANE PLATEAUS", "Intermontane"),
        ("ROCKY MOUNTAIN SYSTEM", "Rocky Mtn."),
        ("PACIFIC MOUNTAIN SYSTEM", "Pacific Mtn."),
        ("INTERIOR HIGHLANDS", "Int. High."),
        ("LAURENTIAN UPLAND", "Laurentian"),
    ]
    comp_models = ["M5", "M2-XGB", "M2-MLP", "M4", "M1"]
    x_c = np.arange(len(div_order))
    width_c = 0.16
    for m_idx, m_key in enumerate(comp_models):
        vals = []
        for raw_div, _ in div_order:
            v = _matrix_val(
                mat,
                model=m_key,
                family="S-LRO",
                target="dbar",
                metric="rmse_log",
                stratum_kind="division",
                stratum=raw_div,
            )
            vals.append(v)
            if m_key in ("M5", "M2-XGB"):
                registry.register(
                    figure_id="figS2",
                    panel_id="c",
                    label=f"{m_key}_S-LRO_dbar_div_{raw_div}_rmse_log",
                    raw_value=v,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_matrix.parquet",
                    source_key=f"{m_key}|primary|S-LRO|dbar|division:{raw_div}|rmse_log",
                    evidence_tier=TIER_DESCRIPTIVE,
                )
        offset = (m_idx - 2.0) * width_c
        ax_c.bar(
            x_c + offset,
            vals,
            width=width_c * 0.92,
            color=MODEL_PALETTE[m_key],
            edgecolor="#1E293B",
            linewidth=0.4,
            label=m_key,
        )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([short for _, short in div_order], rotation=28, ha="right", fontsize=6.0)
    ax_c.set_ylabel(r"S-LRO $\mathrm{RMSE}_{\log}(\bar{d})$", fontsize=7.0)
    ax_c.set_ylim(0.25, 0.96)
    label_panel(
        ax_c,
        "c",
        "S-LRO Depth Accuracy by Division",
        TIER_DESCRIPTIVE,
    )
    ax_c.legend(
        loc="upper left",
        fontsize=5.4,
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
    )

    # ------------------------------------------------------------------
    # Panel (d): S-LRO RMSE_log(dbar) by Strahler stream order (1..10)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    so_orders = [str(k) for k in range(1, 11)]
    x_d = np.arange(len(so_orders))
    markers_d = {"M5": "o", "M2-XGB": "s", "M2-MLP": "^", "M4": "D", "M1": "v"}
    for m_key in comp_models:
        vals = []
        for so in so_orders:
            v = _matrix_val(
                mat,
                model=m_key,
                family="S-LRO",
                target="dbar",
                metric="rmse_log",
                stratum_kind="stream_order",
                stratum=so,
            )
            vals.append(v)
            if m_key in ("M5", "M2-XGB"):
                registry.register(
                    figure_id="figS2",
                    panel_id="d",
                    label=f"{m_key}_S-LRO_dbar_so_{so}_rmse_log",
                    raw_value=v,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_matrix.parquet",
                    source_key=f"{m_key}|primary|S-LRO|dbar|stream_order:{so}|rmse_log",
                    evidence_tier=TIER_DESCRIPTIVE,
                )
        ax_d.plot(
            x_d,
            vals,
            marker=markers_d[m_key],
            markersize=3.8,
            linewidth=1.15,
            color=MODEL_PALETTE[m_key],
            label=m_key,
        )

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([f"SO {s}" for s in so_orders], rotation=25, ha="right", fontsize=6.0)
    ax_d.set_ylabel(r"S-LRO $\mathrm{RMSE}_{\log}(\bar{d})$", fontsize=7.0)
    ax_d.set_ylim(0.20, 1.32)
    label_panel(
        ax_d,
        "d",
        "S-LRO Depth Accuracy by Stream Order",
        TIER_DESCRIPTIVE,
    )
    ax_d.legend(
        loc="upper center",
        fontsize=5.4,
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
    )

    return save_figure_bundle(fig, figures_dir, "figS2_per_division_slro")


def _generate_fig_s3(
    bundle: VerifiedBundle,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Render Fig. S3: SA1-SA6 sensitivity analyses, S-RND, S1 HyG check, and not-computed audit."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 5.6))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.96,
        top=0.92,
        bottom=0.10,
        hspace=0.44,
        wspace=0.32,
    )

    mat = bundle.matrix_df
    desc = bundle.descriptive

    # ------------------------------------------------------------------
    # Panel (a): SA1-SA6 sensitivity analyses on S-LRO RMSE_log(dbar)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    sa_list = ["primary", "SA1", "SA2", "SA3", "SA4", "SA5", "SA6"]
    sa_labels = [
        "Primary",
        "SA1\n(Alt init)",
        "SA2\n(No flags)",
        "SA3\n(Strict QC)",
        "SA4\n(Drop F_Qsel)",
        "SA5\n(Strict eps)",
        "SA6\n(Alt seed)",
    ]
    sa_models = ["M5", "M2-XGB", "M2-MLP", "M1"]
    x_a = np.arange(len(sa_list))
    width_a = 0.19

    for m_idx, m_key in enumerate(sa_models):
        vals = []
        for sa in sa_list:
            if sa == "SA1" and m_key != "M5":
                v = _matrix_val(
                    mat,
                    model=m_key,
                    family="S-LRO",
                    target="dbar",
                    metric="rmse_log",
                    analysis="primary",
                )
            else:
                v = _matrix_val(
                    mat,
                    model=m_key,
                    family="S-LRO",
                    target="dbar",
                    metric="rmse_log",
                    analysis=sa,
                )
                registry.register(
                    figure_id="figS3",
                    panel_id="a",
                    label=f"{m_key}_{sa}_S-LRO_dbar_rmse_log",
                    raw_value=v,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_matrix.parquet",
                    source_key=f"{m_key}|{sa}|S-LRO|dbar|all|rmse_log",
                    evidence_tier=TIER_DESCRIPTIVE,
                )
            vals.append(v)
        offset = (m_idx - 1.5) * width_a
        bars = ax_a.bar(
            x_a + offset,
            vals,
            width=width_a * 0.92,
            color=MODEL_PALETTE[m_key],
            edgecolor="#1E293B",
            linewidth=0.45,
            label=m_key,
        )
        if m_key != "M5":
            bars[1].set_hatch("//")
            bars[1].set_alpha(0.55)

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(sa_labels, fontsize=5.3)
    ax_a.set_ylabel(r"S-LRO $\mathrm{RMSE}_{\log}(\bar{d})$", fontsize=7.0)
    ax_a.set_ylim(0.32, 0.90)
    label_panel(
        ax_a,
        "a",
        "SA1-SA6 Sensitivity on S-LRO (dbar)",
        TIER_DESCRIPTIVE,
    )
    ax_a.text(
        0.02,
        0.96,
        (
            "Hatched SA1 (M2-XGB, M2-MLP, M1):\n"
            "'identical to primary by construction (no Phi); not trained'"
        ),
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=4.9,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "#FFFFFF", "edgecolor": "#CBD5E1"},
    )
    ax_a.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.78),
        fontsize=5.2,
        ncol=4,
        columnspacing=0.5,
        handlelength=1.0,
    )

    # ------------------------------------------------------------------
    # Panel (b): Secondary S-RND ADCP measurement benchmark vs S-ID
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    targets_b = [("dbar", r"$\bar{d}$"), ("A", "$A$"), ("W", "$W$")]
    x_b = np.arange(len(targets_b))
    series_b = [
        ("M1", "S-RND", MODEL_PALETTE["M1"], None, "M1 (S-RND, n=9,673)"),
        ("M2-XGB", "S-RND", MODEL_PALETTE["M2-XGB"], None, "M2-XGB (S-RND, n=9,673)"),
        ("M1", "S-ID", MODEL_PALETTE["M1"], "//", "M1 (S-ID site-level)"),
        ("M2-XGB", "S-ID", MODEL_PALETTE["M2-XGB"], "//", "M2-XGB (S-ID site-level)"),
    ]
    width_b = 0.18
    for idx_b, (m_key, fam, color, hatch, label) in enumerate(series_b):
        vals = []
        for t_code, _ in targets_b:
            v = _matrix_val(mat, model=m_key, family=fam, target=t_code, metric="rmse_log")
            vals.append(v)
            registry.register(
                figure_id="figS3",
                panel_id="b",
                label=f"{m_key}_{fam}_{t_code}_rmse_log",
                raw_value=v,
                fmt_kind="dec4",
                source_file="reports/results/event3_matrix.parquet",
                source_key=f"{m_key}|primary|{fam}|{t_code}|all|rmse_log",
                evidence_tier=TIER_SECONDARY if fam == "S-RND" else TIER_DESCRIPTIVE,
            )
        offset = (idx_b - 1.5) * width_b
        ax_b.bar(
            x_b + offset,
            vals,
            width=width_b * 0.92,
            color=color,
            edgecolor="#1E293B",
            linewidth=0.45,
            hatch=hatch,
            alpha=0.65 if hatch else 0.92,
            label=label,
        )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([lbl for _, lbl in targets_b], fontsize=7.0)
    ax_b.set_ylabel(r"$\mathrm{RMSE}_{\log}$ (weighted $1/|I_j|$ on S-RND)", fontsize=7.0)
    ax_b.set_ylim(0.0, 0.92)
    label_panel(
        ax_b,
        "b",
        "S-RND vs S-ID Benchmark",
        f"{TIER_SECONDARY} / {TIER_DESCRIPTIVE}",
    )
    ax_b.text(
        0.02,
        0.96,
        f"S-RND & S-ID: '{EVENT1_AUDIT_LABEL}'",
        transform=ax_b.transAxes,
        va="top",
        ha="left",
        fontsize=5.0,
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "#FFFFFF", "edgecolor": "#CBD5E1"},
    )
    ax_b.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.84),
        fontsize=5.0,
        ncol=2,
        columnspacing=0.5,
        handlelength=1.1,
    )

    # ------------------------------------------------------------------
    # Panel (c): S1 HyG consistency check (protocol 4.6)
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    s1 = desc["S1"]
    s1_rmse = float(s1["rmse_log"])
    s1_rho = float(s1["spearman_rho"])
    s1_matched = int(s1["n_matched"])
    s1_note = str(s1["note"])
    hyg_counts = s1["hyg_reader_counts"]

    rmse_txt = registry.register(
        figure_id="figS3",
        panel_id="c",
        label="S1_rmse_log",
        raw_value=s1_rmse,
        fmt_kind="dec4",
        source_file="reports/results/event3_descriptive.json",
        source_key="S1.rmse_log",
        evidence_tier=TIER_SECONDARY,
    )
    rho_txt = registry.register(
        figure_id="figS3",
        panel_id="c",
        label="S1_spearman_rho",
        raw_value=s1_rho,
        fmt_kind="dec4",
        source_file="reports/results/event3_descriptive.json",
        source_key="S1.spearman_rho",
        evidence_tier=TIER_SECONDARY,
    )
    matched_txt = registry.register(
        figure_id="figS3",
        panel_id="c",
        label="S1_n_matched",
        raw_value=s1_matched,
        fmt_kind="int",
        source_file="reports/results/event3_descriptive.json",
        source_key="S1.n_matched",
        evidence_tier=TIER_SECONDARY,
    )

    cats_c = [
        "Raw HyG\nrows",
        "Padded\nIDs",
        "Kept valid\nManning n",
        "Missing/<=0\nManning n",
        "Matched\nframe sites",
    ]
    counts_c = [
        int(hyg_counts["rows"]),
        int(hyg_counts["padded_id"]),
        int(hyg_counts["kept"]),
        int(hyg_counts["manning_n_missing_or_nonpositive"]),
        s1_matched,
    ]
    colors_c = [
        OKABE_ITO["grey"],
        OKABE_ITO["sky_blue"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["orange"],
        OKABE_ITO["blue"],
    ]
    x_c = np.arange(len(cats_c))
    bars_c = ax_c.bar(x_c, counts_c, width=0.55, color=colors_c, edgecolor="#1E293B", linewidth=0.5)
    for b, cnt in zip(bars_c, counts_c, strict=True):
        ax_c.text(
            b.get_x() + b.get_width() / 2.0,
            cnt + 250.0,
            f"{cnt:,}",
            ha="center",
            va="bottom",
            fontsize=5.6,
            fontweight="bold",
        )
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(cats_c, fontsize=5.8)
    ax_c.set_ylabel("Site / record count", fontsize=7.0)
    ax_c.set_ylim(0, 17800)
    label_panel(
        ax_c,
        "c",
        "S1 HyG Manning n Consistency",
        TIER_SECONDARY,
    )
    ax_c.text(
        0.98,
        0.96,
        f"M5 $\\hat{{n}}$ vs HyG ($n={matched_txt}$): "
        f"$\\mathrm{{RMSE}}_{{\\log}}={rmse_txt}$, Spearman $\\rho={rho_txt}$\n"
        f"Note: '{s1_note}'",
        transform=ax_c.transAxes,
        va="top",
        ha="right",
        fontsize=5.0,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#F8FAFC", "edgecolor": "#94A3B8"},
    )

    # ------------------------------------------------------------------
    # Panel (d): Pre-registered "not computed" & dropped-row audit card
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    l8_status = str(desc["L8_zarrabi_product"]["status"])
    l8_reason = str(desc["L8_zarrabi_product"]["reason"])
    far_h2_nc = str(bundle.hypotheses["far_not_nested_strata"]["far"]["H2"]["not_computed"])
    far_h2_cnts = bundle.hypotheses["far_not_nested_strata"]["far"]["H2"]["sites_per_repeat"]
    far_s2_cnts = bundle.hypotheses["far_not_nested_strata"]["far"]["S2"]["sites_per_repeat"]
    sind_multi = int(bundle.uq["M5|primary"]["S-IND"]["not_computed_multi_comid"])
    sind_single = int(bundle.uq["M5|primary"]["S-IND"]["n_locations_single"])
    aj_dropped = int(desc["dropped_rows"]["A_J|primary|S-IND"])

    aj_drop_txt = registry.register(
        figure_id="figS3",
        panel_id="d",
        label="dropped_rows_A_J_primary_S-IND",
        raw_value=aj_dropped,
        fmt_kind="int",
        source_file="reports/results/event3_descriptive.json",
        source_key="dropped_rows.A_J|primary|S-IND",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    multi_txt = registry.register(
        figure_id="figS3",
        panel_id="d",
        label="M5_primary_S-IND_not_computed_multi_comid",
        raw_value=sind_multi,
        fmt_kind="int",
        source_file="reports/results/event3_uq.json",
        source_key="M5|primary.S-IND.not_computed_multi_comid",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    label_panel(
        ax_d,
        "d",
        "Pre-Registered Not-Computed Audit",
        TIER_DESCRIPTIVE,
    )
    card_lines = [
        f"1. L8_zarrabi_product: status = '{l8_status}'",
        f"   Reason: {l8_reason[:64]}",
        f"   {l8_reason[64:]}",
        f"2. Event 1 'far' Stratum CIs (H2 & S2): '{far_h2_nc}'",
        f"   Sites/repeat: H2 Q1={far_h2_cnts}, S2={far_s2_cnts} (<30 floor)",
        f"3. S-IND Multi-COMID UQ: not_computed_multi_comid = {multi_txt}",
        f"   Evaluated on n_locations_single = {sind_single} single-COMID (docs/3 13.5)",
        f"4. S-IND Dropped Predictions: A_J|primary|S-IND = {aj_drop_txt} dropped",
        "   (50/131 evaluated; all other models = 0 dropped)",
        "5. F_S Stratum rho(n, S): 'undefined: an input is constant in this stratum'",
    ]
    ax_d.text(
        0.01,
        0.95,
        "\n".join(card_lines),
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontsize=4.9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFBEB", "edgecolor": "#D97706"},
    )

    return save_figure_bundle(fig, figures_dir, "figS3_secondary_analyses")


def _generate_fig_s4(
    bundle: VerifiedBundle,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Render Fig. S4: Training/validation curves, Xi_e schedule, Picard summary, and UQ."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 5.6))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.96,
        top=0.92,
        bottom=0.10,
        hspace=0.44,
        wspace=0.34,
    )

    curves = bundle.load_training_list("reports/training/curves.json")
    xi_list = bundle.load_training_list("reports/training/xi.json")
    summary = bundle.load_training_json("reports/training/summary.json")
    curves_df = pd.DataFrame(curves)
    xi_df = pd.DataFrame(xi_list)

    # ------------------------------------------------------------------
    # Panel (a): Training and validation loss trajectories
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    plot_models = [
        ("M5", MODEL_PALETTE["M5"], "-"),
        ("M4", MODEL_PALETTE["M4"], "-"),
        ("M3", MODEL_PALETTE["M3"], "-"),
        ("M2-MLP", MODEL_PALETTE["M2-MLP"], "-"),
        ("M6-NN", MODEL_PALETTE["M6-NN"], "-"),
        ("M5_A4", OKABE_ITO["sky_blue"], "--"),
    ]
    for m_key, color, ls in plot_models:
        sub = curves_df.loc[curves_df["model"] == m_key].sort_values("epoch")
        ax_a.plot(
            sub["epoch"],
            sub["es_val"],
            color=color,
            linestyle=ls,
            linewidth=1.15,
            label=f"{m_key} (val)",
        )
        ax_a.plot(
            sub["epoch"],
            sub["train"],
            color=color,
            linestyle=":",
            linewidth=0.75,
            alpha=0.65,
        )

    ax_a.set_xlabel("Training epoch $e$", fontsize=7.0)
    ax_a.set_ylabel("Mean loss (solid/dashed: val; dotted: train)", fontsize=6.8)
    label_panel(
        ax_a,
        "a",
        "Training & Validation Loss Curves",
        TIER_DESCRIPTIVE,
    )
    ax_a.legend(loc="upper right", fontsize=5.5, ncol=2)

    # ------------------------------------------------------------------
    # Panel (b): M4 soft-penalty curriculum schedule Xi_e
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    xi_sorted = xi_df.sort_values("epoch")
    ax_b.plot(
        xi_sorted["epoch"],
        xi_sorted["xi_mean"],
        color=MODEL_PALETTE["M4"],
        linewidth=1.3,
        label=r"Mean $\Xi_e$ across active fits",
    )
    ax_b.plot(
        xi_sorted["epoch"],
        xi_sorted["xi_max"],
        color=OKABE_ITO["vermilion"],
        linestyle="--",
        linewidth=1.0,
        label=r"Max $\Xi_e$ across active fits",
    )
    ax_b_twin = ax_b.twinx()
    ax_b_twin.plot(
        xi_sorted["epoch"],
        xi_sorted["n"],
        color=OKABE_ITO["grey"],
        linestyle=":",
        linewidth=1.0,
        label="Active fits $n_e$",
    )
    xi_final = float(summary["models"]["M4"]["xi_final_max"])
    ax_b.set_xlabel("Training epoch $e$", fontsize=7.0)
    ax_b.set_ylabel(r"M4 soft physics residual $\Xi_e$", fontsize=7.0)
    ax_b_twin.set_ylabel("Active fits $n_e$ (of 75)", fontsize=6.8, color=OKABE_ITO["grey"])
    ax_b_twin.tick_params(axis="y", labelsize=6.0, colors=OKABE_ITO["grey"])
    ax_b_twin.spines["right"].set_visible(True)
    label_panel(
        ax_b,
        "b",
        r"M4 Soft Penalty Trajectory $\Xi_e$",
        TIER_DESCRIPTIVE,
    )
    ax_b.text(
        0.96,
        0.52,
        f"Final max $\\Xi_{{\\mathrm{{final}}}} = {xi_final:.4f}$",
        transform=ax_b.transAxes,
        va="center",
        ha="right",
        fontsize=5.8,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax_b.legend(loc="upper right", fontsize=5.8)

    # ------------------------------------------------------------------
    # Panel (c): Stopping epochs & Picard iteration ceiling summary
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    sum_models = [
        "M5",
        "M5_A1",
        "M5_wA1",
        "M5_A2",
        "M5_A3-SP",
        "M5_A3-SR",
        "M5_A4",
        "M4",
        "M3",
        "M2-MLP",
        "M6-NN",
    ]
    sum_labels = ["M5", "A1", "wA1", "A2", "A3-SP", "A3-SR", "A4", "M4", "M3", "M2-MLP", "M6-NN"]
    ep_means = [float(summary["models"][m]["epochs_mean"]) for m in sum_models]
    ep_maxs = [float(summary["models"][m]["epochs_max"]) for m in sum_models]
    m5_train_picard = int(summary["models"]["M5"]["picard_max"])
    m5_eval_picard = int(bundle.consistency["models"]["M5"]["S-ID"]["picard_iterations_max"])
    eval_pic_txt = registry.register(
        figure_id="figS4",
        panel_id="c",
        label="M5_S-ID_picard_iterations_max",
        raw_value=m5_eval_picard,
        fmt_kind="int",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M5.S-ID.picard_iterations_max",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    x_c = np.arange(len(sum_models))
    ax_c.bar(
        x_c - 0.16,
        ep_means,
        width=0.32,
        color=OKABE_ITO["blue"],
        edgecolor="#1E293B",
        linewidth=0.45,
        label="Mean stopping epoch",
    )
    ax_c.bar(
        x_c + 0.16,
        ep_maxs,
        width=0.32,
        color=OKABE_ITO["sky_blue"],
        edgecolor="#1E293B",
        linewidth=0.45,
        label="Max stopping epoch",
    )
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(sum_labels, rotation=28, ha="right", fontsize=5.8)
    ax_c.set_ylabel("Epoch count", fontsize=7.0)
    ax_c.set_ylim(0, 580)
    label_panel(
        ax_c,
        "c",
        "Stopping Epochs & Picard Ceiling",
        TIER_DESCRIPTIVE,
    )
    ax_c.text(
        0.02,
        0.96,
        f"Training Picard max = {m5_train_picard} (M5/A1/wA1/A3/A4)\n"
        f"Evaluation Picard max = {eval_pic_txt} (exact 1-step convergence)",
        transform=ax_c.transAxes,
        va="top",
        ha="left",
        fontsize=5.1,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax_c.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.78),
        fontsize=5.3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
    )

    # ------------------------------------------------------------------
    # Panel (d): Multi-target 90% UQ coverage across dbar, A, W, t
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    uq_doc = bundle.uq
    band_lo, band_hi = float(uq_doc["band"][0]), float(uq_doc["band"][1])
    ax_d.axhspan(band_lo, band_hi, color="#DCFCE7", alpha=0.85, zorder=0, label="[0.85, 0.95] band")
    ax_d.axhline(0.90, color="#15803D", linewidth=0.9, linestyle="--", zorder=1)

    group_specs = [
        ("S-ID", "dbar", r"S-ID $\bar{d}$"),
        ("S-ID", "A", "S-ID $A$"),
        ("S-ID", "W", "S-ID $W$"),
        ("S-ID", "t", "S-ID $t$"),
        ("S-LRO", "dbar", r"S-LRO $\bar{d}$"),
        ("S-LRO", "A", "S-LRO $A$"),
        ("S-LRO", "W", "S-LRO $W$"),
        ("S-LRO", "t", "S-LRO $t$"),
        ("S-IND", "dbar", r"S-IND $\bar{d}$"),
        ("S-IND", "A", "S-IND $A$"),
        ("S-IND", "W", "S-IND $W$"),
    ]
    uq_models = [
        ("M5|primary", "M5", MODEL_PALETTE["M5"], "o"),
        ("M2-MLP|primary", "M2-MLP", MODEL_PALETTE["M2-MLP"], "s"),
        ("M4|primary", "M4", MODEL_PALETTE["M4"], "^"),
        ("M5_A4|primary", "M5 (A4)", OKABE_ITO["sky_blue"], "D"),
    ]
    x_d = np.arange(len(group_specs))
    for m_idx, (u_tag, lbl, color, marker) in enumerate(uq_models):
        ys = []
        xs = []
        for g_idx, (fam, t_code, _) in enumerate(group_specs):
            fam_doc = uq_doc[u_tag][fam]
            if t_code not in fam_doc:
                continue
            c_val = float(fam_doc[t_code]["overall"]["coverage"])
            ys.append(c_val)
            xs.append(g_idx + (m_idx - 1.5) * 0.14)
            if u_tag == "M5|primary":
                registry.register(
                    figure_id="figS4",
                    panel_id="d",
                    label=f"M5_primary_{fam}_{t_code}_coverage",
                    raw_value=c_val,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_uq.json",
                    source_key=f"{u_tag}.{fam}.{t_code}.overall.coverage",
                    evidence_tier=TIER_SECONDARY,
                )
        ax_d.plot(
            xs,
            ys,
            marker=marker,
            markersize=3.6,
            linewidth=0.0,
            color=color,
            label=lbl,
        )

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([lbl for _, _, lbl in group_specs], rotation=32, ha="right", fontsize=5.8)
    ax_d.set_ylabel(r"Empirical coverage $\hat{C}_{0.10}$", fontsize=7.0)
    ax_d.set_ylim(0.48, 1.03)
    label_panel(
        ax_d,
        "d",
        "90% Coverage Across Splits & Targets",
        f"{TIER_SECONDARY} / {TIER_DESCRIPTIVE}",
    )
    ax_d.legend(loc="lower left", fontsize=5.5, ncol=3)

    return save_figure_bundle(fig, figures_dir, "figS4_training_and_conformal")


def _generate_fig_s5(
    bundle: VerifiedBundle,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Render Fig. S5: S-IND USACE eHydro rivers (131 locations / 18 rivers; 33 single-COMID)."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 5.7))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.96,
        top=0.91,
        bottom=0.11,
        hspace=0.45,
        wspace=0.32,
    )

    ehydro = bundle.load_frozen_table("data/processed/ind/ehydro_ind.parquet")
    kept = ehydro.loc[ehydro["dedupe_status"] == "kept"].copy()
    kept["loc_id"] = kept["loc_id"].astype(str)

    m5_preds = bundle.load_sealed_prediction("M5.parquet")
    m2_preds = bundle.load_sealed_prediction("M2-XGB.parquet")
    m2mlp_preds = bundle.load_sealed_prediction("M2-MLP.parquet")
    mat = bundle.matrix_df

    m5_loc = m5_preds.loc[
        (m5_preds["family"] == "S-IND")
        & (m5_preds["unit"] == "location")
        & (m5_preds["ablation"] == "")
        & (m5_preds["analysis"] == "primary")
    ].copy()
    m5_loc["loc_id"] = m5_loc["loc_id"].astype(str)
    m5_merged = m5_loc.merge(kept[["loc_id", "W", "dbar", "A"]], on="loc_id", how="inner")

    m2_loc = m2_preds.loc[
        (m2_preds["family"] == "S-IND")
        & (m2_preds["unit"] == "location")
        & (m2_preds["ablation"] == "")
        & (m2_preds["analysis"] == "primary")
    ].copy()
    m2_loc["loc_id"] = m2_loc["loc_id"].astype(str)
    m2_merged = m2_loc.merge(kept[["loc_id", "W", "dbar", "A"]], on="loc_id", how="inner")

    # ------------------------------------------------------------------
    # Panel (a): S-IND predicted vs observed dbar (m) for M5 and M2-XGB
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    obs_d = m5_merged["dbar"].to_numpy(dtype=np.float64)
    m5_d = np.exp(m5_merged["mu_dbar"].to_numpy(dtype=np.float64))
    m2_d = np.exp(m2_merged["mu_dbar"].to_numpy(dtype=np.float64))

    ax_a.scatter(
        obs_d,
        m5_d,
        s=14,
        alpha=0.80,
        color=MODEL_PALETTE["M5"],
        edgecolors="#1E293B",
        linewidths=0.35,
        label="M5",
    )
    ax_a.scatter(
        obs_d,
        m2_d,
        s=14,
        alpha=0.75,
        marker="s",
        color=MODEL_PALETTE["M2-XGB"],
        edgecolors="#1E293B",
        linewidths=0.35,
        label="M2-XGB ($M^*$)",
    )
    ax_a.plot(
        [0.8, 38.0],
        [0.8, 38.0],
        color="#0F172A",
        linewidth=0.9,
        linestyle="--",
        label="1:1 line",
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(0.8, 38.0)
    ax_a.set_ylim(0.8, 38.0)
    ax_a.set_xlabel(r"Observed eHydro mean depth $\bar{d}$ (m)", fontsize=7.0)
    ax_a.set_ylabel(r"Predicted mean depth $\hat{\bar{d}}$ (m)", fontsize=7.0)
    label_panel(
        ax_a,
        "a",
        r"S-IND Pred. vs Obs. $\bar{d}$",
        f"{TIER_CONFIRMATORY} / {TIER_DESCRIPTIVE}",
    )

    m5_sind_rmse = _matrix_val(mat, model="M5", family="S-IND", target="dbar", metric="rmse_log")
    m2_sind_rmse = _matrix_val(
        mat, model="M2-XGB", family="S-IND", target="dbar", metric="rmse_log"
    )
    m5_sind_bias = _matrix_val(mat, model="M5", family="S-IND", target="dbar", metric="bias_log")
    m2_sind_bias = _matrix_val(
        mat, model="M2-XGB", family="S-IND", target="dbar", metric="bias_log"
    )
    m5_r_txt = registry.register(
        figure_id="figS5",
        panel_id="a",
        label="M5_S-IND_dbar_rmse_log",
        raw_value=m5_sind_rmse,
        fmt_kind="dec4",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M5|primary|S-IND|dbar|all|rmse_log",
        evidence_tier=TIER_CONFIRMATORY,
    )
    m2_r_txt = registry.register(
        figure_id="figS5",
        panel_id="a",
        label="M2-XGB_S-IND_dbar_rmse_log",
        raw_value=m2_sind_rmse,
        fmt_kind="dec4",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M2-XGB|primary|S-IND|dbar|all|rmse_log",
        evidence_tier=TIER_CONFIRMATORY,
    )
    ax_a.text(
        0.03,
        0.96,
        f"S-IND ({S_IND_PANEL_NOTE}):\n"
        f"M5: $\\mathrm{{RMSE}}_{{\\log}}={m5_r_txt}, B_{{\\log}}={m5_sind_bias:+.4f}$\n"
        f"M2-XGB: $\\mathrm{{RMSE}}_{{\\log}}={m2_r_txt}, B_{{\\log}}={m2_sind_bias:+.4f}$",
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=5.2,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax_a.legend(loc="lower right", fontsize=5.8)

    # ------------------------------------------------------------------
    # Panel (b): Per-river location counts (33 single-COMID vs 98 multi-COMID)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    m5_comid = (
        m5_preds.loc[
            (m5_preds["family"] == "S-IND")
            & (m5_preds["unit"] == "comid")
            & (m5_preds["ablation"] == "")
            & (m5_preds["analysis"] == "primary")
        ][["loc_id", "river", "n_comids"]]
        .drop_duplicates("loc_id")
        .copy()
    )
    m5_comid["is_single"] = m5_comid["n_comids"].astype(int) == 1
    river_grp = (
        m5_comid.groupby("river", as_index=False)
        .agg(
            n_total=("loc_id", "count"),
            n_single=("is_single", "sum"),
        )
        .sort_values(["n_total", "river"], ascending=[False, True])
        .reset_index(drop=True)
    )
    river_grp["n_multi"] = river_grp["n_total"] - river_grp["n_single"]

    x_b = np.arange(len(river_grp))
    ax_b.bar(
        x_b,
        river_grp["n_single"],
        width=0.65,
        color=OKABE_ITO["bluish_green"],
        edgecolor="#1E293B",
        linewidth=0.45,
        label=f"Single-COMID (UQ evaluated: n={int(river_grp['n_single'].sum())})",
    )
    ax_b.bar(
        x_b,
        river_grp["n_multi"],
        bottom=river_grp["n_single"],
        width=0.65,
        color=OKABE_ITO["orange"],
        edgecolor="#1E293B",
        linewidth=0.45,
        label=f"Multi-COMID (UQ not computed: n={int(river_grp['n_multi'].sum())})",
    )
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(river_grp["river"], rotation=40, ha="right", fontsize=5.2)
    ax_b.set_ylabel("Locations per river", fontsize=7.0)
    ax_b.set_ylim(0, float(river_grp["n_total"].max()) * 1.18)
    label_panel(
        ax_b,
        "b",
        "eHydro Locations by River (18 Rivers)",
        TIER_DESCRIPTIVE,
    )
    ax_b.text(
        0.98,
        0.96,
        f"S-IND: {S_IND_UQ_NOTE}",
        transform=ax_b.transAxes,
        va="top",
        ha="right",
        fontsize=5.0,
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax_b.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.83),
        fontsize=5.2,
    )

    # ------------------------------------------------------------------
    # Panel (c): S-IND full model ladder comparison on dbar and A
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ladder_models = [
        "M-null",
        "A_J",
        "M0",
        "M1",
        "M2-MLP",
        "M2-XGB",
        "M3",
        "M4",
        "M5",
        "M6-NN",
        "M6-XGB",
    ]
    x_c = np.arange(len(ladder_models))
    dbar_vals: list[float] = []
    a_vals: list[float] = []
    for m in ladder_models:
        av = _matrix_val(mat, model=m, family="S-IND", target="A", metric="rmse_log")
        a_vals.append(av)
        registry.register(
            figure_id="figS5",
            panel_id="c",
            label=f"{m}_S-IND_A_rmse_log",
            raw_value=av,
            fmt_kind="dec4",
            source_file="reports/results/event3_matrix.parquet",
            source_key=f"{m}|primary|S-IND|A|all|rmse_log",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        if m == "A_J":
            dbar_vals.append(float("nan"))
        else:
            dv = _matrix_val(mat, model=m, family="S-IND", target="dbar", metric="rmse_log")
            dbar_vals.append(dv)
            registry.register(
                figure_id="figS5",
                panel_id="c",
                label=f"{m}_S-IND_dbar_rmse_log",
                raw_value=dv,
                fmt_kind="dec4",
                source_file="reports/results/event3_matrix.parquet",
                source_key=f"{m}|primary|S-IND|dbar|all|rmse_log",
                evidence_tier=TIER_DESCRIPTIVE,
            )

    ax_c.bar(
        x_c - 0.17,
        np.nan_to_num(dbar_vals, nan=0.0),
        width=0.34,
        color=OKABE_ITO["blue"],
        edgecolor="#1E293B",
        linewidth=0.45,
        label=r"$\bar{d}$ $\mathrm{RMSE}_{\log}$",
    )
    ax_c.bar(
        x_c + 0.17,
        a_vals,
        width=0.34,
        color=OKABE_ITO["vermilion"],
        edgecolor="#1E293B",
        linewidth=0.45,
        label=r"$A$ $\mathrm{RMSE}_{\log}$",
    )
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(
        [m if m != "A_J" else "A_J\n(n=50)" for m in ladder_models],
        rotation=28,
        ha="right",
        fontsize=5.8,
    )
    ax_c.set_ylabel(r"S-IND $\mathrm{RMSE}_{\log}$", fontsize=7.0)
    ax_c.set_ylim(0.0, max(a_vals) * 1.12)
    label_panel(
        ax_c,
        "c",
        "S-IND Model Ladder Accuracy",
        TIER_DESCRIPTIVE,
    )
    ax_c.legend(loc="upper right", fontsize=5.8)

    # ------------------------------------------------------------------
    # Panel (d): 33-location single-COMID 90% prediction intervals on S-IND
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    m5_single = m5_preds.loc[
        (m5_preds["family"] == "S-IND")
        & (m5_preds["unit"] == "comid")
        & (m5_preds["ablation"] == "")
        & (m5_preds["analysis"] == "primary")
        & m5_preds["n_comids"].eq(1)
    ].copy()
    m5_single["loc_id"] = m5_single["loc_id"].astype(str)
    m5_single = m5_single.merge(kept[["loc_id", "dbar"]], on="loc_id", how="inner")
    m5_single["ln_obs_dbar"] = np.log(m5_single["dbar"].to_numpy(dtype=np.float64))
    m5_single = m5_single.sort_values("ln_obs_dbar").reset_index(drop=True)

    m2mlp_single = m2mlp_preds.loc[
        (m2mlp_preds["family"] == "S-IND")
        & (m2mlp_preds["unit"] == "comid")
        & (m2mlp_preds["ablation"] == "")
        & (m2mlp_preds["analysis"] == "primary")
        & m2mlp_preds["n_comids"].eq(1)
    ].copy()
    m2mlp_single["loc_id"] = m2mlp_single["loc_id"].astype(str)
    m2mlp_single = m5_single[["loc_id", "ln_obs_dbar"]].merge(
        m2mlp_single, on="loc_id", how="inner"
    )

    rank_x = np.arange(1, len(m5_single) + 1)
    ax_d.vlines(
        rank_x - 0.14,
        m5_single["q_lo_dbar"],
        m5_single["q_hi_dbar"],
        color=MODEL_PALETTE["M5"],
        linewidth=1.1,
        alpha=0.85,
        label="M5 90% PI",
    )
    ax_d.vlines(
        rank_x + 0.14,
        m2mlp_single["q_lo_dbar"],
        m2mlp_single["q_hi_dbar"],
        color=MODEL_PALETTE["M2-MLP"],
        linewidth=1.1,
        alpha=0.75,
        label="M2-MLP 90% PI",
    )
    ax_d.scatter(
        rank_x,
        m5_single["ln_obs_dbar"],
        color="#0F172A",
        s=11,
        zorder=5,
        label=r"Observed $\ln \bar{d}$",
    )

    m5_cov = float(bundle.uq["M5|primary"]["S-IND"]["dbar"]["overall"]["coverage"])
    m5_wid = float(bundle.uq["M5|primary"]["S-IND"]["dbar"]["overall"]["mean_log_width"])
    mlp_cov = float(bundle.uq["M2-MLP|primary"]["S-IND"]["dbar"]["overall"]["coverage"])
    mlp_wid = float(bundle.uq["M2-MLP|primary"]["S-IND"]["dbar"]["overall"]["mean_log_width"])

    m5_cov_txt = registry.register(
        figure_id="figS5",
        panel_id="d",
        label="M5_primary_S-IND_dbar_coverage",
        raw_value=m5_cov,
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M5|primary.S-IND.dbar.overall.coverage",
        evidence_tier=TIER_SECONDARY,
    )
    mlp_cov_txt = registry.register(
        figure_id="figS5",
        panel_id="d",
        label="M2-MLP_primary_S-IND_dbar_coverage",
        raw_value=mlp_cov,
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M2-MLP|primary.S-IND.dbar.overall.coverage",
        evidence_tier=TIER_SECONDARY,
    )

    ax_d.set_xlabel("Single-COMID S-IND location rank (sorted by observed depth)", fontsize=6.8)
    ax_d.set_ylabel(r"$\ln \bar{d}$ ($\ln\mathrm{m}$)", fontsize=7.0)
    label_panel(
        ax_d,
        "d",
        "S-IND 90% Prediction Intervals (n=33)",
        f"{TIER_SECONDARY} / {TIER_DESCRIPTIVE}",
    )
    ax_d.text(
        0.02,
        0.95,
        f"M5: $\\hat{{C}}_{{0.10}}={m5_cov_txt}, \\bar{{W}}_{{\\log}}={m5_wid:.4f}$\n"
        f"M2-MLP: $\\hat{{C}}_{{0.10}}={mlp_cov_txt}, \\bar{{W}}_{{\\log}}={mlp_wid:.4f}$",
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontsize=5.4,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax_d.legend(loc="lower right", fontsize=5.6)

    return save_figure_bundle(fig, figures_dir, "figS5_sind_ehydro_rivers")


def generate_supplementary_figures(
    bundle: VerifiedBundle,
    repo_root: Path,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, dict[str, Path]]:
    """Render all five supplementary figures (Fig. S1-Fig. S5)."""
    del repo_root
    return {
        "figS1": _generate_fig_s1(bundle, figures_dir, registry),
        "figS2": _generate_fig_s2(bundle, figures_dir, registry),
        "figS3": _generate_fig_s3(bundle, figures_dir, registry),
        "figS4": _generate_fig_s4(bundle, figures_dir, registry),
        "figS5": _generate_fig_s5(bundle, figures_dir, registry),
    }
