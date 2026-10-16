"""Figure 2: Primary benchmark performance across splits and Confirmatory H1 forest plot.

Implements ``Fig. 2`` from ``.agents/prompts/PHASE_09.md`` §1 (P9-D4 (a)):
- Panel (a) [Descriptive]: RMSE_log(dbar) by model x split
  (S-ID, S-LRO, S-TAIL-DA, S-TAIL-SO, S-IND).
- Panel (b) [Descriptive]: RMSE_log(A) and RMSE_log(W) by model x split.
- Panel (c) [Confirmatory]: H1 forest plot: Delta_dbar and Delta_A (M5 - M*) with 95% CIs on
  S-LRO (n=3,988) and S-IND (131 locations / 18 rivers), zero line and row-2 margin +0.05,
  and the verbatim §4.7 row 4 statement.
- Panel (d) [Descriptive]: Predicted vs observed dbar (log-log, 1:1 line) for M5 and M2-XGB
  on S-LRO out-of-fold (n=3,988 sites).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    VerifiedBundle,
)
from annxs_figures.style import (
    ASCE_DOUBLE_COL_IN,
    MODEL_MARKERS,
    MODEL_PALETTE,
    OKABE_ITO,
    SPLIT_PALETTE,
    AnnotationRegistry,
    apply_asce_style,
    label_panel,
    save_figure_bundle,
)
from annxs_figures.tables import (
    CORE_FAMILIES,
    TIER_CONFIRMATORY,
    TIER_DESCRIPTIVE,
)

PRIMARY_SPLITS: Final[tuple[str, ...]] = CORE_FAMILIES

DBAR_MODELS: Final[tuple[str, ...]] = (
    "M-null",
    "M0",
    "M1",
    "M2-MLP",
    "M2-XGB",
    "M3",
    "M4",
    "M5",
    "M6-NN",
    "M6-XGB",
)
AREA_MODELS: Final[tuple[str, ...]] = (
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
)


def _lookup_metric(
    mat: pd.DataFrame,
    *,
    model: str,
    family: str,
    target: str,
    metric: str = "rmse_log",
) -> float:
    sub = mat.loc[
        (mat["spec"] == model)
        & (mat["analysis"] == "primary")
        & (mat["family"] == family)
        & (mat["target"] == target)
        & (mat["stratum_kind"] == "all")
        & (mat["metric"] == metric),
        "value",
    ]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0])


def _draw_panel_a_dbar(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (a): RMSE_log(dbar) across models and the 5 primary splits."""
    mat = bundle.matrix_df
    x_idx = np.arange(len(DBAR_MODELS), dtype=np.float64)
    offsets = np.linspace(-0.28, 0.28, len(PRIMARY_SPLITS))
    split_markers = {"S-ID": "o", "S-LRO": "s", "S-TAIL-DA": "^", "S-TAIL-SO": "v", "S-IND": "D"}

    for s_idx, fam in enumerate(PRIMARY_SPLITS):
        vals = [
            _lookup_metric(mat, model=m, family=fam, target="dbar", metric="rmse_log")
            for m in DBAR_MODELS
        ]
        ax.plot(
            x_idx + offsets[s_idx],
            vals,
            marker=split_markers[fam],
            color=SPLIT_PALETTE[fam],
            linestyle="-",
            linewidth=0.85,
            markersize=4.0,
            alpha=0.90,
            label=fam,
        )

    # Register and annotate M2-XGB (M*) and M5 on S-ID, S-LRO, S-IND
    for m_name in ("M2-XGB", "M5"):
        for fam in ("S-ID", "S-LRO", "S-IND"):
            val = _lookup_metric(mat, model=m_name, family=fam, target="dbar", metric="rmse_log")
            registry.register(
                figure_id="fig2",
                panel_id="a",
                label=f"{m_name}_{fam}_dbar_rmse_log",
                raw_value=val,
                fmt_kind="dec4",
                source_file="reports/results/event3_matrix.parquet",
                source_key=f"{m_name}|primary|{fam}|dbar|all|rmse_log",
                evidence_tier=TIER_DESCRIPTIVE,
            )

    m2_lro = registry.register(
        figure_id="fig2",
        panel_id="a",
        label="callout_M2-XGB_S-LRO_dbar",
        raw_value=_lookup_metric(mat, model="M2-XGB", family="S-LRO", target="dbar"),
        fmt_kind="dec4",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M2-XGB|primary|S-LRO|dbar|all|rmse_log",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m5_lro = registry.register(
        figure_id="fig2",
        panel_id="a",
        label="callout_M5_S-LRO_dbar",
        raw_value=_lookup_metric(mat, model="M5", family="S-LRO", target="dbar"),
        fmt_kind="dec4",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M5|primary|S-LRO|dbar|all|rmse_log",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.text(
        0.02,
        0.97,
        (
            f"S-LRO dbar: M*={m2_lro}, M5={m5_lro}\n"
            f"S-IND: {S_IND_PANEL_NOTE}\n"
            f'S-ID: "{EVENT1_AUDIT_LABEL}"'
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.8,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(DBAR_MODELS, rotation=30, ha="right", fontsize=6.4)
    ax.set_ylabel("RMSE_log(dbar) [ln m]")
    ax.set_ylim(0.38, 1.48)
    ax.legend(
        loc="upper right",
        fontsize=5.0,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        handlelength=1.1,
    )
    label_panel(
        ax,
        "a",
        "Mean Depth (dbar) RMSE_log",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_b_area_width(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (b): RMSE_log(A) and RMSE_log(W) on S-ID, S-LRO, and S-IND."""
    mat = bundle.matrix_df
    x_idx = np.arange(len(AREA_MODELS), dtype=np.float64)

    for fam, marker, col in (
        ("S-ID", "o", SPLIT_PALETTE["S-ID"]),
        ("S-LRO", "s", SPLIT_PALETTE["S-LRO"]),
        ("S-IND", "D", SPLIT_PALETTE["S-IND"]),
    ):
        a_vals = [_lookup_metric(mat, model=m, family=fam, target="A") for m in AREA_MODELS]
        w_vals = [_lookup_metric(mat, model=m, family=fam, target="W") for m in AREA_MODELS]
        ax.plot(
            x_idx - 0.12,
            a_vals,
            marker=marker,
            color=col,
            linestyle="-",
            linewidth=0.95,
            markersize=4.0,
            label=f"A ({fam})",
        )
        ax.plot(
            x_idx + 0.12,
            w_vals,
            marker=marker,
            color=col,
            linestyle="--",
            linewidth=0.80,
            markersize=3.6,
            markerfacecolor="white",
            label=f"W ({fam})",
        )

    m2_a_lro = registry.register(
        figure_id="fig2",
        panel_id="b",
        label="M2-XGB_S-LRO_A_rmse_log",
        raw_value=_lookup_metric(mat, model="M2-XGB", family="S-LRO", target="A"),
        fmt_kind="dec4",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M2-XGB|primary|S-LRO|A|all|rmse_log",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m5_a_lro = registry.register(
        figure_id="fig2",
        panel_id="b",
        label="M5_S-LRO_A_rmse_log",
        raw_value=_lookup_metric(mat, model="M5", family="S-LRO", target="A"),
        fmt_kind="dec4",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M5|primary|S-LRO|A|all|rmse_log",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m2_w_lro = registry.register(
        figure_id="fig2",
        panel_id="b",
        label="M2-XGB_S-LRO_W_rmse_log",
        raw_value=_lookup_metric(mat, model="M2-XGB", family="S-LRO", target="W"),
        fmt_kind="dec4",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M2-XGB|primary|S-LRO|W|all|rmse_log",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m5_w_lro = registry.register(
        figure_id="fig2",
        panel_id="b",
        label="M5_S-LRO_W_rmse_log",
        raw_value=_lookup_metric(mat, model="M5", family="S-LRO", target="W"),
        fmt_kind="dec4",
        source_file="reports/results/event3_matrix.parquet",
        source_key="M5|primary|S-LRO|W|all|rmse_log",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.text(
        0.02,
        0.97,
        (
            f"S-LRO A: M*={m2_a_lro}, M5={m5_a_lro}\n"
            f"S-LRO W: M*={m2_w_lro}, M5={m5_w_lro}\n"
            f"S-IND: {S_IND_PANEL_NOTE}\n(A_J: -81 rows)"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.9,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(AREA_MODELS, rotation=30, ha="right", fontsize=6.4)
    ax.set_ylabel("RMSE_log(A, W) [ln units]")
    ax.set_ylim(0.32, 2.02)
    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=5.0,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        handlelength=1.2,
        columnspacing=0.5,
    )
    label_panel(
        ax,
        "b",
        "Area (A, solid) & Width (W, dashed)",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_c_h1_forest(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (c): Confirmatory H1 forest plot (M5 - M*, 95% CIs) + §4.7 row 4 verdict."""
    h1 = bundle.hypotheses["H1"]
    items = [
        ("S-LRO", "dbar", "S-LRO dbar (n=3,988)", h1["S-LRO"]["dbar"], SPLIT_PALETTE["S-LRO"]),
        ("S-LRO", "A", "S-LRO A (n=3,988)", h1["S-LRO"]["A"], SPLIT_PALETTE["S-LRO"]),
        (
            "S-IND",
            "dbar",
            f"S-IND dbar\n({S_IND_PANEL_NOTE})",
            h1["S-IND"]["dbar"],
            SPLIT_PALETTE["S-IND"],
        ),
        (
            "S-IND",
            "A",
            f"S-IND A\n({S_IND_PANEL_NOTE})",
            h1["S-IND"]["A"],
            SPLIT_PALETTE["S-IND"],
        ),
    ]

    y_pos = np.arange(len(items), dtype=np.float64)[::-1]
    ax.axvline(
        0.0, color="#111111", linestyle="-", linewidth=0.9, zorder=2, label="Delta = 0 (parity)"
    )
    ax.axvline(
        0.05,
        color=OKABE_ITO["reddish_purple"],
        linestyle="--",
        linewidth=0.9,
        zorder=2,
        label="+0.05 (§4.7 row-2 margin)",
    )

    y_labels: list[str] = []
    for idx, (fam, tgt, y_lbl, blk, col) in enumerate(items):
        y = float(y_pos[idx])
        est = float(blk["estimate"])
        lo = float(blk["lo"])
        hi = float(blk["hi"])
        est_s = registry.register(
            figure_id="fig2",
            panel_id="c",
            label=f"H1_{fam}_{tgt}_estimate",
            raw_value=est,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"H1.{fam}.{tgt}.estimate",
            evidence_tier=TIER_CONFIRMATORY,
        )
        lo_s = registry.register(
            figure_id="fig2",
            panel_id="c",
            label=f"H1_{fam}_{tgt}_lo",
            raw_value=lo,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"H1.{fam}.{tgt}.lo",
            evidence_tier=TIER_CONFIRMATORY,
        )
        hi_s = registry.register(
            figure_id="fig2",
            panel_id="c",
            label=f"H1_{fam}_{tgt}_hi",
            raw_value=hi,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"H1.{fam}.{tgt}.hi",
            evidence_tier=TIER_CONFIRMATORY,
        )

        ax.errorbar(
            est,
            y,
            xerr=[[est - lo], [hi - est]],
            fmt="o" if tgt == "dbar" else "s",
            color=col,
            ecolor=col,
            elinewidth=1.4,
            capsize=3.2,
            markersize=4.6,
            zorder=4,
        )
        ax.text(
            est,
            y + 0.18,
            f"Delta_{tgt} = {est_s} [{lo_s}, {hi_s}]",
            ha="center",
            va="bottom",
            fontsize=5.6,
            fontweight="bold",
            color="#111111",
        )
        y_labels.append(y_lbl)

    o47 = bundle.outcome_4_7
    h1_verdict = "Falsified" if h1["decision"]["falsified"] else "Supported"
    ax.text(
        0.02,
        0.03,
        f"H1 Verdict: {h1_verdict} (M* = M2-XGB)\n§4.7 Row {o47.row}: {o47.outcome}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=5.3,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "#FFF8F6",
            "edgecolor": "#A50E0E",
            "alpha": 0.94,
        },
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=6.0)
    ax.set_xlabel("Delta = RMSE_log(M5) - RMSE_log(M*)  (< 0 favours M5)")
    ax.set_xlim(-0.36, 0.48)
    ax.set_ylim(-0.90, 3.85)
    ax.legend(
        loc="upper left",
        fontsize=5.4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
    )
    label_panel(
        ax,
        "c",
        "H1 Forest Plot: M5 vs M* (M2-XGB)",
        TIER_CONFIRMATORY,
    )


def _draw_panel_d_scatter(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (d): Predicted vs observed dbar (log-log, 1:1 line) on S-LRO OOF."""
    site_table = bundle.load_frozen_table("data/processed/site_table.parquet")
    primary_sites = site_table.loc[
        site_table["variant"].eq("primary") & ~site_table["F_DA"].astype(bool),
        ["site_no", "dbar"],
    ].copy()
    primary_sites["site_no"] = primary_sites["site_no"].astype(str)

    for m_name, col, marker, alpha, zord in (
        ("M2-XGB", MODEL_PALETTE["M2-XGB"], MODEL_MARKERS["M2-XGB"], 0.35, 3),
        ("M5", MODEL_PALETTE["M5"], MODEL_MARKERS["M5"], 0.35, 4),
    ):
        df_p = bundle.load_sealed_prediction(f"{m_name}.parquet")
        sub = df_p.loc[
            df_p["model"].eq(m_name)
            & df_p["ablation"].eq("")
            & df_p["analysis"].eq("primary")
            & df_p["family"].eq("S-LRO")
            & df_p["unit"].eq("site"),
            ["site_no", "mu_dbar"],
        ].copy()
        sub["site_no"] = sub["site_no"].astype(str)
        merged = primary_sites.merge(sub, on="site_no", how="inner")
        obs_dbar = merged["dbar"].to_numpy(dtype=np.float64)
        pred_dbar = np.exp(merged["mu_dbar"].to_numpy(dtype=np.float64))

        rmse_v = _lookup_metric(
            bundle.matrix_df, model=m_name, family="S-LRO", target="dbar", metric="rmse_log"
        )
        bias_v = _lookup_metric(
            bundle.matrix_df, model=m_name, family="S-LRO", target="dbar", metric="bias_log"
        )
        rmse_s = registry.register(
            figure_id="fig2",
            panel_id="d",
            label=f"{m_name}_S-LRO_dbar_rmse_log",
            raw_value=rmse_v,
            fmt_kind="dec4",
            source_file="reports/results/event3_matrix.parquet",
            source_key=f"{m_name}|primary|S-LRO|dbar|all|rmse_log",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        bias_s = registry.register(
            figure_id="fig2",
            panel_id="d",
            label=f"{m_name}_S-LRO_dbar_bias_log",
            raw_value=bias_v,
            fmt_kind="dec4",
            source_file="reports/results/event3_matrix.parquet",
            source_key=f"{m_name}|primary|S-LRO|dbar|all|bias_log",
            evidence_tier=TIER_DESCRIPTIVE,
        )

        ax.scatter(
            obs_dbar,
            pred_dbar,
            s=4.5,
            c=col,
            marker=marker,
            alpha=alpha,
            edgecolors="none",
            zorder=zord,
            rasterized=True,
            label=f"{m_name}: RMSE_log={rmse_s}, B_log={bias_s}",
        )

    grid = np.logspace(-2.0, 2.0, 100)
    ax.plot(grid, grid, color="#111111", linestyle="-", linewidth=0.95, zorder=5, label="1:1 line")
    ax.plot(
        grid,
        2.0 * grid,
        color="#666666",
        linestyle=":",
        linewidth=0.7,
        zorder=2,
        label="x2 / /2 band",
    )
    ax.plot(grid, 0.5 * grid, color="#666666", linestyle=":", linewidth=0.7, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 50.0)
    ax.set_ylim(0.02, 50.0)
    ax.set_xlabel("Observed Mean Depth dbar [m] (S-LRO, n = 3,988)")
    ax.set_ylabel("Predicted Mean Depth dbar_hat [m]")
    ax.legend(
        loc="upper left",
        fontsize=5.6,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        markerscale=2.0,
    )
    label_panel(
        ax,
        "d",
        "Predicted vs Observed dbar (S-LRO)",
        TIER_DESCRIPTIVE,
    )


def generate_fig2(
    bundle: VerifiedBundle,
    repo_root: Path,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Build and save Fig. 2 (Benchmark performance and Confirmatory H1 forest plot)."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 6.6))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        hspace=0.40,
        wspace=0.32,
        left=0.11,
        right=0.98,
        top=0.94,
        bottom=0.13,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _draw_panel_a_dbar(ax_a, bundle, registry)
    _draw_panel_b_area_width(ax_b, bundle, registry)
    _draw_panel_c_h1_forest(ax_c, bundle, registry)
    _draw_panel_d_scatter(ax_d, bundle, registry)

    o47 = bundle.outcome_4_7
    fig.text(
        0.50,
        0.012,
        (
            f"Pre-Registered §4.7 Row {o47.row} Decision Outcome [Confirmatory]: "
            f'"{o47.outcome}" | Venue: "{o47.venue}"\n'
            f"{o47.statement}"
        ),
        ha="center",
        va="bottom",
        fontsize=5.3,
        fontweight="bold",
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#FFF8F6",
            "edgecolor": "#A50E0E",
            "linewidth": 0.75,
        },
    )
    del repo_root

    return save_figure_bundle(fig, figures_dir, "fig2_benchmark_performance")
