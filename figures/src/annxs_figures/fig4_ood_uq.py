"""Figure 4: OOD generalization stress test (H3), UQ calibration ([0.85, 0.95] band), and H2 bias.

Implements ``Fig. 4`` from ``.agents/prompts/PHASE_09.md`` §1 (P9-D4 (a)):
- Panel (a) [Confirmatory (H3-LRO, H3-TAIL) / Secondary (H3-TAIL-SO)]: H3 skill loss L_M5, L_M*,
  and Delta = L(M5) - L(M*) with 95% bootstrap CIs across H3-LRO (n=3,988), H3-TAIL (S-TAIL-DA,
  n=678), and H3-TAIL-SO (n=339).
- Panel (b) [Secondary (S-ID dbar band check) / Descriptive]: 90% empirical coverage C_0.10(dbar)
  by split (S-ID, S-LRO, S-TAIL-DA, S-TAIL-SO, S-IND 33 single-COMID rows) and by S-ID DA quintile
  (Q1-Q5) for M5, M2-MLP, and M4, with the pre-registered [0.85, 0.95] acceptance band shaded.
- Panel (c) [Descriptive]: Probability Integral Transform (PIT) 10-bin histograms on S-ID and
  S-LRO for M5, M2-MLP, and M4 (target dbar) vs uniform 0.10 reference.
- Panel (d) [Descriptive]: Closed-form log-space CRPS and mean log-width W_bar_log(dbar) across
  splits for M5, M2-MLP, and M4.
- Panel (e) [Confirmatory (S-ID Q1) / Secondary (S-LRO Q1 & strata) / Descriptive (Q1-Q5)]:
  H2 signed dbar log-bias B_log(dbar) across drainage-area quintiles Q1-Q5 on S-ID and S-LRO for
  M5 vs M* (M2-XGB), with Q1 highlighted and H2 95% CIs + far/not-nested strata annotated.
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

UQ_SPLITS: Final[tuple[str, ...]] = ("S-ID", "S-LRO", "S-TAIL-DA", "S-TAIL-SO", "S-IND")
UQ_MODELS: Final[tuple[str, ...]] = ("M5", "M2-MLP", "M4")
DA_QUINTILES: Final[tuple[str, ...]] = ("1", "2", "3", "4", "5")


def _lookup_da_bias(
    matrix_df: pd.DataFrame,
    *,
    spec: str,
    family: str,
    q_str: str,
) -> float:
    sub = matrix_df.loc[
        (matrix_df["spec"] == spec)
        & (matrix_df["analysis"] == "primary")
        & (matrix_df["family"] == family)
        & (matrix_df["target"] == "dbar")
        & (matrix_df["stratum_kind"] == "da_q")
        & (matrix_df["stratum"] == q_str)
        & (matrix_df["metric"] == "bias_log")
    ]
    if len(sub) != 1:
        raise KeyError(f"Expected 1 bias_log row for {spec}|{family}|da_q={q_str}, got {len(sub)}")
    return float(sub["value"].iloc[0])


def _draw_panel_a_h3(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (a): H3 OOD skill loss L_M5, L_M*, and Delta = L(M5) - L(M*) with 95% CIs."""
    h3 = bundle.hypotheses["H3"]
    h3_verdict = "Falsified" if h3["decision"]["falsified"] else "Supported"

    contrasts = [
        ("H3-LRO", "H3-LRO (n=3,988) [Conf.]", TIER_CONFIRMATORY),
        ("H3-TAIL", "H3-TAIL DA (n=678) [Conf.]", TIER_CONFIRMATORY),
        ("H3-TAIL-SO", "H3-TAIL SO (n=339) [Sec.]", TIER_SECONDARY),
    ]
    y_centers = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    offsets = {"L_M5": 0.22, "L_Mstar": 0.0, "delta": -0.22}
    colors = {
        "L_M5": MODEL_PALETTE["M5"],
        "L_Mstar": MODEL_PALETTE["M2-XGB"],
        "delta": OKABE_ITO["reddish_purple"],
    }
    markers = {"L_M5": "o", "L_Mstar": "s", "delta": "D"}
    labels = {
        "L_M5": "L(M5) = RMSE_OOD - RMSE_ID",
        "L_Mstar": "L(M*) [M2-XGB]",
        "delta": "Delta_H3 = L(M5) - L(M*)",
    }

    ax.axvline(0.0, color="#111111", linestyle="-", linewidth=0.85, zorder=2)

    for c_idx, (c_key, _, tier) in enumerate(contrasts):
        blk = h3[c_key]
        y_c = float(y_centers[c_idx])
        for sub_k in ("L_M5", "L_Mstar", "delta"):
            s_blk = blk[sub_k]
            est = float(s_blk["estimate"])
            lo = float(s_blk["lo"])
            hi = float(s_blk["hi"])
            est_s = registry.register(
                figure_id="fig4",
                panel_id="a",
                label=f"{c_key}_{sub_k}_estimate",
                raw_value=est,
                fmt_kind="dec4",
                source_file="reports/results/event3_hypotheses.json",
                source_key=f"H3.{c_key}.{sub_k}.estimate",
                evidence_tier=tier,
            )
            lo_s = registry.register(
                figure_id="fig4",
                panel_id="a",
                label=f"{c_key}_{sub_k}_lo",
                raw_value=lo,
                fmt_kind="dec4",
                source_file="reports/results/event3_hypotheses.json",
                source_key=f"H3.{c_key}.{sub_k}.lo",
                evidence_tier=tier,
            )
            hi_s = registry.register(
                figure_id="fig4",
                panel_id="a",
                label=f"{c_key}_{sub_k}_hi",
                raw_value=hi,
                fmt_kind="dec4",
                source_file="reports/results/event3_hypotheses.json",
                source_key=f"H3.{c_key}.{sub_k}.hi",
                evidence_tier=tier,
            )
            y_val = y_c + offsets[sub_k]
            ax.errorbar(
                est,
                y_val,
                xerr=[[est - lo], [hi - est]],
                fmt=markers[sub_k],
                color=colors[sub_k],
                ecolor=colors[sub_k],
                elinewidth=1.15,
                capsize=2.5,
                markersize=4.0,
                label=labels[sub_k] if c_idx == 0 else None,
                zorder=4,
            )
            if sub_k == "delta":
                ax.text(
                    est,
                    y_val - 0.14,
                    f"Delta={est_s} [{lo_s}, {hi_s}]",
                    va="top",
                    ha="center",
                    fontsize=5.1,
                    fontweight="bold",
                    color=colors["delta"],
                )

    ax.text(
        0.02,
        0.03,
        f"H3 Verdict: {h3_verdict} (all 95% CIs for Delta_H3 span 0)",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=5.2,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFF8F6",
            "edgecolor": "#A50E0E",
            "alpha": 0.94,
        },
    )

    ax.set_yticks(y_centers)
    ax.set_yticklabels([c[1] for c in contrasts], fontsize=5.8)
    ax.set_xlabel("OOD Skill Loss L or Contrast Delta_H3 [ln m] (95% Bootstrap CI)")
    ax.set_xlim(-0.48, 0.42)
    ax.set_ylim(-0.92, 2.62)
    ax.legend(
        loc="upper left",
        fontsize=4.9,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        handlelength=1.1,
    )
    label_panel(
        ax,
        "a",
        "H3 OOD Skill Loss",
        f"{TIER_CONFIRMATORY} / {TIER_SECONDARY}",
    )


def _draw_panel_b_coverage(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (b): 90% coverage C_0.10(dbar) by split and S-ID DA quintile with [0.85, 0.95]."""
    uq = bundle.uq
    band_lo = float(uq["band"][0])
    band_hi = float(uq["band"][1])

    x_labels = [
        "S-ID",
        "S-LRO",
        "TAIL-DA",
        "TAIL-SO",
        "S-IND\n(n=33)",
        "ID:Q1",
        "ID:Q2",
        "ID:Q3",
        "ID:Q4",
        "ID:Q5",
    ]
    x_pos = np.arange(len(x_labels), dtype=np.float64)

    ax.axhspan(
        band_lo,
        band_hi,
        color=OKABE_ITO["bluish_green"],
        alpha=0.16,
        label="[0.85, 0.95] UQ acceptance band",
    )
    ax.axhline(0.90, color="#111111", linestyle="--", linewidth=0.85, label="Nominal 1-alpha=0.90")
    ax.axvline(4.5, color="#888888", linestyle=":", linewidth=0.8)

    markers = {"M5": "o", "M2-MLP": "s", "M4": "^"}
    offsets = {"M5": -0.15, "M2-MLP": 0.0, "M4": 0.15}

    for m_name in UQ_MODELS:
        spec_key = f"{m_name}|primary"
        cov_splits: list[float] = []
        for fam in UQ_SPLITS:
            ov = uq[spec_key][fam]["dbar"]["overall"]
            c_val = float(ov["coverage"])
            cov_splits.append(c_val)
            tier = TIER_SECONDARY if (m_name == "M5" and fam == "S-ID") else TIER_DESCRIPTIVE
            if m_name in {"M5", "M2-MLP"} and fam in {"S-ID", "S-LRO", "S-IND"}:
                registry.register(
                    figure_id="fig4",
                    panel_id="b",
                    label=f"{m_name}_{fam}_dbar_coverage",
                    raw_value=c_val,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_uq.json",
                    source_key=f"{spec_key}.{fam}.dbar.overall.coverage",
                    evidence_tier=tier,
                )

        cov_daq: list[float] = []
        for q_str in DA_QUINTILES:
            q_blk = uq[spec_key]["S-ID"]["dbar"]["by"]["da_q"][q_str]
            c_q = float(q_blk["coverage"])
            cov_daq.append(c_q)
            if m_name == "M5" and q_str in {"1", "5"}:
                registry.register(
                    figure_id="fig4",
                    panel_id="b",
                    label=f"{m_name}_S-ID_da_q_{q_str}_coverage",
                    raw_value=c_q,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_uq.json",
                    source_key=f"{spec_key}.S-ID.dbar.by.da_q.{q_str}.coverage",
                    evidence_tier=TIER_DESCRIPTIVE,
                )

        y_all = cov_splits + cov_daq
        ax.plot(
            x_pos + offsets[m_name],
            y_all,
            marker=markers[m_name],
            color=MODEL_PALETTE[m_name],
            linestyle="-",
            linewidth=0.9,
            markersize=3.8,
            label=m_name,
        )

    m5_sid_cov = registry.register(
        figure_id="fig4",
        panel_id="b",
        label="callout_M5_S-ID_dbar_coverage",
        raw_value=float(uq["M5|primary"]["S-ID"]["dbar"]["overall"]["coverage"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M5|primary.S-ID.dbar.overall.coverage",
        evidence_tier=TIER_SECONDARY,
    )
    m5_lro_cov = registry.register(
        figure_id="fig4",
        panel_id="b",
        label="callout_M5_S-LRO_dbar_coverage",
        raw_value=float(uq["M5|primary"]["S-LRO"]["dbar"]["overall"]["coverage"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M5|primary.S-LRO.dbar.overall.coverage",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m5_ind_cov = registry.register(
        figure_id="fig4",
        panel_id="b",
        label="callout_M5_S-IND_dbar_coverage",
        raw_value=float(uq["M5|primary"]["S-IND"]["dbar"]["overall"]["coverage"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M5|primary.S-IND.dbar.overall.coverage",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.text(
        0.02,
        0.03,
        (
            f"M5 C_0.10(dbar): S-ID={m5_sid_cov} (in [0.85, 0.95]), "
            f"S-LRO={m5_lro_cov}, S-IND={m5_ind_cov}\n"
            f"S-IND: {S_IND_UQ_NOTE}\n"
            f'S-ID: "{EVENT1_AUDIT_LABEL}"'
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=4.8,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.94,
        },
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=5.6)
    ax.set_ylabel("Empirical 90% Coverage C_0.10(dbar)")
    ax.set_ylim(0.50, 1.09)
    ax.legend(
        loc="upper right",
        ncol=3,
        fontsize=4.7,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        columnspacing=0.45,
        handlelength=1.0,
    )
    label_panel(
        ax,
        "b",
        "90% Coverage ([0.85, 0.95] Band)",
        f"{TIER_SECONDARY} / {TIER_DESCRIPTIVE}",
    )


def _draw_panel_c_pit(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (c): PIT 10-bin relative frequency histograms on S-ID and S-LRO for dbar."""
    uq = bundle.uq
    bin_edges = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    series_specs = [
        ("M5", "S-ID", "-", MODEL_PALETTE["M5"], "M5 S-ID (n=11,964)"),
        ("M5", "S-LRO", "--", MODEL_PALETTE["M5"], "M5 S-LRO (n=3,988)"),
        ("M2-MLP", "S-LRO", "-", MODEL_PALETTE["M2-MLP"], "M2-MLP S-LRO (n=3,988)"),
        ("M4", "S-LRO", ":", MODEL_PALETTE["M4"], "M4 S-LRO (n=3,988)"),
    ]

    ax.axhline(
        0.10,
        color="#111111",
        linestyle="--",
        linewidth=0.9,
        label="Uniform calibration (0.10 / bin)",
    )

    for m_name, fam, ls, col, lbl in series_specs:
        ov = uq[f"{m_name}|primary"][fam]["dbar"]["overall"]
        counts = [int(c) for c in ov["pit_counts"]]
        n_rows = int(ov["n_rows"])
        rel_freq = [c / n_rows for c in counts]
        registry.register(
            figure_id="fig4",
            panel_id="c",
            label=f"{m_name}_{fam}_dbar_pit_bin0",
            raw_value=float(counts[0]),
            fmt_kind="int",
            source_file="reports/results/event3_uq.json",
            source_key=f"{m_name}|primary.{fam}.dbar.overall.pit_counts[0]",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        registry.register(
            figure_id="fig4",
            panel_id="c",
            label=f"{m_name}_{fam}_dbar_pit_bin9",
            raw_value=float(counts[9]),
            fmt_kind="int",
            source_file="reports/results/event3_uq.json",
            source_key=f"{m_name}|primary.{fam}.dbar.overall.pit_counts[9]",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        ax.step(
            bin_centers,
            rel_freq,
            where="mid",
            linestyle=ls,
            linewidth=1.15,
            color=col,
            label=lbl,
        )

    m5_lro_b0 = int(uq["M5|primary"]["S-LRO"]["dbar"]["overall"]["pit_counts"][0])
    m5_lro_b9 = int(uq["M5|primary"]["S-LRO"]["dbar"]["overall"]["pit_counts"][9])
    m2_lro_b0 = int(uq["M2-MLP|primary"]["S-LRO"]["dbar"]["overall"]["pit_counts"][0])
    m2_lro_b9 = int(uq["M2-MLP|primary"]["S-LRO"]["dbar"]["overall"]["pit_counts"][9])

    ax.text(
        0.02,
        0.04,
        (
            f"S-LRO dbar tail bin counts (bin 1, bin 10 of n=3988):\n"
            f"M5: ({m5_lro_b0}, {m5_lro_b9}); M2-MLP: ({m2_lro_b0}, {m2_lro_b9})"
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=4.9,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.94,
        },
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.04, 0.19)
    ax.set_xlabel("PIT Quantile Bin u in [0, 1] (10 Equal-Probability Bins)")
    ax.set_ylabel("Relative Frequency (pit_counts / n_rows)")
    ax.legend(
        loc="upper right",
        fontsize=4.8,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        handlelength=1.2,
    )
    label_panel(
        ax,
        "c",
        "PIT Histograms (dbar, 10 Bins)",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_d_crps_width(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (d): Log-space CRPS and mean log-width W_bar_log(dbar) across splits."""
    uq = bundle.uq
    x_pos = np.arange(len(UQ_SPLITS), dtype=np.float64)
    width = 0.24

    for m_idx, m_name in enumerate(UQ_MODELS):
        spec_key = f"{m_name}|primary"
        crps_vals: list[float] = []
        w_vals: list[float] = []
        for fam in UQ_SPLITS:
            ov = uq[spec_key][fam]["dbar"]["overall"]
            c_v = float(ov["crps"])
            w_v = float(ov["mean_log_width"])
            crps_vals.append(c_v)
            w_vals.append(w_v)
            if m_name in {"M5", "M2-MLP"} and fam in {"S-ID", "S-LRO", "S-IND"}:
                registry.register(
                    figure_id="fig4",
                    panel_id="d",
                    label=f"{m_name}_{fam}_dbar_crps",
                    raw_value=c_v,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_uq.json",
                    source_key=f"{spec_key}.{fam}.dbar.overall.crps",
                    evidence_tier=TIER_DESCRIPTIVE,
                )
                registry.register(
                    figure_id="fig4",
                    panel_id="d",
                    label=f"{m_name}_{fam}_dbar_mean_log_width",
                    raw_value=w_v,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_uq.json",
                    source_key=f"{spec_key}.{fam}.dbar.overall.mean_log_width",
                    evidence_tier=TIER_DESCRIPTIVE,
                )

        ax.bar(
            x_pos + (m_idx - 1.0) * width,
            crps_vals,
            width=width,
            color=MODEL_PALETTE[m_name],
            edgecolor="#222222",
            linewidth=0.45,
            alpha=0.88,
            label=f"{m_name} CRPS",
        )

    m5_lro_crps = registry.register(
        figure_id="fig4",
        panel_id="d",
        label="callout_M5_S-LRO_dbar_crps",
        raw_value=float(uq["M5|primary"]["S-LRO"]["dbar"]["overall"]["crps"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M5|primary.S-LRO.dbar.overall.crps",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m5_lro_w = registry.register(
        figure_id="fig4",
        panel_id="d",
        label="callout_M5_S-LRO_dbar_wlog",
        raw_value=float(uq["M5|primary"]["S-LRO"]["dbar"]["overall"]["mean_log_width"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M5|primary.S-LRO.dbar.overall.mean_log_width",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m2_lro_crps = registry.register(
        figure_id="fig4",
        panel_id="d",
        label="callout_M2-MLP_S-LRO_dbar_crps",
        raw_value=float(uq["M2-MLP|primary"]["S-LRO"]["dbar"]["overall"]["crps"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M2-MLP|primary.S-LRO.dbar.overall.crps",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m2_lro_w = registry.register(
        figure_id="fig4",
        panel_id="d",
        label="callout_M2-MLP_S-LRO_dbar_wlog",
        raw_value=float(uq["M2-MLP|primary"]["S-LRO"]["dbar"]["overall"]["mean_log_width"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_uq.json",
        source_key="M2-MLP|primary.S-LRO.dbar.overall.mean_log_width",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.text(
        0.02,
        0.97,
        (
            f"S-LRO dbar: M5 CRPS={m5_lro_crps} (W_bar_log={m5_lro_w}); "
            f"M2-MLP CRPS={m2_lro_crps} (W_bar_log={m2_lro_w})\n"
            f"S-IND: {S_IND_UQ_NOTE}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.7,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.94,
        },
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(UQ_SPLITS, fontsize=5.8)
    ax.set_ylabel("Closed-Form CRPS(dbar) [ln m]")
    ax.set_ylim(0.0, 0.82)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.79),
        ncol=3,
        fontsize=4.8,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        columnspacing=0.6,
        handlelength=1.1,
    )
    label_panel(
        ax,
        "d",
        "CRPS & Log-Width Summary (dbar)",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_e_h2(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (e): Confirmatory H2 signed dbar bias by DA quintile (Q1 highlighted)."""
    mat = bundle.matrix_df
    h2 = bundle.hypotheses["H2"]
    fnn = bundle.hypotheses["far_not_nested_strata"]
    cuts = bundle.descriptive["da_quintile_cuts_km2"]
    h2_verdict = "Falsified" if h2["decision"]["falsified"] else "Supported"

    q_pos = np.arange(len(DA_QUINTILES), dtype=np.float64)
    q_labels = [
        f"Q1 (n=798)\n<={cuts[0]:.1f} km2",
        f"Q2 (n=797)\n{cuts[0]:.1f}-{cuts[1]:.1f}",
        f"Q3 (n=798)\n{cuts[1]:.1f}-{cuts[2]:.1f}",
        f"Q4 (n=797)\n{cuts[2]:.1f}-{cuts[3]:.1f}",
        f"Q5 (n=798)\n>{cuts[3]:.1f} km2",
    ]

    ax.axvspan(
        -0.38,
        0.38,
        color=OKABE_ITO["yellow"],
        alpha=0.22,
        label="H2 headwater target stratum Q1 (n=798 sites)",
    )
    ax.axhline(0.0, color="#111111", linestyle="-", linewidth=0.85, zorder=2)

    series = [
        ("M5", "S-ID", "o", "-", MODEL_PALETTE["M5"], "M5 B_log(dbar) [S-ID]"),
        ("M2-XGB", "S-ID", "s", "-", MODEL_PALETTE["M2-XGB"], "M* (M2-XGB) B_log(dbar) [S-ID]"),
        ("M5", "S-LRO", "o", "--", MODEL_PALETTE["M5"], "M5 B_log(dbar) [S-LRO]"),
        ("M2-XGB", "S-LRO", "s", "--", MODEL_PALETTE["M2-XGB"], "M* (M2-XGB) B_log(dbar) [S-LRO]"),
    ]
    offsets = [-0.12, -0.04, 0.04, 0.12]

    for idx, (m_name, fam, mk, ls, col, lbl) in enumerate(series):
        vals = [_lookup_da_bias(mat, spec=m_name, family=fam, q_str=q) for q in DA_QUINTILES]
        for q_str, v in zip(DA_QUINTILES, vals, strict=True):
            if q_str in {"1", "5"}:
                registry.register(
                    figure_id="fig4",
                    panel_id="e",
                    label=f"{m_name}_{fam}_da_q_{q_str}_bias_log",
                    raw_value=v,
                    fmt_kind="dec4",
                    source_file="reports/results/event3_matrix.parquet",
                    source_key=f"{m_name}|primary|{fam}|dbar|da_q={q_str}|bias_log",
                    evidence_tier=TIER_DESCRIPTIVE,
                )
        ax.plot(
            q_pos + offsets[idx],
            vals,
            marker=mk,
            linestyle=ls,
            color=col,
            linewidth=1.1,
            markersize=4.2,
            label=lbl,
            zorder=4,
        )

    # Register H2 Confirmatory (S-ID_Q1) and Secondary (S-LRO_Q1_secondary, not_nested) statistics
    sid_h2 = h2["S-ID_Q1"]["h2"]
    sid_est = registry.register(
        figure_id="fig4",
        panel_id="e",
        label="H2_S-ID_Q1_h2_estimate",
        raw_value=float(sid_h2["estimate"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H2.S-ID_Q1.h2.estimate",
        evidence_tier=TIER_CONFIRMATORY,
    )
    sid_lo = registry.register(
        figure_id="fig4",
        panel_id="e",
        label="H2_S-ID_Q1_h2_lo",
        raw_value=float(sid_h2["lo"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H2.S-ID_Q1.h2.lo",
        evidence_tier=TIER_CONFIRMATORY,
    )
    sid_hi = registry.register(
        figure_id="fig4",
        panel_id="e",
        label="H2_S-ID_Q1_h2_hi",
        raw_value=float(sid_h2["hi"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H2.S-ID_Q1.h2.hi",
        evidence_tier=TIER_CONFIRMATORY,
    )

    lro_h2 = h2["S-LRO_Q1_secondary"]["h2"]
    lro_est = registry.register(
        figure_id="fig4",
        panel_id="e",
        label="H2_S-LRO_Q1_h2_estimate",
        raw_value=float(lro_h2["estimate"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H2.S-LRO_Q1_secondary.h2.estimate",
        evidence_tier=TIER_SECONDARY,
    )
    lro_lo = registry.register(
        figure_id="fig4",
        panel_id="e",
        label="H2_S-LRO_Q1_h2_lo",
        raw_value=float(lro_h2["lo"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H2.S-LRO_Q1_secondary.h2.lo",
        evidence_tier=TIER_SECONDARY,
    )
    lro_hi = registry.register(
        figure_id="fig4",
        panel_id="e",
        label="H2_S-LRO_Q1_h2_hi",
        raw_value=float(lro_h2["hi"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H2.S-LRO_Q1_secondary.h2.hi",
        evidence_tier=TIER_SECONDARY,
    )

    nn_h2 = fnn["not_nested"]["H2"]["h2"]
    nn_est = registry.register(
        figure_id="fig4",
        panel_id="e",
        label="H2_not_nested_Q1_h2_estimate",
        raw_value=float(nn_h2["estimate"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="far_not_nested_strata.not_nested.H2.h2.estimate",
        evidence_tier=TIER_SECONDARY,
    )
    far_reason = str(fnn["far"]["H2"]["not_computed"])

    ax.text(
        0.22,
        0.04,
        (
            f"H2 Verdict: {h2_verdict} | h2 = |B_dbar(M5)(Q1)| - |B_dbar(M*)(Q1)| (95% CI): "
            f"S-ID Q1 [Conf.] = {sid_est} [{sid_lo}, {sid_hi}]; "
            f"S-LRO Q1 [Sec.] = {lro_est} [{lro_lo}, {lro_hi}]\n"
            f"S-ID not-nested Q1 [Sec.]: h2 = {nn_est} | "
            f"S-ID far Q1 [Sec.]: not computed ({far_reason}) | "
            f'S-ID: "{EVENT1_AUDIT_LABEL}"'
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=4.9,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFF8F6",
            "edgecolor": "#A50E0E",
            "alpha": 0.94,
        },
    )

    ax.set_xticks(q_pos)
    ax.set_xticklabels(q_labels, fontsize=5.8)
    ax.set_ylabel("Signed Log-Bias B_log(dbar) [ln m]")
    ax.set_ylim(-0.22, 0.14)
    ax.legend(
        loc="upper right",
        ncol=3,
        fontsize=4.9,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        columnspacing=0.6,
    )
    label_panel(
        ax,
        "e",
        "H2 Headwater (Q1) Signed dbar Log-Bias Across Drainage-Area Quintiles",
        f"{TIER_CONFIRMATORY} / {TIER_SECONDARY} / {TIER_DESCRIPTIVE}",
    )


def generate_fig4(
    bundle: VerifiedBundle,
    repo_root: Path,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Build and save Fig. 4 (OOD stress test H3, UQ calibration, and H2 small-watershed bias)."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 7.6))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.05, 1.05, 1.0],
        hspace=0.44,
        wspace=0.34,
        left=0.09,
        right=0.98,
        top=0.95,
        bottom=0.08,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    _draw_panel_a_h3(ax_a, bundle, registry)
    _draw_panel_b_coverage(ax_b, bundle, registry)
    _draw_panel_c_pit(ax_c, bundle, registry)
    _draw_panel_d_crps_width(ax_d, bundle, registry)
    _draw_panel_e_h2(ax_e, bundle, registry)
    del repo_root

    return save_figure_bundle(fig, figures_dir, "fig4_ood_and_uq_calibration")
