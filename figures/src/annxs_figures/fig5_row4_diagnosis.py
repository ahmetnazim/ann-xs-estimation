"""Figure 5: Pre-registered §4.7 row-4 diagnosis (ablations, SA1/SA5, S2 contrasts, and §5.1).

Implements ``Fig. 5`` from ``.agents/prompts/PHASE_09.md`` §1 (P9-D4 (a)):
- Panel (a) [Descriptive]: M5 S-ID and S-LRO RMSE_log(dbar) under the pre-registered ablations
  A1 (kappa=1), wA1 (weak I_blind), A2 (drop LayerNorm), A3-SP/A3-SR (Strickler n), M4 (soft),
  SA1 (Q_C), SA5 (drop F_S), and A4 (oracle Q_meas), side-by-side with M* (M2-XGB) under the
  same sensitivity/ablation variants, plus the docs/2 §5.1 Manning log-elasticities
  (d ln dbar / d ln S = -3/10; d/d ln W, d/d ln n = -/+ 3/5 at eta -> 0).
- Panel (b) [Secondary (98.75% CIs) / Descriptive (kappa quintiles)]: S2 stepwise physics
  contrasts s2_M4_vs_M5 and s2_M5_vs_M6NN with 98.75% Bonferroni bootstrap CIs (never mixed with
  95% CIs), not_nested / far strata, and descriptive kappa-quintile profiles (Q1-Q5).
- Panel (c) [Descriptive]: Oracle-Q (A4) gain on dbar and A plus implied-velocity V_hat = Q / A_hat
  error decomposition (rmse_log vs oracle_rmse_log, share_Q vs share_A) across M5 and M2-XGB.
- Panel (d) [Descriptive]: Pre-registered ablation H4 parameter plausibility comparison on S-ID
  (pi_lit, rho_S(n_hat, D50), rho_S(n_hat, S)) across M5 primary, A1, A1 Prop 7.6 mapped, wA1,
  A2, A3-SP, A3-SR, and A4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
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


def _lookup_spec_metric(
    matrix_df: pd.DataFrame,
    *,
    spec: str,
    analysis: str,
    family: str,
    target: str = "dbar",
    metric: str = "rmse_log",
) -> float:
    sub = matrix_df.loc[
        (matrix_df["spec"] == spec)
        & (matrix_df["analysis"] == analysis)
        & (matrix_df["family"] == family)
        & (matrix_df["target"] == target)
        & (matrix_df["stratum_kind"] == "all")
        & (matrix_df["stratum"] == "all")
        & (matrix_df["metric"] == metric)
    ]
    if len(sub) != 1:
        raise KeyError(
            f"Expected 1 row for {spec}|{analysis}|{family}|{target}|{metric}, got {len(sub)}"
        )
    return float(sub["value"].iloc[0])


def _draw_panel_a_ablations(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (a): M5 vs M2-XGB S-ID/S-LRO RMSE_log(dbar) under §4.7 row-4 ablations & SAs."""
    mat = bundle.matrix_df
    sa1_note = str(bundle.descriptive["sensitivity_status"]["M2-XGB"]["SA1"])

    variants = [
        ("Primary", "M5", "primary", "M2-XGB", "primary"),
        ("A1 (kappa=1)", "M5_A1", "primary", None, None),
        ("wA1 (weak I_b)", "M5_wA1", "primary", None, None),
        ("A2 (-LayerNorm)", "M5_A2", "primary", None, None),
        ("A3-SP (n(D50))", "M5_A3-SP", "primary", None, None),
        ("A3-SR (n(D50,S))", "M5_A3-SR", "primary", None, None),
        ("M4 (Soft PINN)", "M4", "primary", None, None),
        ("SA1 (Q_C forcing)", "M5", "SA1", None, None),
        ("SA5 (drop F_S)", "M5", "SA5", "M2-XGB", "SA5"),
        ("A4 (oracle Q)", "M5_A4", "primary", "M2-XGB_A4", "primary"),
    ]

    x_idx = np.arange(len(variants), dtype=np.float64)
    width = 0.20

    m5_sid: list[float] = []
    m5_lro: list[float] = []
    for v_lbl, m5_spec, m5_an, m2_spec, m2_an in variants:
        v_sid = _lookup_spec_metric(mat, spec=m5_spec, analysis=m5_an, family="S-ID")
        v_lro = _lookup_spec_metric(mat, spec=m5_spec, analysis=m5_an, family="S-LRO")
        m5_sid.append(v_sid)
        m5_lro.append(v_lro)
        registry.register(
            figure_id="fig5",
            panel_id="a",
            label=f"{m5_spec}_{m5_an}_S-ID_dbar_rmse_log",
            raw_value=v_sid,
            fmt_kind="dec4",
            source_file="reports/results/event3_matrix.parquet",
            source_key=f"{m5_spec}|{m5_an}|S-ID|dbar|all|rmse_log",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        registry.register(
            figure_id="fig5",
            panel_id="a",
            label=f"{m5_spec}_{m5_an}_S-LRO_dbar_rmse_log",
            raw_value=v_lro,
            fmt_kind="dec4",
            source_file="reports/results/event3_matrix.parquet",
            source_key=f"{m5_spec}|{m5_an}|S-LRO|dbar|all|rmse_log",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        if m2_spec is not None and m2_an is not None:
            m2_v_sid = _lookup_spec_metric(mat, spec=m2_spec, analysis=m2_an, family="S-ID")
            m2_v_lro = _lookup_spec_metric(mat, spec=m2_spec, analysis=m2_an, family="S-LRO")
            registry.register(
                figure_id="fig5",
                panel_id="a",
                label=f"{m2_spec}_{m2_an}_S-ID_dbar_rmse_log",
                raw_value=m2_v_sid,
                fmt_kind="dec4",
                source_file="reports/results/event3_matrix.parquet",
                source_key=f"{m2_spec}|{m2_an}|S-ID|dbar|all|rmse_log",
                evidence_tier=TIER_DESCRIPTIVE,
            )
            registry.register(
                figure_id="fig5",
                panel_id="a",
                label=f"{m2_spec}_{m2_an}_S-LRO_dbar_rmse_log",
                raw_value=m2_v_lro,
                fmt_kind="dec4",
                source_file="reports/results/event3_matrix.parquet",
                source_key=f"{m2_spec}|{m2_an}|S-LRO|dbar|all|rmse_log",
                evidence_tier=TIER_DESCRIPTIVE,
            )
            v_pos = float(x_idx[[v[0] for v in variants].index(v_lbl)])
            ax.bar(
                v_pos + 0.5 * width,
                m2_v_sid,
                width=width,
                color=MODEL_PALETTE["M2-XGB"],
                edgecolor="#222222",
                linewidth=0.45,
                alpha=0.70,
                label="M* (M2-XGB) S-ID" if v_lbl == "Primary" else None,
            )
            ax.bar(
                v_pos + 1.5 * width,
                m2_v_lro,
                width=width,
                color=OKABE_ITO["sky_blue"],
                edgecolor="#222222",
                linewidth=0.45,
                alpha=0.92,
                label="M* (M2-XGB) S-LRO" if v_lbl == "Primary" else None,
            )

    ax.bar(
        x_idx - 1.5 * width,
        m5_sid,
        width=width,
        color=MODEL_PALETTE["M5"],
        edgecolor="#222222",
        linewidth=0.45,
        alpha=0.72,
        label="M5 / M4 S-ID",
    )
    ax.bar(
        x_idx - 0.5 * width,
        m5_lro,
        width=width,
        color=OKABE_ITO["orange"],
        edgecolor="#222222",
        linewidth=0.45,
        alpha=0.92,
        label="M5 / M4 S-LRO",
    )

    ax.text(
        0.02,
        0.97,
        (
            "docs/2 §5.1 wide-channel elasticities (eta -> 0): "
            "d ln dbar / d ln S = -3/10 (-0.3000);\n"
            "d ln dbar / d ln W = -3/5 (-0.6000); "
            "d ln dbar / d ln n = +3/5 (+0.6000); "
            "10x slope error -> 1.995x dbar\n"
            f"M2-XGB SA1: {sa1_note}\n"
            f'S-ID: "{EVENT1_AUDIT_LABEL}"'
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.6,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.94,
        },
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels([v[0] for v in variants], rotation=28, ha="right", fontsize=5.5)
    ax.set_ylabel("RMSE_log(dbar) [ln m]")
    ax.set_ylim(0.15, 1.95)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.74),
        ncol=2,
        fontsize=4.7,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        columnspacing=0.5,
    )
    label_panel(
        ax,
        "a",
        "Row-4 Ablations & Sensitivity (dbar)",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_b_s2(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (b): Stepwise physics contrasts S2 (98.75% Bonferroni CIs) & kappa quintiles."""
    s2 = bundle.hypotheses["S2"]
    fnn = bundle.hypotheses["far_not_nested_strata"]
    far_reason = str(fnn["far"]["S2"]["not_computed"])

    rows = [
        (
            "M4 - M5\n[S-ID, 98.75%]",
            s2["S-ID"]["s2_M4_vs_M5"],
            "S2.S-ID.s2_M4_vs_M5",
            MODEL_PALETTE["M4"],
        ),
        (
            "M4 - M5\n[S-LRO, 98.75%]",
            s2["S-LRO"]["s2_M4_vs_M5"],
            "S2.S-LRO.s2_M4_vs_M5",
            MODEL_PALETTE["M4"],
        ),
        (
            "M4 - M5\n[not_nested, 98.75%]",
            fnn["not_nested"]["S2"]["s2_M4_vs_M5"],
            "far_not_nested_strata.not_nested.S2.s2_M4_vs_M5",
            MODEL_PALETTE["M4"],
        ),
        (
            "M5 - M6-NN\n[S-ID, 98.75%]",
            s2["S-ID"]["s2_M5_vs_M6NN"],
            "S2.S-ID.s2_M5_vs_M6NN",
            MODEL_PALETTE["M5"],
        ),
        (
            "M5 - M6-NN\n[S-LRO, 98.75%]",
            s2["S-LRO"]["s2_M5_vs_M6NN"],
            "S2.S-LRO.s2_M5_vs_M6NN",
            MODEL_PALETTE["M5"],
        ),
        (
            "M5 - M6-NN\n[not_nested, 98.75%]",
            fnn["not_nested"]["S2"]["s2_M5_vs_M6NN"],
            "far_not_nested_strata.not_nested.S2.s2_M5_vs_M6NN",
            MODEL_PALETTE["M5"],
        ),
    ]

    y_pos = np.arange(len(rows), dtype=np.float64)[::-1]
    ax.axvline(0.0, color="#111111", linestyle="-", linewidth=0.85, zorder=2)

    for idx, (_, blk, skey, col) in enumerate(rows):
        y = float(y_pos[idx])
        est = float(blk["estimate"])
        lo = float(blk["lo"])
        hi = float(blk["hi"])
        est_s = registry.register(
            figure_id="fig5",
            panel_id="b",
            label=f"{skey}.estimate",
            raw_value=est,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"{skey}.estimate",
            evidence_tier=TIER_SECONDARY,
        )
        lo_s = registry.register(
            figure_id="fig5",
            panel_id="b",
            label=f"{skey}.lo",
            raw_value=lo,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"{skey}.lo",
            evidence_tier=TIER_SECONDARY,
        )
        hi_s = registry.register(
            figure_id="fig5",
            panel_id="b",
            label=f"{skey}.hi",
            raw_value=hi,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"{skey}.hi",
            evidence_tier=TIER_SECONDARY,
        )
        ax.errorbar(
            est,
            y,
            xerr=[[est - lo], [hi - est]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=1.25,
            capsize=2.8,
            markersize=4.2,
            zorder=4,
        )
        ax.text(
            hi + 0.004,
            y,
            f"{est_s} [{lo_s}, {hi_s}]",
            va="center",
            ha="left",
            fontsize=5.1,
            fontweight="bold",
            color="#111111",
        )

    # Register S-LRO kappa Q1 and Q5 descriptive contrasts
    kq_lro = s2["kappa_quintiles_descriptive"]["S-LRO"]
    q1_m4 = registry.register(
        figure_id="fig5",
        panel_id="b",
        label="S2_S-LRO_kappa_Q1_M4_vs_M5",
        raw_value=float(kq_lro["Q1"]["s2_M4_vs_M5"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="S2.kappa_quintiles_descriptive.S-LRO.Q1.s2_M4_vs_M5",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    q5_m4 = registry.register(
        figure_id="fig5",
        panel_id="b",
        label="S2_S-LRO_kappa_Q5_M4_vs_M5",
        raw_value=float(kq_lro["Q5"]["s2_M4_vs_M5"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="S2.kappa_quintiles_descriptive.S-LRO.Q5.s2_M4_vs_M5",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.text(
        0.02,
        0.02,
        (
            f"Bonferroni level = 98.75% (never mixed with 95% CIs)\n"
            f"S-LRO kappa quintiles (M4-M5): Q1 (narrow)={q1_m4}, Q5 (wide)={q5_m4}\n"
            f"S-ID far stratum: not computed ({far_reason})"
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=4.7,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.94,
        },
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.3)
    ax.set_xlabel("Stepwise Physics Contrast Delta RMSE_log(dbar) [ln m] (98.75% CI)")
    ax.set_xlim(-0.135, 0.095)
    ax.set_ylim(-1.35, 5.65)
    label_panel(
        ax,
        "b",
        "S2 Stepwise Contrasts (98.75% CI)",
        f"{TIER_SECONDARY} / {TIER_DESCRIPTIVE}",
    )


def _draw_panel_c_oracle_implied_v(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (c): Oracle-Q (A4) gain and implied-V error decomposition (§4.3.2)."""
    cons_models = bundle.consistency["models"]
    mat = bundle.matrix_df

    groups = [
        "M5 dbar\n(S-LRO)",
        "M2-XGB dbar\n(S-LRO)",
        "M5 V_hat\n(S-LRO)",
        "M2-XGB V_hat\n(S-LRO)",
    ]
    x_pos = np.arange(len(groups), dtype=np.float64)
    width = 0.34

    m5_dbar_er = _lookup_spec_metric(mat, spec="M5", analysis="primary", family="S-LRO")
    m5_dbar_or = _lookup_spec_metric(mat, spec="M5_A4", analysis="primary", family="S-LRO")
    m2_dbar_er = _lookup_spec_metric(mat, spec="M2-XGB", analysis="primary", family="S-LRO")
    m2_dbar_or = _lookup_spec_metric(mat, spec="M2-XGB_A4", analysis="primary", family="S-LRO")

    m5_v = cons_models["M5"]["S-LRO"]["implied_V"]
    m2_v = cons_models["M2-XGB"]["S-LRO"]["implied_V"]

    m5_v_er = float(m5_v["rmse_log"])
    m5_v_or = float(m5_v["oracle_rmse_log"])
    m2_v_er = float(m2_v["rmse_log"])
    m2_v_or = float(m2_v["oracle_rmse_log"])

    for lbl_k, val_k, skey in (
        ("M5_S-LRO_implied_V_rmse_log", m5_v_er, "models.M5.S-LRO.implied_V.rmse_log"),
        (
            "M5_S-LRO_implied_V_oracle_rmse_log",
            m5_v_or,
            "models.M5.S-LRO.implied_V.oracle_rmse_log",
        ),
        ("M2-XGB_S-LRO_implied_V_rmse_log", m2_v_er, "models.M2-XGB.S-LRO.implied_V.rmse_log"),
        (
            "M2-XGB_S-LRO_implied_V_oracle_rmse_log",
            m2_v_or,
            "models.M2-XGB.S-LRO.implied_V.oracle_rmse_log",
        ),
    ):
        registry.register(
            figure_id="fig5",
            panel_id="c",
            label=lbl_k,
            raw_value=val_k,
            fmt_kind="dec4",
            source_file="reports/results/event3_consistency.json",
            source_key=skey,
            evidence_tier=TIER_DESCRIPTIVE,
        )

    m5_sh_q = registry.register(
        figure_id="fig5",
        panel_id="c",
        label="M5_S-LRO_implied_V_share_Q",
        raw_value=float(m5_v["share_Q"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M5.S-LRO.implied_V.share_Q",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m5_sh_a = registry.register(
        figure_id="fig5",
        panel_id="c",
        label="M5_S-LRO_implied_V_share_A",
        raw_value=float(m5_v["share_A"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M5.S-LRO.implied_V.share_A",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m2_sh_q = registry.register(
        figure_id="fig5",
        panel_id="c",
        label="M2-XGB_S-LRO_implied_V_share_Q",
        raw_value=float(m2_v["share_Q"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M2-XGB.S-LRO.implied_V.share_Q",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m2_sh_a = registry.register(
        figure_id="fig5",
        panel_id="c",
        label="M2-XGB_S-LRO_implied_V_share_A",
        raw_value=float(m2_v["share_A"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M2-XGB.S-LRO.implied_V.share_A",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    erom_vals = [m5_dbar_er, m2_dbar_er, m5_v_er, m2_v_er]
    orac_vals = [m5_dbar_or, m2_dbar_or, m5_v_or, m2_v_or]

    ax.bar(
        x_pos - 0.5 * width,
        erom_vals,
        width=width,
        color=MODEL_PALETTE["M5"],
        edgecolor="#222222",
        linewidth=0.45,
        alpha=0.88,
        label="Ungauged Q_E (EROM primary)",
    )
    ax.bar(
        x_pos + 0.5 * width,
        orac_vals,
        width=width,
        color=OKABE_ITO["bluish_green"],
        edgecolor="#222222",
        linewidth=0.45,
        alpha=0.88,
        label="Oracle Q_meas (A4 / oracle V_hat)",
    )

    for idx, (ev, ov) in enumerate(zip(erom_vals, orac_vals, strict=True)):
        ax.text(
            float(x_pos[idx]) - 0.54 * width,
            ev + 0.02,
            f"{ev:.4f}",
            ha="center",
            va="bottom",
            fontsize=4.6,
            fontweight="bold",
        )
        ax.text(
            float(x_pos[idx]) + 0.54 * width,
            ov + 0.02,
            f"{ov:.4f}",
            ha="center",
            va="bottom",
            fontsize=4.6,
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.97,
        (
            f"Implied V_hat = Q / A_hat variance shares on S-LRO (diagnostic, never headline):\n"
            f"M5: share_Q={m5_sh_q}, share_A={m5_sh_a} | "
            f"M2-XGB: share_Q={m2_sh_q}, share_A={m2_sh_a}"
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
    ax.set_xticklabels(groups, fontsize=5.8)
    ax.set_ylabel("RMSE_log [ln units]")
    ax.set_ylim(0.0, 1.65)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.80),
        ncol=2,
        fontsize=4.7,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        columnspacing=0.6,
    )
    label_panel(
        ax,
        "c",
        "Oracle-Q (A4) & Implied-V Decomposition",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_d_ablation_h4(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (d): Pre-registered ablation H4 parameter plausibility points on S-ID."""
    h4_sid = bundle.hypotheses["H4"]["S-ID"]
    ab_h4 = bundle.descriptive["ablations_M5"]

    specs = [
        ("M5 Primary", h4_sid, "H4.S-ID", "reports/results/event3_hypotheses.json"),
        (
            "M5_A1 (raw)",
            ab_h4["M5_A1"]["H4_points_S-ID"],
            "ablations_M5.M5_A1.H4_points_S-ID",
            "reports/results/event3_descriptive.json",
        ),
        (
            "M5_A1 (Prop 7.6)",
            ab_h4["M5_A1"]["H4_points_S-ID_prop_7_6_mapped"],
            "ablations_M5.M5_A1.H4_points_S-ID_prop_7_6_mapped",
            "reports/results/event3_descriptive.json",
        ),
        (
            "M5_wA1",
            ab_h4["M5_wA1"]["H4_points_S-ID"],
            "ablations_M5.M5_wA1.H4_points_S-ID",
            "reports/results/event3_descriptive.json",
        ),
        (
            "M5_A2",
            ab_h4["M5_A2"]["H4_points_S-ID"],
            "ablations_M5.M5_A2.H4_points_S-ID",
            "reports/results/event3_descriptive.json",
        ),
        (
            "M5_A3-SP",
            ab_h4["M5_A3-SP"]["H4_points_S-ID"],
            "ablations_M5.M5_A3-SP.H4_points_S-ID",
            "reports/results/event3_descriptive.json",
        ),
        (
            "M5_A3-SR",
            ab_h4["M5_A3-SR"]["H4_points_S-ID"],
            "ablations_M5.M5_A3-SR.H4_points_S-ID",
            "reports/results/event3_descriptive.json",
        ),
        (
            "M5_A4",
            ab_h4["M5_A4"]["H4_points_S-ID"],
            "ablations_M5.M5_A4.H4_points_S-ID",
            "reports/results/event3_descriptive.json",
        ),
    ]

    x_pos = np.arange(len(specs), dtype=np.float64)
    width = 0.25

    pi_vals: list[float] = []
    d50_vals: list[float] = []
    ns_vals: list[float] = []

    for lbl, blk, skey, sfile in specs:
        pi_v = float(blk["pi_lit"])
        d50_v = float(blk["rho_n_d50"])
        ns_v = float(blk["rho_n_s"])
        pi_vals.append(pi_v)
        d50_vals.append(d50_v)
        ns_vals.append(ns_v)
        tier = TIER_CONFIRMATORY if lbl == "M5 Primary" else TIER_DESCRIPTIVE
        registry.register(
            figure_id="fig5",
            panel_id="d",
            label=f"{skey}.pi_lit",
            raw_value=pi_v,
            fmt_kind="dec4",
            source_file=sfile,
            source_key=f"{skey}.pi_lit",
            evidence_tier=tier,
        )
        registry.register(
            figure_id="fig5",
            panel_id="d",
            label=f"{skey}.rho_n_d50",
            raw_value=d50_v,
            fmt_kind="dec4",
            source_file=sfile,
            source_key=f"{skey}.rho_n_d50",
            evidence_tier=tier,
        )
        registry.register(
            figure_id="fig5",
            panel_id="d",
            label=f"{skey}.rho_n_s",
            raw_value=ns_v,
            fmt_kind="dec4",
            source_file=sfile,
            source_key=f"{skey}.rho_n_s",
            evidence_tier=tier,
        )

    ax.axhline(0.90, color="#666666", linestyle=":", linewidth=0.8, label="H4 pi_lit >= 0.90")
    ax.axhline(0.0, color="#111111", linestyle="-", linewidth=0.75)

    ax.bar(
        x_pos - width,
        pi_vals,
        width=width,
        color=OKABE_ITO["bluish_green"],
        edgecolor="#222222",
        linewidth=0.4,
        alpha=0.88,
        label="pi_lit (N_lit=[0.012, 0.33])",
    )
    ax.bar(
        x_pos,
        d50_vals,
        width=width,
        color=OKABE_ITO["vermilion"],
        edgecolor="#222222",
        linewidth=0.4,
        alpha=0.88,
        label="rho_S(n_hat, D50)",
    )
    ax.bar(
        x_pos + width,
        ns_vals,
        width=width,
        color=OKABE_ITO["blue"],
        edgecolor="#222222",
        linewidth=0.4,
        alpha=0.88,
        label="rho_S(n_hat, S)",
    )

    a3sp_d50 = registry.register(
        figure_id="fig5",
        panel_id="d",
        label="callout_A3-SP_rho_n_d50",
        raw_value=float(ab_h4["M5_A3-SP"]["H4_points_S-ID"]["rho_n_d50"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_descriptive.json",
        source_key="ablations_M5.M5_A3-SP.H4_points_S-ID.rho_n_d50",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    a3sr_d50 = registry.register(
        figure_id="fig5",
        panel_id="d",
        label="callout_A3-SR_rho_n_d50",
        raw_value=float(ab_h4["M5_A3-SR"]["H4_points_S-ID"]["rho_n_d50"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_descriptive.json",
        source_key="ablations_M5.M5_A3-SR.H4_points_S-ID.rho_n_d50",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.text(
        0.02,
        0.97,
        (
            f"A3 grain-roughness prior forces rho_S(n_hat, D50) = {a3sp_d50} (A3-SP), "
            f"{a3sr_d50} (A3-SR)\n"
            f'but degrades dbar RMSE_log (Panel a) | S-ID: "{EVENT1_AUDIT_LABEL}"'
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
    ax.set_xticklabels([s[0] for s in specs], rotation=25, ha="right", fontsize=5.5)
    ax.set_ylabel("Share pi_lit or Rank Correlation rho_S")
    ax.set_ylim(-0.12, 1.50)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.80),
        ncol=2,
        fontsize=4.7,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        columnspacing=0.5,
    )
    label_panel(
        ax,
        "d",
        "Ablation H4 Plausibility on S-ID",
        f"{TIER_CONFIRMATORY} / {TIER_DESCRIPTIVE}",
    )


def generate_fig5(
    bundle: VerifiedBundle,
    repo_root: Path,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Build and save Fig. 5 (Pre-registered §4.7 row-4 diagnosis)."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 6.6))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        hspace=0.44,
        wspace=0.42,
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.13,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _draw_panel_a_ablations(ax_a, bundle, registry)
    _draw_panel_b_s2(ax_b, bundle, registry)
    _draw_panel_c_oracle_implied_v(ax_c, bundle, registry)
    _draw_panel_d_ablation_h4(ax_d, bundle, registry)

    o47 = bundle.outcome_4_7
    fig.text(
        0.50,
        0.012,
        (
            f"Pre-Registered §4.7 Row {o47.row} Decision Outcome [{TIER_CONFIRMATORY}]: "
            f'"{o47.outcome}" | Venue: "{o47.venue}"\n'
            f"{o47.statement}"
        ),
        ha="center",
        va="bottom",
        fontsize=5.4,
        fontweight="bold",
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#FFF8F6",
            "edgecolor": "#A50E0E",
            "alpha": 0.96,
        },
    )
    del repo_root

    return save_figure_bundle(fig, figures_dir, "fig5_row4_diagnosis")
