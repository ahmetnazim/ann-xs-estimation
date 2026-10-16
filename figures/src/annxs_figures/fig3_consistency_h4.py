"""Figure 3: Physical consistency analogues (§4.3.2) and Confirmatory H4 parameter plausibility.

Implements ``Fig. 3`` from ``.agents/prompts/PHASE_09.md`` §1 (P9-D4 (a)):
- Panel (a) [Descriptive]: Predicted area closure residual eps_A_hat distribution for M2-MLP
  and M2-XGB vs exact closure (eps_A_hat = 0) for hard/physics models and QC R8 observed closure.
- Panel (b) [Descriptive]: Share of predictions with no admissible Manning roughness n_back
  (Prop 6.2 / §7.4) by model and split, plus shape violation share (d_max_hat <= dbar_hat).
- Panel (c) [Descriptive]: Hard-model Picard solver residual (<= tau_l = 1e-10) vs M4 soft-loss
  test-site Manning residual R_M across S-ID, S-LRO, and S-IND.
- Panel (d) [Confirmatory (S-ID) / Descriptive (S-LRO & caveats)]: H4 latent n_hat distribution
  vs N_lit = [0.012, 0.33] and box bounds [0.005, 0.60], chan_mat_cd quantiles, and Spearman
  rank correlations rho_S(n_hat, D50), rho_S(n_hat, S) (caveat C2), rho_S(r_hat, r_obs) with
  95% bootstrap CIs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    VerifiedBundle,
)
from annxs_figures.style import (
    ASCE_DOUBLE_COL_IN,
    MODEL_PALETTE,
    OKABE_ITO,
    SPLIT_PALETTE,
    AnnotationRegistry,
    apply_asce_style,
    label_panel,
    save_figure_bundle,
)
from annxs_figures.tables import TIER_CONFIRMATORY, TIER_DESCRIPTIVE

BACK_CALC_MODELS: Final[tuple[str, ...]] = (
    "M0",
    "M1",
    "M2-MLP",
    "M2-XGB",
    "M3",
    "M5",
)


def _draw_panel_a_closure(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (a): Predicted area closure eps_A_hat = ln(A_hat / (W_hat * dbar_hat))."""
    cons = bundle.consistency
    bins: list[float] = np.linspace(-0.25, 0.25, 81).tolist()

    sid_notes: list[str] = []
    for m_name, col, ls in (
        ("M2-MLP", MODEL_PALETTE["M2-MLP"], "-"),
        ("M2-XGB", MODEL_PALETTE["M2-XGB"], "--"),
    ):
        df_p = bundle.load_sealed_prediction(f"{m_name}.parquet")
        sub = df_p.loc[
            df_p["model"].eq(m_name)
            & df_p["ablation"].eq("")
            & df_p["analysis"].eq("primary")
            & df_p["family"].eq("S-LRO")
            & df_p["unit"].eq("site")
        ]
        eps_a = (
            sub["mu_A"].to_numpy(dtype=np.float64)
            - sub["mu_W"].to_numpy(dtype=np.float64)
            - sub["mu_dbar"].to_numpy(dtype=np.float64)
        )
        eps_clipped = np.clip(eps_a, -0.249, 0.249)

        c_sid = cons["models"][m_name]["S-ID"]["predicted_closure"]
        c_lro = cons["models"][m_name]["S-LRO"]["predicted_closure"]
        med_sid = registry.register(
            figure_id="fig3",
            panel_id="a",
            label=f"{m_name}_S-ID_closure_median_abs",
            raw_value=float(c_sid["median_abs"]),
            fmt_kind="dec4",
            source_file="reports/results/event3_consistency.json",
            source_key=f"models.{m_name}.S-ID.predicted_closure.median_abs",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        gt_sid = registry.register(
            figure_id="fig3",
            panel_id="a",
            label=f"{m_name}_S-ID_closure_share_gt_ln105",
            raw_value=float(c_sid["share_gt_ln105"]),
            fmt_kind="dec4",
            source_file="reports/results/event3_consistency.json",
            source_key=f"models.{m_name}.S-ID.predicted_closure.share_gt_ln105",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        sid_notes.append(f"{m_name} med={med_sid} (>5%={gt_sid})")
        med_lro = registry.register(
            figure_id="fig3",
            panel_id="a",
            label=f"{m_name}_S-LRO_closure_median_abs",
            raw_value=float(c_lro["median_abs"]),
            fmt_kind="dec4",
            source_file="reports/results/event3_consistency.json",
            source_key=f"models.{m_name}.S-LRO.predicted_closure.median_abs",
            evidence_tier=TIER_DESCRIPTIVE,
        )
        gt_lro = registry.register(
            figure_id="fig3",
            panel_id="a",
            label=f"{m_name}_S-LRO_closure_share_gt_ln105",
            raw_value=float(c_lro["share_gt_ln105"]),
            fmt_kind="dec4",
            source_file="reports/results/event3_consistency.json",
            source_key=f"models.{m_name}.S-LRO.predicted_closure.share_gt_ln105",
            evidence_tier=TIER_DESCRIPTIVE,
        )

        ax.hist(
            eps_clipped,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.2,
            linestyle=ls,
            color=col,
            label=f"{m_name} S-LRO (med={med_lro}, >5%={gt_lro})",
        )

    obs_c = cons["observed_closure"]
    obs_med = registry.register(
        figure_id="fig3",
        panel_id="a",
        label="observed_closure_median_abs",
        raw_value=float(obs_c["median_abs"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_consistency.json",
        source_key="observed_closure.median_abs",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    obs_gt = registry.register(
        figure_id="fig3",
        panel_id="a",
        label="observed_closure_share_gt_ln105",
        raw_value=float(obs_c["share_gt_ln105"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_consistency.json",
        source_key="observed_closure.share_gt_ln105",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    tol = math.log(1.05)
    ax.axvline(
        0.0,
        color=MODEL_PALETTE["M5"],
        linewidth=1.5,
        linestyle="-",
        label="M3-M6: eps_A_hat = 0 (exact)",
    )
    ax.axvline(-tol, color="#444444", linewidth=0.8, linestyle=":", label="+/- ln(1.05) QC R8 tol")
    ax.axvline(tol, color="#444444", linewidth=0.8, linestyle=":")

    ax.text(
        0.02,
        0.65,
        (
            f"S-ID: {sid_notes[0]}\n"
            f"S-ID: {sid_notes[1]}\n"
            f"QC R8 obs: med |eps_A|={obs_med}, >ln(1.05)={obs_gt}\n"
            f'S-ID: "{EVENT1_AUDIT_LABEL}"'
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

    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(0.0, 29.0)
    ax.set_xlabel("Predicted Area Closure Residual eps_A_hat = ln(A_hat / (W_hat * dbar_hat))")
    ax.set_ylabel("Density")
    ax.legend(
        loc="upper left",
        fontsize=5.2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
    )
    label_panel(
        ax,
        "a",
        "Area Closure Residual eps_A_hat (§4.3.2)",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_b_back_calc(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (b): Share of predictions with no admissible Manning roughness n_back."""
    cons_models = bundle.consistency["models"]
    x_idx = np.arange(len(BACK_CALC_MODELS), dtype=np.float64)
    width = 0.20
    splits = ("S-ID", "S-LRO", "S-TAIL-DA", "S-TAIL-SO")

    for s_idx, fam in enumerate(splits):
        shares: list[float] = []
        for m_name in BACK_CALC_MODELS:
            if m_name in {"M3", "M5"}:
                shares.append(0.0)
            else:
                sh = float(cons_models[m_name][fam]["back_calc"]["share_outside"])
                shares.append(sh)
                if m_name in {"M2-XGB", "M2-MLP"}:
                    registry.register(
                        figure_id="fig3",
                        panel_id="b",
                        label=f"{m_name}_{fam}_back_calc_share_outside",
                        raw_value=sh,
                        fmt_kind="dec4",
                        source_file="reports/results/event3_consistency.json",
                        source_key=f"models.{m_name}.{fam}.back_calc.share_outside",
                        evidence_tier=TIER_DESCRIPTIVE,
                    )
        ax.bar(
            x_idx + (s_idx - 1.5) * width,
            shares,
            width=width,
            color=SPLIT_PALETTE[fam],
            edgecolor="#222222",
            linewidth=0.4,
            alpha=0.88,
            label=fam,
        )

    m2xgb_shape_lro = registry.register(
        figure_id="fig3",
        panel_id="b",
        label="M2-XGB_S-LRO_shape_violation_share",
        raw_value=float(cons_models["M2-XGB"]["S-LRO"]["shape_violation_share"]),
        fmt_kind="auto_share",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M2-XGB.S-LRO.shape_violation_share",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m2xgb_shape_sid = registry.register(
        figure_id="fig3",
        panel_id="b",
        label="M2-XGB_S-ID_shape_violation_share",
        raw_value=float(cons_models["M2-XGB"]["S-ID"]["shape_violation_share"]),
        fmt_kind="auto_share",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M2-XGB.S-ID.shape_violation_share",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.text(
        0.02,
        0.81,
        (
            "M3, M5 non-admissible share = 0.0000 (n_hat in [0.005, 0.60])\n"
            f"Shape viol. d_max <= dbar: M2-XGB S-LRO={m2xgb_shape_lro}, "
            f"S-ID={m2xgb_shape_sid} (M5=0)"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=5.2,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(BACK_CALC_MODELS, rotation=24, ha="right", fontsize=6.2)
    ax.set_ylabel("Non-Admissible n_back Share")
    ax.set_ylim(0.0, 0.56)
    ax.legend(
        loc="upper center",
        ncol=4,
        fontsize=5.2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        columnspacing=0.6,
        handletextpad=0.3,
    )
    label_panel(
        ax,
        "b",
        "Manning Back-Calc Non-Admissibility (§7.4)",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_c_residuals(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> None:
    """Draw Panel (c): Hard-model Picard solver residual vs M4 test-site Manning residual R_M."""
    cons_models = bundle.consistency["models"]
    splits = ("S-ID", "S-LRO", "S-TAIL-DA", "S-TAIL-SO")
    hard_models = ("M3", "M5", "M6-NN", "M6-XGB")

    x_pos = np.arange(len(splits), dtype=np.float64)
    width = 0.15

    # Plot M4 RMS(R_M)
    m4_rms: list[float] = []
    for fam in splits:
        rm_blk = cons_models["M4"][fam]["manning_residual_M4"]
        rms_v = float(rm_blk["rms"])
        m4_rms.append(rms_v)
        registry.register(
            figure_id="fig3",
            panel_id="c",
            label=f"M4_{fam}_manning_residual_rms",
            raw_value=rms_v,
            fmt_kind="dec4",
            source_file="reports/results/event3_consistency.json",
            source_key=f"models.M4.{fam}.manning_residual_M4.rms",
            evidence_tier=TIER_DESCRIPTIVE,
        )

    ax.bar(
        x_pos - 2.0 * width,
        m4_rms,
        width=width,
        color=MODEL_PALETTE["M4"],
        edgecolor="#222222",
        linewidth=0.45,
        label="M4 Soft PINN: RMS(R_M)",
    )

    for idx, m_name in enumerate(hard_models):
        res_vals: list[float] = []
        for fam in splits:
            rv = float(cons_models[m_name][fam]["solver_residual_max"])
            res_vals.append(rv)
            if m_name == "M5":
                registry.register(
                    figure_id="fig3",
                    panel_id="c",
                    label=f"M5_{fam}_solver_residual_max",
                    raw_value=rv,
                    fmt_kind="sci2",
                    source_file="reports/results/event3_consistency.json",
                    source_key=f"models.M5.{fam}.solver_residual_max",
                    evidence_tier=TIER_DESCRIPTIVE,
                )
        ax.bar(
            x_pos + (idx - 1.0) * width,
            res_vals,
            width=width,
            color=MODEL_PALETTE[m_name],
            edgecolor="#222222",
            linewidth=0.45,
            label=f"{m_name}: max |res|",
        )

    ax.axhline(
        1e-10,
        color="#A50E0E",
        linestyle="--",
        linewidth=0.9,
        label="Picard tol tau_l = 1e-10",
    )

    m4_sid_s = registry.register(
        figure_id="fig3",
        panel_id="c",
        label="callout_M4_S-ID_rms",
        raw_value=float(cons_models["M4"]["S-ID"]["manning_residual_M4"]["rms"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M4.S-ID.manning_residual_M4.rms",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    m5_sid_s = registry.register(
        figure_id="fig3",
        panel_id="c",
        label="callout_M5_S-ID_solver_max",
        raw_value=float(cons_models["M5"]["S-ID"]["solver_residual_max"]),
        fmt_kind="sci2",
        source_file="reports/results/event3_consistency.json",
        source_key="models.M5.S-ID.solver_residual_max",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.text(
        0.02,
        0.74,
        (
            f"M4 test R_M RMS (S-ID)={m4_sid_s} vs M5 max res={m5_sid_s}\n"
            f"S-IND ({S_IND_PANEL_NOTE}): Picard tau_l <= 1.00e-10"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=5.1,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-13, 1e6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(splits, fontsize=6.2)
    ax.set_ylabel("Manning / Picard Residual [ln m] (log)")
    ax.legend(
        loc="upper left",
        ncol=2,
        fontsize=5.0,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        columnspacing=0.5,
        handletextpad=0.3,
    )
    label_panel(
        ax,
        "c",
        "M4 Soft R_M vs Hard Solver Residual",
        TIER_DESCRIPTIVE,
    )


def _draw_panel_d_h4(
    ax: Axes,
    bundle: VerifiedBundle,
    registry: AnnotationRegistry,
) -> str:
    """Draw Panel (d): Confirmatory H4 parameter plausibility and rank correlations."""
    h4 = bundle.hypotheses["H4"]
    sid = h4["S-ID"]
    lro = h4["S-LRO_reported"]
    cav = h4["caveats"]
    h4_verdict = "Falsified" if h4["decision"]["falsified"] else "Supported"

    pi_lit_sid = registry.register(
        figure_id="fig3",
        panel_id="d",
        label="H4_S-ID_pi_lit",
        raw_value=float(sid["pi_lit"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H4.S-ID.pi_lit",
        evidence_tier=TIER_CONFIRMATORY,
    )
    pi_nu_sid = registry.register(
        figure_id="fig3",
        panel_id="d",
        label="H4_S-ID_pi_nu",
        raw_value=float(sid["pi_nu"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H4.S-ID.pi_nu",
        evidence_tier=TIER_CONFIRMATORY,
    )
    pi_r_sid = registry.register(
        figure_id="fig3",
        panel_id="d",
        label="H4_S-ID_pi_r",
        raw_value=float(sid["pi_r"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H4.S-ID.pi_r",
        evidence_tier=TIER_CONFIRMATORY,
    )
    pi_lit_lro = registry.register(
        figure_id="fig3",
        panel_id="d",
        label="H4_S-LRO_pi_lit",
        raw_value=float(lro["pi_lit"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H4.S-LRO_reported.pi_lit",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    # Plot H4 rank correlations with 95% CIs (S-ID Confirmatory, S-LRO & not_F_S Descriptive)
    corr_items = [
        (
            "rho_S(n_hat, D50) [S-ID, Conf.]",
            sid["ci"]["n_d50"],
            "H4.S-ID.ci.n_d50",
            TIER_CONFIRMATORY,
            OKABE_ITO["vermilion"],
        ),
        (
            "rho_S(n_hat, D50) [S-LRO, Desc.]",
            lro["ci"]["n_d50"],
            "H4.S-LRO_reported.ci.n_d50",
            TIER_DESCRIPTIVE,
            OKABE_ITO["orange"],
        ),
        (
            "rho_S(n_hat, S) [S-ID all, Conf.]",
            sid["ci"]["n_s"],
            "H4.S-ID.ci.n_s",
            TIER_CONFIRMATORY,
            OKABE_ITO["blue"],
        ),
        (
            "rho_S(n_hat, S) [not_F_S n=3,167]",
            cav["rho_n_s_ci_by_F_S"]["not_F_S"]["n_s"],
            "H4.caveats.rho_n_s_ci_by_F_S.not_F_S.n_s",
            TIER_DESCRIPTIVE,
            OKABE_ITO["sky_blue"],
        ),
        (
            "rho_S(r_hat, r_obs) [S-ID, Conf.]",
            sid["ci"]["r_robs"],
            "H4.S-ID.ci.r_robs",
            TIER_CONFIRMATORY,
            OKABE_ITO["bluish_green"],
        ),
        (
            "rho_S(r_hat, r_obs) [S-LRO, Desc.]",
            lro["ci"]["r_robs"],
            "H4.S-LRO_reported.ci.r_robs",
            TIER_DESCRIPTIVE,
            OKABE_ITO["reddish_purple"],
        ),
    ]

    y_pos = np.arange(len(corr_items), dtype=np.float64)[::-1]
    ax.axvline(0.0, color="#111111", linestyle="-", linewidth=0.85, zorder=2)

    y_labels: list[str] = []
    for idx, (lbl, blk, skey, tier, col) in enumerate(corr_items):
        y = float(y_pos[idx])
        est = float(blk["estimate"])
        lo = float(blk["lo"])
        hi = float(blk["hi"])
        est_s = registry.register(
            figure_id="fig3",
            panel_id="d",
            label=f"{skey}.estimate",
            raw_value=est,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"{skey}.estimate",
            evidence_tier=tier,
        )
        lo_s = registry.register(
            figure_id="fig3",
            panel_id="d",
            label=f"{skey}.lo",
            raw_value=lo,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"{skey}.lo",
            evidence_tier=tier,
        )
        hi_s = registry.register(
            figure_id="fig3",
            panel_id="d",
            label=f"{skey}.hi",
            raw_value=hi,
            fmt_kind="dec4",
            source_file="reports/results/event3_hypotheses.json",
            source_key=f"{skey}.hi",
            evidence_tier=tier,
        )
        ax.errorbar(
            est,
            y,
            xerr=[[est - lo], [hi - est]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=1.3,
            capsize=3.0,
            markersize=4.5,
            zorder=4,
        )
        ax.text(
            hi + 0.015,
            y,
            f"{est_s} [{lo_s}, {hi_s}]",
            va="center",
            ha="left",
            fontsize=5.6,
            fontweight="bold",
            color="#111111",
        )
        y_labels.append(lbl)

    # Register chan_mat_cd medians for SAND and CBLS as provenance check
    sand_med = registry.register(
        figure_id="fig3",
        panel_id="d",
        label="H4_chan_mat_SAND_median",
        raw_value=float(cav["n_hat_by_chan_mat_cd"]["SAND"]["median"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H4.caveats.n_hat_by_chan_mat_cd.SAND.median",
        evidence_tier=TIER_DESCRIPTIVE,
    )
    cbls_med = registry.register(
        figure_id="fig3",
        panel_id="d",
        label="H4_chan_mat_CBLS_median",
        raw_value=float(cav["n_hat_by_chan_mat_cd"]["CBLS"]["median"]),
        fmt_kind="dec4",
        source_file="reports/results/event3_hypotheses.json",
        source_key="H4.caveats.n_hat_by_chan_mat_cd.CBLS.median",
        evidence_tier=TIER_DESCRIPTIVE,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=5.8)
    ax.set_xlabel("Spearman Rank Correlation rho_S (95% Bootstrap CI)")
    ax.set_xlim(-0.14, 0.78)
    ax.set_ylim(-0.65, 5.65)
    label_panel(
        ax,
        "d",
        "H4 Parameter Plausibility & Rank Correlations",
        f"{TIER_CONFIRMATORY} / {TIER_DESCRIPTIVE}",
    )
    return (
        f"Panel (d) H4 Summary — Verdict: {h4_verdict} (rho_S(n_hat, D50) CI spans 0) | "
        f"N_lit=[0.012, 0.33]: pi_lit={pi_lit_sid} (S-ID), {pi_lit_lro} (S-LRO) >= 0.80\n"
        f"Boundary shares: pi_nu={pi_nu_sid}, pi_r={pi_r_sid} <= 0.10 | "
        f"chan_mat_cd med n_hat: SAND={sand_med}, CBLS={cbls_med} | "
        "F_S (n=821) rho_n_s: undefined (S=S_floor=1e-5)"
    )


def generate_fig3(
    bundle: VerifiedBundle,
    repo_root: Path,
    figures_dir: Path,
    registry: AnnotationRegistry,
) -> dict[str, Path]:
    """Build and save Fig. 3 (Physical consistency analogues and Confirmatory H4)."""
    apply_asce_style()
    fig = plt.figure(figsize=(ASCE_DOUBLE_COL_IN, 6.6))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        hspace=0.40,
        wspace=0.42,
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.12,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _draw_panel_a_closure(ax_a, bundle, registry)
    _draw_panel_b_back_calc(ax_b, bundle, registry)
    _draw_panel_c_residuals(ax_c, bundle, registry)
    h4_banner = _draw_panel_d_h4(ax_d, bundle, registry)
    fig.text(
        0.50,
        0.015,
        h4_banner,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#FFF8F6",
            "edgecolor": "#A50E0E",
            "alpha": 0.96,
        },
    )
    del repo_root

    return save_figure_bundle(fig, figures_dir, "fig3_physical_consistency_and_h4")
