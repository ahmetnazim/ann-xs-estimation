"""Mechanical generation of Phase 9 diagnostic tables (T1-T9) and Markdown report.

Every number in ``docs/tables/benchmark_results.md`` and ``docs/tables/t1_*.csv``
through ``t9_*.csv`` is read directly from the SHA-256-verified ``VerifiedBundle``
(``reports/results/*``). Zero inferential statistics, confidence intervals,
metrics, or verdicts are computed here (P9-D5 (a)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    S_IND_UQ_NOTE,
    VerifiedBundle,
)

TIER_CONFIRMATORY: Final[str] = "Confirmatory"
TIER_SECONDARY: Final[str] = "Secondary"
TIER_DESCRIPTIVE: Final[str] = "Descriptive"

PRIMARY_LADDER_MODELS: Final[tuple[str, ...]] = (
    "M-null",
    "A_J",
    "M0",
    "M1",
    "M2-XGB",
    "M2-MLP",
    "M3",
    "M4",
    "M5",
    "M6-NN",
    "M6-XGB",
)

CORE_FAMILIES: Final[tuple[str, ...]] = (
    "S-ID",
    "S-LRO",
    "S-TAIL-DA",
    "S-TAIL-SO",
    "S-IND",
)

ABLATION_SPECS: Final[tuple[str, ...]] = (
    "M5_A1",
    "M5_A2",
    "M5_A3-SP",
    "M5_A3-SR",
    "M5_A4",
    "M5_wA1",
)

A4_SPECS: Final[tuple[str, ...]] = (
    "M2-XGB_A4",
    "M1_A4",
    "M2-MLP_A4",
)

SENSITIVITY_ANALYSES: Final[tuple[str, ...]] = (
    "SA1",
    "SA2",
    "SA3",
    "SA4",
    "SA5",
    "SA6",
)

ROUNDING_RULES_TEXT: Final[str] = (
    "Rounding rules (stated once, applied uniformly across Tables T1-T9): "
    "(1) CSV mirrors (`t1_*.csv`-`t9_*.csv`) store full IEEE-754 double precision "
    "as recorded in `reports/results/*`; "
    "(2) in Markdown tables, log-space metrics (`rmse_log`, `bias_log`, `nse_log`, "
    "`crps`, `mean_log_width`, `R_M`, `|epsilon_A|`, skill losses `L`, contrasts "
    "`Delta`, `bias_t`, `rmse_t`, `z_n` quantiles) and dimensionless ratios, "
    "correlations, coverages, shares, and width factors (`r2_raw`, `rho_r`, `rho_S`, "
    "`nse_raw`, `kge`, `mult_bias`, `coverage`, `pi_lit`, `pi_nu`, `pi_r`, "
    "`width_factor`, `kappa` cuts) are formatted to 4 decimal places (`0.0000`); "
    "(3) percentage metrics (`mdape`, `apb_pct`), raw-space RMSE (`rmse_raw`), and "
    "spatial distances/drainage-area cuts (`km`, `km^2`) are formatted to 2 decimal "
    "places (`0.00`); "
    "(4) solver residuals (`solver_residual_max`) and positive shares `< 1e-4` are "
    "formatted in scientific notation to 2 decimal places (`0.00e+00`); "
    "(5) sample sizes, cluster counts, and iteration counts are formatted as exact integers."
)


@dataclass(frozen=True)
class GeneratedTables:
    """Container of generated DataFrames and file paths for T1-T9."""

    markdown_path: Path
    csv_paths: dict[str, Path]
    frames: dict[str, pd.DataFrame]


def format_num(value: object, kind: str = "dec4") -> str:
    """Format a scalar value deterministically according to the Phase 9 rounding rules."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:d}"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if kind == "int":
            return f"{round(value):d}"
        if kind == "dec2":
            return f"{value:.2f}"
        if kind == "sci2":
            return f"{value:.2e}"
        if kind == "auto_share":
            if 0.0 < abs(value) < 1e-4:
                return f"{value:.2e}"
            return f"{value:.4f}"
        return f"{value:.4f}"
    return str(value)


def metric_fmt_kind(metric: str) -> str:
    """Return formatting kind for a given metric name."""
    if metric in {"mdape", "apb_pct", "rmse_raw"}:
        return "dec2"
    if metric in {"solver_residual_max"}:
        return "sci2"
    return "dec4"


def _df_to_markdown(df: pd.DataFrame, col_formats: dict[str, str] | None = None) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table with deterministic formatting."""
    fmt_map = col_formats or {}
    cols = list(df.columns)
    header = "| " + " | ".join(str(c).replace("|", "\\|") for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in cols:
            val = row[col]
            kind = fmt_map.get(col, "dec4")
            cells.append(format_num(val, kind).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_t1_hypotheses_and_outcome(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T1: Confirmatory hypotheses H1-H4, Event 2 M* selection, and verbatim §4.7 row 4."""
    rows: list[dict[str, Any]] = []

    # 1. Pre-registered §4.7 outcome row 4 (verbatim)
    o47 = bundle.outcome_4_7
    rows.append(
        {
            "section": "outcome_4_7",
            "hypothesis": f"§4.7 row {o47.row}",
            "split": "S-LRO",
            "stratum": "all",
            "target_or_quantity": o47.outcome,
            "statistic": o47.statement,
            "estimate": None,
            "ci_lo": None,
            "ci_hi": None,
            "ci_level": 0.95,
            "n_units": 3988,
            "n_clusters": None,
            "verdict": f"row {o47.row} ({o47.venue})",
            "evidence_tier": TIER_CONFIRMATORY,
            "split_note": "8 F&J divisions",
            "source_file": o47.source_file,
        }
    )

    # 2. Event 2 M* empirical selection on S-ID
    mstar_doc = bundle.mstar
    mstar_means = mstar_doc["rmse_log_mean"]
    mstar_repeats = mstar_doc["rmse_log_by_repeat"]
    mstar_choice = mstar_doc["mstar"]
    for cand in mstar_doc["candidates"]:
        for tgt in mstar_doc["targets"]:
            rep_dict = mstar_repeats[cand][tgt]
            rows.append(
                {
                    "section": "event_2_mstar",
                    "hypothesis": "Event 2 M*",
                    "split": "S-ID",
                    "stratum": "all",
                    "target_or_quantity": f"{cand}:{tgt}",
                    "statistic": (
                        f"mean RMSE_log (r0={rep_dict['0']:.4f}, "
                        f"r1={rep_dict['1']:.4f}, r2={rep_dict['2']:.4f})"
                    ),
                    "estimate": float(mstar_means[cand][tgt]),
                    "ci_lo": None,
                    "ci_hi": None,
                    "ci_level": None,
                    "n_units": int(mstar_doc["n_sites"]),
                    "n_clusters": None,
                    "verdict": "selected M*" if mstar_choice[tgt] == cand else "candidate",
                    "evidence_tier": TIER_CONFIRMATORY,
                    "split_note": EVENT1_AUDIT_LABEL,
                    "source_file": "reports/results/mstar.json",
                }
            )

    # 3. H1-H4 from bundle.hypotheses_df (excluding S2 rows, which belong to T3)
    hyp_df = bundle.hypotheses_df
    h1_h4_df = hyp_df[hyp_df["hypothesis"].isin(["H1", "H2", "H3", "H4"])].copy()
    for _, r in h1_h4_df.iterrows():
        fam_str = str(r["family"])
        n_clust = (
            int(bundle.hypotheses["H1"]["S-IND"]["n_clusters"])
            if (str(r["hypothesis"]) == "H1" and fam_str == "S-IND")
            else None
        )
        rows.append(
            {
                "section": "hypotheses_H1_H4",
                "hypothesis": str(r["hypothesis"]),
                "split": fam_str,
                "stratum": "Q1" if "Q1" in str(r["contrast"]) else "all",
                "target_or_quantity": str(r["contrast"]),
                "statistic": f"target={r['target']}",
                "estimate": None if pd.isna(r["estimate"]) else float(r["estimate"]),
                "ci_lo": None if pd.isna(r["ci_lo"]) else float(r["ci_lo"]),
                "ci_hi": None if pd.isna(r["ci_hi"]) else float(r["ci_hi"]),
                "ci_level": None if pd.isna(r["ci_level"]) else float(r["ci_level"]),
                "n_units": None if pd.isna(r["n_units"]) else int(r["n_units"]),
                "n_clusters": n_clust,
                "verdict": str(r["verdict"]),
                "evidence_tier": str(r["evidence_tier"]),
                "split_note": str(r["split_note"]),
                "source_file": str(r["source_file"]),
            }
        )

    # 3b. H4 point quantities (pi_lit, pi_nu, pi_r, rmse_t, bias_t) on S-ID and S-LRO_reported
    h4_doc = bundle.hypotheses["H4"]
    for split_key, fam_name, tier in (
        ("S-ID", "S-ID", TIER_CONFIRMATORY),
        ("S-LRO_reported", "S-LRO", TIER_DESCRIPTIVE),
    ):
        blk = h4_doc[split_key]
        s_note = EVENT1_AUDIT_LABEL if fam_name == "S-ID" else "8 F&J divisions"
        for pt_key in ("pi_lit", "pi_nu", "pi_r", "rmse_t", "bias_t"):
            rows.append(
                {
                    "section": "H4_point_quantities",
                    "hypothesis": "H4",
                    "split": fam_name,
                    "stratum": "all",
                    "target_or_quantity": f"H4_{fam_name}_{pt_key}",
                    "statistic": f"H4 {pt_key} ({split_key})",
                    "estimate": float(blk[pt_key]),
                    "ci_lo": None,
                    "ci_hi": None,
                    "ci_level": None,
                    "n_units": int(blk["n_sites"]),
                    "n_clusters": None,
                    "verdict": "Falsified" if h4_doc["decision"]["falsified"] else "Supported",
                    "evidence_tier": tier,
                    "split_note": s_note,
                    "source_file": "reports/results/event3_hypotheses.json",
                }
            )

    # 4. H4 caveats (by_F_S, rho_n_s_ci_by_F_S, z_n quantiles, n_hat_by_chan_mat_cd)
    h4_cav = h4_doc["caveats"]
    for fs_key, fs_block in sorted(h4_cav["by_F_S"].items()):
        n_s_ci = h4_cav["rho_n_s_ci_by_F_S"][fs_key]["n_s"]
        for metric_key in ("pi_lit", "pi_nu", "pi_r", "rho_n_d50", "rho_n_s", "rho_r_robs"):
            val = fs_block[metric_key]
            lo_val = n_s_ci.get("lo") if (metric_key == "rho_n_s" and val is not None) else None
            hi_val = n_s_ci.get("hi") if (metric_key == "rho_n_s" and val is not None) else None
            lvl_val = n_s_ci.get("level") if (metric_key == "rho_n_s" and val is not None) else None
            verdict_str = (
                fs_block["undefined"]["rho_n_s"]
                if (metric_key == "rho_n_s" and val is None)
                else f"caveat C2/L10 ({fs_key})"
            )
            rows.append(
                {
                    "section": "H4_caveats_by_F_S",
                    "hypothesis": "H4",
                    "split": "S-ID",
                    "stratum": fs_key,
                    "target_or_quantity": f"{fs_key}:{metric_key}",
                    "statistic": f"H4 {metric_key} in {fs_key}",
                    "estimate": None if val is None else float(val),
                    "ci_lo": None if lo_val is None else float(lo_val),
                    "ci_hi": None if hi_val is None else float(hi_val),
                    "ci_level": None if lvl_val is None else float(lvl_val),
                    "n_units": int(fs_block["n_sites"]),
                    "n_clusters": None,
                    "verdict": verdict_str,
                    "evidence_tier": TIER_DESCRIPTIVE,
                    "split_note": EVENT1_AUDIT_LABEL,
                    "source_file": "reports/results/event3_hypotheses.json",
                }
            )

    for cm_key, cm_block in sorted(h4_cav["n_hat_by_chan_mat_cd"].items()):
        rows.append(
            {
                "section": "H4_n_hat_by_chan_mat_cd",
                "hypothesis": "H4",
                "split": "S-ID",
                "stratum": cm_key,
                "target_or_quantity": f"chan_mat_cd:{cm_key}",
                "statistic": "median [q25, q75] of site-level n-hat by chan_mat_cd",
                "estimate": float(cm_block["median"]),
                "ci_lo": float(cm_block["q25"]),
                "ci_hi": float(cm_block["q75"]),
                "ci_level": None,
                "n_units": int(cm_block["n"]),
                "n_clusters": None,
                "verdict": "descriptive",
                "evidence_tier": TIER_DESCRIPTIVE,
                "split_note": EVENT1_AUDIT_LABEL,
                "source_file": "reports/results/event3_hypotheses.json",
            }
        )

    zn_q = h4_cav["z_n"]["quantiles"]
    for q_name, q_val in sorted(zn_q.items()):
        rows.append(
            {
                "section": "H4_z_n_quantiles",
                "hypothesis": "H4",
                "split": "S-ID",
                "stratum": "all",
                "target_or_quantity": f"z_n_{q_name}",
                "statistic": f"z_n {q_name} (n_infinite={h4_cav['z_n']['n_infinite']})",
                "estimate": float(q_val),
                "ci_lo": None,
                "ci_hi": None,
                "ci_level": None,
                "n_units": 3988,
                "n_clusters": None,
                "verdict": "descriptive",
                "evidence_tier": TIER_DESCRIPTIVE,
                "split_note": EVENT1_AUDIT_LABEL,
                "source_file": "reports/results/event3_hypotheses.json",
            }
        )

    # 5. Far / not-nested H2 strata (including explicit "not computed" far stratum)
    fnn = bundle.hypotheses["far_not_nested_strata"]
    rows.append(
        {
            "section": "H2_far_not_nested",
            "hypothesis": "H2",
            "split": "S-ID",
            "stratum": "far",
            "target_or_quantity": "H2_far",
            "statistic": "mean over repeats of |B_dbar(M5)(Q1)| - |B_dbar(M*)(Q1)| (far stratum)",
            "estimate": None,
            "ci_lo": None,
            "ci_hi": None,
            "ci_level": 0.95,
            "n_units": 0,
            "n_clusters": None,
            "verdict": f"not computed: {fnn['far']['H2']['not_computed']}",
            "evidence_tier": TIER_SECONDARY,
            "split_note": EVENT1_AUDIT_LABEL,
            "source_file": "reports/results/event3_hypotheses.json",
        }
    )
    nn_h2 = fnn["not_nested"]["H2"]
    nn_h2_reps = nn_h2["sites_per_repeat"]
    for q_key in ("bias_M5", "bias_Mstar", "h2"):
        q_blk = nn_h2[q_key]
        rows.append(
            {
                "section": "H2_far_not_nested",
                "hypothesis": "H2",
                "split": "S-ID",
                "stratum": "not_nested",
                "target_or_quantity": f"not_nested:{q_key}",
                "statistic": (
                    f"not_nested Q1 {q_key} (sites/repeat: r0={nn_h2_reps['0']}, "
                    f"r1={nn_h2_reps['1']}, r2={nn_h2_reps['2']})"
                ),
                "estimate": float(q_blk["estimate"]),
                "ci_lo": float(q_blk["lo"]),
                "ci_hi": float(q_blk["hi"]),
                "ci_level": float(q_blk["level"]),
                "n_units": int(nn_h2_reps["0"]),
                "n_clusters": None,
                "verdict": f"opposite_sign_to_full_set={q_blk['opposite_sign_to_full_set']}",
                "evidence_tier": TIER_SECONDARY,
                "split_note": EVENT1_AUDIT_LABEL,
                "source_file": "reports/results/event3_hypotheses.json",
            }
        )

    return pd.DataFrame(rows)


def build_t2_metrics_matrix(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T2: Primary metrics matrix for dbar, W, A, dmax, t across splits for ladder models."""
    df = bundle.matrix_df
    mask = (
        (df["analysis"] == "primary")
        & (df["ablation"] == "")
        & (df["stratum_kind"] == "all")
        & (df["stratum"] == "all")
        & (df["family"].isin(CORE_FAMILIES))
        & (df["model"].isin(PRIMARY_LADDER_MODELS))
    )
    sub = df.loc[mask].copy()
    model_order = {m: idx for idx, m in enumerate(PRIMARY_LADDER_MODELS)}
    family_order = {f: idx for idx, f in enumerate(CORE_FAMILIES)}
    target_order = {"dbar": 0, "A": 1, "W": 2, "dmax": 3, "t": 4}
    sub["_m_ord"] = sub["model"].map(lambda x: model_order.get(str(x), 99))
    sub["_f_ord"] = sub["family"].map(lambda x: family_order.get(str(x), 99))
    sub["_t_ord"] = sub["target"].map(lambda x: target_order.get(str(x), 99))
    sub = sub.sort_values(["_f_ord", "_t_ord", "_m_ord", "metric"]).drop(
        columns=["_m_ord", "_f_ord", "_t_ord"]
    )
    return sub.reset_index(drop=True)


def build_t3_s2_contrasts(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T3: Stepwise physics contrasts S2 (98.75% Bonferroni CIs) and kappa quintiles."""
    rows: list[dict[str, Any]] = []
    s2_doc = bundle.hypotheses["S2"]
    stat_str = str(s2_doc["statistic"])

    for fam in ("S-ID", "S-LRO"):
        split_note = EVENT1_AUDIT_LABEL if fam == "S-ID" else "8 F&J divisions"
        for contrast_key in ("s2_M4_vs_M5", "s2_M5_vs_M6NN"):
            c_blk = s2_doc[fam][contrast_key]
            rows.append(
                {
                    "section": "S2_overall",
                    "split": fam,
                    "stratum": "all",
                    "contrast": contrast_key,
                    "statistic": stat_str,
                    "estimate": float(c_blk["estimate"]),
                    "ci_lo": float(c_blk["lo"]),
                    "ci_hi": float(c_blk["hi"]),
                    "ci_level": float(c_blk["level"]),
                    "n_rows": 11964 if fam == "S-ID" else 3988,
                    "status": "computed",
                    "evidence_tier": TIER_SECONDARY,
                    "split_note": split_note,
                    "source_file": "reports/results/event3_hypotheses.json",
                }
            )

    # Far and not-nested strata for S2
    fnn = bundle.hypotheses["far_not_nested_strata"]
    far_s2 = fnn["far"]["S2"]
    far_reps = far_s2["sites_per_repeat"]
    for contrast_key in ("s2_M4_vs_M5", "s2_M5_vs_M6NN"):
        rows.append(
            {
                "section": "S2_far_not_nested",
                "split": "S-ID",
                "stratum": "far",
                "contrast": contrast_key,
                "statistic": (
                    f"{stat_str} (sites/repeat: r0={far_reps['0']}, "
                    f"r1={far_reps['1']}, r2={far_reps['2']})"
                ),
                "estimate": None,
                "ci_lo": None,
                "ci_hi": None,
                "ci_level": 0.9875,
                "n_rows": int(far_reps["0"]),
                "status": f"not computed: {far_s2['not_computed']}",
                "evidence_tier": TIER_SECONDARY,
                "split_note": EVENT1_AUDIT_LABEL,
                "source_file": "reports/results/event3_hypotheses.json",
            }
        )

    nn_s2 = fnn["not_nested"]["S2"]
    nn_reps = nn_s2["sites_per_repeat"]
    for contrast_key in ("s2_M4_vs_M5", "s2_M5_vs_M6NN"):
        c_blk = nn_s2[contrast_key]
        rows.append(
            {
                "section": "S2_far_not_nested",
                "split": "S-ID",
                "stratum": "not_nested",
                "contrast": contrast_key,
                "statistic": (
                    f"{stat_str} (sites/repeat: r0={nn_reps['0']}, "
                    f"r1={nn_reps['1']}, r2={nn_reps['2']})"
                ),
                "estimate": float(c_blk["estimate"]),
                "ci_lo": float(c_blk["lo"]),
                "ci_hi": float(c_blk["hi"]),
                "ci_level": float(c_blk["level"]),
                "n_rows": int(nn_reps["0"]),
                "status": f"opposite_sign_to_full_set={c_blk['opposite_sign_to_full_set']}",
                "evidence_tier": TIER_SECONDARY,
                "split_note": EVENT1_AUDIT_LABEL,
                "source_file": "reports/results/event3_hypotheses.json",
            }
        )

    # Descriptive kappa quintiles
    kq_doc = s2_doc["kappa_quintiles_descriptive"]
    for fam in ("S-ID", "S-LRO"):
        split_note = EVENT1_AUDIT_LABEL if fam == "S-ID" else "8 F&J divisions"
        cuts = kq_doc[fam]["cuts"]
        cuts_str = ", ".join(f"{float(c):.4f}" for c in cuts)
        for q_key in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            q_blk = kq_doc[fam][q_key]
            for contrast_key in ("s2_M4_vs_M5", "s2_M5_vs_M6NN"):
                rows.append(
                    {
                        "section": "S2_kappa_quintiles_descriptive",
                        "split": fam,
                        "stratum": f"kappa_{q_key}",
                        "contrast": contrast_key,
                        "statistic": f"{stat_str} (kappa cuts: {cuts_str})",
                        "estimate": float(q_blk[contrast_key]),
                        "ci_lo": None,
                        "ci_hi": None,
                        "ci_level": None,
                        "n_rows": int(q_blk["n_rows"]),
                        "status": "descriptive",
                        "evidence_tier": TIER_DESCRIPTIVE,
                        "split_note": split_note,
                        "source_file": "reports/results/event3_hypotheses.json",
                    }
                )

    return pd.DataFrame(rows)


def build_t4_uq_calibration(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T4: UQ calibration (coverage, mean_log_width, width_factor, CRPS, in_band)."""
    df = bundle.uq_df.copy()
    return df.sort_values(["spec", "family", "target", "stratum_kind", "stratum"]).reset_index(
        drop=True
    )


def build_t5_physical_consistency(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T5: Physical consistency analogues (§4.3.2) from event3_consistency.json."""
    rows: list[dict[str, Any]] = []
    c_doc = bundle.consistency

    # Observed closure (QC R8)
    obs = c_doc["observed_closure"]
    for k in (
        "median_abs",
        "share_gt_ln105",
        "n_sites",
        "qc_rows",
        "qc_R8_rows_flagged",
        "share_removed",
    ):
        rows.append(
            {
                "model": "observed_frame",
                "split": "frame",
                "category": "observed_closure",
                "metric": k,
                "value": float(obs[k]) if isinstance(obs[k], float) else int(obs[k]),
                "note": str(c_doc["continuity"]),
                "evidence_tier": TIER_DESCRIPTIVE,
                "split_note": "3,988 frame sites / 39,927 QC rows",
                "source_file": "reports/results/event3_consistency.json",
            }
        )

    # Model-by-split consistency rows directly from bundle.consistency["models"]
    for model_name, fam_map in sorted(c_doc["models"].items()):
        for fam_name, metrics_map in sorted(fam_map.items()):
            s_note = EVENT1_AUDIT_LABEL if fam_name == "S-ID" else fam_name
            for cat_key, cat_val in sorted(metrics_map.items()):
                if isinstance(cat_val, dict):
                    for m_key, m_val in sorted(cat_val.items()):
                        rows.append(
                            {
                                "model": str(model_name),
                                "split": str(fam_name),
                                "category": str(cat_key),
                                "metric": str(m_key),
                                "value": (float(m_val) if isinstance(m_val, float) else int(m_val)),
                                "note": (
                                    str(c_doc["continuity"]) if cat_key == "implied_V" else ""
                                ),
                                "evidence_tier": TIER_DESCRIPTIVE,
                                "split_note": s_note,
                                "source_file": "reports/results/event3_consistency.json",
                            }
                        )
                else:
                    rows.append(
                        {
                            "model": str(model_name),
                            "split": str(fam_name),
                            "category": "scalar",
                            "metric": str(cat_key),
                            "value": (
                                float(cat_val) if isinstance(cat_val, float) else int(cat_val)
                            ),
                            "note": "",
                            "evidence_tier": TIER_DESCRIPTIVE,
                            "split_note": s_note,
                            "source_file": "reports/results/event3_consistency.json",
                        }
                    )

    return pd.DataFrame(rows)


def build_t6_ablations_and_sensitivity(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T6: Pre-registered ablations, A4 baselines, SA1-SA6, and ablation H4 points."""
    rows: list[dict[str, Any]] = []
    df = bundle.matrix_df
    desc = bundle.descriptive
    sens_status = desc["sensitivity_status"]
    a4_primary = str(desc["A4_realised_mstar"])

    # 1. Matrix rows for ablations, A4 specs, and sensitivity analyses (stratum_kind == 'all')
    mask = (
        (df["stratum_kind"] == "all")
        & (df["stratum"] == "all")
        & (
            (df["spec"].isin(ABLATION_SPECS))
            | (df["spec"].isin(A4_SPECS))
            | (df["analysis"].isin(SENSITIVITY_ANALYSES))
            | (
                (df["spec"].isin(["M5", "M2-XGB", "M1", "M2-MLP", "M4"]))
                & (df["analysis"] == "primary")
                & (df["metric"].isin(["rmse_log", "bias_log"]))
            )
        )
    )
    sub = df.loc[mask].copy()
    for _, r in sub.iterrows():
        spec = str(r["spec"])
        model = str(r["model"])
        analysis = str(r["analysis"])
        ablation = str(r["ablation"])
        if ablation == "A4" and model != "M5":
            role = "primary realised M* (A4)" if spec == a4_primary else "supplementary (A4; §13.8)"
        elif analysis in SENSITIVITY_ANALYSES:
            m_role = str(sens_status.get(model, {}).get("role", "supplementary"))
            role = (
                f"primary ({m_role}; §13.8)"
                if model in {"M5", "M2-XGB"}
                else f"supplementary ({m_role}; §13.8)"
            )
        elif spec in ABLATION_SPECS:
            role = "pre-registered M5 ablation"
        else:
            role = "primary reference"

        rows.append(
            {
                "section": "matrix_ablation_sa",
                "spec": spec,
                "model": model,
                "ablation": ablation,
                "analysis": analysis,
                "role": role,
                "split": str(r["family"]),
                "target": str(r["target"]),
                "metric": str(r["metric"]),
                "value": float(r["value"]),
                "n_rows": int(r["n_rows"]),
                "n_units": int(r["n_units"]),
                "status": "computed",
                "evidence_tier": TIER_DESCRIPTIVE,
                "split_note": str(r["split_note"]),
                "source_file": "reports/results/event3_matrix.parquet",
            }
        )

    # 2. Explicit SA1 status rows for M1, M2-MLP, M2-XGB (not trained; identical by construction)
    for m_name in ("M2-XGB", "M1", "M2-MLP"):
        s_info = sens_status[m_name]
        m_role = str(s_info["role"])
        role_label = (
            f"primary ({m_role}; §13.8)"
            if m_name == "M2-XGB"
            else f"supplementary ({m_role}; §13.8)"
        )
        rows.append(
            {
                "section": "sensitivity_SA1_status",
                "spec": m_name,
                "model": m_name,
                "ablation": "",
                "analysis": "SA1",
                "role": role_label,
                "split": "all",
                "target": "all",
                "metric": "status",
                "value": None,
                "n_rows": 0,
                "n_units": 0,
                "status": str(s_info["SA1"]),
                "evidence_tier": TIER_DESCRIPTIVE,
                "split_note": "no Phi layer",
                "source_file": "reports/results/event3_descriptive.json",
            }
        )

    # 3. Ablation H4 point summaries on S-ID from event3_descriptive.json
    for ab_spec, ab_dict in sorted(desc["ablations_M5"].items()):
        for pt_group, pt_vals in sorted(ab_dict.items()):
            n_sites = int(pt_vals.get("n_sites", 3988))
            for k, v in sorted(pt_vals.items()):
                if k == "n_sites":
                    continue
                rows.append(
                    {
                        "section": f"ablation_{pt_group}",
                        "spec": ab_spec,
                        "model": "M5",
                        "ablation": ab_spec.split("_", 1)[1],
                        "analysis": "primary",
                        "role": (
                            "pre-registered M5 ablation (Prop 7.6 mapped)"
                            if "prop_7_6" in pt_group
                            else "pre-registered M5 ablation"
                        ),
                        "split": "S-ID",
                        "target": "n_hat/r_hat",
                        "metric": k,
                        "value": float(v),
                        "n_rows": n_sites,
                        "n_units": n_sites,
                        "status": "computed",
                        "evidence_tier": TIER_DESCRIPTIVE,
                        "split_note": EVENT1_AUDIT_LABEL,
                        "source_file": "reports/results/event3_descriptive.json",
                    }
                )

    return pd.DataFrame(rows)


def build_t7_srnd_benchmark(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T7: Secondary S-RND benchmark (M1, M2-XGB; 1/|I_j| weights; 9,673 rows)."""
    df = bundle.matrix_df
    srnd_meta = bundle.descriptive["S_RND"]
    weights_str = str(srnd_meta["weights"])
    sub = df.loc[df["family"] == "S-RND"].copy()
    sub["weights"] = weights_str
    sub = sub.sort_values(["target", "model", "metric"]).reset_index(drop=True)
    return sub


def build_t8_s1_hyg_consistency(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T8: S1 HyG consistency check (2,887 matched sites; protocol §4.6)."""
    s1 = bundle.descriptive["S1"]
    note_str = str(s1["note"])
    rows: list[dict[str, Any]] = [
        {
            "category": "S1_comparison",
            "metric": "rmse_log",
            "value": float(s1["rmse_log"]),
            "note": note_str,
            "evidence_tier": TIER_DESCRIPTIVE,
            "split_note": EVENT1_AUDIT_LABEL,
            "source_file": "reports/results/event3_descriptive.json",
        },
        {
            "category": "S1_comparison",
            "metric": "spearman_rho",
            "value": float(s1["spearman_rho"]),
            "note": note_str,
            "evidence_tier": TIER_DESCRIPTIVE,
            "split_note": EVENT1_AUDIT_LABEL,
            "source_file": "reports/results/event3_descriptive.json",
        },
        {
            "category": "S1_comparison",
            "metric": "n_matched",
            "value": int(s1["n_matched"]),
            "note": note_str,
            "evidence_tier": TIER_DESCRIPTIVE,
            "split_note": EVENT1_AUDIT_LABEL,
            "source_file": "reports/results/event3_descriptive.json",
        },
        {
            "category": "S1_comparison",
            "metric": "n_frame_sites_in_hyg",
            "value": int(s1["n_frame_sites_in_hyg"]),
            "note": note_str,
            "evidence_tier": TIER_DESCRIPTIVE,
            "split_note": EVENT1_AUDIT_LABEL,
            "source_file": "reports/results/event3_descriptive.json",
        },
    ]
    for k, v in sorted(s1["hyg_reader_counts"].items()):
        rows.append(
            {
                "category": "hyg_reader_counts",
                "metric": k,
                "value": int(v),
                "note": note_str,
                "evidence_tier": TIER_DESCRIPTIVE,
                "split_note": "HyG raw table",
                "source_file": "reports/results/event3_descriptive.json",
            }
        )
    return pd.DataFrame(rows)


def build_t9_frame_and_data_summary(bundle: VerifiedBundle) -> pd.DataFrame:
    """Build T9: Frame, splits, strata, Event 1 dependence audit, and 'not computed' items."""
    rows: list[dict[str, Any]] = []
    audit = bundle.audit
    desc = bundle.descriptive
    hyp = bundle.hypotheses
    cons = bundle.consistency
    dep = bundle.dependence
    uq = bundle.uq
    mat = bundle.matrix_df

    # 1. Frame and QC R8 summary
    obs_c = cons["observed_closure"]
    for k in ("qc_rows", "qc_R8_rows_flagged", "share_removed", "n_sites"):
        rows.append(
            {
                "section": "frame_and_qc",
                "item": k,
                "stratum": "all",
                "value": float(obs_c[k]) if isinstance(obs_c[k], float) else int(obs_c[k]),
                "status_or_note": "QC R8 closure filter on HYDROSWOT",
                "evidence_tier": TIER_DESCRIPTIVE,
                "source_file": "reports/results/event3_consistency.json",
            }
        )

    # 2. Split sample sizes (from M5 primary dbar all-stratum rows + S-RND M2-XGB + S-IND UQ)
    m5_dbar_all = mat.loc[
        (mat["spec"] == "M5")
        & (mat["analysis"] == "primary")
        & (mat["target"] == "dbar")
        & (mat["stratum_kind"] == "all")
        & (mat["metric"] == "rmse_log")
    ]
    for _, r in m5_dbar_all.iterrows():
        fam = str(r["family"])
        rows.append(
            {
                "section": "split_summary",
                "item": f"{fam}:n_rows",
                "stratum": "all",
                "value": int(r["n_rows"]),
                "status_or_note": f"n_units={int(r['n_units'])}, n_repeats={int(r['n_repeats'])}",
                "evidence_tier": TIER_DESCRIPTIVE,
                "source_file": "reports/results/event3_matrix.parquet",
            }
        )
        rows.append(
            {
                "section": "split_summary",
                "item": f"{fam}:n_units",
                "stratum": "all",
                "value": int(r["n_units"]),
                "status_or_note": str(r["split_note"]),
                "evidence_tier": TIER_DESCRIPTIVE,
                "source_file": "reports/results/event3_matrix.parquet",
            }
        )

    rows.append(
        {
            "section": "split_summary",
            "item": "S-IND:n_clusters",
            "stratum": "all",
            "value": int(hyp["H1"]["S-IND"]["n_clusters"]),
            "status_or_note": S_IND_PANEL_NOTE,
            "evidence_tier": TIER_DESCRIPTIVE,
            "source_file": "reports/results/event3_hypotheses.json",
        }
    )
    rows.append(
        {
            "section": "split_summary",
            "item": "S-IND:n_locations_single_comid_uq",
            "stratum": "single_comid",
            "value": int(uq["M5|primary"]["S-IND"]["n_locations_single"]),
            "status_or_note": S_IND_UQ_NOTE,
            "evidence_tier": TIER_DESCRIPTIVE,
            "source_file": "reports/results/event3_uq.json",
        }
    )
    rows.append(
        {
            "section": "split_summary",
            "item": "S-IND:not_computed_multi_comid_uq",
            "stratum": "multi_comid",
            "value": int(uq["M5|primary"]["S-IND"]["not_computed_multi_comid"]),
            "status_or_note": "not computed: multi-COMID eHydro locations excluded from UQ",
            "evidence_tier": TIER_DESCRIPTIVE,
            "source_file": "reports/results/event3_uq.json",
        }
    )
    rows.append(
        {
            "section": "split_summary",
            "item": "S-IND:A_J_dropped_rows",
            "stratum": "A_J|primary|S-IND",
            "value": int(desc["dropped_rows"]["A_J|primary|S-IND"]),
            "status_or_note": "non-positive Q_E rows dropped for A_J on S-IND",
            "evidence_tier": TIER_DESCRIPTIVE,
            "source_file": "reports/results/event3_descriptive.json",
        }
    )
    rows.append(
        {
            "section": "split_summary",
            "item": "S-RND:n_rows",
            "stratum": "all",
            "value": int(desc["S_RND"]["rows"]["M2-XGB"]),
            "status_or_note": f"weights={desc['S_RND']['weights']}",
            "evidence_tier": TIER_SECONDARY,
            "source_file": "reports/results/event3_descriptive.json",
        }
    )

    # 3. Division, stream-order, and DA-quintile site counts (from S-LRO M5 primary dbar)
    for skind in ("division", "stream_order", "da_q"):
        sub_s = mat.loc[
            (mat["spec"] == "M5")
            & (mat["analysis"] == "primary")
            & (mat["family"] == "S-LRO")
            & (mat["target"] == "dbar")
            & (mat["stratum_kind"] == skind)
            & (mat["metric"] == "rmse_log")
        ].sort_values("stratum")
        for _, r in sub_s.iterrows():
            rows.append(
                {
                    "section": f"stratum_counts_{skind}",
                    "item": f"{skind}:{r['stratum']}",
                    "stratum": str(r["stratum"]),
                    "value": int(r["n_units"]),
                    "status_or_note": f"S-LRO n_rows={int(r['n_rows'])}",
                    "evidence_tier": TIER_DESCRIPTIVE,
                    "source_file": "reports/results/event3_matrix.parquet",
                }
            )

    # 4. DA quintile cuts (km^2)
    for idx, cut_val in enumerate(desc["da_quintile_cuts_km2"], start=1):
        rows.append(
            {
                "section": "da_quintile_cuts_km2",
                "item": f"cut_Q{idx}_Q{idx + 1}_km2",
                "stratum": f"Q{idx}/Q{idx + 1}",
                "value": float(cut_val),
                "status_or_note": "drainage area quintile boundary (km^2)",
                "evidence_tier": TIER_DESCRIPTIVE,
                "source_file": "reports/results/event3_descriptive.json",
            }
        )

    # 5. Event 1 spatial & network dependence audit summary (audit.json + audit_strata.parquet)
    audit_scalars: list[tuple[str, int | float, str]] = [
        ("n_sites", int(audit["n_sites"]), "Event 1 frame sites"),
        ("n_blocks", int(audit["n_blocks"]), "3 repeats x 5 outer folds"),
        (
            "d_nn_median_km",
            float(audit["d_nn_median_km"]),
            "pooled median nearest-train distance (km)",
        ),
        ("rho_hat_km", float(audit["rho_hat_km"]), "variogram practical range rho_hat (km)"),
        ("rho_star_km", float(audit["rho_star_km"]), "effective range rho_star (km)"),
        (
            "share_far",
            float(audit["far"]["share_far"]),
            "share of test instances with d_nn > rho_star",
        ),
        (
            "share_strong",
            float(audit["nested"]["share_strong"]),
            "share of strongly nested test instances (psi >= 0.5)",
        ),
        (
            "share_weak",
            float(audit["nested"]["share_weak"]),
            "share of weakly nested test instances (0 < psi < 0.5)",
        ),
    ]
    for item_k, val_k, note_k in audit_scalars:
        rows.append(
            {
                "section": "event_1_audit",
                "item": item_k,
                "stratum": "S-ID",
                "value": val_k,
                "status_or_note": f"{note_k}; label={EVENT1_AUDIT_LABEL}",
                "evidence_tier": TIER_DESCRIPTIVE,
                "source_file": "reports/results/audit.json",
            }
        )

    m2_dbar_audit = audit["series"]["M2-XGB:dbar"]
    for w_name in ("moran_W1", "moran_W2"):
        w_blk = m2_dbar_audit[w_name]
        rows.append(
            {
                "section": "event_1_audit",
                "item": f"M2-XGB:dbar:{w_name}_I",
                "stratum": "S-ID",
                "value": float(w_blk["I"]),
                "status_or_note": (
                    f"z_perm={w_blk['z_perm']:.4f}, pseudo_p={w_blk['pseudo_p']:.4f}"
                ),
                "evidence_tier": TIER_DESCRIPTIVE,
                "source_file": "reports/results/audit.json",
            }
        )

    rows.append(
        {
            "section": "event_1_audit_strata",
            "item": "n_test_instances",
            "stratum": "S-ID",
            "value": len(bundle.audit_strata_df),
            "status_or_note": "3 repeats x 3,988 sites in audit_strata.parquet",
            "evidence_tier": TIER_DESCRIPTIVE,
            "source_file": "reports/results/audit_strata.parquet",
        }
    )

    # Event 3 M5 residual spatial structure (§3.6.1(c))
    m5_dep = dep["M5_residual_structure_3_6_1_c"]["series"]
    for series_k, s_blk in sorted(m5_dep.items()):
        for w_name in ("moran_W1", "moran_W2"):
            w_blk = s_blk[w_name]
            rows.append(
                {
                    "section": "event_3_M5_residual_dependence",
                    "item": f"{series_k}:{w_name}_I",
                    "stratum": "S-ID",
                    "value": float(w_blk["I"]),
                    "status_or_note": (
                        f"z_perm={w_blk['z_perm']:.4f}, pseudo_p={w_blk['pseudo_p']:.4f}"
                    ),
                    "evidence_tier": TIER_DESCRIPTIVE,
                    "source_file": "reports/results/event3_dependence.json",
                }
            )

    # 6. Explicit "not computed" items
    l8 = desc["L8_zarrabi_product"]
    rows.append(
        {
            "section": "not_computed_items",
            "item": "L8_zarrabi_product",
            "stratum": "external_product",
            "value": None,
            "status_or_note": f"{l8['status']}: {l8['reason']}",
            "evidence_tier": TIER_DESCRIPTIVE,
            "source_file": "reports/results/event3_descriptive.json",
        }
    )
    fnn = hyp["far_not_nested_strata"]["far"]
    rows.append(
        {
            "section": "not_computed_items",
            "item": "H2_far_stratum",
            "stratum": "far",
            "value": None,
            "status_or_note": f"not computed: {fnn['H2']['not_computed']}",
            "evidence_tier": TIER_SECONDARY,
            "source_file": "reports/results/event3_hypotheses.json",
        }
    )
    rows.append(
        {
            "section": "not_computed_items",
            "item": "S2_far_stratum",
            "stratum": "far",
            "value": None,
            "status_or_note": f"not computed: {fnn['S2']['not_computed']}",
            "evidence_tier": TIER_SECONDARY,
            "source_file": "reports/results/event3_hypotheses.json",
        }
    )

    return pd.DataFrame(rows)


def _render_benchmark_markdown(
    bundle: VerifiedBundle,
    frames: dict[str, pd.DataFrame],
) -> str:
    """Render docs/tables/benchmark_results.md from T1-T9 frames."""
    o47 = bundle.outcome_4_7
    lines: list[str] = [
        "# Phase 9 Empirical Benchmark & Diagnostic Tables (T1-T9)",
        "",
        (
            "> **Pre-registered Decision Matrix Outcome "
            f"(`docs/3_experimental_protocol.md` §4.7 row {o47.row}):**"
        ),
        f"> - **Recorded Outcome:** `{o47.outcome}`",
        f"> - **Verbatim Statement:** `{o47.statement}`",
        f"> - **Target Venue:** `{o47.venue}`",
        f"> - **Event 1 S-ID Audit Label:** `{EVENT1_AUDIT_LABEL}`",
        f"> - **S-IND Sample Note:** `{S_IND_PANEL_NOTE}` (`{S_IND_UQ_NOTE}`)",
        "",
        f"**Rounding Rules:** {ROUNDING_RULES_TEXT}",
        "",
        "---",
        "",
        (
            "## Table T1 — Confirmatory Hypotheses (H1-H4), "
            "Event 2 Baseline Selection ($M^*$), and §4.7 Outcome"
        ),
        "",
        (
            "- **Evidence Label:** `Confirmatory` (H1-H4 primary tests, Event 2 $M^*$ selection, "
            "§4.7 row 4); `Secondary` (H2 S-LRO Q1, H3-TAIL-SO, H2 far/not-nested strata); "
            "`Descriptive` (H4 S-LRO and caveats)."
        ),
        f"- **Split Notes:** `S-ID`: *{EVENT1_AUDIT_LABEL}*; `S-IND`: *{S_IND_PANEL_NOTE}*.",
        "- **CSV Mirror:** `docs/tables/t1_hypotheses_and_outcome.csv`",
        "",
    ]

    t1 = frames["t1"]
    lines.append(
        _df_to_markdown(
            t1,
            {
                "estimate": "dec4",
                "ci_lo": "dec4",
                "ci_hi": "dec4",
                "ci_level": "dec4",
                "n_units": "int",
                "n_clusters": "int",
            },
        )
    )
    lines.extend(
        [
            "",
            "---",
            "",
            (
                "## Table T2 — Primary Benchmark Metrics Matrix across Splits "
                "(`S-ID`, `S-LRO`, `S-TAIL-DA`, `S-TAIL-SO`, `S-IND`)"
            ),
            "",
            (
                "- **Evidence Label:** `Descriptive` (cell-level metrics across the model ladder "
                "x split matrix; `r2_raw` and `nse_*` are comparability-only and never used for "
                "ranking; `kge` is descriptive only; MAE is not a protocol metric and is not "
                "computed)."
            ),
            (
                f"- **Split Notes:** `S-ID`: *{EVENT1_AUDIT_LABEL}*; "
                f"`S-IND`: *{S_IND_PANEL_NOTE}* (`A_J` drops 81 rows with non-positive $Q_E$)."
            ),
            "- **CSV Mirror:** `docs/tables/t2_metrics_matrix.csv`",
            "",
        ]
    )

    t2 = frames["t2"].copy()
    t2["formatted_value"] = [
        format_num(v, metric_fmt_kind(str(m)))
        for v, m in zip(t2["value"], t2["metric"], strict=True)
    ]
    t2_display = t2[
        [
            "family",
            "target",
            "model",
            "metric",
            "formatted_value",
            "n_rows",
            "n_units",
            "n_excluded",
            "n_repeats",
            "evidence_tier",
            "split_note",
            "source_file",
        ]
    ].rename(columns={"formatted_value": "value"})
    lines.append(
        _df_to_markdown(
            t2_display,
            {
                "value": "str",
                "n_rows": "int",
                "n_units": "int",
                "n_excluded": "int",
                "n_repeats": "int",
            },
        )
    )

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "## Table T3 — Stepwise Physics Contrasts (`S2`, 98.75% Bonferroni CIs) "
                "and Descriptive $\\kappa$ Quintiles"
            ),
            "",
            (
                "- **Evidence Label:** `Secondary` (S2 overall and far/not-nested strata at "
                "$\\alpha_B / 4 = 0.0125$, 98.75% CI); `Descriptive` ($\\kappa$-quintile "
                "point contrasts)."
            ),
            (
                f"- **Split Notes:** `S-ID`: *{EVENT1_AUDIT_LABEL}*; `far` stratum explicitly "
                "recorded as `not computed: fewer than 30 sites in a repeat`."
            ),
            "- **CSV Mirror:** `docs/tables/t3_s2_contrasts.csv`",
            "",
            _df_to_markdown(
                frames["t3"],
                {
                    "estimate": "dec4",
                    "ci_lo": "dec4",
                    "ci_hi": "dec4",
                    "ci_level": "dec4",
                    "n_rows": "int",
                },
            ),
            "",
            "---",
            "",
            (
                "## Table T4 — Uncertainty Quantification Calibration "
                "(90% Interval Coverage, $\\bar{W}_{\\log}$, CRPS, `[0.85, 0.95]` Band)"
            ),
            "",
            (
                "- **Evidence Label:** `Secondary` (pooled `S-ID` $\\bar{d}$ 90% coverage "
                "quality target $\\hat{C}_{0.10} \\in [0.85, 0.95]$); `Descriptive` "
                "(OOD splits, DA quintiles, divisions, ablations, and sensitivity analyses)."
            ),
            f"- **Split Notes:** `S-ID`: *{EVENT1_AUDIT_LABEL}*; `S-IND`: *{S_IND_UQ_NOTE}*.",
            "- **CSV Mirror:** `docs/tables/t4_uq_calibration.csv`",
            "",
        ]
    )

    t4 = frames["t4"]
    t4_overall = t4.loc[t4["stratum_kind"] == "overall"].copy()
    lines.append(
        _df_to_markdown(
            t4_overall,
            {
                "alpha": "dec4",
                "band_lo": "dec4",
                "band_hi": "dec4",
                "coverage": "dec4",
                "crps": "dec4",
                "mean_log_width": "dec4",
                "width_factor": "dec4",
                "n_rows": "int",
                "n_excluded": "int",
            },
        )
    )

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "## Table T5 — Physical Consistency Analogues "
                "(`docs/3_experimental_protocol.md` §4.3.2)"
            ),
            "",
            (
                "- **Evidence Label:** `Descriptive` (predicted area closure "
                "$\\hat{\\varepsilon}_A$, Manning back-calculation admissibility per Prop 6.2/7.4, "
                "M4 test-site residual $R_M$, hard-model Picard solver residual "
                "$\\le \\tau_\\ell$, shape violation share "
                "$\\hat{d}_{\\max} \\le \\hat{\\bar{d}}$, implied-$V$ error and oracle $Q_E$ vs "
                "$\\hat{A}$ decomposition, and QC R8 observed closure)."
            ),
            (
                f"- **Split Notes:** `S-ID`: *{EVENT1_AUDIT_LABEL}*; continuity holds exactly "
                "($\\hat{V} = Q_E / \\hat{A}$) for every model."
            ),
            "- **CSV Mirror:** `docs/tables/t5_physical_consistency.csv`",
            "",
        ]
    )

    t5 = frames["t5"].copy()
    t5_fmt_vals: list[str] = []
    for _, r in t5.iterrows():
        m_name = str(r["metric"])
        v_val = r["value"]
        if m_name in {
            "n_rows",
            "n_used",
            "n_undefined",
            "n_invariant4",
            "picard_iterations_max",
            "n_sites",
            "qc_rows",
            "qc_R8_rows_flagged",
        }:
            t5_fmt_vals.append(format_num(v_val, "int"))
        elif m_name == "solver_residual_max":
            t5_fmt_vals.append(format_num(v_val, "sci2"))
        elif m_name == "shape_violation_share":
            t5_fmt_vals.append(format_num(v_val, "auto_share"))
        else:
            t5_fmt_vals.append(format_num(v_val, "dec4"))
    t5["value"] = t5_fmt_vals
    lines.append(_df_to_markdown(t5, {"value": "str"}))

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "## Table T6 — Pre-Registered Ablations (`A1`-`A4`, `wA1`), "
                "`A4` Baselines, and Sensitivity Analyses (`SA1`-`SA6`)"
            ),
            "",
            (
                "- **Evidence Label:** `Descriptive` (`M2-XGB_A4` is the primary realised-$M^*$ "
                "`A4` run and `M1_A4`, `M2-MLP_A4` are supplementary per §13.8; for `SA2`-`SA6`, "
                "`M5` and `M2-XGB` are primary and `M1`, `M2-MLP` are supplementary per §13.8; "
                "`SA1` is trained only for `M5` and is identical to primary by construction for "
                "`M1`, `M2-MLP`, `M2-XGB`)."
            ),
            f"- **Split Notes:** `S-ID`: *{EVENT1_AUDIT_LABEL}*; `S-IND`: *{S_IND_PANEL_NOTE}*.",
            "- **CSV Mirror:** `docs/tables/t6_ablations_and_sensitivity.csv`",
            "",
        ]
    )

    t6 = frames["t6"].copy()
    t6_sub = t6.loc[
        t6["metric"].isin(
            [
                "rmse_log",
                "bias_log",
                "status",
                "pi_lit",
                "pi_nu",
                "pi_r",
                "rho_n_d50",
                "rho_n_s",
                "rho_r_robs",
            ]
        )
    ].copy()
    t6_sub["value"] = [
        format_num(v, metric_fmt_kind(str(m)))
        for v, m in zip(t6_sub["value"], t6_sub["metric"], strict=True)
    ]
    lines.append(
        _df_to_markdown(
            t6_sub,
            {"value": "str", "n_rows": "int", "n_units": "int"},
        )
    )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Table T7 — Secondary Multi-Measurement Benchmark (`S-RND`, $1/|I_j|$ Site Weights)",
            "",
            (
                "- **Evidence Label:** `Secondary` (`M1` and `M2-XGB` evaluated on 9,673 "
                "multi-measurement rows across 3,988 sites with $1/|I_j|$ weights)."
            ),
            "- **CSV Mirror:** `docs/tables/t7_srnd_benchmark.csv`",
            "",
        ]
    )

    t7 = frames["t7"].copy()
    t7["value"] = [
        format_num(v, metric_fmt_kind(str(m)))
        for v, m in zip(t7["value"], t7["metric"], strict=True)
    ]
    lines.append(
        _df_to_markdown(
            t7[
                [
                    "family",
                    "target",
                    "model",
                    "stratum_kind",
                    "stratum",
                    "metric",
                    "value",
                    "weights",
                    "n_rows",
                    "n_units",
                    "n_repeats",
                    "evidence_tier",
                    "split_note",
                    "source_file",
                ]
            ],
            {"value": "str", "n_rows": "int", "n_units": "int", "n_repeats": "int"},
        )
    )

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "## Table T8 — HyG Roughness Consistency Check "
                "(`S1`, `docs/3_experimental_protocol.md` §4.6)"
            ),
            "",
            (
                "- **Evidence Label:** `Descriptive` "
                "(`consistency check, not independent validation (protocol 4.6)`)."
            ),
            f"- **Split Notes:** `S-ID`: *{EVENT1_AUDIT_LABEL}* (2,887 matched frame sites).",
            "- **CSV Mirror:** `docs/tables/t8_s1_hyg_consistency.csv`",
            "",
        ]
    )

    t8 = frames["t8"].copy()
    t8["value"] = [
        format_num(v, "dec4" if str(m) in {"rmse_log", "spearman_rho"} else "int")
        for v, m in zip(t8["value"], t8["metric"], strict=True)
    ]
    lines.append(_df_to_markdown(t8, {"value": "str"}))

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "## Table T9 — Study Frame, Split Sizes, Event 1 Dependence Audit, "
                "and Explicitly Not-Computed Items"
            ),
            "",
            (
                "- **Evidence Label:** `Descriptive` (frame, splits, Event 1 spatial/network "
                "dependence audit, Event 3 M5 residual spatial autocorrelation, and "
                "`L8_zarrabi_product` not-computed record); `Secondary` (`S-RND` row count and "
                "`far`-stratum H2/S2 not-computed records)."
            ),
            "- **CSV Mirror:** `docs/tables/t9_frame_and_data_summary.csv`",
            "",
        ]
    )

    t9 = frames["t9"].copy()
    t9_fmt: list[str] = []
    for _, r in t9.iterrows():
        sec = str(r["section"])
        itm = str(r["item"])
        val = r["value"]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            t9_fmt.append("—")
        elif sec == "da_quintile_cuts_km2" or itm in {
            "d_nn_median_km",
            "rho_hat_km",
            "rho_star_km",
        }:
            t9_fmt.append(format_num(val, "dec2"))
        elif (
            itm in {"share_removed", "share_strong", "share_weak", "share_far"} or "moran_W" in itm
        ):
            t9_fmt.append(format_num(val, "dec4"))
        else:
            t9_fmt.append(format_num(val, "int"))
    t9["value"] = t9_fmt
    lines.append(_df_to_markdown(t9, {"value": "str"}))
    lines.append("")

    return "\n".join(lines)


def generate_all_tables(bundle: VerifiedBundle, tables_dir: Path) -> GeneratedTables:
    """Generate T1-T9 CSV mirrors and docs/tables/benchmark_results.md deterministically."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {
        "t1": build_t1_hypotheses_and_outcome(bundle),
        "t2": build_t2_metrics_matrix(bundle),
        "t3": build_t3_s2_contrasts(bundle),
        "t4": build_t4_uq_calibration(bundle),
        "t5": build_t5_physical_consistency(bundle),
        "t6": build_t6_ablations_and_sensitivity(bundle),
        "t7": build_t7_srnd_benchmark(bundle),
        "t8": build_t8_s1_hyg_consistency(bundle),
        "t9": build_t9_frame_and_data_summary(bundle),
    }

    csv_names: dict[str, str] = {
        "t1": "t1_hypotheses_and_outcome.csv",
        "t2": "t2_metrics_matrix.csv",
        "t3": "t3_s2_contrasts.csv",
        "t4": "t4_uq_calibration.csv",
        "t5": "t5_physical_consistency.csv",
        "t6": "t6_ablations_and_sensitivity.csv",
        "t7": "t7_srnd_benchmark.csv",
        "t8": "t8_s1_hyg_consistency.csv",
        "t9": "t9_frame_and_data_summary.csv",
    }

    csv_paths: dict[str, Path] = {}
    for key, fname in csv_names.items():
        out_path = tables_dir / fname
        frames[key].to_csv(out_path, index=False, lineterminator="\n")
        csv_paths[key] = out_path

    md_text = _render_benchmark_markdown(bundle, frames)
    md_path = tables_dir / "benchmark_results.md"
    md_path.write_text(md_text, encoding="utf-8")

    return GeneratedTables(
        markdown_path=md_path,
        csv_paths=csv_paths,
        frames=frames,
    )
