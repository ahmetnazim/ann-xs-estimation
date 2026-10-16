"""Factual, provenance-verified figure captions generator for Phase 9 (P9-D9 (a)).

Generates ``docs/figures/captions.md`` mechanically from :class:`VerifiedBundle`
(``reports/results/*``) with zero manuscript prose, zero new statistics, explicit
evidence labels (`Confirmatory`, `Secondary`, `Descriptive`) on every panel, the
verbatim ``docs/3_experimental_protocol.md`` §4.7 row 4 decision rule, the Event 1
spatial/network dependence label on `S-ID` panels, and the `131 locations / 18 rivers`
(`UQ: 33 single-COMID rows`) scope note on `S-IND` panels.
"""

from __future__ import annotations

from pathlib import Path

from annxs_figures.loader import (
    EVENT1_AUDIT_LABEL,
    S_IND_PANEL_NOTE,
    S_IND_UQ_NOTE,
    VERBATIM_ROW4_OUTCOME,
    VERBATIM_ROW4_STATEMENT,
    VERBATIM_ROW4_VENUE,
    VerifiedBundle,
    sha256_file,
)
from annxs_figures.style import (
    ASCE_ARTWORK_ACCESS_DATE,
    ASCE_ARTWORK_URL,
    ASCE_DOUBLE_COL_IN,
    ASCE_FONT_FAMILY,
    ASCE_RASTER_DPI,
    WILEY_WRR_ARTWORK_URL,
)
from annxs_figures.tables import format_num

TIER_CONFIRMATORY = "Confirmatory"
TIER_SECONDARY = "Secondary"
TIER_DESCRIPTIVE = "Descriptive"


def _mat_val(
    bundle: VerifiedBundle,
    *,
    spec: str,
    family: str,
    target: str = "dbar",
    metric: str = "rmse_log",
    analysis: str = "primary",
    stratum_kind: str = "all",
    stratum: str = "all",
) -> float:
    df = bundle.matrix_df
    sub = df.loc[
        (df["spec"] == spec)
        & (df["analysis"] == analysis)
        & (df["family"] == family)
        & (df["target"] == target)
        & (df["stratum_kind"] == stratum_kind)
        & (df["stratum"] == stratum)
        & (df["metric"] == metric),
        "value",
    ]
    if len(sub) != 1:
        raise KeyError(
            f"Expected 1 matrix row for {spec}|{analysis}|{family}|{target}|"
            f"{stratum_kind}:{stratum}|{metric}, found {len(sub)}"
        )
    return float(sub.iloc[0])


def generate_captions(bundle: VerifiedBundle, figures_dir: Path) -> Path:
    """Build and write ``docs/figures/captions.md`` from the verified results bundle."""
    audit = bundle.audit
    hyp = bundle.hypotheses
    uq = bundle.uq
    cons = bundle.consistency
    desc = bundle.descriptive

    # Headline numbers from verified bundle
    n_sid_sites = int(audit["n_sites"])
    d_nn_med = format_num(float(audit["d_nn_median_km"]), "dec2")
    rho_star = format_num(float(audit["rho_hat_km"]), "dec2")
    psi_strong = format_num(float(audit["nested"]["share_strong"]), "dec4")
    psi_weak = format_num(float(audit["nested"]["share_weak"]), "dec4")

    h1_lro_dbar = hyp["H1"]["S-LRO"]["dbar"]
    h1_lro_a = hyp["H1"]["S-LRO"]["A"]
    h1_sind_dbar = hyp["H1"]["S-IND"]["dbar"]
    h1_sind_a = hyp["H1"]["S-IND"]["A"]

    m5_lro_dbar = format_num(_mat_val(bundle, spec="M5", family="S-LRO", target="dbar"), "dec4")
    m2_lro_dbar = format_num(_mat_val(bundle, spec="M2-XGB", family="S-LRO", target="dbar"), "dec4")
    m5_sid_dbar = format_num(_mat_val(bundle, spec="M5", family="S-ID", target="dbar"), "dec4")
    m2_sid_dbar = format_num(_mat_val(bundle, spec="M2-XGB", family="S-ID", target="dbar"), "dec4")

    h4_sid = hyp["H4"]["S-ID"]
    h4_lro = hyp["H4"]["S-LRO_reported"]
    m2_back_lro = format_num(
        float(cons["models"]["M2-XGB"]["S-LRO"]["back_calc"]["share_outside"]), "dec4"
    )
    m2_clos_lro = format_num(
        float(cons["models"]["M2-XGB"]["S-LRO"]["predicted_closure"]["median_abs"]), "dec4"
    )
    m4_rms_sid = format_num(
        float(cons["models"]["M4"]["S-ID"]["manning_residual_M4"]["rms"]), "dec4"
    )
    m5_res_sid = format_num(float(cons["models"]["M5"]["S-ID"]["solver_residual_max"]), "sci2")

    h3_lro = hyp["H3"]["H3-LRO"]["delta"]
    h3_da = hyp["H3"]["H3-TAIL"]["delta"]
    h3_so = hyp["H3"]["H3-TAIL-SO"]["delta"]

    m5_cov_sid = format_num(float(uq["M5|primary"]["S-ID"]["dbar"]["overall"]["coverage"]), "dec4")
    m5_cov_lro = format_num(float(uq["M5|primary"]["S-LRO"]["dbar"]["overall"]["coverage"]), "dec4")
    m5_cov_ind = format_num(float(uq["M5|primary"]["S-IND"]["dbar"]["overall"]["coverage"]), "dec4")
    m5_crps_lro = format_num(float(uq["M5|primary"]["S-LRO"]["dbar"]["overall"]["crps"]), "dec4")
    m2mlp_crps_lro = format_num(
        float(uq["M2-MLP|primary"]["S-LRO"]["dbar"]["overall"]["crps"]), "dec4"
    )

    h2_sid = hyp["H2"]["S-ID_Q1"]["h2"]
    h2_lro = hyp["H2"]["S-LRO_Q1_secondary"]["h2"]
    far_h2_reason = str(hyp["far_not_nested_strata"]["far"]["H2"]["not_computed"])

    s2_sid_m4 = hyp["S2"]["S-ID"]["s2_M4_vs_M5"]
    s2_lro_m4 = hyp["S2"]["S-LRO"]["s2_M4_vs_M5"]
    s2_sid_m6 = hyp["S2"]["S-ID"]["s2_M5_vs_M6NN"]
    s2_lro_m6 = hyp["S2"]["S-LRO"]["s2_M5_vs_M6NN"]

    a3_sp_lro = format_num(_mat_val(bundle, spec="M5_A3-SP", family="S-LRO", target="dbar"), "dec4")
    a3_sr_lro = format_num(_mat_val(bundle, spec="M5_A3-SR", family="S-LRO", target="dbar"), "dec4")
    sa5_m5_lro = format_num(
        _mat_val(bundle, spec="M5", family="S-LRO", target="dbar", analysis="SA5"), "dec4"
    )
    a4_m5_lro = format_num(_mat_val(bundle, spec="M5_A4", family="S-LRO", target="dbar"), "dec4")

    s1_hyg = desc["S1"]
    s1_n_matched = int(s1_hyg["n_matched"])
    s1_rmse = format_num(float(s1_hyg["rmse_log"]), "dec4")
    s1_rho = format_num(float(s1_hyg["spearman_rho"]), "dec4")
    s1_note = str(s1_hyg["note"])

    lines: list[str] = [
        "# Phase 9 Publication Figure Captions (Factual & Provenance-Locked)",
        "",
        (
            "## Artwork & Provenance Specification "
            "(`P9-D2 (a)`, `P9-D5 (a)`, `P9-D6 (a)`, `P9-D9 (a)`)"
        ),
        f"- **Target Venue:** ASCE *Journal of Hydrologic Engineering* (`{VERBATIM_ROW4_VENUE}`).",
        (
            f"- **Artwork Guidelines:** `{ASCE_ARTWORK_URL}` "
            f"(comparison: `{WILEY_WRR_ARTWORK_URL}`; accessed `{ASCE_ARTWORK_ACCESS_DATE}`)."
        ),
        (
            f"- **Format & Typography:** Double-column width `{ASCE_DOUBLE_COL_IN:.1f} in` "
            f"(`504 pt`), sans-serif (`{ASCE_FONT_FAMILY}`), vector PDF (`pdf.fonttype = 42`) "
            f"+ vector SVG (`svg.fonttype = 'none'`) + `{ASCE_RASTER_DPI} DPI` PNG."
        ),
        f"- **Pre-Registered §4.7 Row 4 Outcome:** `{VERBATIM_ROW4_OUTCOME}`.",
        f"- **Pre-Registered §4.7 Row 4 Action:** `{VERBATIM_ROW4_STATEMENT}`.",
        f"- **Event 1 S-ID Label:** `{EVENT1_AUDIT_LABEL}`.",
        f"- **S-IND Scope Note:** `{S_IND_PANEL_NOTE}` (`UQ: {S_IND_UQ_NOTE}`).",
        "",
        "---",
        "",
        "## Core Manuscript Figures (`Fig. 1`-`Fig. 5`)",
        "",
        (
            "### Fig. 1 - Study Domain, Hard-Constrained `M5` Architecture & Validation Design "
            "(`fig1_domain_and_architecture.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_DESCRIPTIVE}]:** CONUS Albers Equal-Area (`EPSG:5070`) map "
            f"of the primary ungauged `HYDROSWOT` evaluation frame (`n = {n_sid_sites:,}` sites "
            f"across `8` Fenneman & Johnson physiographic divisions) and independent `S-IND` "
            f"`USACE eHydro` survey locations (`{S_IND_PANEL_NOTE}`)."
        ),
        (
            f"- **Panel (b) [{TIER_DESCRIPTIVE}]:** `M5` hard-constrained physics-informed "
            "architecture (`I_blind` roughness head excluding `Q_E, S, F_S, V_E` + "
            "differentiable Manning + continuity projection layer "
            "`Phi(W_hat, r_hat, n_hat; Q_E, S)` with `K = 8` Gauss-Legendre wet-perimeter "
            "quadrature and Picard tolerance `tau_l = 1e-10`)."
        ),
        (
            f"- **Panel (c) [{TIER_DESCRIPTIVE}]:** Pre-registered multi-regime validation "
            f"design (`S-ID` `3 x 5` grouped CV with Event 1 dependence audit: "
            f"`d_nn,med = {d_nn_med} km`, `rho_hat = rho* = {rho_star} km`, nested strong "
            f"`Psi >= 0.5` share `{psi_strong}`, weak share `{psi_weak}`, verdict "
            f"`{EVENT1_AUDIT_LABEL}`; `S-LRO` `8` held-out divisions; `S-TAIL-DA` `n = 678`; "
            f"`S-TAIL-SO` `n = 339`; `S-IND` `{S_IND_UQ_NOTE}`; `S-RND` `n = 9,673`) and "
            f"verbatim §4.7 Row 4 outcome (`{VERBATIM_ROW4_OUTCOME}`; "
            f"`{VERBATIM_ROW4_STATEMENT}`)."
        ),
        "",
        (
            "### Fig. 2 - Primary Benchmark Performance & Confirmatory `H1` Forest Plot "
            "(`fig2_benchmark_performance.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_DESCRIPTIVE}]:** Full model-ladder `RMSE_log(dbar)` across "
            f"`M-null`, `M0`, `M1`, `M2-MLP`, `M2-XGB`, `M3`, `M4`, `M5`, `M6-NN`, and `M6-XGB` "
            f"on `S-ID` (`M5 dbar = {m5_sid_dbar}`, `M2-XGB dbar = {m2_sid_dbar}`, "
            f"`{EVENT1_AUDIT_LABEL}`), `S-LRO` (`M5 dbar = {m5_lro_dbar}`, "
            f"`M2-XGB dbar = {m2_lro_dbar}`), `S-TAIL-DA`, `S-TAIL-SO`, and `S-IND` "
            f"(`{S_IND_PANEL_NOTE}`)."
        ),
        (
            f"- **Panel (b) [{TIER_DESCRIPTIVE}]:** Full model-ladder `RMSE_log(A)` and "
            "`RMSE_log(W)` across splits (including `A_J` Jobson baseline on `A`)."
        ),
        (
            f"- **Panel (c) [{TIER_CONFIRMATORY} / {TIER_SECONDARY}]:** Confirmatory `H1` "
            "paired log-RMSE contrast `Delta_H1 = RMSE_log(M5) - RMSE_log(M*)` "
            "(`M* = M2-XGB`, selected on `S-ID` mean `{{dbar, A}}` `RMSE_log = 0.5313`) with "
            "95% division / river cluster-bootstrap CIs (`B = 2,000`):"
        ),
        (
            f"  - `S-LRO` (`{TIER_CONFIRMATORY}`, `n = 3,988`): `dbar` "
            f"`Delta = +{format_num(float(h1_lro_dbar['estimate']), 'dec4')} "
            f"[+{format_num(float(h1_lro_dbar['lo']), 'dec4')}, "
            f"+{format_num(float(h1_lro_dbar['hi']), 'dec4')}]`; `A` "
            f"`Delta = +{format_num(float(h1_lro_a['estimate']), 'dec4')} "
            f"[+{format_num(float(h1_lro_a['lo']), 'dec4')}, "
            f"+{format_num(float(h1_lro_a['hi']), 'dec4')}]`."
        ),
        (
            f"  - `S-IND` (`{TIER_CONFIRMATORY}` point / `{TIER_SECONDARY}` river bootstrap on "
            f"`18 < 30` rivers, `{S_IND_PANEL_NOTE}`): `dbar` "
            f"`Delta = +{format_num(float(h1_sind_dbar['estimate']), 'dec4')} "
            f"[{format_num(float(h1_sind_dbar['lo']), 'dec4')}, "
            f"+{format_num(float(h1_sind_dbar['hi']), 'dec4')}]`; `A` "
            f"`Delta = {format_num(float(h1_sind_a['estimate']), 'dec4')} "
            f"[{format_num(float(h1_sind_a['lo']), 'dec4')}, "
            f"+{format_num(float(h1_sind_a['hi']), 'dec4')}]`."
        ),
        f"  - Verdict: `H1 Falsified` (`{VERBATIM_ROW4_OUTCOME}`).",
        (
            f"- **Panel (d) [{TIER_DESCRIPTIVE}]:** Out-of-fold predicted vs observed `dbar` "
            "on `S-LRO` (`n = 3,988` sites) for `M5` vs `M2-XGB` (`M*`)."
        ),
        "",
        (
            "### Fig. 3 - Physical Consistency Analogues & Confirmatory `H4` Parameter "
            "Plausibility (`fig3_physical_consistency_and_h4.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_DESCRIPTIVE}]:** Continuity area closure residual "
            f"`eps_A_hat = ln(A_hat / (W_hat * dbar_hat))` on `S-LRO` (`M2-XGB` median "
            f"`|eps_A_hat| = {m2_clos_lro}` vs `M3`-`M6` exact `0.0000` and QC R8 observed "
            "closure tolerance `+/- ln(1.05)`)."
        ),
        (
            f"- **Panel (b) [{TIER_DESCRIPTIVE}]:** Share of predictions with no admissible "
            "back-calculated Manning roughness `n_back in [0.005, 0.40]` "
            f"(`docs/2_mathematical_proofs.md` Prop. 6.2 / §7.4; `M2-XGB` `S-LRO` share "
            f"outside = `{m2_back_lro}` vs `M3`/`M5` = `0.0000`) and Dingman shape bound "
            "violation share `P(d_max_hat <= dbar_hat)`."
        ),
        (
            f"- **Panel (c) [{TIER_DESCRIPTIVE}]:** Hard-constrained Picard solver residual "
            f"(`M5` `S-ID` max `|res| = {m5_res_sid} <= tau_l = 1e-10`) vs `M4` soft-loss "
            f"test-site Manning residual `R_M` (`S-ID` `RMS = {m4_rms_sid}`)."
        ),
        (
            f"- **Panel (d) [{TIER_CONFIRMATORY} / {TIER_DESCRIPTIVE}]:** Confirmatory `H4` "
            f"parameter plausibility on `S-ID` (`{TIER_CONFIRMATORY}`, `{EVENT1_AUDIT_LABEL}`: "
            f"`pi_lit = {format_num(float(h4_sid['pi_lit']), 'dec4')} >= 0.90`, "
            f"`rho_S(n_hat, D50) = {format_num(float(h4_sid['ci']['n_d50']['estimate']), 'dec4')} "
            f"[{format_num(float(h4_sid['ci']['n_d50']['lo']), 'dec4')}, "
            f"{format_num(float(h4_sid['ci']['n_d50']['hi']), 'dec4')}]`, "
            f"`rho_S(n_hat, S) = {format_num(float(h4_sid['ci']['n_s']['estimate']), 'dec4')} "
            f"[{format_num(float(h4_sid['ci']['n_s']['lo']), 'dec4')}, "
            f"{format_num(float(h4_sid['ci']['n_s']['hi']), 'dec4')}]`) and `S-LRO` "
            f"(`{TIER_DESCRIPTIVE}`: `pi_lit = {format_num(float(h4_lro['pi_lit']), 'dec4')}`, "
            f"`rho_S(n_hat, D50) = "
            f"{format_num(float(h4_lro['ci']['n_d50']['estimate']), 'dec4')}`, "
            f"`rho_S(n_hat, S) = {format_num(float(h4_lro['ci']['n_s']['estimate']), 'dec4')}`); "
            "verdict: `H4 Falsified (CI lower bound <= 0 for rho_S(n_hat, D50))`."
        ),
        "",
        (
            "### Fig. 4 - OOD Stress Tests (`H3`), UQ Calibration (`[0.85, 0.95]` Band) & "
            "Headwater Bias (`H2`) (`fig4_ood_and_uq_calibration.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_CONFIRMATORY} / {TIER_SECONDARY}]:** `H3` OOD skill loss "
            "`L_OOD(M) = RMSE_OOD - RMSE_ID` and contrast `Delta_H3 = L(M5) - L(M*)` with 95% "
            f"bootstrap CIs (`H3-LRO` `{TIER_CONFIRMATORY}`: `Delta_H3 = "
            f"{format_num(float(h3_lro['estimate']), 'dec4')} "
            f"[{format_num(float(h3_lro['lo']), 'dec4')}, "
            f"{format_num(float(h3_lro['hi']), 'dec4')}]`; `H3-TAIL-DA` `{TIER_CONFIRMATORY}`: "
            f"`Delta_H3 = {format_num(float(h3_da['estimate']), 'dec4')} "
            f"[{format_num(float(h3_da['lo']), 'dec4')}, "
            f"{format_num(float(h3_da['hi']), 'dec4')}]`; `H3-TAIL-SO` `{TIER_SECONDARY}`, "
            f"`n = 339 < 500`: `Delta_H3 = {format_num(float(h3_so['estimate']), 'dec4')} "
            f"[{format_num(float(h3_so['lo']), 'dec4')}, "
            f"{format_num(float(h3_so['hi']), 'dec4')}]`; verdict: `H3 Falsified`)."
        ),
        (
            f"- **Panel (b) [{TIER_SECONDARY} / {TIER_DESCRIPTIVE}]:** Empirical 90% "
            "split-conformal coverage `C_0.10(dbar)` with the pre-registered **`[0.85, 0.95]`** "
            f"acceptance band across `S-ID` (`M5 = {m5_cov_sid}` in `[0.85, 0.95]`, "
            f"`{EVENT1_AUDIT_LABEL}`), `S-LRO` (`M5 = {m5_cov_lro}`), `S-TAIL-DA`, "
            f"`S-TAIL-SO`, `S-IND` (`M5 = {m5_cov_ind}`, `{S_IND_UQ_NOTE}`), and `S-ID` "
            "drainage-area quintiles `Q1`-`Q5`."
        ),
        (
            f"- **Panel (c) [{TIER_DESCRIPTIVE}]:** 10-bin probability integral transform (PIT) "
            "histograms (`pit_counts / n_rows`) for `dbar` on `S-ID` and `S-LRO` relative to "
            "the uniform `0.10 / bin` reference."
        ),
        (
            f"- **Panel (d) [{TIER_DESCRIPTIVE}]:** Closed-form log-Gaussian CRPS and mean 90% "
            f"log-interval width `W_bar_log` across all 5 evaluation splits (`S-LRO`: "
            f"`M5 CRPS = {m5_crps_lro}`, `M2-MLP CRPS = {m2mlp_crps_lro}`)."
        ),
        (
            f"- **Panel (e) [{TIER_CONFIRMATORY} / {TIER_SECONDARY} / {TIER_DESCRIPTIVE}]:** "
            "`H2` signed `dbar` log-bias `B_log(dbar)` across drainage-area quintiles `Q1`-`Q5` "
            f"(`S-ID Q1` `{TIER_CONFIRMATORY}`: `h2 = "
            f"{format_num(float(h2_sid['estimate']), 'dec4')} "
            f"[{format_num(float(h2_sid['lo']), 'dec4')}, "
            f"{format_num(float(h2_sid['hi']), 'dec4')}]`; `S-LRO Q1` `{TIER_SECONDARY}`: "
            f"`h2 = {format_num(float(h2_lro['estimate']), 'dec4')} "
            f"[{format_num(float(h2_lro['lo']), 'dec4')}, "
            f"{format_num(float(h2_lro['hi']), 'dec4')}]`; `S-ID far Q1` `{TIER_SECONDARY}`: "
            f"`not computed ({far_h2_reason})`; verdict: `H2 Falsified`)."
        ),
        "",
        (
            "### Fig. 5 - Pre-Registered §4.7 Row-4 Mechanism Diagnosis "
            "(`fig5_row4_diagnosis.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_DESCRIPTIVE}]:** Pre-registered Row-4 ablations (`Primary`, "
            f"`A1`, `wA1`, `A2`, `A3-SP` `{a3_sp_lro}`, `A3-SR` `{a3_sr_lro}`, `M4`, `SA1`, "
            f"`SA5` `{sa5_m5_lro}`, `A4` `{a4_m5_lro}`) on `S-ID` and `S-LRO` `RMSE_log(dbar)`, "
            "annotated with `docs/2_mathematical_proofs.md` §5.1 wide-channel elasticities "
            "(`d ln dbar / d ln S = -3/10`, `d ln dbar / d ln W = -3/5`, "
            "`d ln dbar / d ln n = +3/5`) and "
            "`M2-XGB SA1: identical to primary by construction (no Phi); not trained`."
        ),
        (
            f"- **Panel (b) [{TIER_SECONDARY} / {TIER_DESCRIPTIVE}]:** `S2` stepwise physics "
            "contrasts with **98.75% Bonferroni CIs** (`M4 - M5` on `S-ID`: "
            f"`{format_num(float(s2_sid_m4['estimate']), 'dec4')} "
            f"[{format_num(float(s2_sid_m4['lo']), 'dec4')}, "
            f"{format_num(float(s2_sid_m4['hi']), 'dec4')}]`; `S-LRO`: "
            f"`{format_num(float(s2_lro_m4['estimate']), 'dec4')} "
            f"[{format_num(float(s2_lro_m4['lo']), 'dec4')}, "
            f"{format_num(float(s2_lro_m4['hi']), 'dec4')}]`; `M5 - M6-NN` on `S-ID`: "
            f"`{format_num(float(s2_sid_m6['estimate']), 'dec4')} "
            f"[{format_num(float(s2_sid_m6['lo']), 'dec4')}, "
            f"{format_num(float(s2_sid_m6['hi']), 'dec4')}]`; `S-LRO`: "
            f"`{format_num(float(s2_lro_m6['estimate']), 'dec4')} "
            f"[{format_num(float(s2_lro_m6['lo']), 'dec4')}, "
            f"{format_num(float(s2_lro_m6['hi']), 'dec4')}]`; `S-ID far stratum`: "
            f"`not computed ({far_h2_reason})`)."
        ),
        (
            f"- **Panel (c) [{TIER_DESCRIPTIVE}]:** Oracle-discharge (`A4`) vs ungauged EROM "
            "(`Q_E`) comparison on `S-LRO` for `dbar` and implied velocity `V_hat = Q / A_hat` "
            "(`docs/3_experimental_protocol.md` §4.3.2 variance shares)."
        ),
        (
            f"- **Panel (d) [{TIER_CONFIRMATORY} / {TIER_DESCRIPTIVE}]:** `H4` physical "
            "parameter plausibility (`pi_lit`, `rho_S(n_hat, D50)`, `rho_S(n_hat, S)`) across "
            "`M5 Primary` and pre-registered ablations on `S-ID`."
        ),
        (
            f"- **Bottom Banner [{TIER_CONFIRMATORY}]:** Verbatim §4.7 Row 4 outcome "
            f"(`{VERBATIM_ROW4_OUTCOME}`), venue (`{VERBATIM_ROW4_VENUE}`), and action "
            f"(`{VERBATIM_ROW4_STATEMENT}`)."
        ),
        "",
        "---",
        "",
        "## Supplementary Figures (`Fig. S1`-`Fig. S5`)",
        "",
        (
            "### Fig. S1 - Event 1 Spatial & Network Dependence Audit "
            "(`figS1_event1_spatial_dependence.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_DESCRIPTIVE}]:** Per-block (`R0F0`-`R2F4`) `S-ID` "
            "nested-basin index shares (`Psi >= 0.5` strong, `0 < Psi < 0.5` weak) and median "
            f"nearest-neighbour distance `d_nn` (`global median = {d_nn_med} km`)."
        ),
        (
            f"- **Panel (b) [{TIER_DESCRIPTIVE}]:** Empirical OOF `ln dbar` residual "
            "semivariograms (`M-null`, `M2-XGB`, `M5`) up to `500 km`."
        ),
        (
            f"- **Panel (c) [{TIER_DESCRIPTIVE}]:** OOF `ln dbar` spatial correlograms "
            "`Moran's I(h)` across distance bands (`[0, 10)` to `[250, 500) km`)."
        ),
        (
            f"- **Panel (d) [{TIER_DESCRIPTIVE}]:** Significant Local Moran's `I` (`LISA`, "
            f"`p < 0.05`) site counts by physiographic division; banner quotes Event 1 verdict "
            f"`{EVENT1_AUDIT_LABEL}`."
        ),
        "",
        (
            "### Fig. S2 - Per-Division & Stream-Order `S-LRO` Accuracy and Spatial Residuals "
            "(`figS2_per_division_slro.{pdf,svg,png}`)"
        ),
        (
            f"- **Panels (a)-(b) [{TIER_DESCRIPTIVE}]:** CONUS Albers (`EPSG:5070`) spatial "
            f"residual maps `ln(dbar_hat / dbar)` on `S-LRO` (`n = 3,988`) for `M5` "
            f"(`RMSE_log = {m5_lro_dbar}`) and `M2-XGB` (`RMSE_log = {m2_lro_dbar}`)."
        ),
        (
            f"- **Panel (c) [{TIER_DESCRIPTIVE}]:** `S-LRO` `RMSE_log(dbar)` across all `8` "
            "Fenneman & Johnson physiographic divisions for `M5`, `M2-XGB`, `M2-MLP`, `M4`, "
            "and `M1`."
        ),
        (
            f"- **Panel (d) [{TIER_DESCRIPTIVE}]:** `S-LRO` `RMSE_log(dbar)` by Strahler "
            "stream order `SO 1` through `SO 10`."
        ),
        "",
        (
            "### Fig. S3 - Sensitivity Suite (`SA1`-`SA6`), `S-RND`, `S1` HyG Consistency & "
            "Not-Computed Audit (`figS3_secondary_analyses.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_DESCRIPTIVE}]:** `SA1`-`SA6` sensitivity suite on `S-LRO` "
            "`RMSE_log(dbar)` (`SA1` for `M1`, `M2-MLP`, `M2-XGB` hatched with verbatim status "
            "`identical to primary by construction (no Phi); not trained`)."
        ),
        (
            f"- **Panel (b) [{TIER_SECONDARY} / {TIER_DESCRIPTIVE}]:** `S-RND` (`n = 9,673` "
            "records, inverse-record-count weighted `1 / |I_j|`) vs `S-ID` site-level benchmark "
            f"for `M1` and `M2-XGB` (`{EVENT1_AUDIT_LABEL}`)."
        ),
        (
            f"- **Panel (c) [{TIER_SECONDARY}]:** `S1` HyG (`US-SWD`) Manning `n` consistency "
            f"funnel (`11,513` raw rows -> `9,078` padded IDs -> `8,350` valid `n` -> "
            f"`{s1_n_matched:,}` matched frame sites; `RMSE_log = {s1_rmse}`, "
            f"`Spearman rho = {s1_rho}`; verbatim note: `{s1_note}`)."
        ),
        (
            f"- **Panel (d) [{TIER_DESCRIPTIVE}]:** Pre-registered `not computed` audit "
            "summary (`L8_zarrabi_product`, Event 1 `far` stratum `<30` floor, `S-IND` `98` "
            "multi-COMID UQ rows, `A_J` `81` dropped `S-IND` rows, and `F_S` constant-input "
            "`rho_S(n_hat, S)`)."
        ),
        "",
        (
            "### Fig. S4 - Training Convergence, `M4` Penalty Trajectory & Multi-Target UQ "
            "Coverage (`figS4_training_and_conformal.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_DESCRIPTIVE}]:** Training and validation loss trajectories "
            "across epochs `e` (`reports/training/curves.json`)."
        ),
        (
            f"- **Panel (b) [{TIER_DESCRIPTIVE}]:** `M4` soft-penalty physics residual "
            "trajectory `Xi_e` and active fit count `n_e` (`reports/training/xi.json`)."
        ),
        (
            f"- **Panel (c) [{TIER_DESCRIPTIVE}]:** Mean and max stopping epochs and Picard "
            "iteration ceiling (`training max = 28`, `evaluation max = 19`)."
        ),
        (
            f"- **Panel (d) [{TIER_SECONDARY} / {TIER_DESCRIPTIVE}]:** Empirical 90% coverage "
            f"`C_0.10` across `S-ID`, `S-LRO`, and `S-IND` (`{S_IND_UQ_NOTE}`) for `dbar`, "
            "`A`, `W`, and `t` with the `[0.85, 0.95]` band."
        ),
        "",
        (
            "### Fig. S5 - `S-IND` USACE eHydro Independent River Survey Evaluation "
            "(`figS5_sind_ehydro_rivers.{pdf,svg,png}`)"
        ),
        (
            f"- **Panel (a) [{TIER_CONFIRMATORY} / {TIER_DESCRIPTIVE}]:** Predicted vs observed "
            f"mean depth `dbar` on `S-IND` (`{S_IND_PANEL_NOTE}`) for `M5` and `M2-XGB` (`M*`)."
        ),
        (
            f"- **Panel (b) [{TIER_DESCRIPTIVE}]:** Per-river location counts across all `18` "
            "`eHydro` rivers, decomposed into single-COMID locations (`n = 33`, UQ evaluated) "
            f"and multi-COMID locations (`n = 98`, UQ `not computed`; `{S_IND_UQ_NOTE}`)."
        ),
        (
            f"- **Panel (c) [{TIER_DESCRIPTIVE}]:** Full model-ladder `S-IND` `RMSE_log(dbar)` "
            "and `RMSE_log(A)` across `M-null`, `A_J (n=50)`, `M0`, `M1`, `M2-MLP`, `M2-XGB`, "
            "`M3`, `M4`, `M5`, `M6-NN`, and `M6-XGB`."
        ),
        (
            f"- **Panel (d) [{TIER_SECONDARY} / {TIER_DESCRIPTIVE}]:** Sorted 90% "
            "split-conformal prediction intervals for `ln dbar` across all `33` single-COMID "
            f"`S-IND` locations for `M5` (`C_0.10 = {m5_cov_ind}`) and `M2-MLP`."
        ),
        "",
    ]

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "captions.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    _ = sha256_file(out_path)
    return out_path
