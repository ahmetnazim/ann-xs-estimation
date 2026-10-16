# Phase 9 Publication Figure Captions (Factual & Provenance-Locked)

## Artwork & Provenance Specification (`P9-D2 (a)`, `P9-D5 (a)`, `P9-D6 (a)`, `P9-D9 (a)`)
- **Target Venue:** ASCE *Journal of Hydrologic Engineering* (`ASCE; fallback per ADL-014`).
- **Artwork Guidelines:** `https://ascelibrary.org/author-center/proceedings-style-guide#figures` (comparison: `https://authorservices.wiley.com/asset/Guidelines-for-Preparing-Figures.html`; accessed `2026-10-15`).
- **Format & Typography:** Double-column width `7.0 in` (`504 pt`), sans-serif (`DejaVu Sans / Arial (sans-serif, 7-9 pt)`), vector PDF (`pdf.fonttype = 42`) + vector SVG (`svg.fonttype = 'none'`) + `600 DPI` PNG.
- **Pre-Registered §4.7 Row 4 Outcome:** `M5 worse (S-LRO CI lower bound > 0)`.
- **Pre-Registered §4.7 Row 4 Action:** `row 4: "Diagnose with the pre-registered ablations (A1, A3, M4, SA1, SA5) and report. No post-hoc headline switch: any new variant is labelled exploratory"`.
- **Event 1 S-ID Label:** `optimistic under spatial/network dependence`.
- **S-IND Scope Note:** `131 locations / 18 rivers` (`UQ: 131 locations / 18 rivers (UQ: 33 single-COMID rows)`).

---

## Core Manuscript Figures (`Fig. 1`-`Fig. 5`)

### Fig. 1 - Study Domain, Hard-Constrained `M5` Architecture & Validation Design (`fig1_domain_and_architecture.{pdf,svg,png}`)
- **Panel (a) [Descriptive]:** CONUS Albers Equal-Area (`EPSG:5070`) map of the primary ungauged `HYDROSWOT` evaluation frame (`n = 3,988` sites across `8` Fenneman & Johnson physiographic divisions) and independent `S-IND` `USACE eHydro` survey locations (`131 locations / 18 rivers`).
- **Panel (b) [Descriptive]:** `M5` hard-constrained physics-informed architecture (`I_blind` roughness head excluding `Q_E, S, F_S, V_E` + differentiable Manning + continuity projection layer `Phi(W_hat, r_hat, n_hat; Q_E, S)` with `K = 8` Gauss-Legendre wet-perimeter quadrature and Picard tolerance `tau_l = 1e-10`).
- **Panel (c) [Descriptive]:** Pre-registered multi-regime validation design (`S-ID` `3 x 5` grouped CV with Event 1 dependence audit: `d_nn,med = 15.01 km`, `rho_hat = rho* = 169.23 km`, nested strong `Psi >= 0.5` share `0.4213`, weak share `0.4879`, verdict `optimistic under spatial/network dependence`; `S-LRO` `8` held-out divisions; `S-TAIL-DA` `n = 678`; `S-TAIL-SO` `n = 339`; `S-IND` `131 locations / 18 rivers (UQ: 33 single-COMID rows)`; `S-RND` `n = 9,673`) and verbatim §4.7 Row 4 outcome (`M5 worse (S-LRO CI lower bound > 0)`; `row 4: "Diagnose with the pre-registered ablations (A1, A3, M4, SA1, SA5) and report. No post-hoc headline switch: any new variant is labelled exploratory"`).

### Fig. 2 - Primary Benchmark Performance & Confirmatory `H1` Forest Plot (`fig2_benchmark_performance.{pdf,svg,png}`)
- **Panel (a) [Descriptive]:** Full model-ladder `RMSE_log(dbar)` across `M-null`, `M0`, `M1`, `M2-MLP`, `M2-XGB`, `M3`, `M4`, `M5`, `M6-NN`, and `M6-XGB` on `S-ID` (`M5 dbar = 0.7036`, `M2-XGB dbar = 0.4396`, `optimistic under spatial/network dependence`), `S-LRO` (`M5 dbar = 0.7284`, `M2-XGB dbar = 0.4782`), `S-TAIL-DA`, `S-TAIL-SO`, and `S-IND` (`131 locations / 18 rivers`).
- **Panel (b) [Descriptive]:** Full model-ladder `RMSE_log(A)` and `RMSE_log(W)` across splits (including `A_J` Jobson baseline on `A`).
- **Panel (c) [Confirmatory / Secondary]:** Confirmatory `H1` paired log-RMSE contrast `Delta_H1 = RMSE_log(M5) - RMSE_log(M*)` (`M* = M2-XGB`, selected on `S-ID` mean `{{dbar, A}}` `RMSE_log = 0.5313`) with 95% division / river cluster-bootstrap CIs (`B = 2,000`):
  - `S-LRO` (`Confirmatory`, `n = 3,988`): `dbar` `Delta = +0.2502 [+0.2262, +0.2765]`; `A` `Delta = +0.3094 [+0.2728, +0.3520]`.
  - `S-IND` (`Confirmatory` point / `Secondary` river bootstrap on `18 < 30` rivers, `131 locations / 18 rivers`): `dbar` `Delta = +0.1589 [-0.2778, +0.3907]`; `A` `Delta = -0.0343 [-0.1588, +0.0746]`.
  - Verdict: `H1 Falsified` (`M5 worse (S-LRO CI lower bound > 0)`).
- **Panel (d) [Descriptive]:** Out-of-fold predicted vs observed `dbar` on `S-LRO` (`n = 3,988` sites) for `M5` vs `M2-XGB` (`M*`).

### Fig. 3 - Physical Consistency Analogues & Confirmatory `H4` Parameter Plausibility (`fig3_physical_consistency_and_h4.{pdf,svg,png}`)
- **Panel (a) [Descriptive]:** Continuity area closure residual `eps_A_hat = ln(A_hat / (W_hat * dbar_hat))` on `S-LRO` (`M2-XGB` median `|eps_A_hat| = 0.0607` vs `M3`-`M6` exact `0.0000` and QC R8 observed closure tolerance `+/- ln(1.05)`).
- **Panel (b) [Descriptive]:** Share of predictions with no admissible back-calculated Manning roughness `n_back in [0.005, 0.40]` (`docs/2_mathematical_proofs.md` Prop. 6.2 / §7.4; `M2-XGB` `S-LRO` share outside = `0.0552` vs `M3`/`M5` = `0.0000`) and Dingman shape bound violation share `P(d_max_hat <= dbar_hat)`.
- **Panel (c) [Descriptive]:** Hard-constrained Picard solver residual (`M5` `S-ID` max `|res| = 6.65e-11 <= tau_l = 1e-10`) vs `M4` soft-loss test-site Manning residual `R_M` (`S-ID` `RMS = 0.1341`).
- **Panel (d) [Confirmatory / Descriptive]:** Confirmatory `H4` parameter plausibility on `S-ID` (`Confirmatory`, `optimistic under spatial/network dependence`: `pi_lit = 0.9265 >= 0.90`, `rho_S(n_hat, D50) = -0.0203 [-0.0500, 0.0083]`, `rho_S(n_hat, S) = 0.4431 [0.4181, 0.4674]`) and `S-LRO` (`Descriptive`: `pi_lit = 0.9519`, `rho_S(n_hat, D50) = -0.0257`, `rho_S(n_hat, S) = 0.4425`); verdict: `H4 Falsified (CI lower bound <= 0 for rho_S(n_hat, D50))`.

### Fig. 4 - OOD Stress Tests (`H3`), UQ Calibration (`[0.85, 0.95]` Band) & Headwater Bias (`H2`) (`fig4_ood_and_uq_calibration.{pdf,svg,png}`)
- **Panel (a) [Confirmatory / Secondary]:** `H3` OOD skill loss `L_OOD(M) = RMSE_OOD - RMSE_ID` and contrast `Delta_H3 = L(M5) - L(M*)` with 95% bootstrap CIs (`H3-LRO` `Confirmatory`: `Delta_H3 = -0.0138 [-0.0494, 0.0209]`; `H3-TAIL-DA` `Confirmatory`: `Delta_H3 = -0.0485 [-0.1750, 0.0759]`; `H3-TAIL-SO` `Secondary`, `n = 339 < 500`: `Delta_H3 = -0.2088 [-0.4357, 0.0151]`; verdict: `H3 Falsified`).
- **Panel (b) [Secondary / Descriptive]:** Empirical 90% split-conformal coverage `C_0.10(dbar)` with the pre-registered **`[0.85, 0.95]`** acceptance band across `S-ID` (`M5 = 0.8997` in `[0.85, 0.95]`, `optimistic under spatial/network dependence`), `S-LRO` (`M5 = 0.9025`), `S-TAIL-DA`, `S-TAIL-SO`, `S-IND` (`M5 = 0.8182`, `131 locations / 18 rivers (UQ: 33 single-COMID rows)`), and `S-ID` drainage-area quintiles `Q1`-`Q5`.
- **Panel (c) [Descriptive]:** 10-bin probability integral transform (PIT) histograms (`pit_counts / n_rows`) for `dbar` on `S-ID` and `S-LRO` relative to the uniform `0.10 / bin` reference.
- **Panel (d) [Descriptive]:** Closed-form log-Gaussian CRPS and mean 90% log-interval width `W_bar_log` across all 5 evaluation splits (`S-LRO`: `M5 CRPS = 0.3938`, `M2-MLP CRPS = 0.2724`).
- **Panel (e) [Confirmatory / Secondary / Descriptive]:** `H2` signed `dbar` log-bias `B_log(dbar)` across drainage-area quintiles `Q1`-`Q5` (`S-ID Q1` `Confirmatory`: `h2 = 0.0299 [-0.0335, 0.0740]`; `S-LRO Q1` `Secondary`: `h2 = 0.0753 [0.0435, 0.1057]`; `S-ID far Q1` `Secondary`: `not computed (fewer than 30 sites in a repeat)`; verdict: `H2 Falsified`).

### Fig. 5 - Pre-Registered §4.7 Row-4 Mechanism Diagnosis (`fig5_row4_diagnosis.{pdf,svg,png}`)
- **Panel (a) [Descriptive]:** Pre-registered Row-4 ablations (`Primary`, `A1`, `wA1`, `A2`, `A3-SP` `1.2070`, `A3-SR` `1.2325`, `M4`, `SA1`, `SA5` `0.6067`, `A4` `0.7288`) on `S-ID` and `S-LRO` `RMSE_log(dbar)`, annotated with `docs/2_mathematical_proofs.md` §5.1 wide-channel elasticities (`d ln dbar / d ln S = -3/10`, `d ln dbar / d ln W = -3/5`, `d ln dbar / d ln n = +3/5`) and `M2-XGB SA1: identical to primary by construction (no Phi); not trained`.
- **Panel (b) [Secondary / Descriptive]:** `S2` stepwise physics contrasts with **98.75% Bonferroni CIs** (`M4 - M5` on `S-ID`: `-0.0231 [-0.0464, -0.0074]`; `S-LRO`: `-0.0420 [-0.0657, -0.0250]`; `M5 - M6-NN` on `S-ID`: `-0.0996 [-0.1096, -0.0903]`; `S-LRO`: `-0.1016 [-0.1110, -0.0930]`; `S-ID far stratum`: `not computed (fewer than 30 sites in a repeat)`).
- **Panel (c) [Descriptive]:** Oracle-discharge (`A4`) vs ungauged EROM (`Q_E`) comparison on `S-LRO` for `dbar` and implied velocity `V_hat = Q / A_hat` (`docs/3_experimental_protocol.md` §4.3.2 variance shares).
- **Panel (d) [Confirmatory / Descriptive]:** `H4` physical parameter plausibility (`pi_lit`, `rho_S(n_hat, D50)`, `rho_S(n_hat, S)`) across `M5 Primary` and pre-registered ablations on `S-ID`.
- **Bottom Banner [Confirmatory]:** Verbatim §4.7 Row 4 outcome (`M5 worse (S-LRO CI lower bound > 0)`), venue (`ASCE; fallback per ADL-014`), and action (`row 4: "Diagnose with the pre-registered ablations (A1, A3, M4, SA1, SA5) and report. No post-hoc headline switch: any new variant is labelled exploratory"`).

---

## Supplementary Figures (`Fig. S1`-`Fig. S5`)

### Fig. S1 - Event 1 Spatial & Network Dependence Audit (`figS1_event1_spatial_dependence.{pdf,svg,png}`)
- **Panel (a) [Descriptive]:** Per-block (`R0F0`-`R2F4`) `S-ID` nested-basin index shares (`Psi >= 0.5` strong, `0 < Psi < 0.5` weak) and median nearest-neighbour distance `d_nn` (`global median = 15.01 km`).
- **Panel (b) [Descriptive]:** Empirical OOF `ln dbar` residual semivariograms (`M-null`, `M2-XGB`, `M5`) up to `500 km`.
- **Panel (c) [Descriptive]:** OOF `ln dbar` spatial correlograms `Moran's I(h)` across distance bands (`[0, 10)` to `[250, 500) km`).
- **Panel (d) [Descriptive]:** Significant Local Moran's `I` (`LISA`, `p < 0.05`) site counts by physiographic division; banner quotes Event 1 verdict `optimistic under spatial/network dependence`.

### Fig. S2 - Per-Division & Stream-Order `S-LRO` Accuracy and Spatial Residuals (`figS2_per_division_slro.{pdf,svg,png}`)
- **Panels (a)-(b) [Descriptive]:** CONUS Albers (`EPSG:5070`) spatial residual maps `ln(dbar_hat / dbar)` on `S-LRO` (`n = 3,988`) for `M5` (`RMSE_log = 0.7284`) and `M2-XGB` (`RMSE_log = 0.4782`).
- **Panel (c) [Descriptive]:** `S-LRO` `RMSE_log(dbar)` across all `8` Fenneman & Johnson physiographic divisions for `M5`, `M2-XGB`, `M2-MLP`, `M4`, and `M1`.
- **Panel (d) [Descriptive]:** `S-LRO` `RMSE_log(dbar)` by Strahler stream order `SO 1` through `SO 10`.

### Fig. S3 - Sensitivity Suite (`SA1`-`SA6`), `S-RND`, `S1` HyG Consistency & Not-Computed Audit (`figS3_secondary_analyses.{pdf,svg,png}`)
- **Panel (a) [Descriptive]:** `SA1`-`SA6` sensitivity suite on `S-LRO` `RMSE_log(dbar)` (`SA1` for `M1`, `M2-MLP`, `M2-XGB` hatched with verbatim status `identical to primary by construction (no Phi); not trained`).
- **Panel (b) [Secondary / Descriptive]:** `S-RND` (`n = 9,673` records, inverse-record-count weighted `1 / |I_j|`) vs `S-ID` site-level benchmark for `M1` and `M2-XGB` (`optimistic under spatial/network dependence`).
- **Panel (c) [Secondary]:** `S1` HyG (`US-SWD`) Manning `n` consistency funnel (`11,513` raw rows -> `9,078` padded IDs -> `8,350` valid `n` -> `2,887` matched frame sites; `RMSE_log = 1.3257`, `Spearman rho = 0.3015`; verbatim note: `consistency check, not independent validation (protocol 4.6)`).
- **Panel (d) [Descriptive]:** Pre-registered `not computed` audit summary (`L8_zarrabi_product`, Event 1 `far` stratum `<30` floor, `S-IND` `98` multi-COMID UQ rows, `A_J` `81` dropped `S-IND` rows, and `F_S` constant-input `rho_S(n_hat, S)`).

### Fig. S4 - Training Convergence, `M4` Penalty Trajectory & Multi-Target UQ Coverage (`figS4_training_and_conformal.{pdf,svg,png}`)
- **Panel (a) [Descriptive]:** Training and validation loss trajectories across epochs `e` (`reports/training/curves.json`).
- **Panel (b) [Descriptive]:** `M4` soft-penalty physics residual trajectory `Xi_e` and active fit count `n_e` (`reports/training/xi.json`).
- **Panel (c) [Descriptive]:** Mean and max stopping epochs and Picard iteration ceiling (`training max = 28`, `evaluation max = 19`).
- **Panel (d) [Secondary / Descriptive]:** Empirical 90% coverage `C_0.10` across `S-ID`, `S-LRO`, and `S-IND` (`131 locations / 18 rivers (UQ: 33 single-COMID rows)`) for `dbar`, `A`, `W`, and `t` with the `[0.85, 0.95]` band.

### Fig. S5 - `S-IND` USACE eHydro Independent River Survey Evaluation (`figS5_sind_ehydro_rivers.{pdf,svg,png}`)
- **Panel (a) [Confirmatory / Descriptive]:** Predicted vs observed mean depth `dbar` on `S-IND` (`131 locations / 18 rivers`) for `M5` and `M2-XGB` (`M*`).
- **Panel (b) [Descriptive]:** Per-river location counts across all `18` `eHydro` rivers, decomposed into single-COMID locations (`n = 33`, UQ evaluated) and multi-COMID locations (`n = 98`, UQ `not computed`; `131 locations / 18 rivers (UQ: 33 single-COMID rows)`).
- **Panel (c) [Descriptive]:** Full model-ladder `S-IND` `RMSE_log(dbar)` and `RMSE_log(A)` across `M-null`, `A_J (n=50)`, `M0`, `M1`, `M2-MLP`, `M2-XGB`, `M3`, `M4`, `M5`, `M6-NN`, and `M6-XGB`.
- **Panel (d) [Secondary / Descriptive]:** Sorted 90% split-conformal prediction intervals for `ln dbar` across all `33` single-COMID `S-IND` locations for `M5` (`C_0.10 = 0.8182`) and `M2-MLP`.
