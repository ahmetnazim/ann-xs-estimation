# Phase 0 Gate Deliverable: Point ZERO Problem Statement

> **Project:** ANN_XS_Estimation, the successor study to the v1 manuscript *"A Data-Driven Framework for National-Scale Hydraulic Geometry Estimation: Integrating HYDROSWOT, NHDPlus, and ANNs"* (frozen as v2; ADL-001).
> **PI:** A. N. Sahin (NCCHE, University of Mississippi). **Funding:** USDA-ARS Cooperative Agreement 6060-13000-030-00D.
> **Phase:** 0 (Point ZERO). The code-generation lock is ON. This document contains no executable code.
> **Date:** 2026-10-07. **Decisions in force:** ADL-011 to ADL-025 (D1 NHDPlusV2.1; D2 v1 table unavailable; D3 rebuild from raw HYDROSWOT; D4 Option A; C1 ASCE venue).
> **Companion document:** `docs/0a_open_data_questions.md` (resolutions of Q1–Q6, findings F1–F6).

---

## 1. Problem statement

One-dimensional hydrodynamic models (for example CCHE1D) and National Water Model (NWM)–type channel parameterizations need reach-scale cross-section geometry at every reach of the river network: top width $W$, mean depth $\bar d$, maximum depth $d_{max}$, flow area $A$, and the roughness that links them to discharge. Most of the roughly 2.6–2.7 million NHDPlusV2.1 reaches in the conterminous United States (CONUS) have never been surveyed, so these values must be predicted from attributes that are available at ungauged reaches.

Current CONUS-scale machine-learning products estimate width and depth as **independent statistical targets** (Doyle et al., 2023; Chang et al., 2024; Modaresi Rad et al., 2024; Zarrabi et al., 2025). Three limitations follow:

1. **Depth transfers poorly.** Zarrabi et al. (2025) report test-set R² of 0.63 (bankfull depth) and 0.68 (mean-flow depth) under a random split. On independent datasets, depth R² falls to 0.32 (mean flow, eHydro) and 0.37 (bankfull), while width stays at 0.84 and 0.65.
2. **The low end is biased.** The same study reports Tier-2 biases at minimum values of 142% (bankfull width), 186.9% (bankfull depth), 81.3% (mean-flow width) and 155.7% (mean-flow depth).
3. **Outputs are not hydraulically consistent.** Predicted $W$, $\bar d$ and discharge are not required to satisfy continuity or a flow-resistance law. These products also provide no shape or roughness, which a 1D model needs.

The v1 study had the same structural issue, plus evaluation defects that this study must not repeat: a random split with repeated sites, a measured-discharge predictor, and unverified references (ADL-001, ADL-006).

**Problem:** we do not know whether **building flow resistance and continuity into the predictor as a hard constraint** gives more accurate and more transferable depth and area estimates at ungauged reaches than purely statistical models trained on the same data, and whether the hydraulic parameters it learns are physically meaningful.

## 2. Research questions

- **RQ1 (accuracy and transfer).** Does a hard-constrained hybrid, in which a neural network predicts $W$, a shape exponent $r$ and an effective Manning roughness $n$, and a differentiable Manning + continuity layer computes $\bar d$, $d_{max}$, $A$ and $V$, reduce out-of-sample depth and area error relative to pure ML and regional hydraulic-geometry curves?
- **RQ2 (size dependence).** Does the hard constraint reduce the systematic bias of statistical products on the smallest streams?
- **RQ3 (extrapolation).** Does the hybrid lose less skill than pure ML when moving from in-distribution tests to unseen regions and stream-size tails?
- **RQ4 (physical meaning).** Are the learned $n$ and $r$ physically plausible, that is, inside literature ranges and related to grain size, slope and the observed shape in the expected directions?

## 3. Falsifiable hypotheses

Each hypothesis is pre-registered here. Its test and threshold are fixed in the Phase 3 protocol (`3_experimental_protocol.md`) before any model is trained. "Pure ML" means the strongest of the M1/M2 baselines (see §4) under identical site-grouped folds.

| ID | Hypothesis | Primary test (definition fixed in Phase 3) | Falsified if |
|---|---|---|---|
| **H1** | The hard hybrid reduces out-of-sample error in $\bar d$ (and $A$) relative to pure ML | Paired bootstrap over sites of Δlog-RMSE($\bar d$) = hybrid − pure ML, on leave-region-out folds and independent sets | The 95% CI of Δlog-RMSE($\bar d$) includes 0 or is positive |
| **H2** | The hybrid reduces bias for the smallest streams | Log-space bias of $\bar d$ in the lowest drainage-area quintile | Bottom-quintile absolute bias is not lower than pure ML's (paired bootstrap 95% CI) |
| **H3** | The hybrid loses less skill under extrapolation | Skill(in-distribution) − Skill(OOD), for leave-region-out and size-tail splits | The hybrid's skill loss is equal to or larger than pure ML's |
| **H4** | The learned $n$ and $r$ are physically plausible | Share of $n$ inside the bounded literature range; sign and strength of the rank correlation of $n$ with D50 and slope; agreement of learned $r$ with observed $r$ from HYDROSWOT | Learned parameters cluster at the bounds or outside plausible ranges, or show no relation (or the wrong sign) to D50, slope and observed shape |

**Pre-registered interpretation:**
- If H1 and H3 hold, the claim is improved transferability, and escalating the venue to WRR or HESS is permitted (ADL-015).
- If H1 fails but H4 holds and accuracy is not worse, the claim becomes "hydraulically consistent geometry, shape, roughness and velocity at no loss of accuracy".
- If the hybrid is worse, the failure is diagnosed and reported, not hidden.

The full decision matrix is carried into Phase 3 from `literature/synthesis_notes/option_a_research_design.md` §8.

## 4. Scope and non-goals

### In scope (core)
- **Target condition:** mean flow first. Discharge at ungauged reaches comes from NHDPlusV2.1 EROM mean annual flow (`QE_MA`).
- **Targets:** $W$, $\bar d$, $d_{max}$, $A$ and implied $V$ at HYDROSWOT measurement sites, rebuilt from raw HYDROSWOT (ADL-013).
- **Model ladder (names only; definitions belong to Phases 2–3):**
  - M0: regional curves (Bieger et al., 2015).
  - M1: Zarrabi-style tree models retrained on our folds.
  - M2: pure ML (MLP and gradient boosting).
  - M3: physics-only (ML width plus empirical $n$ plus Manning).
  - M4: soft hybrid (Manning residual penalty).
  - M5: hard hybrid (proposed).
  - M6: two-stage back-calculated $n$.
  - Phase 3 may add a training-free reference, $A_J$ = `QE_MA`/`VE_MA`.
- **Evaluation:**
  - site-grouped cross-validation;
  - leave-region-out tests;
  - size-tail extrapolation;
  - independent evaluation sets;
  - uncertainty quantification with empirical interval coverage.

### Secondary (only after the core is complete)
- Bankfull geometry as a second stage, with an explicit decomposition of the error that comes from bankfull discharge.

### Non-goals (this study)
- Geometry as a continuous function of flow (synthetic rating curves for every reach). Deferred to a follow-up.
- Option B, the CCHE1D value demonstration (Yazoo / Goodwin Creek). Not core (ADL-014); it may be added only if surveyed cross-sections and a CCHE1D setup are confirmed.
- Graph or network-topology architectures.
- Operational or real-time inference services. There are no latency targets (ADL-010).
- Reuse of any v1 result, model or merged table as a result or target (ADL-001, ADL-012).
- Recovering "true" Manning's $n$. The learned $n$ is an **effective** conveyance coefficient. It absorbs the difference between reach slope and water-surface slope, departures from uniform flow, and slope error.

## 5. Novelty boundary

This study makes **no "first" claims**. Its contribution is bounded as follows against the closest verified work.

| Work | What it does | What this study adds (to be confirmed in Phase 1) |
|---|---|---|
| Zarrabi et al. (2025) | Tree-based ML on HYDROSWOT linked to NHDPlusV2.1; bankfull and mean-flow $W$ and depth for 2,642,127 reaches; random split; independent-set evaluation | Hard hydraulic constraint; shape, roughness, area and velocity outputs; site-grouped and spatial evaluation; retrained head-to-head comparison on identical folds |
| Chang et al. (2024) | Multi-model ML for CONUS river geometry | As above. Phase 1 must check whether any physics coupling is present |
| Doyle et al. (2023) | Random-forest bankfull and low-flow $W$ and depth across CONUS | As above |
| Modaresi Rad et al. (2024) | ML channel dimensions for the NWM hydrofabric | As above |
| Heldmyer et al. (2022) | Observation-based NWM channel parameterization (width, side slope, roughness) | **Highest overlap risk.** Phase 1 must establish how parameters were derived and whether learned roughness or shape overlaps with M5 |
| Bjerklie et al. (2020) | Analysis of cross-section hydraulics in the HYDROSWOT ADCP data set | **Overlap risk** for the $r$ and $n$ components. Phase 1 must read it before any shape or resistance claim |

**Methodological lineage (cited, not claimed):**
- The power-law cross-section and analytical at-a-station relations follow Dingman (2007).
- The hard-constrained, differentiable parameter-learning design follows Tsai et al. (2021) and Shen et al. (2023).
- Enforcing analytic constraints in network outputs follows Beucler et al. (2021).
- The bankfull and mean-flow condition-extraction steps follow Zarrabi et al. (2025) and are not presented as new. The v1 variant, an IQR on the width/depth ratio (finding F1), is kept as a sensitivity case.

The final novelty statement is a Phase 1 deliverable.

## 6. Data constraints

| Item | Constraint | Reference / ADL |
|---|---|---|
| Targets | HYDROSWOT (Canova et al., 2016). 223,022 ADCP measurements at 10,081 sites (counts as reported by Zarrabi et al., 2025), one row per measurement. It includes `max_depth_va`, so observed $r$ is computable (F2). No access or use constraints | doi:10.5066/F7D798H6; ADL-013 |
| Hydrography | NHDPlusV2.1 (COMID). Gages are linked by the V2.1 gage–reach identifier; disagreement with a k-NN snap is a QC diagnostic | ADL-011; McKay et al. (2012) ⚠️ |
| Headline predictors | Only attributes available at ungauged reaches. **Measured discharge is banned.** EROM `VE_MA` is allowed but redundant | ADL-005, ADL-006 |
| Grouping and splits | `site_no` groups every split. Random splits are secondary results only. Split first, then scale (fit scalers on training folds only) | §10.2 of the research state |
| Comparator product | Zarrabi et al. (2024) dataset, CC BY 4.0. Used **only** on independent evaluation sets, or via models retrained on our folds. Never compared in-sample at HYDROSWOT sites | doi:10.5281/zenodo.13883263 |
| Channel material | `chan_mat_cd` is a **diagnostic** for H4 only. It is observed at gages, so it is never a predictor | F3 |
| Acquisition | No data is downloaded before the Phase 3 protocol and the Phase 5 DVC pipeline | ADL-007, ADL-009 |
| Licences still to check | NHDPlusV2.1; the Zarrabi GitHub code (Phase 3) | §10.2 |

## 7. Physical invariants (statements only; proofs in Phase 2)

Symbols: $Q$ discharge, $V$ mean velocity, $P$ wetted perimeter, $R = A/P$ hydraulic radius, $S$ slope, $n$ effective Manning roughness, $r$ power-law shape exponent.

1. **Area closure:** $A = W\,\bar d$.
2. **Continuity:** $Q = A\,V$.
3. **Flow resistance (Manning):** $Q = \tfrac{1}{n}\,A\,R^{2/3}\,S^{1/2}$, with $R = A/P$.
4. **Power-law shape:** $\bar d = d_{max}\,\tfrac{r}{r+1}$, so $d_{max} \ge \bar d$ for every $r > 0$.
5. **Positivity:** $W > 0$, $\bar d > 0$, $A > 0$, $n > 0$, $r > 0$.
6. **Monotonicity:** $\partial \bar d / \partial Q > 0$ at fixed $W$, $n$, $r$ and $S$.
7. **Bounded parameters:** $n$ and $r$ lie inside physical ranges. The provisional ranges are $n \in [0.015, 0.25]$ and $r \in [0.5, 10]$; Phase 3 sets the final ranges from data.
8. **Units:** HYDROSWOT (ft, ft², ft/s, cfs) and NHDPlus EROM (cfs, ft/s) are converted to SI in an explicit, testable step.
9. **Slope validity:** NHDPlus smoothed slopes are floored at 10⁻⁵, and −9998 means missing. Both conditions become QC flags, never silent imputations.

Phase 2 formalizes invariants 1–7, including the closed-form depth solution, the conveyance correction $\kappa = R/\bar d$ and the elasticity analysis. Phase 7 encodes them as Pytest physical assertions.

---

## References

All DOIs were verified on 2026-10-07 against Crossref (journal articles) or DataCite (datasets). Where a reference has no DOI, it is marked ⚠️ and its publisher URL is given.

- Beucler, T., Pritchard, M., Rasp, S., Ott, J., Baldi, P., & Gentine, P. (2021). Enforcing analytic constraints in neural networks emulating physical systems. *Physical Review Letters*, 126(9), 098302. https://doi.org/10.1103/PhysRevLett.126.098302
- Bieger, K., Rathjens, H., Allen, P. M., & Arnold, J. G. (2015). Development and evaluation of bankfull hydraulic geometry relationships for the physiographic regions of the United States. *JAWRA*, 51(3), 842–858. https://doi.org/10.1111/jawr.12282
- Bjerklie, D. M., et al. (2020). Fundamental hydraulics of cross sections in natural rivers: Preliminary analysis of a large data set of acoustic Doppler flow measurements. *Water Resources Research*, 56, e2019WR025986. https://doi.org/10.1029/2019WR025986
- Canova, M. G., Fulton, J. W., & Bjerklie, D. M. (2016). *USGS HYDRoacoustic dataset in support of the Surface Water Oceanographic Topography satellite mission (HYDRoSWOT)* [Data set]. U.S. Geological Survey. https://doi.org/10.5066/F7D798H6
- Chang, S. Y., Ghahremani, Z., Manuel, L., Erfani, S. M. H., Shen, C., Cohen, S., Van Meter, K. J., Pierce, J. L., Meselhe, E. A., & Goharian, E. (2024). The geometry of flow: Advancing predictions of river geometry with multi-model machine learning. *Water Resources Research*, 60, e2023WR036733. https://doi.org/10.1029/2023WR036733
- Dingman, S. L. (2007). Analytical derivation of at-a-station hydraulic–geometry relations. *Journal of Hydrology*, 334, 17–27. https://doi.org/10.1016/j.jhydrol.2006.09.021
- Doyle, J. M., Hill, R. A., Leibowitz, S. G., & Ebersole, J. L. (2023). Random forest models to estimate bankfull and low flow channel widths and depths across the conterminous United States. *JAWRA*, 59(5), 1099–1114. https://doi.org/10.1111/1752-1688.13116
- Heldmyer, A., Livneh, B., McCreight, J., Read, L., Kasprzyk, J., & Minear, T. (2022). Evaluation of a new observationally based channel parameterization for the National Water Model. *Hydrology and Earth System Sciences*, 26, 6121–6136. https://doi.org/10.5194/hess-26-6121-2022
- McKay, L., et al. (2012). *NHDPlus Version 2: User guide*. U.S. EPA / USGS. ⚠️ No DOI. https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus (author list to be confirmed against the guide's title page in Phase 3).
- Modaresi Rad, A., Johnson, J. M., Ghahremani, Z., Coll, J., & Frazier, N. (2024). Enhancing river channel dimension estimation: A machine learning approach leveraging the National Water Model, hydrographic networks, and landscape characteristics. *Journal of Geophysical Research: Machine Learning and Computation*, 1, e2024JH000173. https://doi.org/10.1029/2024JH000173
- Shen, C., et al. (2023). Differentiable modelling to unify machine learning and physical models for geosciences. *Nature Reviews Earth & Environment*, 4, 552–567. https://doi.org/10.1038/s43017-023-00450-9
- Tsai, W.-P., et al. (2021). From calibration to parameter learning: Harnessing the scaling effects of big data in geoscientific modeling. *Nature Communications*, 12, 5988. https://doi.org/10.1038/s41467-021-26107-z
- Zarrabi, R., McDermott, R., Erfani, S. M. H., & Cohen, S. (2024). *Bankfull and mean-flow channel geometry dataset across the CONtiguous United States (CONUS)* [Data set, CC BY 4.0]. Zenodo. https://doi.org/10.5281/zenodo.13883263
- Zarrabi, R., McDermott, R., Erfani, S. M. H., & Cohen, S. (2025). Bankfull and mean-flow channel geometry estimation through machine learning algorithms across the CONtiguous United States (CONUS). *Water Resources Research*, 61, e2024WR037997. https://doi.org/10.1029/2024WR037997
