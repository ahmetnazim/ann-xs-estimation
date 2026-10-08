# Phase 1 Gate Deliverable: Literature Synthesis Matrix, Novelty Audit, Gap Statement and Core Hypothesis

> **Project:** ANN_XS_Estimation, the successor study to the v1 manuscript (frozen as v2; ADL-001).
> **Phase:** 1, SOTA Synthesis. The code-generation lock is ON, and this document contains no executable code.
> **Date:** 2026-10-08. **Refines:** `docs/0_point_zero.md` (commit `3723971`); it does not replace it.
> **Decisions in force:** ADL-011 to ADL-028, plus Phase 1 decisions P1-D1 to P1-D4 (2026-10-07; logged as ADL-029, PI approval 2026-10-08):
> - P1-D1: `master.bib` lives in `literature/synthesis_notes/`, with a tracked copy at `docs/references.bib`.
> - P1-D2: open-access full text may be fetched and saved to `raw_pdfs/`.
> - P1-D3: the core corpus is 30–40 papers; rows Elicit labels "Low" go to the appendix.
> - P1-D4: limited read-only OpenAlex/Crossref search is allowed.
>
> **Citation keys** (`[@Key]`) resolve in `docs/references.bib`.

---

## 1. Corpus and method

### 1.1 Sources
1. **Elicit:** 15 query groups (A1–A4, B1–B4, C1–C3, D1–D4), exported to `Elicit - Literature_Matrix.csv`, plus the 15 `Query <ID> - Refs` source CSVs, which serve as the DOI join. The Elicit narrative answers and the Q1–Q3 novelty notes were treated as **leads only**.
2. **Mandatory additions** (PHASE_01 Step 1): the six comparators and five lineage papers, all already DOI-verified in `0_point_zero.md`.
3. **Q1–Q3 note papers**, verified in this session.
4. **Supplementary search** (P1-D4, 2026-10-08). Six OpenAlex queries aimed at the novelty risk, i.e. learned roughness or geometry coupled to Manning or shallow-water physics, plus Crossref title lookups for seed and must-read papers. Queries used:
   - *physics-informed neural network hydraulic geometry Manning channel depth prediction*
   - *differentiable Manning equation neural network river channel geometry roughness*
   - *physics-guided machine learning river width depth Manning roughness national*
   - *machine learning Manning roughness coefficient river reaches HYDRoSWOT*
   - *hybrid machine learning hydraulic geometry continuity constraint river cross-section*
   - *differentiable routing neural network learned Manning roughness channel geometry parameters large basin*

### 1.2 Verification protocol
- **Metadata.** Every DOI was resolved against Crossref (`api.crossref.org/works/{doi}`) or, for arXiv records and datasets, DataCite. Title, first author, year and venue were matched.
- **Abstracts and open-access status** came from OpenAlex.
- **Full text** came from open-access copies (P1-D2). It was read with read-only PyMuPDF extraction into the agent scratch folder.
- **Blocked full text.** Wiley, IWA and the NOAA repository refused scripted access, and the browser fallback was unavailable (server capacity). Those papers are therefore marked **Verified-abstract**.

**Evidence status (last matrix column):**
- **Verified-full-text:** numbers and design facts were checked in the paper body.
- **Verified-abstract:** checked against the publisher abstract only.
- **Lineage:** DOI verified in Phase 0; the content is cited at the level of the general method.
- *(Elicit, unverified)*: an individual cell value that comes from Elicit and could not be confirmed.

### 1.3 PRISMA-lite flow

| Stage | Count | Notes |
|---|---|---|
| Elicit matrix rows read | **74** | Not "about 54". One more CSV line is an export footer (`[1] matrix-excerpts-normalized.txt`) and was discarded |
| Elicit relevance labels | 16 High / 45 Medium / 13 Low | Recounted exactly |
| Rows with a DOI in the Refs CSVs | 65 | Title join: 74/74 rows matched a Refs entry |
| … resolved in Crossref/DataCite | 64 | 1 failed: `10.5555/3692070.3693060` |
| Rows without a DOI in Elicit | 9 | 2 resolved this session (MC-LSTM via DataCite; Meyer & Pebesma via Crossref); 7 not pursued (excluded on scope) |
| **Elicit rows DOI-verified** | **66** | |
| Elicit rows included | **24** | 10 of the 16 High rows; Beucler 2021 included as mandatory lineage despite Elicit's "Low" label |
| Elicit rows excluded | **50** | 12 "Low" (P1-D3) and 38 on scope, redundancy or verifiability. Listed in Appendix A |
| Added: mandatory comparators or lineage missing from the CSV | 3 | Bjerklie 2020, Dingman 2007, Tsai 2021 |
| Added: Q1–Q3 note papers | 6 (+1 audit-only) | Huthoff 2008 has no DOI: ⚠️ audit-only, not a matrix row |
| Added: supplementary search (P1-D4) | 6 | Bjerklie 2005, Roberts 2017, Ploton 2020, **Al Mehedi 2024, Bindas 2024, Liu & Song 2025** |
| **Core matrix** | **39 rows** | All DOI-verified. Inside the P1-D3 range |

### 1.4 Elicit data-quality findings (why Elicit is leads, not evidence)
- **Row count.** The matrix has 74 rows, about 37% more than the profile expected.
- **Wrong years.** 8 Elicit years disagree with the registry: Feng 2023 (PINN), Liu 2024, Beucler 2021, Nearing 2021, Feng 2021, Tyralis 2023, Pasche 2024 and Wang 2025.
- **Discussion-paper DOIs** were given where a final version exists:
  - Heldmyer: `hess-2021-552` → `hess-26-6121-2022`;
  - Frame: `hess-2021-423` → `hess-26-3377-2022`.
- **One DOI did not resolve:** prefix `10.5555`.
- **Q1 note.** It reported finding no continental ML roughness product. The supplementary search found **Al Mehedi et al. (2024)**: a continental random-forest model of Manning's n trained on about 200,000 U.S. surveys [@AlMehedi_2024_ChannelRoughness]. It also found **Bindas et al. (2024)**: a neural network that infers n and channel geometry inside a differentiable routing model [@Bindas_2024_DifferentiableRouting]. **Elicit missed both of the strongest novelty risks.**

---

## 2. Literature synthesis matrix

**Column codes:**
- **Physical constraints:** None / Soft (loss penalty) / Hard (architectural) / Physics-only (no learned component, or physics used only to build the target).
- **Coupling:** n/a; serial (physics before or after ML, trained separately); end-to-end (gradients flow through the physics); NN→physics (an NN predicts parameters of a physics model).
- **H-link:** which of H1–H4 in `0_point_zero.md` §3 the paper informs. "M#" refers to the model ladder in §4 of that document.
- **Validation & OOD behavior** always has three parts:
  - **Split:** the in-distribution evaluation design.
  - **OOD:** the out-of-distribution test the paper ran.
  - **ID→OOD Δ:** the reported change in skill from in-distribution to OOD.
- **Vocabulary for that column:**
  - **Not tested:** the full text, or an unambiguous description of the design, shows no OOD test.
  - **Not reported (abstract only):** the full text was not read, so the absence of a test cannot be claimed.
  - **n/a:** theory, review, dataset descriptor or single-system inverse problem.

### Block A — National and CONUS channel geometry and channel parameters (direct comparators)

| Key / DOI | First author (year) | Method / architecture | Dataset & scale | Physical constraints | Coupling | Validation & OOD behavior | Key result | Primary limitation | H-link | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| `Zarrabi_2025_ChannelGeometryCONUS`<br>10.1029/2024WR037997 | Zarrabi (2025) | RF and XGB. Two tiers (Tier 2 uses NHDPlus EROM discharge). Separate models per target and flow condition | HYDROSWOT bankfull and mean-flow records at 2,626 gauge sites; NHDPlusV2.1; predictions for ~2.7M CONUS reaches | None | n/a. W and d predicted independently | **Split:** random 75:25 over site records. **OOD:** external independent sets (eHydro for mean flow; 11 sources for bankfull). **ID→OOD Δ:** depth falls from an average test R² of 0.69 (averaged over tiers and flow conditions, so not like-for-like) to **0.32** (mean flow) and **0.37** (bankfull). Width: 0.84 and 0.65 on the same sets | Average R² 0.85 (W) and 0.69 (d); +30% (bankfull W) and +76% (bankfull d) over regional HG | Tier-2 bias at minimum values: 142% / 186.9% / 81.3% / 155.7% (bankfull W / bankfull d / mean-flow W / mean-flow d). No hydraulic consistency | H1, H2, H3 (M1) | Verified-full-text |
| `Chang_2024_GeometryOfFlow`<br>10.1029/2023WR036733 | Chang (2024) | RF, XGB and MLP (multi-model) | Site-median HYDROSWOT W (5,269 sites) and d (2,678 sites); watershed predictors; STREAM-geo product for ~2.7M NHDPlus reaches | None. No resistance or continuity term in the methods | n/a. W and d separate | **Split:** random 75:25 over site medians. **OOD:** Not tested (only ecoregion and stream-order breakdowns of the random test set; no held-out region). **ID→OOD Δ:** n/a | R² up to 0.75 (W) and 0.67 (d), vs 0.45 / 0.18 for regional HG | U-shaped error across stream orders; worst performance and greatest overprediction at the lowest and highest orders; first-order streams under-sampled | H1, H2 (M1/M2) | Verified-full-text (arXiv preprint; numbers match the published abstract) |
| `Doyle_2023_RandomForestGeometry`<br>10.1111/1752-1688.13116 | Doyle (2023) | Random forest | Several thousand NRSA field measurements; wetted W, thalweg d, bankfull W and d; joinable to 1.1M NHDPlusV2.1 segments | None | n/a | **Split:** Not reported (abstract only). **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | Pseudo-R² 0.70–0.91; median absolute error within ±6–21% of IQR | Independent targets; design unverified | H1 | Verified-abstract |
| `ModaresiRad_2024_ChannelDimensions`<br>10.1029/2024JH000173 | Modaresi Rad (2024) | ML (algorithm not confirmed from the abstract) | >200,000 USGS ADCP records; NWM discharge, NHGF, StreamCat and satellite/reanalysis predictors; CONUS | None stated | n/a. Uses NWM discharge as a predictor | **Split:** held-out testing set (design not stated in the abstract). **OOD:** Not reported (abstract only). Regional breakdown: **lowest skill in mountainous and plateau regions**. **ID→OOD Δ:** Not reported | Log-space R² 0.80 bankfull d (0.77 in-channel); 0.76 bankfull TW (0.66 in-channel) | Regional skill loss | H1, H3 | Verified-abstract |
| `Lin_2020_GlobalBankfullWidth`<br>10.1029/2019GL086405 | Lin (2020) | ML with 16 environmental covariates | Landsat-derived widths over ~1.6M km of rivers; global | None | n/a. Width only | **Split:** "two testing cases" (design not stated in the abstract). **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | R² 0.81 and 0.77; existing schemes capture 30–40% of width variance | No depth or shape | H1 (context) | Verified-abstract |
| `Heldmyer_2022_NWMChannelParameterization`<br>10.5194/hess-26-6121-2022 | Heldmyer (2022) | Sobol sensitivity; at-a-station HG fits; Manning inversion n = R^(2/3)·S^(1/2)/v with R = wd/(w+2d) and NHDPlus ElevSlope; trapezoid bottom width and side slope from a linear fit of d on w/2; log-log regression of n on slope at HUC4/HUC2/CONUS | HyG, ~48,000 gauges; NWM v2.1 (2.7M reaches); streamflow evaluation at ~7,400 gauges | Physics-only (used for inversion) | Serial two-stage: observe → invert → regionalize → NWM routing. No NN | **Split:** **random** k-fold CV (k = 10) of the n regionalization. **OOD:** Not tested for n. Downstream check is NWM streamflow at ~7,400 gauges in the same domain. **ID→OOD Δ:** n/a | Mean streamflow R² 0.479 → 0.494 (significant). n moved from the 0.06 default to **0.006–0.537 (median 0.077)** | Slope proxy and the R approximation; the land model dominates flow volume | H4 (n range); M6 analogue | Verified-full-text |
| `Enzminger_2024_HyGDataset`<br>10.1038/s41597-024-03916-7 | Enzminger (2024) | At-a-station HG fits; Manning inversion with NHDPlusV2 ElevSlope; HUC4 regionalized downstream HG | 4,051,682 measurements at 66,841 gauges; n at 11,513 gauges that passed QC; CONUS | Physics-only (inversion) | Serial. No NN | **Split:** n/a (dataset descriptor). **OOD:** Not tested (the HUC4 downstream HG regressions are not externally validated). **ID→OOD Δ:** n/a | Public at-a-station HG, at-a-station n, discharge percentiles and HUC4 downstream HG | Slope and hydraulic-radius substitutions propagate into n | H4 (external n benchmark) | Verified-full-text |
| `AlMehedi_2024_ChannelRoughness`<br>10.1029/2023EF004257 | Al Mehedi (2024) | Random forest predicting n (and alternative resistance parameters) in space and time | n computed from observed V, S and R in ~200,000 surveys at ~5,000 U.S. sites; continental | Physics-only target construction; RF unconstrained | Serial two-stage (back-calculate n, then RF). No physics layer for geometry | **Split:** Not reported (abstract only). NSE > 0.7 "in space and time". **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | NSE > 0.7 for n. Temporal variability explains ~35% of n variance, spatial 50% | Inherits slope and R error; n is time-variable | H4; **M6 analogue; novelty risk** | Verified-abstract |
| `Bjerklie_2020_CrossSectionHydraulics`<br>10.1029/2019WR025986 | Bjerklie (2020) | Exploratory statistics of hydraulic ratios | QC subset of HYDROSWOT (20,625 measurements at 3,519 stations, *(Elicit, unverified)*); national | None (descriptive) | n/a | **Split:** n/a (descriptive analysis; no predictive model). **OOD:** n/a. **ID→OOD Δ:** n/a | The d_max/d̄ ratio is "highly predictable" and consistent across river sizes and discharges; ≈1.5 *(Elicit, unverified)* | Repeated measurements are not necessarily at the same cross-section *(Elicit, unverified)* | H4 (observed-shape prior) | Verified-abstract |
| `Bjerklie_2026_BaselineFlowResistance`<br>10.1111/1752-1688.70142 | Bjerklie (2026) | Multiple regression for baseline (near-bankfull) n | 14 calibrated Alaskan reaches and 754 U.S. ADCP-derived reaches; SWORD and S3 morphometrics | Physics-only target (calibrated modified Manning) | Serial | **Split:** Not reported (abstract only). Two datasets: 14 calibrated Alaskan reaches and 754 U.S. ADCP reaches. **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | Significant predictors: water-surface slope, width and elevation variation, number of active channels, maximum width, sinuosity, meander length | Less certain n for the 754-reach set | H4 | Verified-abstract |
| `Guillon_2020_ChannelTypesML`<br>10.1029/2019WR026691 | Guillon (2020) | Random forest selected from 20,000 model combinations of 287 predictors | 290 field surveys; 108,943 reaches; Sacramento basin | None | n/a (classification) | **Split:** cross-validation (type not stated in the abstract); median accuracy 61%. **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | Six times the 10% random baseline | Channel-type classification, not geometry; regional | Context | Verified-abstract |

### Block B — Hydraulic-geometry theory, regional curves and flow resistance

| Key / DOI | First author (year) | Method / architecture | Dataset & scale | Physical constraints | Coupling | Validation & OOD behavior | Key result | Primary limitation | H-link | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| `Dingman_2007_AnalyticalHG`<br>10.1016/j.jhydrol.2006.09.021 | Dingman (2007) | Analytical derivation of at-a-station HG for power-law cross-sections | Theory | Physics-only | n/a | **Split:** n/a (analytical theory). **OOD:** n/a. **ID→OOD Δ:** n/a | At-a-station exponents follow from the shape exponent and the resistance law | Assumes uniform flow and an idealized section | M5 shape layer (lineage) | Lineage |
| `Bieger_2015_RegionalCurves`<br>10.1111/jawr.12282 | Bieger (2015) | Regional power-law regressions on drainage area | U.S. physiographic divisions and provinces | None (empirical) | n/a | **Split:** regression fits compared across regionalization levels (no hold-out stated in the abstract). **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | Division-level curves are more reliable than a nationwide curve; drainage area predicts less well than bankfull Q | Small samples in some provinces | M0; H1 | Verified-abstract |
| `BlackburnLynch_2017_HLRRegionalCurves`<br>10.1111/1752-1688.12540 | Blackburn-Lynch (2017) | Regional curves by HLR and physiographic unit | 2,856 sites; CONUS | None | n/a | **Split:** regression fits compared across stratifications (no hold-out stated in the abstract). **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | Stratifying by HLR improved reliability | Bankfull; drainage area only | M0 alternative | Verified-abstract |
| `Bastola_2023_DimensionlessBankfull`<br>10.1029/2022WR032688 | Bastola (2023) | Dimensionless regression (with Q as predictor, or with precipitation) | 538 sand and gravel reaches (UK and USA) | Dimensional-analysis structure | n/a | **Split:** Not reported (abstract only). **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | Bankfull geometry related to watershed and precipitation characteristics | Bankfull only; small sample | M0 context | Verified-abstract |
| `Bjerklie_2005_ManningChezy`<br>10.1029/2004WR003776 | Bjerklie (2005) | Manning- and Chezy-based discharge models vs multiple regression | 1,037 measurements in 103 rivers (US, NZ) | Physics-only | n/a | **Split:** separate calibration and validation sets (the partition is not stated in the abstract). **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | A slope exponent of 0.33 instead of 0.5 reduces the variance of the resistance estimate | In-bank only | H4; resistance form for Phase 2 | Verified-abstract |
| `Mir_2023_FlowResistanceML`<br>10.2166/wst.2023.396 | Mir (2023) | RF, extra trees, XGB, Lasso | Laboratory alluvial channels with bedforms; six inputs | None | n/a | **Split:** K-fold CV on laboratory data. **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | R² ≥ 0.99 for the tree models | Laboratory scale; not a network product | Context (n as an ML target) | Verified-abstract |
| `Mohanta_2025_RoughnessSinuousChannels`<br>10.66132/ngce20250202 | Mohanta (2025) | ANN, GEP, ELM, GPR | Laboratory two-stage sinuous channels plus one flood on the Baitarani River | None | n/a | **Split:** training and testing phases (design not stated in the abstract). **OOD:** laboratory-to-field application to one Baitarani flood; skill not stated in the abstract. **ID→OOD Δ:** Not reported | R² 0.82 (GPR, ANN) | Laboratory scale. The venue is new (volume 1) | Context | Verified-abstract |

### Block C — Physics-constrained and differentiable ML

| Key / DOI | First author (year) | Method / architecture | Dataset & scale | Physical constraints | Coupling | Validation & OOD behavior | Key result | Primary limitation | H-link | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| `Tsai_2021_ParameterLearning`<br>10.1038/s41467-021-26107-z | Tsai (2021) | Differentiable parameter learning (dPL): an NN maps inputs to process-model parameters | Soil moisture and streamflow; CONUS | Hard (process model in the loop) | End-to-end NN→physics | **Split:** temporal test at the training locations. **OOD:** spatial generalization to neighbouring out-of-training grid cells (soil moisture); global PUB (streamflow). **ID→OOD Δ:** dPL showed "almost no degradation" from training cells to neighbouring cells, while SCE-UA's ubRMSE increased (Wilcoxon test, significant) *(full text)* | Better performance and physical coherence as data grow; needed ~12.5% of the training data for similar skill; orders of magnitude cheaper than calibration | Depends on process-model structure | M5 design (lineage); H3, H4 | Verified-abstract; OOD cell verified in full text |
| `Shen_2023_DifferentiableModelling`<br>10.1038/s43017-023-00450-9 | Shen (2023) | Perspective on differentiable modelling | Geosciences | Hard / flexible | NN→physics | **Split:** n/a (perspective). **OOD:** n/a. **ID→OOD Δ:** n/a | Claims better interpretability, generalizability and extrapolation than pure ML at similar accuracy | Perspective, not a test | M5 framing (lineage) | Verified-abstract |
| `Feng_2022_DifferentiableHBV`<br>10.1029/2022WR032404 | Feng (2022) | δ models: HBV backbone with embedded NNs | 671 CAMELS basins | Hard (mass-conserving backbone) | End-to-end | **Split:** temporal hold-out *(Elicit, unverified)*. **OOD:** Not reported (abstract only); spatial tests appear in the follow-up paper (Feng 2023). **ID→OOD Δ:** Not reported | Median NSE 0.732 vs 0.748 for LSTM (Daymet); outputs untrained variables | Small accuracy gap to LSTM | H1 (accuracy cost of hard constraints) | Verified-abstract |
| `Feng_2023_DifferentiableUngauged`<br>10.5194/hess-27-2357-2023 | Feng (2023) | δ models vs LSTM | CONUS basins | Hard | End-to-end | **Split:** PUB = random 10-group basin CV. **OOD:** **PUR = 7 contiguous CONUS regions, leave-one-region-out** *(full text)*. **ID→OOD Δ:** δ models approached or surpassed LSTM in PUB and surpassed it in PUR (relative result; numbers not extracted) | δ models approached or surpassed LSTM in PUB and **surpassed LSTM in PUR**; parameter fields stayed spatially stable | Worse low-flow trends | **H3** | Verified-abstract; OOD cell verified in full text |
| `Bindas_2024_DifferentiableRouting`<br>10.1029/2023WR035337 | Bindas (2024) | Differentiable Muskingum-Cunge routing with an embedded NN that infers Manning's n and channel geometry from reach attributes | Juniata River basin; trained only on downstream hydrographs | Hard (routing physics) | **End-to-end NN→physics (n, geometry)** | **Split:** trained on downstream hydrographs over short periods. **OOD:** untrained inner gauges in the same basin, and other time periods; synthetic identifiability tests. **ID→OOD Δ:** Not reported (abstract only). n trained on short periods performed well in other periods | Better long-term routing for subbasins >2,000 km². **The geometry parameter was unidentifiable; n was identifiable with moderate precision.** The learned n pattern matched literature expectations | Absolute n varies with the training period | **H4; novelty risk; identifiability** | Verified-abstract |
| `Liu_2025_UniversalSWEResistance`<br>10.1029/2025WR040265 | Liu (2025) | Universal SWE solver (Hydrograd) with automatic differentiation; an NN learns an n relationship | A real river channel (single site) | Hard (2D SWE) | End-to-end NN→physics | **Split:** forward-model validation, then one real-world inverse case. **OOD:** "out-of-sample scenarios" claimed but not quantified in the abstract. **ID→OOD Δ:** Not reported | Gradient-based inversion of n; an NN learns n as a function of hydraulic parameters and flow | Single site | H4; mechanism | Verified-abstract |
| `Beucler_2021_AnalyticConstraints`<br>10.1103/PhysRevLett.126.098302 | Beucler (2021) | Architecture-constrained (ACnet) vs loss-constrained (LCnet) vs unconstrained NN | Convection emulation (climate) | **Hard vs Soft compared** | Hard constraint layer | **Split:** training, validation and test sets from the same simulated climate. **OOD:** **Not tested** in this paper. The authors caution that obeying conservation laws "does not guarantee the ability to generalize well far outside of the training set" (e.g. a warmer climate) *(full text)*. **ID→OOD Δ:** n/a | Architectural constraints enforce conservation to machine precision **without degrading performance**, and reduce error in the constrained outputs | Different domain | **H1 (M4 vs M5)** | Verified-full-text (arXiv version) |
| `Read_2019_ProcessGuidedDL`<br>10.1029/2019WR024922 | Read (2019) | Process-guided DL: LSTM with an energy-conservation penalty and process-model pretraining | 2 detailed lakes; 68 lakes | Soft (loss penalty) | Serial pretraining plus soft loss | **Split:** test period (2 detailed lakes; extended to 68 lakes). **OOD:** Not reported (abstract only). The gain held only when pretraining data had more variability than the training period. **ID→OOD Δ:** Not reported | Median RMSE 1.65 °C vs 1.78 (DL) and 2.03 (process-based) | Benefit depended on the variability of the pretraining data | M4 analogue | Verified-abstract |
| `Hoedt_2021_MCLSTM`<br>10.48550/arXiv.2101.05186 | Hoedt (2021) | Mass-conserving LSTM | Benchmarks including hydrology | Hard (architectural) | Hard layer | **Split:** benchmark splits (not stated in the record). **OOD:** Not reported (abstract only); a peak-flow benchmark is reported. **ID→OOD Δ:** Not reported | State of the art for peak flows (as claimed); states are interpretable | arXiv record (the comment says ICLR 2021 submission) | H1 | Verified-abstract |
| `Wang_2025_MCRLSTM`<br>10.1029/2024WR039131 | Wang (2025) | Mass-conservation-relaxed LSTM vs MC-LSTM, LSTM and SAC-SMA | 531 CONUS watersheds | Hard vs relaxed | Hard layer | **Split:** train/test over 531 CONUS watersheds (design not stated in the abstract); regional comparison. **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | MCR beats MC-LSTM but **underperforms LSTM**; strict conservation is unrealistic where fluxes are unobserved | A mis-specified constraint costs skill | **H1 risk** | Verified-abstract |
| `Ohara_2024_PINNRiverInversion`<br>10.1063/5.0232852 | Ohara (2024) | PINN (motion, continuity and constant-discharge constraints) | Flume experiment and simulations with alternating bars | Soft (PINN residual loss) | Inverse problem | **Split:** n/a (single-system inverse problem). Validated against flume and numerical data; robustness to data sparsity tested. **OOD:** n/a. **ID→OOD Δ:** n/a | Inverts depth, bed elevation and roughness without direct training data for them; mass conservation improved convergence | Laboratory and reach scale | H4 (mechanism) | Verified-abstract |
| `Strijhak_2025_PINNRoughness`<br>10.3390/w17182731 | Strijhak (2025) | PINN for 1D/2D Saint-Venant; inverse roughness | Synthetic channels (Delft3D) | Soft | Inverse problem | **Split:** n/a (synthetic single-system inverse problem; Delft3D data). **OOD:** n/a. **ID→OOD Δ:** n/a | n error ≤10% (2D). Training took 2.5–3.5 h on an A100 GPU | Synthetic; cost reported | Context (cost) | Verified-abstract |
| `Liu_2024_BathymetryInversion`<br>10.1029/2023WR035890 | Liu (2024) | Differentiable CNN surrogate of SWE plus gradient-based inversion | River reach | Soft (range regularizations) | Surrogate inversion | **Split:** n/a (single-reach inverse problem). **OOD:** n/a. **ID→OOD Δ:** n/a | Bathymetry inversion works; the authors suggest the method transfers to roughness | Needs regularization to contract the solution space | Identifiability context | Verified-abstract |
| `Feng_2023_PINNSaintVenant`<br>10.1029/2022WR033168 | Feng (2023) | PINN solving 1D Saint-Venant; Fourier-feature tidal boundary | Synthetic floodplain and channel; tidal river | Soft | Downscaling | **Split:** n/a (single-system solver checked against analytical and numerical solutions). **OOD:** n/a. **ID→OOD Δ:** n/a | Satisfactory depth accuracy with limited observations | Single-system solver | Context | Verified-abstract |

### Block D — Out-of-distribution and spatial validation

| Key / DOI | First author (year) | Method / architecture | Dataset & scale | Physical constraints | Coupling | Validation & OOD behavior | Key result | Primary limitation | H-link | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| `Roberts_2017_BlockCV`<br>10.1111/ecog.02881 | Roberts (2017) | Review and simulations of blocked CV | Ecology | n/a | n/a | **Split:** n/a (methods review). Simulations compare random CV with spatial, temporal and hierarchical blocking. **OOD:** blocked CV. **ID→OOD Δ:** random CV "serious[ly]" underestimates predictive error | Block CV is "nearly universally more appropriate" than random CV for prediction to new data | Blocking can itself induce extrapolation | H3 design (site grouping) | Verified-abstract |
| `Ploton_2020_SpatialValidation`<br>10.1038/s41467-020-18321-y | Ploton (2020) | RF biomass mapping; random vs spatial validation | 11.8M trees; central Africa | n/a | n/a | **Split:** random K-fold CV. **OOD:** spatial K-fold CV and buffered leave-one-out (0–150 km buffers). **ID→OOD Δ:** R² **0.53 → 0.14** (spatial K-fold); buffered LOO R² 0.50 at a 0 km buffer → about 0.15 at large buffers *(full text)* | Non-spatial validation suggested >50% variance explained; spatial validation showed **quasi-null** predictive power | Domain differs | H3 (random splits are optimistic) | Verified-abstract; OOD cell verified in full text |
| `Meyer_2021_AreaOfApplicability`<br>10.1111/2041-210X.13650 | Meyer (2021) | Dissimilarity index and area of applicability (AOA) | Simulation studies | n/a | n/a | **Split:** random and spatially clustered CV (simulation). **OOD:** prediction outside the area of applicability. **ID→OOD Δ:** inside the AOA, error is comparable to CV error; outside it, the CV error "does not apply" | Recommends reporting the AOA and DI-dependent performance | Simulated case | H3; UQ reporting | Verified-abstract |
| `Kratzert_2019_UngaugedLSTM`<br>10.1029/2019WR026065 | Kratzert (2019) | LSTM, k-fold over basins | 531 CAMELS basins | None | n/a | **Split:** k-fold over 531 basins (test basins supply no training data); 15-year validation period. **OOD:** **prediction in ungauged basins**. **ID→OOD Δ:** Not reported (abstract only) | Out-of-sample median NSE 0.69 vs 0.64 (calibrated SAC-SMA) and 0.58 (NWM) | Suggests physical constraints *might* help | H3 (strong pure-ML baseline) | Verified-abstract |
| `Krabbenhoft_2022_GaugePlacementBias`<br>10.1038/s41893-022-00873-0 | Krabbenhoft (2022) | Placement-bias analysis of the global gauge network | Global gauges | n/a | n/a | **Split:** n/a (representativeness analysis; no predictive model). **OOD:** n/a. **ID→OOD Δ:** n/a | Gauges sit disproportionately on **large, perennial rivers** | Global, not CONUS-specific | **H2** (small-stream under-sampling) | Verified-abstract |
| `Frame_2022_ExtremeEvents`<br>10.5194/hess-26-3377-2022 | Frame (2022) | LSTM and mass-conserving LSTM vs SAC-SMA and NWM | CAMELS basins | Hard (MC-LSTM) | Hard layer | **Split:** three train/test schemes. **OOD:** trained on water years with peaks at or below a 5-year return period; tested on years above it (1996–2014), with at least 1 year of separation *(full text)*. **ID→OOD Δ:** LSTM "remained relatively accurate" on extremes; MC-LSTM lost skill (numbers not extracted) | LSTM remained relatively accurate for extremes; **adding mass balance reduced skill during extreme events** | Streamflow, not geometry | **H1/H3 risk** | Verified-abstract; OOD cell verified in full text |

### Block E — Network-aware and GNN models (background only; non-goal in `0_point_zero.md` §4)

| Key / DOI | First author (year) | Method / architecture | Dataset & scale | Physical constraints | Coupling | Validation & OOD behavior | Key result | Primary limitation | H-link | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| `Sun_2022_GNNRiverNetwork`<br>10.5194/hess-26-5163-2022 | Sun (2022) | Physics-guided GNN surrogate of a vector river-network model, fine-tuned on observations; graph data fusion | Snow-dominated western U.S. basin, then a larger basin | Soft (pretraining on physics model output) | Serial | **Split:** cross-validation gauges. **OOD:** Not reported (abstract only). **ID→OOD Δ:** Not reported | Median KGE >0.97 as a surrogate; fusion improved KGE by up to 50% at some gauges; graph coarsening achieves comparable skill at a fraction of training cost | Streamflow; out of scope | Background | Verified-abstract |

---

## 3. Novelty audit against M5

**M5 (hard hybrid), as defined in `0_point_zero.md`:**
- An NN predicts top width W, shape exponent r and effective Manning n from attributes available at ungauged reaches.
- A differentiable Manning + continuity layer computes d̄, d_max, A and V from EROM `QE_MA` and NHDPlusV2.1 slope.
- The model is trained on observed cross-section geometry at HYDROSWOT sites, at CONUS scale.
- It is evaluated with site-grouped, leave-region-out, size-tail and independent-set tests.

**Verdicts:**
- **Overlap:** same mechanism, supervision and scale.
- **Partial:** shares one or more components.
- **None:** no shared mechanism.

| Paper | What is derived | Observed or learned | Scale | NN feeds a physics layer? | Supervision target | OOD tested? | Verdict vs M5 |
|---|---|---|---|---|---|---|---|
| Heldmyer 2022 [@Heldmyer_2022_NWMChannelParameterization] | n (Manning inversion), trapezoid bottom width and side slope, top width at Q99/Q99.9 | **Observed** at gauges, then regionalized by log-log regression (n on slope) | CONUS (NWM) | **No.** Physics is used to invert, not inside a learned predictor | NWM streamflow | Random 10-fold CV of the regionalization only | **Partial.** Shares the n and shape output set. Methodologically it is the **M6** pattern (two-stage back-calculated n), not M5 |
| Bjerklie 2020 [@Bjerklie_2020_CrossSectionHydraulics] | Distributions of d_max/d̄, W/d̄, Froude number, V_max/V̄ | **Observed** (descriptive) | National (HYDROSWOT) | No model | None | No | **Partial.** Supplies an empirical shape prior; d_max/d̄ ≈ 1.5 *(Elicit, unverified)*. No learned r or n |
| Chang 2024 [@Chang_2024_GeometryOfFlow] | Site-median W and d̄ (plus W/D ratio and %RSSA) | Learned (RF/XGB/MLP) | CONUS | **No.** Full text shows no physics coupling | Observed W and d (separately) | Random 75:25 only; ecoregion breakdown | **None** on mechanism. Head-to-head comparator for M1/M2 |
| Enzminger 2024 (Q3) [@Enzminger_2024_HyGDataset] | At-a-station n (Manning inversion with ElevSlope); HUC4 downstream HG | Observed and regionalized | CONUS | No | n/a (dataset) | Not reported | **Partial.** A national observed n already exists. M5's learned n must be benchmarked against it (H4) |
| Bjerklie 2026 (Q3) [@Bjerklie_2026_BaselineFlowResistance] | Baseline n via calibrated modified Manning; regression on morphometrics | Observed n, then regression | 754 U.S. reaches plus 14 in Alaska | No | n | Not in the abstract | **Partial.** n predicted from morphology, but not coupled to a geometry prediction |
| Huthoff 2008 (Q3) ⚠️ (no DOI) [@Huthoff_2008_ArbitraryCrossSections] | Resistance in arbitrarily shaped sections (lateral momentum exchange) | Theory | n/a | No | n/a | n/a | **None.** It is a caveat: shape and roughness are not separable, which supports treating n as *effective* (`0_point_zero.md` §4) |
| Lin 2020 (Q1) [@Lin_2020_GlobalBankfullWidth] | Bankfull width | Learned | Global | No | Width | Two test cases | **None** |
| Guillon 2020 (Q1) [@Guillon_2020_ChannelTypesML] | Channel type (classes) | Learned (RF) | Regional basin | No | Field labels | CV | **None** |
| Mir 2023 (Q1) [@Mir_2023_FlowResistanceML] | n in bedform channels | Learned (tree models) | Laboratory | No | n | K-fold | **None** |
| Mohanta 2025 (Q1) [@Mohanta_2025_RoughnessSinuousChannels] | n in two-stage sinuous channels | Learned (ANN/GEP/GPR) | Laboratory plus one river | No | n | Test phase | **None** |
| Doyle 2023 (Q1) [@Doyle_2023_RandomForestGeometry] | W, d (bankfull, low flow) | Learned (RF) | CONUS | No | W, d | Unverified | **None** |
| Zarrabi 2025 (Q1) [@Zarrabi_2025_ChannelGeometryCONUS] | W, d (bankfull, mean flow) | Learned (RF/XGB) | CONUS | No | W, d | Random split plus independent sets | **None** on mechanism. Primary comparator (M1) |
| *Supplementary:* Al Mehedi 2024 [@AlMehedi_2024_ChannelRoughness] | n in space and time | Observed n, then **learned (RF)** | Continental (~5,000 sites) | **No.** Two-stage | Back-calculated n | Not verified | **Partial (high).** Continental ML prediction of n already exists. It is an M6 analogue |
| *Supplementary:* Bindas 2024 [@Bindas_2024_DifferentiableRouting] | n and a channel-geometry parameter | **Learned end-to-end** | One basin (Juniata) | **Yes.** NN → differentiable Muskingum-Cunge | Downstream **hydrographs** | Inner gauges and other periods | **Partial (high).** Same mechanism class (NN → n and geometry → differentiable physics). It differs in supervision (hydrographs, not observed cross-sections), output (routing, not geometry), scale (one basin, not CONUS) and evaluation (no leave-region-out). Its geometry parameter was **unidentifiable** |
| *Supplementary:* Liu & Song 2025 [@Liu_2025_UniversalSWEResistance] | n relationship | Learned end-to-end | Single channel | Yes. NN → 2D SWE | Flow observations | Claimed | **Partial.** Mechanism at local scale |

**Audit outcome.**
- No audited paper received an **overlap** verdict.
- Three received **partial (high)** or **partial** verdicts on components that M5 had been expected to "own":
  - national n estimation: Enzminger 2024, Heldmyer 2022, Al Mehedi 2024;
  - NN → n → differentiable physics: Bindas 2024, Liu & Song 2025;
  - empirical shape regularity: Bjerklie 2020.
- **Novelty claims M5 cannot make:**
  - national prediction of n;
  - ML prediction of n;
  - an NN that feeds roughness to a differentiable hydraulic model.
- **What remains distinct (bounded, no "first" claim):**
  - supervising the hard layer **directly on observed cross-section geometry** (W, d̄, A) at thousands of gauges;
  - with an explicit shape exponent r;
  - at CONUS scale;
  - evaluated on **site-grouped, leave-region-out and size-tail** splits;
  - against comparators retrained on identical folds (M1, M2) **and** against the established two-stage n approach (M6; cf. Heldmyer 2022, Enzminger 2024, Al Mehedi 2024).

This is a result of searching the audited corpus and the P1-D4 queries. It is not proof that no such work exists.

---

## 4. Cross-cutting limitations and trade-offs

The template frames this as *latency vs conservation*. Latency is not a target (ADL-010), so the trade-offs are restated for this project:

1. **Statistical flexibility vs hydraulic consistency.**
   - Every audited CONUS product models W and d̄ as independent statistical targets with no continuity or resistance closure [@Zarrabi_2025_ChannelGeometryCONUS; @Chang_2024_GeometryOfFlow; @Doyle_2023_RandomForestGeometry; @ModaresiRad_2024_ChannelDimensions].
   - Products that do respect Manning use it only to **construct** n targets, never to constrain depth predictions [@Heldmyer_2022_NWMChannelParameterization; @Enzminger_2024_HyGDataset; @AlMehedi_2024_ChannelRoughness].
2. **Soft vs hard constraints.**
   - Architectural constraints can enforce physics to machine precision without losing accuracy [@Beucler_2021_AnalyticConstraints]. The same authors caution that conservation does not by itself guarantee generalization far outside the training set [@Beucler_2021_AnalyticConstraints].
   - Differentiable models with hard process backbones approach LSTM accuracy [@Feng_2022_DifferentiableHBV] and beat it under regional holdout [@Feng_2023_DifferentiableUngauged].
   - However, hard mass conservation **reduced** skill for extremes [@Frame_2022_ExtremeEvents], and a strict constraint underperformed an unconstrained LSTM where fluxes are unobserved [@Wang_2025_MCRLSTM].
   - **A hard constraint helps only if the physics is well specified.** Here, specification error enters through the slope proxy and non-uniform flow, and the effective n absorbs it (`0_point_zero.md` §4).
3. **In-distribution skill vs OOD transfer.**
   - Random splits overstate skill under spatial autocorrelation [@Roberts_2017_BlockCV; @Ploton_2020_SpatialValidation; @Meyer_2021_AreaOfApplicability].
   - Both large comparators used random splits [@Zarrabi_2025_ChannelGeometryCONUS; @Chang_2024_GeometryOfFlow]. Depth skill fell sharply on independent data (R² 0.32 and 0.37) while width held up [@Zarrabi_2025_ChannelGeometryCONUS].
   - Pure ML is a strong ungauged baseline in streamflow [@Kratzert_2019_UngaugedLSTM], so H3 is not a foregone conclusion.
4. **Identifiability.**
   - When trained on hydrographs only, a learned channel-geometry parameter was unidentifiable while n was moderately identifiable [@Bindas_2024_DifferentiableRouting].
   - Inverse problems also need regularization to contract the solution space [@Liu_2024_BathymetryInversion].
   - M5 supervises on W, d̄ and A directly, which should constrain r, but the joint (r, n) identifiability given (W, d̄, Q, S) is an open Phase 2 item.
5. **Sampling bias toward large rivers.**
   - Gauges sit disproportionately on large perennial rivers [@Krabbenhoft_2022_GaugePlacementBias].
   - First-order streams are under-sampled in HYDROSWOT-based models, with the largest biases at the size extremes [@Chang_2024_GeometryOfFlow; @Zarrabi_2025_ChannelGeometryCONUS]. This motivates H2.
6. **Range of effective n.** Back-calculated n spans 0.006–0.537 [@Heldmyer_2022_NWMChannelParameterization], wider than the provisional bounds [0.015, 0.25] in `0_point_zero.md` §7. Phase 3 must set the bounds with this in mind, or H4's "clusters at the bounds" test will be confounded.
7. **Computational cost** is recorded only where reported:
   - PINN roughness inversion took 2.5–3.5 h per training run on an A100 [@Strijhak_2025_PINNRoughness];
   - dPL was orders of magnitude cheaper than evolutionary calibration [@Tsai_2021_ParameterLearning];
   - graph coarsening cut GNN training cost [@Sun_2022_GNNRiverNetwork].

---

## 5. Scientific Gap Statement

**(1) Statistical national products.** Reach-scale geometry for the CONUS river network is currently supplied by statistical models that predict width and depth as separate targets from watershed and hydrographic attributes [@Zarrabi_2025_ChannelGeometryCONUS; @Chang_2024_GeometryOfFlow; @Doyle_2023_RandomForestGeometry; @ModaresiRad_2024_ChannelDimensions]. These models clearly outperform drainage-area regional curves [@Bieger_2015_RegionalCurves; @BlackburnLynch_2017_HLRRegionalCurves]. For example, Chang et al. report R² of 0.75 and 0.67 for width and depth, against 0.45 and 0.18 for regional curves [@Chang_2024_GeometryOfFlow]. However:
- The two largest HYDROSWOT-based products were evaluated on random splits.
- Depth transferred poorly to independent data: R² of 0.32 at mean flow and 0.37 at bankfull, while width held at 0.84 and 0.65 [@Zarrabi_2025_ChannelGeometryCONUS].
- Errors are size-dependent, with Tier-2 biases of 142–187% at the low end [@Zarrabi_2025_ChannelGeometryCONUS] and the largest errors at the extremes of stream order [@Chang_2024_GeometryOfFlow]. This is consistent with gauges being concentrated on large rivers [@Krabbenhoft_2022_GaugePlacementBias].

Random validation is known to overstate predictive power under spatial structure [@Roberts_2017_BlockCV; @Ploton_2020_SpatialValidation]. The reported depth skill is therefore an optimistic estimate of performance at ungauged reaches. None of these products requires its predicted width, depth and discharge to satisfy continuity or a resistance law, and none supplies the shape or roughness that 1D models need.

**(2) Observation-based hydraulic parameterizations.** A second line of work does supply roughness and shape, but by a two-stage route:
- The authors fit at-a-station relations at gauges.
- They back-calculate Manning's n from the measured width, depth and velocity and an NHDPlus slope proxy.
- They then regionalize n with regression [@Heldmyer_2022_NWMChannelParameterization; @Enzminger_2024_HyGDataset; @Bjerklie_2026_BaselineFlowResistance] or with a random forest [@AlMehedi_2024_ChannelRoughness].

In parallel, large ADCP archives show regular bulk cross-section shape [@Bjerklie_2020_CrossSectionHydraulics], which analytical hydraulic geometry links to the section's shape exponent [@Dingman_2007_AnalyticalHG]. In these studies the resistance law is used to *construct* parameters, not to *constrain* a depth predictor: depth at an ungauged reach is never computed from learned parameters through the resistance law and then scored against observed geometry. Two consequences follow:
- **Errors cannot be corrected.** Errors in the slope proxy and in the hydraulic-radius approximation pass directly into n and cannot be corrected by fitting to observed geometry [@Enzminger_2024_HyGDataset].
- **Validation is weak.** Validation has been random k-fold CV of the regionalization, or downstream streamflow skill; the latter improved only modestly, from R² 0.479 to 0.494 [@Heldmyer_2022_NWMChannelParameterization].

**(3) Differentiable, hard-constrained learning.** Methods for learning physical parameters end-to-end through a differentiable process model are established in hydrology:
- They can improve spatial extrapolation over pure ML [@Tsai_2021_ParameterLearning; @Feng_2022_DifferentiableHBV; @Feng_2023_DifferentiableUngauged; @Shen_2023_DifferentiableModelling].
- They have been used to infer Manning's n inside routing and shallow-water models [@Bindas_2024_DifferentiableRouting; @Liu_2025_UniversalSWEResistance] and in reach-scale PINN inversions [@Ohara_2024_PINNRiverInversion; @Strijhak_2025_PINNRoughness].
- Architectural constraints can be imposed without loss of accuracy [@Beucler_2021_AnalyticConstraints].

The evidence is not one-sided. Hard conservation constraints have reduced skill for extremes and where the constrained physics is mis-specified [@Frame_2022_ExtremeEvents; @Wang_2025_MCRLSTM]. Also, when learned only from hydrographs, channel-geometry parameters were unidentifiable [@Bindas_2024_DifferentiableRouting]. Within the 39 audited papers and the supplementary search, we found no study that does all of the following:
- learns W, a shape exponent and an effective n through a hard Manning + continuity layer;
- supervises them directly on observed cross-section geometry at national scale;
- tests whether the constraint improves depth and area transfer under site-grouped and leave-region-out evaluation, compared with statistical models retrained on identical folds and with the two-stage back-calculated-n approach.

The open question is therefore not whether such hybrids can be built. It is **whether imposing the resistance law as a hard constraint improves out-of-sample depth and area at ungauged reaches, or only adds structure, and whether the n and r it learns are physically meaningful.**

*Reconciliation with `0_point_zero.md`:*
- **§1 (problem statement):** all three limitations are confirmed by the matrix evidence.
- **§5 (novelty boundary):** it must add Enzminger 2024, Al Mehedi 2024, Bindas 2024 and Liu & Song 2025 as partial-overlap works. This change is proposed as an ADL row; see the cover note.

---

## 6. Core testable hypothesis and H1–H4 mapping

**Core hypothesis (H-core).** For reach-scale cross-section geometry at ungauged CONUS reaches, compare two models:
- the hybrid: an NN predicts top width W, shape exponent r and effective roughness n, and a **hard, differentiable Manning + continuity layer** computes d̄, d_max, A and V from EROM mean flow and NHDPlusV2.1 slope (M5);
- the strongest unconstrained ML model (best of M1/M2) trained on **identical site-grouped folds**.

H-core states that, relative to that model, M5 gives depth and area estimates that:
- (i) have lower out-of-sample log-error;
- (ii) are less biased on the smallest streams;
- (iii) lose less skill from in-distribution to OOD tests.

And it does so (iv) with learned n and r that are physically plausible.

**Mechanism** (rationale, evidence-cited):
- The hard layer ties d̄ to Q, S, W, r and n. That removes free degrees of freedom from depth, the target that transfers worst [@Zarrabi_2025_ChannelGeometryCONUS].
- It carries size information through Q at small streams.
- It restricts the hypothesis space in a way that has improved regional extrapolation in differentiable hydrology [@Feng_2023_DifferentiableUngauged; @Tsai_2021_ParameterLearning].
- The documented failure mode is a mis-specified constraint [@Frame_2022_ExtremeEvents; @Wang_2025_MCRLSTM]. That keeps H-core falsifiable.

| Sub-hypothesis (from `0_point_zero.md` §3, unchanged) | Measurable outcome | Comparator | Falsified if (restated verbatim) | Literature anchor |
|---|---|---|---|---|
| **H1:** lower out-of-sample error in d̄ (and A) | Δlog-RMSE(d̄) and Δlog-RMSE(A) = hybrid − pure ML; paired bootstrap over sites; leave-region-out folds and independent sets | Best of M1/M2 | "The 95% CI of Δlog-RMSE(d̄) includes 0 or is positive" | Zarrabi 2025 (depth transfer); Beucler 2021; Frame 2022; Wang 2025 |
| **H2:** lower small-stream bias | Log-space bias of d̄ in the lowest drainage-area quintile | Best of M1/M2 | "Bottom-quintile absolute bias is not lower than pure ML's (paired bootstrap 95% CI)" | Zarrabi 2025; Chang 2024; Krabbenhoft 2022 |
| **H3:** smaller in-distribution → OOD skill loss | Skill(in-distribution) − Skill(OOD), for leave-region-out and size-tail splits | Best of M1/M2 | "The hybrid's skill loss is equal to or larger than pure ML's" | Feng 2023; Kratzert 2019; Roberts 2017; Ploton 2020 |
| **H4:** physically plausible n and r | Share of n inside the bounded range; rank correlation of n with D50 and slope (sign and strength); agreement of learned r with observed r from HYDROSWOT | Literature ranges; HYDROSWOT observed shape | "Learned parameters cluster at the bounds or outside plausible ranges, or show no relation (or the wrong sign) to D50, slope and observed shape" | Heldmyer 2022; Enzminger 2024; Al Mehedi 2024; Bjerklie 2020; Bindas 2024 |

**Numeric thresholds, test statistics and fold definitions are deferred to the Phase 3 protocol.**

**Proposed sharpenings.** None of these is applied. Each needs a PI decision and an ADL row, and none replaces H1–H4:
- **S1 (H4 diagnostic):** report the agreement of learned n with independently back-calculated n (HyG at-a-station n [@Enzminger_2024_HyGDataset]) at gauged sites. Diagnostic only; not a falsification criterion unless Phase 3 adopts it.
- **S2 (comparators):** name M6 explicitly as the two-stage back-calculated-n analogue of Heldmyer 2022, Enzminger 2024 and Al Mehedi 2024. Report M5 vs M6 and M4 vs M5 (soft vs hard; Beucler 2021) as secondary contrasts for H1 and H3.
- **S3 (Phase 2):** analyze the joint identifiability of (r, n) given (W, d̄, Q, S), following the unidentifiability result in Bindas 2024.
- **S4 (Phase 3):** set the bounds on n with reference to back-calculated ranges (0.006–0.537 in Heldmyer 2022), so that H4's bound test is not confounded.

---

## 7. References (DOI-verified 2026-10-08 against Crossref, or DataCite where marked; full entries in `docs/references.bib`)

Al Mehedi et al. 2024, 10.1029/2023EF004257 · Bastola & Diplas 2023, 10.1029/2022WR032688 · Beucler et al. 2021, 10.1103/PhysRevLett.126.098302 · Bieger et al. 2015, 10.1111/jawr.12282 · Bindas et al. 2024, 10.1029/2023WR035337 · Bjerklie et al. 2005, 10.1029/2004WR003776 · Bjerklie et al. 2020, 10.1029/2019WR025986 · Bjerklie et al. 2026, 10.1111/1752-1688.70142 · Blackburn-Lynch et al. 2017, 10.1111/1752-1688.12540 · Canova et al. 2016 (DataCite), 10.5066/F7D798H6 · Chang et al. 2024, 10.1029/2023WR036733 · Dingman 2007, 10.1016/j.jhydrol.2006.09.021 · Doyle et al. 2023, 10.1111/1752-1688.13116 · Enzminger et al. 2024, 10.1038/s41597-024-03916-7 · Feng D. (Dapeng) et al. 2022, 10.1029/2022WR032404 · Feng D. (Dapeng) et al. 2023, 10.5194/hess-27-2357-2023 · Feng D. (Dongyu) et al. 2023, 10.1029/2022WR033168 · Frame et al. 2022, 10.5194/hess-26-3377-2022 · Guillon et al. 2020, 10.1029/2019WR026691 · Heldmyer et al. 2022, 10.5194/hess-26-6121-2022 · Hoedt et al. 2021 (DataCite/arXiv), 10.48550/arXiv.2101.05186 · Huthoff et al. 2008 ⚠️ no DOI, https://research.utwente.nl/en/publications/8803bfc2-96c0-47ec-a44f-fb28b877b141 · Krabbenhoft et al. 2022, 10.1038/s41893-022-00873-0 · Kratzert et al. 2019, 10.1029/2019WR026065 · Lin et al. 2020, 10.1029/2019GL086405 · Liu, Song & Shen 2024, 10.1029/2023WR035890 · Liu & Song 2025, 10.1029/2025WR040265 · Meyer & Pebesma 2021, 10.1111/2041-210X.13650 · Mir & Patel 2023, 10.2166/wst.2023.396 · Modaresi Rad et al. 2024, 10.1029/2024JH000173 · Mohanta & Choudhary 2025, 10.66132/ngce20250202 · Ohara et al. 2024, 10.1063/5.0232852 · Ploton et al. 2020, 10.1038/s41467-020-18321-y · Read et al. 2019, 10.1029/2019WR024922 · Roberts et al. 2017, 10.1111/ecog.02881 · Shen et al. 2023, 10.1038/s43017-023-00450-9 · Strijhak et al. 2025, 10.3390/w17182731 · Sun et al. 2022, 10.5194/hess-26-5163-2022 · Tsai et al. 2021, 10.1038/s41467-021-26107-z · Wang et al. 2025, 10.1029/2024WR039131 · Zarrabi et al. 2024 (DataCite), 10.5281/zenodo.13883263 · Zarrabi et al. 2025, 10.1029/2024WR037997

---

## Appendix A — Excluded Elicit rows (50)

The row number is the 0-based position in `Elicit - Literature_Matrix.csv`. The year is the registry year; Elicit's year is shown in brackets where it differs.

| Row | First author (year) | Short title | Elicit relevance | Reason |
|---|---|---|---|---|
| 3 | Yin (2025) | PINN, 1D unsteady SWE in rivers | Medium | Forward solver for a single system; no cross-site parameter learning; represented by `Feng_2023_PINNSaintVenant` |
| 4 | Cedillo (2022) | PINN water surface, 1D steady channels | Medium | Benchmark cases; same reason |
| 7 | Zhang (2023) | Flexible monotone NN | Low | P1-D3 (Low) |
| 10 | You (2017) | Deep lattice networks | Low | P1-D3 (Low); no DOI in Elicit |
| 11 | Bandai (2020) | PINN for Richards equation with monotonicity | Low | P1-D3 (Low); preprint DOI |
| 12 | Garber (2024) | AI bankfull channel extents | High | DEM-based bankfull delineation; relevant only to the secondary bankfull stage (`0_point_zero.md` §4) |
| 14 | Gleason (2014) | At-many-stations HG | Medium | Satellite discharge via AMHG, not reach-geometry prediction; moved out to keep the core ≤ 40 |
| 17 | Lu (2021) | Bayesian and PIML streamflow | Medium | Streamflow serial hybrid; superseded by the δ-model rows |
| 18 | López-Chacón (2024) | Hybrid model for high streamflow | Medium | Single-basin streamflow; redundant |
| 19 | Konapala (2020) | ML-assisted hybrid streamflow | Low | P1-D3 (Low) |
| 20 | Prieto (2025) | PINN for open-channel velocity | Medium | Velocity-field prediction, not reach-scale geometry |
| 21 | Nearing (2021) [2020] | Role of hydrologic science in ML | Low | P1-D3 (Low); perspective |
| 22 | Reichstein (2019) | DL and process understanding | Low | P1-D3 (Low); perspective |
| 24 | Feng (2021) [2020] | DL streamflow error with ensembles and soft data | Medium | Streamflow; redundant with `Feng_2023_DifferentiableUngauged` |
| 25 | Arsenault (2023) | LSTM in ungauged basins | Medium | Streamflow PUB; redundant with `Kratzert_2019_UngaugedLSTM` |
| 26 | Blöschl et al., eds. (2013) | *Runoff Prediction in Ungauged Basins* ("Contents") | Medium | Elicit row is book front matter; background only |
| 28 | Mahoney (2023) | Performance of spatial CV approaches | Medium | No DOI in Elicit; redundant with Roberts / Ploton / Meyer |
| 29 | Vojtek (2026) | Transferability of flood-extent prediction | Medium | Flood-extent mapping; out of scope |
| 30 | Puche (2025) | LSTM temporal and spatial generalization (France) | Medium | Streamflow; redundant |
| 31 | Stock (2025) | Choosing blocks for spatial CV | Medium | Marine remote sensing; redundant with `Roberts_2017_BlockCV` |
| 32 | Schneider (2025) | KGML transferability (drain flow) | Medium | Different domain; redundant |
| 35 | Guan (n.d.) | Beyond observed extremes (hybrid DL) | Medium | **Unverifiable:** no year, no DOI |
| 36 | Frame (2021) | LSTM post-processing of the NWM | Medium | Streamflow post-processing; not geometry |
| 38 | Tyralis (2023) [2022] | Post-processing for extreme quantiles | Low | P1-D3 (Low) |
| 39 | Pasche (2024) [2022] | NN extreme quantile regression | Medium | UQ method background; Phase 3 may revisit |
| 40 | Ma (2021) | Transferring data across continents | Medium | Transfer learning is not in the study design |
| 42 | Yao (2023) | Transfer learning in Alpine regions | Medium | Same |
| 43 | Ghosh (2022) | Meta-transfer learning | Medium | Same |
| 44 | Cai (2024) | Meta-LSTM | Medium | Same |
| 45 | Oruche (2023) | Attention-based domain adaptation | Medium | arXiv preprint; same |
| 46 | Hu (2025) | HydroDCM | Medium | arXiv preprint; reservoir inflow |
| 48 | Mosaffa (2026) | GNN routing module for LSTM | Medium | GNN is a non-goal; background represented by `Sun_2022_GNNRiverNetwork` |
| 49 | Sun (2021) | Spatio-temporal GNN, large-sample hydrology | Medium | Same |
| 50 | Moshe (2020) | HydroNets | Medium | Same; no DOI in Elicit |
| 51 | Wang (2025) [2024] | River-network topology for flood warnings | Medium | GNN non-goal |
| 52 | Gilmer (2017) | Neural message passing | Low | P1-D3 (Low) |
| 53 | Liu (2022) | Directed graph DNN streamflow | Medium | GNN non-goal |
| 54 | Li (2003) | ANN for flow and sediment in a river system | Medium | Flow and sediment, not cross-section geometry |
| 55 | Kirschstein (2024) | Merit of river-network topology | Medium | **Unverifiable:** DOI `10.5555/…` resolved in neither Crossref nor DataCite; also a GNN non-goal |
| 56 | Moore et al. (2019) [Elicit: Johnston] | NHDPlus HR user guide | Medium | Data documentation; HR not used (ADL-011); already cited in ADL-004 |
| 63 | Khadangi (2009) | ANN regime-channel relationships | High | Conference paper; regime scale; abstract not audited |
| 64 | Riahi-Madvar (2011) | ANN expert system for alluvial channel geometry | High | Closed access with no retrievable abstract, so content would be Elicit-only; regime scale |
| 65 | Güven (2011) | Multi-output NN for scour geometry | Medium | Scour below structures; not channel geometry |
| 66 | Bonakdari (2019) | BP-NN stable alluvial profiles | High | Book chapter; regime scale; redundant |
| 67 | Gholami (2016) | SVM vs regression for alluvial geometry | High | No DOI in Elicit; not verified |
| 68 | Borchani (2015) | Survey of multi-output regression | Low | P1-D3 (Low) |
| 69 | Grinsztajn (2022) | Tree models vs DL on tabular data | Low | P1-D3 (Low); candidate method citation for the Phase 3 baseline choice |
| 70 | Chen (2016) | XGBoost | Low | P1-D3 (Low); method citation for Phase 3 |
| 71 | Sit (2020) | Review of DL in hydrology | Low | P1-D3 (Low); the Elicit DOI is the OSF preprint |
| 72 | Amanambu (2025) | ML bankfull stage, lowland river | High | Single river; bankfull stage; secondary scope |

## Appendix B — Watch list (found by the supplementary search; not included)

| Item | Status | Why it is not in the core |
|---|---|---|
| Xu et al. (2026), *X-River: A differentiable Saint-Venant river routing framework…*, 10.22541/essoar.15009809/v1 | Crossref `posted-content`, 2026-10-02 | Not peer-reviewed. A differentiable routing model with spatial parameter learning (Delaware basin). Re-check at the Phase 10 audit |
| Radfar et al. (2026), inverse-PINN benchmark for Manning friction, 10.1063/5.0349054 | Found via OpenAlex; not yet Crossref-checked | Redundant with `Strijhak_2025_PINNRoughness` and `Ohara_2024_PINNRiverInversion` |
