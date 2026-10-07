# Phase 0 pre-work: resolving the open data questions carried over from v1

| Field | Value |
|---|---|
| S-SDLC phase | 0 (Point ZERO). This is pre-work for the gate deliverable `docs/0_point_zero.md` |
| Date | 2026-10-07 |
| Code-generation lock | 🔒 ON. No project code was written. The evidence below comes from documents, dataset metadata, and read-only text extraction |
| Scope | The six open questions in `.agents/.research_state.md` §4 |
| Reference policy | Every source cited here was checked on 2026-10-07 against Crossref, DataCite, the USGS Publications Warehouse, or the publisher. See §4 |

> **Constraint encountered.** The v1 merged HYDROSWOT–NHDPlus table is **not in the workspace**. `data/raw/`, `data/processed/` and `data/checkpoints/` are empty, and no tabular data files (csv, xlsx, parquet, gpkg, shp, h5, pkl) exist anywhere in the project tree. The answers below therefore come from the v1 manuscript text, the official dataset documentation, and Zarrabi et al. (2025). Where an answer is inferred rather than read directly from the v1 data, it is labelled as an inference.

---

## 1. Summary

| # | Question | Status | Answer | Confidence |
|---|---|---|---|---|
| Q1 | Which NHDPlus version did v1 use? | **Resolved by inference** | **NHDPlus High Resolution (HR)** | High. Confirm against the PostGIS source table |
| Q2 | How many unique sites are in the 53,576 mean-flow records? | **Open (blocked)** | About 9 records per site, *if* the 5,942 bankfull rows are unique sites. So multiple records per site is near-certain | Medium (bounded inference) |
| Q3 | Is `discharge_cms` measured at the same time as the geometry? | **Resolved** | **Yes.** It is the discharge from the same ADCP measurement | High |
| Q4 | Where does `vema` come from? | **Resolved by inference** | **NHDPlus HR EROM `VEMA`**: mean annual velocity at the gage-adjusted mean annual flow, computed with the Jobson (1996) method | High. Confirm against the PostGIS join |
| Q5 | How were the Table 1 ANN R² values computed? | **Not resolvable from documents. Out of scope** | Unknown. Not needed for the new study (ADL-001) | — |
| Q6 | Where is the raw data, and what are the licences? | **Partly resolved** | All public sources are identified and their licences recorded (§2.6). The v1 merged table's location is unknown | — |

---

## 2. Evidence and implications for each question

### Q1. NHDPlus version → NHDPlus HR (inferred)

**Evidence**
1. **Primary key.** v1 §2.2.1 says each site was given "a unique NHDPlusID". `NHDPlusID` is the feature key throughout NHDPlus HR (Moore et al., 2019). NHDPlusV2.1 identifies reaches by `COMID`, as Zarrabi et al. (2025) do.
2. **EROM field names.** The v1 predictors `qema` and `vema` match the NHDPlus HR fields `NHDPlusEROMMA.QEMA` and `NHDPlusEROMMA.VEMA` letter for letter (Moore et al., 2019, Table 17). The V2 equivalents are named differently:
   - `QE_MA` / `VE_MA`, as used by Zarrabi et al. (2025);
   - `Q0001E` / `V0001E` in the EROM tables of the NHDPlusV2 User Guide (McKay et al., 2012).

   Neither V2 form becomes `qema` without deliberate renaming.
3. The other v1 predictors (`totdasqkm`, `pathlength`, `streamorde`, `areasqkm`, `slope`) exist in both versions, so they don't distinguish between them.

**Still to confirm:** the source table or schema name of the v1 PostGIS k-NN join.

**Implication for the new study. Decision D1 is needed.** The closest comparator product (Zarrabi et al., 2025, about 2.64 M reaches) and the D50 and aridity predictors it uses are all keyed to **NHDPlusV2.1 COMIDs** (Zarrabi et al., 2025, §2.1). Using V2.1 allows a like-for-like baseline (M1) and reuse of their predictor stack. Using HR would require a crosswalk for every comparator layer. → **Recommendation: NHDPlusV2.1 for the new study.** v1's use of HR stays a v1-only fact.

**A related method difference worth carrying into Phase 3:**
- **v1** linked gages to reaches with a spatial k-NN snap.
- **Zarrabi et al. (2025)** used the gage-to-reach identifier provided in NHDPlusV2.1.

A k-NN snap can assign a gage to the wrong reach, for example a tributary near a confluence. The Phase 3 protocol should use the identifier-based link and report the disagreement rate between the two methods as a QC diagnostic.

### Q2. Unique sites in the mean-flow dataset → open, but bounded

**Known counts (verified sources)**

| Dataset | Observations | Sites | Obs/site | Source |
|---|---|---|---|---|
| HYDROSWOT, raw | 223,022 | 10,081 | ≈ 22 | Zarrabi et al. (2025, §2.1). The DataCite abstract gives "200,000+ records … more than 5,000 stations" (Canova et al., 2016), which is consistent but less precise |
| HYDROSWOT after Zarrabi QC | 38,191 | 4,607 | ≈ 8 | Zarrabi et al. (2025, §2.1) |
| v1 national bankfull | 5,942 | 5,942 (v1 text: "each data point represents … a unique river reach") | 1 | v1 §2.2.1 |
| v1 national mean flow | 53,576 | **unknown** | **≈ 9.0 if the sites match the bankfull set** (53,576 / 5,942) | Inference |
| v1 Yazoo, bankfull / mean flow | 25 / 267 | 25 (?) | ≈ 10.7 | Inference |

**What can be concluded now**
- The 53,576 records exceed the entire Zarrabi-cleaned HYDROSWOT set (38,191). So v1 either skipped the 5% discharge-consistency filter or applied a different QC, and it **kept multiple measurements per site**.
- The v1 mean-flow model was evaluated on a single random 80/20 split of measurements. The same gages were therefore almost certainly in both training and test sets. **The leakage is confirmed in kind; its size is unknown.**

**For the new study the question is largely moot.** Targets will be rebuilt from raw HYDROSWOT with `site_no` as the grouping key (ADL-001; split-then-scale; GroupKFold). Q2 matters only if v1 numbers are ever quoted.

**Action when the v1 table is located:** count distinct `site_no` / `NHDPlusID` values with a read-only shell query (no project code needed).

### Q3. `discharge_cms` concurrency → yes, concurrent

**Evidence**
- HYDROSWOT is a flat file with **one row per ADCP discharge measurement**, identified by `q_meas_no` (and `q_meas_chan_nu` for multi-channel measurements) (Canova et al., 2016, DataCite abstract).
- The discharge fields `q_va` and `meas_q_va` sit in the **same row** as the geometry fields `stream_wdth_va`, `mean_depth_va`, `max_depth_va` and `xsecond_area_va` (HYDROSWOT `FieldDefinitions.txt`).
- v1 Table 2 defines `discharge_cms` as "the measured bankfull discharge at the site". The bankfull target is the single maximum-discharge in-channel measurement (v1 §2.2.1), so its discharge comes from the same measurement as its geometry.
- The release includes the identity check `q3_CalcQ = mean_depth_va × stream_wdth_va × mean_vel_va` (HYDROSWOT `FieldDefinitions.txt`). After Zarrabi-style QC, W·d̄·V matches Q within 5%. Measured Q together with *measured* velocity would therefore nearly determine area. (This applies to measured velocity only; see Q4.)

**Still open (minor):** which field (`q_va` or `meas_q_va`) v1 converted to m³/s. It makes no difference to the new study.

**Implication (consistent with research-state §5):**
- Measured discharge is **not allowed** as a predictor in any headline model.
- It is allowed in only two places:
  1. a gauged "upper bound" ablation, analogous to Zarrabi's Tier 1;
  2. an **"oracle-Q" diagnostic** if Option A is chosen. Feeding the measured Q into the Manning layer separates depth error caused by Q (QE versus measured Q) from depth error caused by the learned W and n.

### Q4. Origin of `vema` → NHDPlus HR EROM velocity (Jobson method)

**Evidence**
- NHDPlus HR defines `NHDPlusEROMMA.VEMA` as "Velocity from gage adjustment" (alias `VelocityEGageAdjustedMA`). It is given in ft/s at the downstream end of each flowline.
- All EROM velocities are computed with the Jobson (1996) regressions from drainage area, flowline slope, the flow for which velocity is computed, and mean annual flow. For mean annual velocity, the two flow inputs are the same value (Moore et al., 2019, "Velocity Computation").
- The user guide calls `QEMA` and `VEMA` the "best" EROM estimates.
- v1 itself calls `vema` an "estimated mean velocity" (v1 §3.1, SHAP discussion).
- HYDROSWOT's measured velocity field is named `mean_vel_va`, which doesn't match.

**Still to confirm:** that the v1 join pulled `VEMA` from `NHDPlusEROMMA` and did not rename HYDROSWOT's `mean_vel_va`.

**Implications**
1. **Correction to the 2026-10-05 evaluation report (M1).** That report treated `vema` as if continuity "nearly hands it the area". If `vema` is EROM, it is a **modelled reach attribute available at every ungauged reach**, so it is *not* measurement leakage. The real issue is **redundancy**: VEMA = f(drainage area, slope, QEMA), a deterministic function of other v1 predictors. That collinearity spreads SHAP attributions across correlated features. The M1 concern about `discharge_cms` (Q3) still holds.
2. **A free physics baseline.** A_J ≡ QEMA / VEMA is a Jobson-implied mean-annual-flow cross-sectional area, available nationwide. It can serve as a zero-training physics baseline for mean-flow area and, with an assumed width, for depth. It sits alongside M0 (regional curves) and M3 (empirical-n Manning) in the Option A model ladder. *Phase 2/3 item; not a commitment.*
3. **Units and QC flags for Phase 3.**
   - EROM flows are in cfs and velocities in ft/s. Check that v1 converted both.
   - Smoothed slopes are floored at 1e-5.
   - Flowlines with slope = −9998 use the Jobson "unknown-slope" equation (Moore et al., 2019). Both should become QC flags.

### Q5. Table 1 proof-of-concept R² → not resolvable; out of scope

**What the v1 text records (v1 §2.1):**
- Data compiled by Bieger et al. (2015, Appendix B): 1,887 sites, of which only **82** have complete records.
- ANN with 2 hidden layers of 6 ReLU units, Adam optimizer, MSE loss.
- Width-type "proxies for hydraulic radius" used as inputs.

The text doesn't record the split, the number of test points or the seeds, and the code isn't in the workspace. **Status: closed for the new study** (ADL-001: v1 results aren't reused; Option A drops the proof-of-concept). It remains an open item for the v1 co-author response only.

### Q6. Raw data location and licensing → partly resolved

| Source | Identifier (verified) | Access / files | Licence or use constraint | Status |
|---|---|---|---|---|
| HYDROSWOT (Canova et al., 2016) | doi:10.5066/F7D798H6 → ScienceBase item `57435ae5e4b07e28b660af55` | `SWOT_ADCP_Dataset.txt` (tab-delimited, 87.4 MB); `.xlsx` (73.2 MB); `FieldDefinitions.txt`, `CodeDefinitions.txt`, `EquationDefinitions.txt`, `DataDictionary.xlsx`, `ReadMe.txt`, FGDC XML | FGDC metadata: access constraints "none", use constraints "none" | ✅ |
| NHDPlusV2.1 (McKay et al., 2012) | User guide has **no DOI**. Publisher URL verified (EPA) | EPA NHDPlus distribution | Not yet checked | ⏳ Phase 3 |
| NHDPlus HR (Moore et al., 2019) | doi:10.3133/ofr20191096 (user guide) | USGS National Map | Not yet checked | ⏳ (only needed if D1 = HR) |
| Zarrabi et al. CONUS product | doi:10.5281/zenodo.13883263 (Zarrabi et al., 2024a); doi:10.4211/hs.63ae139ccd2445959470d0e5a2ebf6a5 (Zarrabi et al., 2024b) | Zenodo; HydroShare | Zenodo DataCite rights: CC BY 4.0 (an MIT licence is also listed). HydroShare: CC BY | ✅ |
| Zarrabi et al. code | GitHub `Reizrb/Bankfull-and-mean-flow-channel-geometry-for-CONUS` (repository exists; cited in Zarrabi et al., 2025, Data Availability) | GitHub | Not yet checked. **Check before reusing it for the M1 baseline** | ⏳ |
| **v1 merged HYDROSWOT–NHDPlus table** (PostGIS k-NN) | — | **Location unknown. Not in the workspace** | Derived from public data | ❌ **Blocker for confirming Q1–Q4 directly** |

No data was downloaded. Acquisition belongs to the Phase 3 protocol and the Phase 5 DVC pipeline, so that data lineage is recorded from the first byte.

---

## 3. Other findings from this audit

| ID | Finding | Consequence |
|---|---|---|
| F1 | **The v1 and Zarrabi bankfull filters differ.** v1 applies the IQR to the **width/depth ratio** (v1 §2.2.1). Zarrabi et al. (2025, §2.1) apply Q3 + 1.5·IQR to **width**, then take the maximum-Q observation, falling back to the largest observation when the IQR finds no pattern | The research state's description "adapted from Zarrabi" is accurate. The new study must state which variant it uses and treat the other as a sensitivity case |
| F2 | HYDROSWOT includes `max_depth_va`, which allows the observed shape exponent r = (d̄/d_max)/(1 − d̄/d_max) assumed in Option A §3.1 | Confirms Option A's shape target is computable |
| F3 | HYDROSWOT also includes `chan_mat_cd` (channel material, e.g. sand, gravel, silt), `chan_stability_cd`, `q_meas_type_cd` (wading, bridge, boat, …), `q_meas_qual_cd`, and the built-in QC flags `q1`–`q10` | `chan_mat_cd` is a candidate **diagnostic** for H4 (plausibility of roughness). It is only observed at gages, so it can't be a predictor. `q_meas_type_cd` can be used to check how representative gage sites are, for example at bridges |
| F4 | `Originals/Sahin_ANN-HydraulicGeometry_v2_2026-10-06_tracked.docx` has **19 comments**, while `ANN_draft_Nazim_v2_tracked.docx` has **24**. The five data queries (Table 1 split, NHDPlus version, Table 2 `vema`/`discharge_cms`, Tables 3/4 runs, unique sites and concurrent Q) are **missing** from the 2026-10-06 file | If the 2026-10-06 file is the one sent to co-authors, **they were never asked Q1–Q5**. Confirm whether this was intended |
| F5 | Registry and file-name mismatches: research state §3 and the session brief refer to `literature/elicit_search_queries.md`, but the files on disk are `elicit_search_queries_verified.md` and `elicit_search_queries_2026-10-02_unverified.md`. There is also a **0-byte** `PoC/Scripts/.research_state.md` duplicate at the repo root, git-excluded | Use `_verified.md` as the canonical seed list. The empty duplicate was left in place (non-destructive rule). Delete it only if you confirm |
| F6 | `src/config.py`, `pyproject.toml`, `README.md` and `CITATION.cff` exist but are empty (0 bytes) | They are pre-created placeholders and don't break the code lock. They weren't touched |

---

## 4. Sources checked in this audit

| Reference | Identifier | Checked via (2026-10-07) |
|---|---|---|
| Canova, M. G., Fulton, J. W., & Bjerklie, D. M. (2016). *USGS HYDRoacoustic dataset in support of the Surface Water Oceanographic Topography satellite mission (HYDRoSWOT)*. U.S. Geological Survey data release. | doi:10.5066/F7D798H6 | DataCite API, ScienceBase JSON, FGDC XML, FieldDefinitions.txt, CodeDefinitions.txt |
| Moore, R. B., McKay, L. D., Rea, A. H., Bondelid, T. R., Price, C. V., Dewald, T. G., & Johnston, C. M. (2019). *User's guide for the National Hydrography Dataset Plus (NHDPlus) High Resolution*. USGS Open-File Report 2019-1096. | doi:10.3133/ofr20191096 | Crossref (first author Moore; title matches); full PDF from pubs.usgs.gov |
| Jobson, H. E. (1996). *Prediction of traveltime and longitudinal dispersion in rivers and streams*. USGS Water-Resources Investigations Report 96-4013. | doi:10.3133/wri964013 | Crossref; USGS Publications Warehouse (author, year, series) |
| McKay, L., et al. (2012). *NHDPlus Version 2: User Guide*. | No DOI. https://www.epa.gov/system/files/documents/2023-04/NHDPlusV2_User_Guide.pdf | Publisher URL fetched; EROM field names read from the PDF. ⚠️ Author list to be checked against the title page before formal citation |
| Zarrabi, R., McDermott, R., Erfani, S. M. H., & Cohen, S. (2025). Bankfull and mean-flow channel geometry estimation through machine learning algorithms across the CONtiguous United States (CONUS). *Water Resources Research*, 61, e2024WR037997. | doi:10.1029/2024WR037997 | Crossref (title, first author, journal, 2025); local PDF |
| Zarrabi, R., McDermott, R., Erfani, S. M. H., & Cohen, S. (2024a). *Bankfull and Mean-flow Channel Geometry Dataset across the CONtiguous United States (CONUS)* [Dataset]. Zenodo. | doi:10.5281/zenodo.13883263 | DataCite API (state: findable; rights) |
| Zarrabi, R., McDermott, R., Erfani, M., & Cohen, S. (2024b). Same title [Dataset]. HydroShare. | doi:10.4211/hs.63ae139ccd2445959470d0e5a2ebf6a5 | DataCite API |
| Bieger, K., Rathjens, H., Allen, P. M., & Arnold, J. G. (2015). Development and evaluation of bankfull hydraulic geometry relationships for the physiographic regions of the United States. *JAWRA*, 51(3), 842–858. | doi:10.1111/jawr.12282 | Crossref (title, first author, journal, 2015) |

---

## 5. Decisions needed before `docs/0_point_zero.md` can be finalized

| ID | Decision | Recommendation |
|---|---|---|
| D1 | NHDPlus version for the new study | **NHDPlusV2.1 (COMID)**, for comparability with Zarrabi et al. (2025) and their D50 and aridity predictors |
| D2 | Where is the v1 merged table / PostGIS database? | Needed only to close Q1–Q4 *directly*. The new study doesn't depend on it |
| D3 | Rebuild targets from raw HYDROSWOT, independently of v1? | **Yes.** Acquisition is set out in the Phase 3 protocol and implemented in the Phase 5 DVC pipeline |
| D4 | Research direction (ADL-002): Option A, A+B, B or C | Needed to write the problem statement, RQs and hypotheses in `0_point_zero.md` |
