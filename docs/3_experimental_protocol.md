# Phase 3 Gate Deliverable: Experimental Protocol & Data Provenance

| Field | Value |
|---|---|
| S-SDLC phase | 3 (Experimental Protocol & Data Provenance) |
| Status | **Approved by the PI (2026-10-08)**: draft approved; O-1 to O-13 resolved as recommended, with O-2 (b) and O-5 (c) (§6). **Middle-prompt revision (2026-10-08):** §1.5 DVC specification (stage-based acquisition, `params.yaml`, `dvc.yaml`, `.dvcignore`), §3.6 dependence audit (diagnostic only, per P3-D4), §4.3 formal metric definitions, L3/L8 and §5 updated. **Gate-review fix (2026-10-08):** raw-unit schemas added to §2.3 (gate item 3); §1.5 DAG aligned with the `dvc.yaml` specification; final ADL mapping in §7 |
| Date | 2026-10-08 |
| Code-generation lock | 🔒 ON. No project code, installs or data downloads. Every command block below is a **non-executed specification** for Phase 5 (P3-D1 (a)). Evidence comes from dataset metadata (DataCite, Crossref, ScienceBase, Zenodo, figshare, GitHub APIs) and read-only text extraction of PDFs |
| Inputs | `0_point_zero.md` §3, §4, §6, §7; `0a_open_data_questions.md`; `1_literature_matrix.md` §4, §6; `2_mathematical_proofs.md` §1.2, §1.3, §7, §9.5, §10.3, §11 (frozen); design note `option_a_research_design.md` §5–§8 (leads only) |
| PI decisions | P3-D1 to P3-D9, answered 2026-10-08 (§8) |
| Frozen | H1–H4 wording (`0_point_zero.md` §3) and all of `2_mathematical_proofs.md`. This protocol fixes **numbers and procedures only** |
| Reference policy | Every key is DOI-verified (Crossref or DataCite, 2026-10-08) or ⚠️-flagged with a publisher URL |

**Notation** follows `2_mathematical_proofs.md` §1. Logs are natural logs. $w=\ln W$, $\ell=\ln\bar d$, $t=\ln(d_{\max}/\bar d)$, $\nu=\ln n$, $q=\ln Q$, $s=\ln S$.

---

## 1. Data lineage and provenance manifest

### 1.1 Manifest

Each source is a **DVC tracking unit**: one `acquire_*` stage whose output directory sits under `data/raw/<unit>/`. The file names, sizes and checksums are the publisher's values where published, read on 2026-10-08. Where no checksum is published, the SHA-256 is computed at acquisition (Phase 5) and recorded in `dvc.lock` (DVC MD5) and in `data/raw/<unit>/MANIFEST.sha256`.

| Unit | Source | Identifier | Version / pin | Licence or terms (verified) | Access | Files used (size; checksum) | Role |
|---|---|---|---|---|---|---|---|
| `hydroswot` | Canova, Fulton & Bjerklie (2016) [@Canova_2016_HYDROSWOT] | doi:10.5066/F7D798H6 → ScienceBase `57435ae5e4b07e28b660af55` | Single release (2016) | FGDC metadata: access constraints "none"; use constraints "none" | HTTPS, anonymous | `SWOT_ADCP_Dataset.txt` (87,445,985 B); `FieldDefinitions.txt` (7,156 B); `CodeDefinitions.txt` (3,342 B); `EquationDefinitions.txt` (2,965 B); `USGS_HYDRoSWOT_metadata.xml` (24,239 B); `ReadMe.txt` (2,295 B). No published checksum → SHA-256 at acquisition. The `.xlsx` copy (73 MB) is **not** acquired | Targets, QC, oracle-Q |
| `nhdplusv21` | NHDPlusV2.1 [@McKay_2012_NHDPlusV2] | ⚠️ **No DOI.** EPA NHDPlus distribution: https://www.epa.gov/waterdata/nhdplus-national-data | NHDPlusV2.1 national seamless geodatabase (Lower 48), or per-VPU attribute archives. User guide Data Model v2.1, 13 March 2019. Pin each file name and its embedded version number in `MANIFEST` | ⚠️ **No licence statement** in the user guide or on the EPA page (EPA disclaimers only). Treated as a U.S. Government work, cited with attribution (O-6) | HTTPS, anonymous | Flowline network attributes (VAA, EROM mean-annual `QE_MA`/`VE_MA` ≡ EROM_MA0001 `Q0001E`/`V0001E` ⚠️ equivalence to be asserted in Phase 5; `QC_MA` ≡ `Q0001C` for SA1), `ElevSlope.Slope`, `GageLoc` (`Source_Fea`, `FLComID`), flowline geometry. **No grids.** SHA-256 at acquisition | Q_E, slope, link, predictors |
| `streamcat` | StreamCat [@Hill_2016_StreamCat] | doi:10.1111/1752-1688.12372 (paper). Data: ⚠️ EPA URL https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset | Metric files pinned by name and EPA release date in `MANIFEST` | ⚠️ EPA disclaimer only; no licence stated. EPA asks users to cite Hill et al. (2016) | HTTPS | Selected national metric files (list fixed in Phase 5 from §2.3 predictor list). SHA-256 at acquisition | Predictors (P3-D6) |
| `usgs_attrs` | Wieczorek et al. (2018), USGS NHDPlusV2.1 catchment attributes [@Wieczorek_2018_NHDPlusAttributes] | doi:10.5066/F7765D7V; parent ScienceBase item `5669a79ee4b08895842a1d47`. ⚠️ doi.org currently resolves to child item `66573d81d34ef3137d35edc8` | Parent release ver. 6.0 (August 2026). ⚠️ Zarrabi et al. used ver. 4.0 (August 2023). Pin the child-item files and the version in `MANIFEST` | ⚠️ No rights field on the parent item; the child items checked carry CC0. USGS data release (O-6) | HTTPS | Child items for dams, 2010 population density, seasonal EVI, soil clay/silt/sand and land cover (developed/forest/agriculture), as listed in Zarrabi Table 1. SHA-256 at acquisition | **M1 features only** (O-2 (b)) |
| `d50` | Abeshu et al. (2022) [@Abeshu_2022_D50]; data [@Li_2021_D50Map] | doi:10.5194/essd-14-929-2022; data doi:10.5281/zenodo.4921987 | Zenodo v1.1 | CC BY 4.0 (DataCite) | HTTPS | `NHDPlus_MediumResolution_D50.dbf` (244,912,011 B; md5 `3617b250addbab3cd81bad15cd93c262`). Optional `.shp` (1,906,768,684 B; md5 `d3f64d2dbcdde10ef1e0afab3069ad2b`) only if the COMID join needs geometry; `.prj`, `.cpg`, `.shx` with the published md5 values | Predictor; H4 (b) |
| `aridity` | Zomer, Xu & Trabucco (2022) [@Zomer_2022_GlobalAridityV3]; data [@Zomer_2025_GlobalAIv31] | doi:10.1038/s41597-022-01493-1; data doi:10.6084/m9.figshare.7504448.v7 | Global-AI v3.1 (figshare v7). ⚠️ Zarrabi et al. used v2 (Trabucco & Zomer 2019, figshare v3) | CC BY 4.0 (DataCite) | HTTPS | `Global-AI_ET0__annual_v3_1.zip` (645,783,906 B; md5 `ef3f6f53a082fb07365954b468cca55f`); `Global AI_ET0 v3.1 - Readme.docx` (645,897 B; md5 `ad026eba14bd5f50c6be2abc603ed74f`). Monthly files are **not** acquired | Predictor (zonal mean per catchment). Adopted (O-1) |
| `physiography` | USGS (2023), Physiographic divisions of the conterminous U.S. (digitized Fenneman & Johnson 1946 map) [@USGS_2023_PhysiographicDivisions] | doi:10.5066/P9B1S3K8 → ScienceBase `631405bbd34e36012efa304e` | 2023 release | ⚠️ No rights field in DataCite; USGS product, treated as a U.S. Government work (O-6) | HTTPS | `physio_shp.zip` (451,315 B; md5 `ba126a4526fb75ab5e13100e692613df`); `physio.xml` (18,470 B; md5 `1868489eb8f734eddbef4a4a954401cb`) | S-LRO regions (P3-D5); M0 regions |
| `ehydro` | USACE eHydro surveys, as processed by Zarrabi et al. (2025) [@Zarrabi_2025_ChannelGeometryCONUS] | ⚠️ No DOI. https://www.sam.usace.army.mil/Missions/Spatial-Data-Branch/eHydro/ | **Route O-5 (c).** The processed 76-location table is **not** on Zenodo or in the Zarrabi GitHub repo (checked 2026-10-08). (1) Request it, with an explicit licence, from the authors; (2) re-derive it from USACE eHydro in Phase 5 following Zarrabi §2.5, both to verify (1) and as the fallback if there is no reply | ⚠️ USACE (U.S. Government) data, treated as a U.S. Government work (O-6). The author-supplied table is used only under the licence they grant | HTTPS (USACE); e-mail (authors) | Author table (if granted) **and** re-derived survey rasters (60 surveys, 394 COMIDs, 25 rivers → 76 locations). SHA-256 at acquisition. The two versions are compared location by location before S-IND is frozen | **Sole independent set** S-IND (P3-D7) |
| `zarrabi_product` | Zarrabi et al. (2024) [@Zarrabi_2024_GeometryDataset] | doi:10.5281/zenodo.13883263 | Zenodo record (single file) | CC BY 4.0 (DataCite also lists MIT) | HTTPS | `Bankfull_Meanflow_CONUS.txt` (216,303,769 B; md5 `d7c22261bfdc4515ed8a6a627c034d5e`). Note (2026-10-08): the code README now links a newer version of the same Zenodo concept (record 19208847, concept 11307036), whose `Bankfull_Meanflow_CONUS_Stream_Reaches.txt` has the **identical** md5, plus a USGS-gage file that is **not** acquired. The pin stays at 13883263 | Comparator **on S-IND only** (`0_point_zero.md` §6) |
| `zarrabi_code` | Zarrabi et al. code [@Zarrabi_2024_Code] | ⚠️ No DOI. GitHub `Reizrb/Bankfull-and-mean-flow-channel-geometry-for-CONUS` | Commit `a30f1f53392ae0cb650e322b31357347d6ba9c4a` (HEAD, 2026-10-08) | **MIT** (LICENSE plus README: "Code: MIT. Data: CC-BY-4.0") | HTTPS | Notebooks read **as text** to recover the QC sequence and XGB hyperparameters. The `Best_Param/*` files are binary model objects and are **never deserialized** (pickle-type code-execution risk) | M1 fidelity |
| `hyg` | Enzminger, Minear & Livneh HyG [@Enzminger_2024_HyGDataset]; data [@Enzminger_2023_HyGData] | doi:10.5281/zenodo.7868764 | Zenodo record (2023) | CC BY 4.0 (DataCite) | HTTPS | `HyG_20230426.csv` (6,491,937 B; md5 `1847a0ef304e3403cfa091e5065d119b`) | S1 diagnostic only (P3-D9) |

**Not acquired** (P3-D7): the Zarrabi 11-source bankfull compilation and the NSL/NCCHE surveyed cross-sections. NHDPlus HR is not used (ADL-011).

**Correction recorded:** the design note attributes StreamCat to "Wieczorek et al. (2018)". StreamCat is Hill et al. (2016). Wieczorek, Jackson & Schwarz (2018; doi:10.5066/F7765D7V) is the **USGS NHDPlusV2.1 catchment-attribute release** that Zarrabi et al. used for their dams, population, EVI, soil and land-cover predictors (Zarrabi et al. 2025, Data Availability). ⚠️ The DataCite metadata for that DOI currently shows a child item's title ("…Difference between Precipitation Potential Evapotranspiration", Wieczorek & Staub, CC0), so the record has drifted. The parent ScienceBase item `5669a79ee4b08895842a1d47` still carries the DOI and the citation "Wieczorek, M.E., Jackson, S.E., Schwarz, G.E., Sekellick, A.J., and Staub, L.E., 2018, … (ver. 6.0, August 2026)". Adopted for M1 features only (O-2 (b)).

### 1.2 Versioning, checksums and immutability

1. **Raw data are immutable.** Each `acquire_*` stage downloads, verifies the publisher checksum where one exists, and writes `MANIFEST.sha256` together with the URL, the retrieval timestamp (UTC) and the HTTP `ETag`/`Last-Modified` headers. On a checksum mismatch the stage fails; it never overwrites.
2. **Processed data** are Parquet (tables) and GeoPackage (geometry, EPSG:5070 for distances, EPSG:4269 for storage), git-ignored and DVC-tracked. Every processed file records the `dvc.lock` hash of its inputs in its metadata.
3. **Frozen artefacts:** the post-QC site table and the fold-assignment table (§3.3). Their SHA-256 values are written into this protocol by a Phase 5 amendment (ADL row). Any later change needs a new ADL row and invalidates every downstream result.
4. **Archival (Phase 11):** Zenodo deposit of the processed site table, the fold table, the model outputs and the code release. Raw third-party data are cited, not redeposited, unless their licence allows it (CC BY 4.0 / CC0 / U.S. Government).

### 1.3 FAIR mapping

| Principle | Implementation |
|---|---|
| **F**indable | Every input carries a DOI or ⚠️ URL (§1.1). The Phase 11 output gets a Zenodo DOI. `CITATION.cff` is completed in Phase 11. The fold table and site table are identified by SHA-256 |
| **A**ccessible | All inputs are open HTTPS downloads without authentication. eHydro: an author request plus re-derivation from USACE (O-5 (c)). DVC remote: local (P3-D2); Zenodo for the final archive |
| **I**nteroperable | Parquet + GeoPackage; explicit EPSG codes; SI units after $\mathcal U$ (§2.1); `site_no` stored as a zero-padded string; COMID as int64; schema tables in §2.3 (P3-D3 (a)), implemented in Phase 4/5 with Pydantic plus Pandera or Great Expectations |
| **R**eusable | Licences recorded per unit; `dvc.lock` gives byte-level lineage; QC flags kept as columns rather than silently dropping rows; this protocol is the human-readable provenance record |

### 1.4 Data scale and storage (estimates; to be confirmed by the Phase 5 timing/size pilot)

| Unit | Raw size (est.) | Basis |
|---|---|---|
| hydroswot | ≈ 0.09 GB | Published sizes |
| nhdplusv21 | ≈ 5–10 GB compressed | ⚠️ Estimate. Attributes, GageLoc and flowlines for the Lower 48; no grids |
| streamcat | ≈ 1–3 GB | ⚠️ Estimate. ≈ 10–20 national metric files |
| d50 | 0.25 GB (dbf only) or 2.2 GB (with shp) | Published sizes |
| aridity | 0.65 GB | Published size (annual only) |
| physiography, hyg, zarrabi_product | ≈ 0.22 GB | Published sizes |
| usgs_attrs | ≈ 0.2–1 GB | ⚠️ Estimate; ≈ 8–12 child-item CSVs |
| ehydro | ≈ 0.001 GB (table) plus several GB (rasters) | O-5 (c) acquires both |
| **Raw total** | **≈ 8–16 GB** | |
| Processed (site table, features, folds) | < 0.5 GB | ≈ 2.6–4.6 k sites × ≈ 60 columns |
| Model outputs (OOF predictions, ensembles) | ≈ 2–5 GB | §4.8 |
| **DVC cache + local remote** | **≈ 25–45 GB** under `~/dvc_store/ANN_XS_Estimation/{cache,remote}` | Cache and remote each hold one copy |

The local machine is the Apple M4 Max (36 GB RAM; ADL-016). Peak RAM is dominated by the NHDPlus flowline attribute join (≈ 2.7 M rows) and the aridity zonal statistics. Both fit in memory if done per VPU.

### 1.5 DVC specification (P3-D1 (a), P3-D2 (a); non-executed)

**Layout.**

```text
PoC/Scripts/                      # git repo (on Google Drive)
├── data/raw/<unit>/              # DVC-managed links into the external cache; NOT synced by Drive
├── data/processed/               # DVC-managed links; NOT synced by Drive
├── data/checkpoints/             # DVC-managed (Phase 6+)
├── dvc.yaml, dvc.lock            # committed only after the phase is ≥ 4 (G2 hook blocks dvc.yaml while ≤ 3)
└── .dvc/config (shared), .dvc/config.local (machine paths; git-ignored by DVC)
~/dvc_store/ANN_XS_Estimation/cache    # DVC cache (outside Drive)
~/dvc_store/ANN_XS_Estimation/remote   # DVC local remote (outside Drive)
```

**Reconciliation with `.gitignore` and ADL-009.**

1. **Facts.** `.gitignore` excludes `data/raw/`, `data/processed/` and `data/checkpoints/` as **directory** patterns, plus `*.parquet`, `*.gpkg` and `*.gdb/` anywhere. Git cannot re-include a file whose parent directory is excluded, so a `!data/raw/*.dvc` exception would have no effect.
2. **Consequence.** A per-source `dvc add data/raw/<source>` writes `data/raw/<source>.dvc` *inside* an ignored directory, so the pointer would never be committed and lineage would break silently.
3. **Adopted rule (P3-D1 (a), middle-prompt item 2 adjusted).**
   - Every source is acquired by a `dvc.yaml` stage (`acquire@<unit>`).
   - Its outputs are recorded in **`dvc.lock` at the repo root**, which is tracked.
   - `dvc add` and `dvc import-url` are **not used** for project data.
   - `dvc.lock` takes over the role ADL-009 gave to `.dvc` pointer files (ADL-043).
   - `.gitignore` needs **no change**. The `.gitignore` files DVC writes inside output parents are themselves ignored, which is harmless.
4. **Author-supplied files without a URL** (the eHydro table, O-5 (c)) are dropped by the PI into `~/dvc_store/ANN_XS_Estimation/inbox/` (outside the repo and Drive). Stage `acquire@ehydro_author` copies the file into `data/raw/ehydro_author/` only if its SHA-256 matches the value pinned in `params.yaml`. Lineage is therefore the pinned hash plus the `dvc.lock` output hash.
5. **Phase 7 checks.**
   - `git check-ignore -v dvc.yaml dvc.lock params.yaml .dvcignore .dvc/config` reports nothing.
   - `git check-ignore data/raw/hydroswot/SWOT_ADCP_Dataset.txt` reports the file as ignored.
   - `dvc status -c` is clean after `dvc push`.
6. **Google Drive and links (P3-D2 follow-up, O-13).** Before any real data, a symlinked 1 MB dummy file is created and Drive's behaviour observed. If Drive follows links, the working tree moves out of Drive. `cache.type` is confirmed by this test.

**Command sequence (Phase 5 only; NOT executed in Phase 3).** Machine-specific paths go to `.dvc/config.local` (`--local`), so the shared `.dvc/config` stays portable. This has the same effect as the prompt's `dvc config cache.dir` / `dvc remote add -d`, without committing a home-directory path.

```bash
# --- Phase 5 only; specification, not executed ---
dvc init                                                     # at the git root PoC/Scripts
dvc config --local cache.dir ~/dvc_store/ANN_XS_Estimation/cache
dvc config --local cache.type symlink                        # confirmed by the O-13 Drive test
dvc remote add --local -d localstore ~/dvc_store/ANN_XS_Estimation/remote
git add .dvc/config .dvc/.gitignore .dvcignore dvc.yaml params.yaml   # after the phase is >= 4 (G2 hook)
dvc dag                                                      # visual check against the Mermaid DAG below
dvc repro                                                    # runs acquire -> ... -> split
git add dvc.lock reports/
dvc push && dvc status -c
```

**`params.yaml` (keys and values fixed by this protocol; non-executed specification).**

```yaml
seeds:   {global: 20261008, sid_repeats: [20261008, 20261009, 20261010]}
acquire: {<unit>: {urls: [...], md5: [...] or null, sha256_pinned: null}}   # filled from section 1.1 in Phase 5
qc:      {q_tol: 0.05, epsA_tol: 0.05, W_max_m: 5000, dbar_max_m: 50, dmax_max_m: 70, V_max_ms: 6}
link:    {snap_max_m: 500, da_ratio_max: 1.25}
condition: {qsel_ratio_flag: 2.0, tie_break: latest_date}
domain:  {n_bounds: [0.005, 0.6], n_lit: [0.012, 0.33], r_bounds: [0.5, 10], s_floor: 1.0e-5}
splits:  {sid_k: 5, sid_r: 3, inner_k: 3, es_frac: 0.10, lro_min_groups: 50,
          tail_deciles: [1, 10], tail_order_min: 7, ind_dedupe_m: 1000}
audit:   {knn_k: 8, idw_band_km: 100, permutations: 999, lisa_alpha: 0.01,
          dist_classes_km: [10, 25, 50, 100, 250, 500], vgm_bins: 20, vgm_max_km: 500,
          rho_cap_km: 250, moran_material: 0.10, nested_psi: 0.5, nested_share_material: 0.20}
```

**`dvc.yaml` stages (non-executed specification; syntax checked with `dvc dag` in Phase 5).** The module paths under `src/` are placeholders that Phase 4 designs.

```yaml
stages:
  acquire:
    foreach: [hydroswot, nhdplusv21, streamcat, usgs_attrs, d50, aridity,
              physiography, ehydro, ehydro_author, zarrabi_product, hyg]
    do:
      cmd: python -m src.data.acquire --unit ${item}
      deps: [src/data/acquire.py]
      params: [acquire]
      outs: [data/raw/${item}]
  validate_raw:
    cmd: python -m src.data.validate_raw
    deps: [src/data/validate_raw.py, src/data/schemas.py, data/raw/hydroswot, data/raw/nhdplusv21,
           data/raw/streamcat, data/raw/usgs_attrs, data/raw/d50, data/raw/aridity, data/raw/physiography]
    metrics: [{reports/validate_raw.json: {cache: false}}]
  unit_map:
    cmd: python -m src.data.unit_map
    deps: [src/data/unit_map.py, reports/validate_raw.json, data/raw/hydroswot, data/raw/nhdplusv21]
    params: [domain.s_floor]
    outs: [data/processed/unit_mapped]
  qc:
    cmd: python -m src.data.qc
    deps: [src/data/qc.py, data/processed/unit_mapped, data/raw/physiography]
    params: [qc]
    outs: [data/processed/qc/hydroswot_init.parquet]
    metrics: [{reports/qc_counts.json: {cache: false}}]
  link:
    cmd: python -m src.data.link
    deps: [src/data/link.py, data/processed/qc/hydroswot_init.parquet, data/processed/unit_mapped]
    params: [link]
    outs: [data/processed/link/site_comid.parquet]
    metrics: [{reports/link_counts.json: {cache: false}}]
  condition_extract:
    cmd: python -m src.data.condition_extract
    deps: [src/data/condition_extract.py, data/processed/qc/hydroswot_init.parquet, data/processed/link/site_comid.parquet]
    params: [condition]
    outs: [data/processed/site_core.parquet]
  features:
    cmd: python -m src.data.features
    deps: [src/data/features.py, data/processed/site_core.parquet, data/processed/unit_mapped, data/raw/streamcat,
           data/raw/usgs_attrs, data/raw/d50, data/raw/aridity, data/raw/physiography]
    outs: [data/processed/site_table.parquet]
    metrics: [{reports/feature_missing.json: {cache: false}}]
  split:
    cmd: python -m src.data.split
    deps: [src/data/split.py, data/processed/site_table.parquet]
    params: [seeds, splits]
    outs: [data/processed/splits/fold_assignment.parquet]
    metrics: [{reports/split_summary.json: {cache: false}}]
  ind_prepare:
    cmd: python -m src.data.ind_prepare
    deps: [src/data/ind_prepare.py, data/raw/ehydro, data/raw/ehydro_author, data/raw/zarrabi_product,
           data/processed/site_table.parquet]
    params: [splits.ind_dedupe_m]
    outs: [data/processed/ind/ehydro_ind.parquet]
```

`reports/*.json` are small metrics files committed to Git (`cache: false`), so QC and link counts are reviewable in diffs. The groups step (§3.1) runs inside `split`.

**`.dvcignore` (proposed).**

```text
# OS and editor files
.DS_Store
**/Icon?
**/.~lock.*
**/*.swp
**/*~
# temporary and sync artefacts
**/*.tmp
**/.gdrive*
# notebooks and scratch
**/.ipynb_checkpoints/
scratch/
**/scratch/
```

**Stage DAG** (mirrors the `dvc.yaml` dependencies above; `dvc.yaml` is authoritative).

```mermaid
flowchart TD
  subgraph acquire["acquire@unit (one stage per unit; checksum-verified)"]
    A1["hydroswot"]
    A2["nhdplusv21"]
    A3["streamcat"]
    A10["usgs_attrs"]
    A4["d50"]
    A5["aridity"]
    A6["physiography"]
    A7["ehydro"]
    A11["ehydro_author"]
    A8["zarrabi_product"]
    A9["hyg"]
  end
  A1 --> V["validate_raw: checksums + raw schemas (section 2.3)"]
  A2 --> V
  A3 --> V
  A10 --> V
  A4 --> V
  A5 --> V
  A6 --> V
  V --> U["unit_map: sentinels to missing, then U (SI)"]
  U --> Q["qc: row-wise rules R1-R14, flags kept"]
  A6 --> Q
  Q --> L["link: site_no to COMID via GageLoc; snap fallback; DA check"]
  L --> C["condition_extract: mean-flow row per site"]
  C --> F["features: join VAA, EROM, StreamCat, USGS attrs, D50, AI zonal mean"]
  A3 --> F
  A10 --> F
  A4 --> F
  A5 --> F
  A6 --> F
  F --> S["split: site-COMID groups, then frozen fold-assignment table (SHA-256)"]
  F --> I["ind_prepare: eHydro dedupe vs training COMIDs; author vs re-derived check"]
  A7 --> I
  A11 --> I
  A8 --> I
  S -.-> T["Phase 6+: train / evaluate (not specified here)"]
  I -.-> T
  A9 -.-> D["Phase 6+: S1 diagnostic only (HyG schema checked at use)"]
```

---

## 2. Data quality and schema rules

### 2.1 Unit map $\mathcal U$ (exact factors; extends `2_mathematical_proofs.md` §1.2)

$\lambda_L=0.3048$ m ft⁻¹ exactly. Sentinels become *missing* **before** $\mathcal U$ and are never scaled.

| Field (source) | Source unit | Factor (exact) | SI / model unit | Notes |
|---|---|---|---|---|
| `stream_wdth_va` → $W$ | ft | 0.3048 | m | |
| `mean_depth_va` → $\bar d$; `max_depth_va` → $d_{\max}$ | ft | 0.3048 | m | |
| `xsecond_area_va` → $A$ | ft² | 0.09290304 | m² | |
| `mean_vel_va` → $V$; `max_vel_va` | ft s⁻¹ | 0.3048 | m s⁻¹ | `max_vel_va` is used for QC only |
| `q_va` → $Q_{obs}$; `meas_q_va` | cfs | 0.028316846592 | m³ s⁻¹ | QC, condition extraction and oracle-Q only. **Never a predictor** (ADL-006) |
| `drain_area_va` | mi² | 2.589988110336 | km² | Link QC only; never a predictor |
| `QE_MA` (≡ `Q0001E`) → $Q_E$; `QC_MA` (≡ `Q0001C`, SA1) | cfs | 0.028316846592 | m³ s⁻¹ | EROM mean annual, 1971–2000 (user guide) |
| `VE_MA` (≡ `V0001E`) | ft s⁻¹ | 0.3048 | m s⁻¹ | Used in $A_J$ only. Not a predictor in M2/M5 (redundant; ADL-005) |
| `Slope` (ElevSlope; smoothed) → $S$ | m m⁻¹ | 1 | – | −9998 → missing |
| `TotDASqKm` | km² | 1 | km² | Feature (log) and size strata |
| Smoothed minimum elevation | cm | 0.01 | m | Feature |
| `D50` (Abeshu) | mm ⚠️ (Zarrabi Table 1 reports mm; confirm from the dbf) | 10⁻³ | m | |
| Global-AI v3.1 raster value | ⚠️ integer-scaled (confirm the scale factor from the v3.1 Readme) | per Readme | – | Zonal mean over the NHDPlusV2.1 catchment |
| `dec_lat_va`, `dec_long_va` + `coord_datum_cd` | degrees | datum transform to NAD83 | EPSG:4269 → EPSG:5070 | NAD27 rows are transformed, never mixed |

### 2.2 Bounds

| Quantity | Bound | Type | Source / rationale |
|---|---|---|---|
| $n$ transform bounds $[n_{\min},n_{\max}]$ | **[0.005, 0.6]**, so $\Delta_n=\ln120=4.7875$ | Model domain (P3-01) | Encloses the back-calculated range 0.006–0.537 of [@Heldmyer_2022_NWMChannelParameterization]. The effective $n$ absorbs slope and non-uniform-flow error (`0_point_zero.md` §4), so the transform band is deliberately wider than the physical range |
| $\mathcal N_{lit}$ (H4) | **[0.012, 0.33]** | Literature range (P3-16) | [@Arcement_1989_ManningSelection], Cowan procedure $n=(n_b+n_1+n_2+n_3+n_4)m$. Lower end: smallest natural-bed base value, sand 0.2 mm, 0.012 (Table 1). Upper end: largest base value (boulder, 0.070) plus the largest adjustments (0.020 + 0.015 + 0.050 + 0.100; Table 2) times the severe-meander factor 1.30 = 0.3315 → 0.33. $\mathcal N_{lit}\subsetneq[n_{\min},n_{\max}]$ ✓ |
| $r$ transform bounds | **[0.5, 10]** (provisional retained) | Model domain (P3-02) | $t\in[\ln1.1,\ln3]=[0.0953,1.0986]$. $r_{\min}=\tfrac12$ is the boundary case of Proposition 2.6; kept at 0.5 (O-9) |
| $S_{floor}$ | $10^{-5}$ m m⁻¹ | Data property (P3-03) | User guide: slopes "constrained to be greater than or equal to 0.00001 m/m". **(A1.1) confirmed** |
| Gross-error bounds (a priori; not tuned) | $0<W\le5000$ m; $0<\bar d\le50$ m; $0<d_{\max}\le70$ m; $0<V\le6$ m s⁻¹ | QC R5 | Rows outside are flagged $F_{bnd}$ and excluded; counts reported |
| $\epsilon_A$ | $\lvert\epsilon_A\rvert\le\ln1.05$ | QC R8 | Equivalent to HYDROSWOT flag q4 (±5 %) |

### 2.3 Schemas (Markdown; P3-D3 (a))

**`hydroswot_raw` (as delivered; tab-delimited).** Exact field names come from `FieldDefinitions.txt` (P3-05 closed):

| Field | Type | Unit | Use |
|---|---|---|---|
| `site_no` | string (keep leading zeros) | – | Key |
| `dec_lat_va`, `dec_long_va`, `coord_datum_cd` | float, float, string | deg | Location |
| `drain_area_va` | float | mi² | Link QC |
| `site_tp_cd` | string | – | Keep `ST` |
| `q_meas_no`, `q_meas_dt`, `q_meas_chan_nu` | string, datetime, int | – | Measurement identity; multi-channel exclusion |
| `q_va`, `meas_q_va`, `q_adj_va` | float | cfs | Discharge (QC, oracle) |
| `stream_wdth_va`, `xsecond_area_va`, `mean_depth_va`, `max_depth_va` | float | ft, ft², ft, ft | Targets |
| `mean_vel_va`, `max_vel_va` | float | ft s⁻¹ | Diagnostic target; QC |
| `q_meas_qual_cd`, `q_meas_used_fg`, `q_meas_type_cd`, `data_aging_cd` | string | – | Quality flags |
| `chan_mat_cd`, `chan_stability_cd`, `base_flw_cd` | string | – | **Diagnostic only** (`chan_mat_cd`: H4 (d)); never predictors |
| Built-in flags `q1`–`q10` (as named in the release) | int / string | – | QC R6–R9 |

**`site_table` (processed; one row per site after condition extraction).**

| Column | Type | Unit | Constraint |
|---|---|---|---|
| `group_id` | string (16 hex) | – | SHA-256 prefix of the sorted member `site_no` list of the component (§3.1) |
| `site_no` | string | – | unique |
| `comid` | int64 | – | present in NHDPlusV2.1 network |
| `link_method` | enum {`gageloc`, `snap`} | – | |
| `division` | string | – | F&J division (after the merge rule, §3.2) |
| `x5070`, `y5070` | float | m | |
| `q_meas_dt` | date | – | |
| `W`, `dbar`, `dmax`, `A`, `V` | float64 | m, m, m, m², m s⁻¹ | > 0 |
| `t_obs`, `eps_A` | float64 | – | `t_obs` ≥ 0 |
| `Q_obs`, `Q_meas`, `Q_E`, `Q_C`, `V_E` | float64 | m³ s⁻¹, m s⁻¹ | `Q_E` > 0 |
| `S` | float64 | – | ≥ 1e-5 |
| `DA_km2`, `stream_order`, `z_min_m` | float64, int8, float64 | | |
| StreamCat metrics (list fixed in Phase 5) | float64 | as published | no sentinel values |
| `D50_m`, `AI` | float64 | m, – | > 0 |
| `A_J` | float64 | m² | `Q_E / V_E` |
| Flags `F_S`, `F_Qsel`, `F_snap`, `F_DA`, `F_t`, `F_qual`, `F_eps`, `F_tout` | bool | – | §2.4 |

**`fold_assignment`** is specified in §3.3. **`ehydro_ind`**: `loc_id`, member COMIDs, river, `W`, `dbar`, `A` (= `W`·`dbar` by construction of the Zarrabi procedure), `dedupe_status`.

**Raw auxiliary units (validated by `validate_raw`, or at first use where marked).** Rules common to every unit:

- **V1** file present, size and checksum match §1.1 or `MANIFEST.sha256`;
- **V2** every required field present, with a lossless cast to the stated type;
- **V3** the join key is unique (or the stated multiplicity is logged);
- **V4** sentinels become missing **before** any check on values;
- **V5** counts of missing and out-of-rule values are written to `reports/validate_raw.json`; a unit fails only on V1, V2 or V3.

Field names marked ⚠️ were not visible in metadata read during Phase 3; they are **asserted at acquisition** and any deviation is recorded by an ADL amendment, not silently remapped.

| Unit / entity | Join key | Required fields (type; unit as delivered) | Sentinels and value rules | Source of field names |
|---|---|---|---|---|
| `nhdplusv21` / PlusFlowlineVAA | `ComID` (int64), unique | `HydroSeq`, `UpHydroSeq`, `DnHydroSeq`, `LevelPathI`, `TerminalPat` (⚠️ DBF 10-character truncation may give `TerminalPa`), `StartFlag`, `TerminalFl`, `Divergence` (int); `PathLength`, `ArbolateSu` (float; km); `TotDASqKm` (float; km²); `StreamOrde` (int8) | `Divergence` ∈ {0, 1, 2}; `StartFlag`, `TerminalFl` ∈ {0, 1}; `TotDASqKm` > 0 for linked COMIDs | NHDPlusV2 User Guide, PlusFlowlineVAA table |
| `nhdplusv21` / EROM_MA0001 | `ComID` (int64), unique | `Q0001E`, `Q0001C` (float; cfs); `V0001E` (float; ft s⁻¹) | `Q0001E` ≤ 0 or missing → R13; the equivalence `Q0001E` ≡ `QE_MA` is asserted (§1.1) | User Guide, EROM tables |
| `nhdplusv21` / ElevSlope | `ComID` (int64), unique | `Slope` (float; m m⁻¹); `MinElevSmo` (float; **cm**, so $z_{\min}$ in m = value/100) | `Slope` = −9998 → missing (R12); otherwise `Slope` ≥ 10⁻⁵ or the row fails V5 | User Guide, ElevSlope table |
| `nhdplusv21` / GageLoc | `Source_Fea` (string, zero-padded `site_no`); multiplicity logged | `FLComID` (int64) | `FLComID` must exist in the VAA (else link fallback, §2.5) | User Guide, GageLoc table |
| `nhdplusv21` / PlusFlow | (`FromComID`, `ToComID`) (int64) | both fields | IDs absent from the VAA (e.g. 0 at network ends, ⚠️) → "no link" in the nested-basin graph | User Guide, PlusFlow table |
| `streamcat` | `COMID` (int64), unique | metric columns from the Phase 5 list (float) ⚠️ | no sentinel values allowed after V4; missing → R14 | ⚠️ per metric file |
| `usgs_attrs` | `COMID` (int64), unique | dams, 2010 population density, seasonal EVI × 4, clay/silt/sand, developed/forest/agriculture (float) ⚠️ | negative sentinel codes (⚠️ e.g. −9999) → missing; missing → R14 (M1 only) | ⚠️ child-item metadata (Zarrabi Table 1 for the list) |
| `d50` | COMID field ⚠️ (int64), unique | D50 value field ⚠️ (float; unit/scale asserted at acquisition, P3-05) | D50 > 0 after scaling; missing → R14 | ⚠️ DBF header (Zenodo describes the content, not the fields) |
| `aridity` | raster; zonal mean per NHDPlus catchment | single annual AI band (integer, scaled ⚠️; scale factor from the v3.1 readme, P3-05) | nodata excluded from the zonal mean; AI > 0 after scaling; CRS read from the file | ⚠️ v3.1 readme |
| `physiography` | polygon ID | division name field ⚠️ (string); valid polygon geometry | geometry valid; every post-QC site falls in exactly one polygon or is dropped (R2) | ⚠️ `physio.xml` (FGDC) |
| `hyg` (at first use, S1) | gage ID (column 1; string, zero-padded) | latitude/longitude (columns 2–3), drainage area (4), slope (5), AHG parameters (6–11), Manning's $n$ (12); 86 columns in total | column count = 86; $n$ > 0 where present | Zenodo record 7868764 description (column positions) |
| `zarrabi_product` | `COMID` (int64) ⚠️, unique | bankfull and mean-flow width and depth (float) ⚠️ names and units | values > 0; md5 match (§1.1) | ⚠️ file header |
| `ehydro` (rasters, at `ind_prepare`) | survey ID | depth or elevation raster; CRS; vertical datum; nodata | CRS and datum recorded per survey; nodata excluded; the derived per-location `W`, `dbar` follow Zarrabi §2.5 | USACE eHydro survey metadata ⚠️ |
| `ehydro_author` (at `ind_prepare`) | `loc_id` | the `ehydro_ind` fields above | SHA-256 equals the value pinned in `params.yaml`; compared location by location with the re-derived set before S-IND is frozen | author table ⚠️ |

### 2.4 QC rules (order fixed; every rule logs counts of rows and sites removed or flagged)

Every rule is **row-wise and deterministic**. No rule uses thresholds estimated from the pooled data, so QC cannot leak across folds (L5).

| ID | Rule | Action | Rationale / source |
|---|---|---|---|
| R1 | `site_tp_cd = ST` | drop others | Zarrabi QC |
| R2 | Site inside the CONUS F&J polygons | drop others | S-LRO needs a division |
| R3 | `q_meas_chan_nu` indicates a multi-channel measurement | drop | Single-section geometry only |
| R4 | Null, zero or negative `drain_area_va`, `q_va`, `meas_q_va`, `stream_wdth_va`, `xsecond_area_va`, `mean_depth_va`, `max_depth_va`, `mean_vel_va`, `max_vel_va` | drop | Zarrabi QC; invariant 5 |
| R5 | Gross-error bounds (§2.2) | drop, flag $F_{bnd}$ | a priori |
| R6 | `mean_depth_va > max_depth_va` (flag q5 = fail) or `mean_vel_va > max_vel_va` | drop | Zarrabi QC; invariant 4 |
| R7 | Discharge consistency: any of the three pairs {`meas_q_va`, `q_va`}, {$A\cdot V$, `meas_q_va`}, {$A\cdot V$, `q_va`} outside ±5 % (HYDROSWOT q1/q2 flags) | drop | Exactly Zarrabi's three pairs (2025, §2.1) |
| R8 | **$\epsilon_A$ closure:** q4 outside ±5 % ($\lvert\epsilon_A\rvert>\ln1.05$) | drop from primary, keep in SA3 | Our addition. Proposition 7.1 (d): M5 cannot fit $\epsilon_A\ne0$ |
| R9 | `max_depth_va = mean_depth_va` ($t_{obs}=0$) | flag $F_t$, drop from primary, keep in SA3 | $r=\infty$; implausible for an ADCP section |
| R10 | `q_meas_qual_cd = POOR`, `q_meas_used_fg = N`, or provisional `data_aging_cd` | flag $F_{qual}$; retained | Reported; SA option |
| R11 | Duplicate (`site_no`, `q_meas_no`) rows | keep first, log | |
| R12 | Slope −9998 | site excluded (∉ 𝒟) | §1.3; never imputed |
| R13 | $Q_E\le0$ or missing (P3-04) | site excluded (∉ 𝒟) | §1.3 |
| R14 | Missing predictor after the join | site excluded; **no imputation**; counts per predictor reported | Avoids imputation leakage |
| – | $S=S_{floor}$ | flag $F_S$; retained; $F_S$ is an admissible feature (P3-03) | A1.1 confirmed. HyG notes most implausibly low $n$ occur at the slope floor |
| – | $t_{obs}\notin[0.0953,1.0986]$ | flag $F_{tout}$; retained (irreducible residual); share reported (P3-05) | §7.1 |

Rows that survive R1–R11 form `hydroswot_init` (the analogue of Zarrabi's HYDRoSWOT_init; ≈ 38 k rows at ≈ 4.6 k sites expected, plus our R2/R3/R5/R8/R9 removals).

### 2.5 Gage–reach link

1. **Primary:** `GageLoc.FLComID` where `GageLoc.Source_Fea = site_no` (string match).
2. **Fallback:** if there is no GageLoc record, the nearest network flowline within 500 m (EPSG:5070) whose drainage area passes check 3. Flag $F_{snap}$.
3. **Drainage-area check:** $\lvert\ln(\texttt{TotDASqKm}/(2.589988110336\cdot\texttt{drain\_area\_va}))\rvert\le\ln1.25$. Failures are flagged $F_{DA}$ and **excluded from the primary set** (likely mis-link), included in SA6.
4. **Diagnostic (ADL-011):** the k-NN-snap COMID is computed for every site, and the GageLoc vs k-NN disagreement rate is reported.

### 2.6 Condition extraction (mean flow; core)

- Following Zarrabi et al. (2025), the mean-flow row of each site is $\arg\min_i\lvert\ln(Q_{obs,i}/Q_E)\rvert$ over its `hydroswot_init` rows, with $Q_{obs}:=$ `q_va` (official discharge). **Q3 closed:** `q_va` is primary, `meas_q_va` is SA2.
- Ties go to the most recent `q_meas_dt`.
- Flag $F_{Qsel}$ if $\lvert\ln(Q_{obs}/Q_E)\rvert>\ln2$; SA4 drops these sites.
- The dataset therefore has **one row per site** ($\lvert I_j\rvert=1$, so $\omega_i=1/\lvert\mathcal J_{tr}\rvert$).
- Zarrabi aggregate to one value per COMID; we keep sites as rows and handle shared COMIDs through the group key (§3.1).
- Bankfull extraction (secondary stage only): IQR on width plus max-Q (Zarrabi), with the v1 width/depth-ratio variant as a sensitivity case (finding F1). Not executed before the core is complete.

### 2.7 Missing-value strategy (summary)

| Situation | Strategy |
|---|---|
| Sentinel (−9998, −9999, empty) | → missing before $\mathcal U$ |
| Target missing or non-positive | row dropped (R4) |
| Physics input missing ($Q_E$, $S$) | site ∉ 𝒟, excluded (R12, R13) |
| Predictor missing | site excluded (R14); counts reported per predictor and per division |
| Imputation | **none**, in any model |

---

## 3. Formal partitioning protocol

### 3.1 Grouping key (P3-D4 (i))

Build the bipartite graph $G=(\{\text{site\_no}\}\cup\{\text{COMID}\},E)$ with an edge for every final link (§2.5). **A group is a connected component of $G$.** Sites sharing a COMID, including renamed or co-located gages, are therefore always in the same fold. `group_id` = first 16 hex characters of SHA-256 over the sorted, comma-joined member `site_no` strings.

**No S-BLK family** (P3-D4). Residual spatial dependence between neighbouring reaches is an **accepted risk** (L3), mitigated by S-LRO, S-TAIL, the near-neighbour stratification and the autocorrelation audit (§3.6).

### 3.2 Split families

| Family | Role | Construction | Seeds |
|---|---|---|---|
| **S-ID** | In-distribution reference; H2 primary; $\mathrm M^*$ selection; UQ coverage target | `StratifiedGroupKFold`, $K=5$, $R=3$ repeats, groups = `group_id`, strata = drainage-area quintile of the group (max `TotDASqKm`) | 20261008, 20261009, 20261010 |
| **S-LRO** | H1 primary; H3 (LRO) | Leave one F&J division out [@USGS_2023_PhysiographicDivisions], the same divisions as Bieger et al. (2015) (P3-D5 (a)). A group is assigned to the division of its first-listed site. **Merge rule:** a division with < 50 groups after QC is merged into the neighbouring division sharing the longest boundary (expected: Laurentian Upland → Interior Plains). The merges are recorded in the fold table | deterministic |
| **S-TAIL-DA** | H3 (TAIL) primary | Train on drainage-area deciles 2–9; test on deciles 1 and 10. Decile cutpoints from `TotDASqKm` over the post-QC site set (covariate only, no target information) | deterministic |
| **S-TAIL-SO** | H3 (TAIL) secondary | Train on stream order ≤ 6; test on ≥ 7 | deterministic |
| **S-IND** | H1 corroboration | Models trained on **all** post-QC sites, evaluated on the eHydro set (76 locations) after deduplication (L8) | 20261008 |
| **S-RND** | Secondary only; a leakage illustration | Row-level random 75:25 on `hydroswot_init` (all measurements), mirroring Zarrabi's design. Never used for any hypothesis | 20261008 |

**Common support rule.** Every model is evaluated on the identical site set in every split. A site that any model cannot predict (for example, M0 without a division curve) gets the documented fallback (§4.1); it is not dropped for one model only.

### 3.3 Frozen fold-assignment table

`data/processed/splits/fold_assignment.parquet`, long format:

| Column | Type | Meaning |
|---|---|---|
| `group_id`, `site_no` | string | keys |
| `family` | enum {S-ID, S-LRO, S-TAIL-DA, S-TAIL-SO, S-IND, S-RND} | |
| `repeat` | int8 | 0–2 for S-ID, else 0 |
| `outer_fold` | int8 | held-out fold index (division index for S-LRO) |
| `inner_fold` | int8 | 0–2: grouped inner CV inside the outer training set |
| `es_member_{m}` | bool, m = 1–10 | membership of the member-specific grouped 10 % early-stopping split |
| `seed` | int64 | |
| `site_table_sha256` | string | hash of the input site table |

The table is generated once by the `split` stage. Its SHA-256 is recorded in `dvc.lock` and copied into this protocol through an ADL amendment in Phase 5. **No model code may create folds;** all code reads this table.

### 3.4 Split-then-scale and nested selection

1. Every transformation that learns parameters (standardization, log offsets, XGB/NN hyperparameters, early-stopping epochs, $\bar r$ for M3, M0 curve coefficients, M-null medians, size-class cutpoints inside M-null) is fitted on the **outer training set only**.
2. **Nested selection:** hyperparameters are chosen by grouped inner 3-fold CV inside each outer training set, using the inner-fold mean log-RMSE of $\bar d$ (NN models) or of the model's own targets (XGB). This is repeated for every outer fold of every family.
3. Final ensemble member $m$ trains on the outer training set minus its member-specific grouped 10 % early-stopping split (§3.3).
4. The outer test fold is touched **once**, for scoring.

### 3.5 Leakage audit (L1–L9 as requested; L10–L11 added from the provenance review)

| ID | Leakage channel | Where it could enter | Control | Residual risk / audit output |
|---|---|---|---|---|
| L1 | Same site in train and test | Multiple measurements per site | One row per site; grouped folds | Assert: no `group_id` in both train and test in any fold (Phase 7 test) |
| L2 | Co-located sites (same COMID, renamed gages) | Different `site_no` on one reach | Connected-component groups (§3.1) | Assert: no COMID in both train and test |
| L3 | Spatial and network dependence (neighbouring and nested reaches) | Nearby gages; upstream/downstream gages whose catchments contain one another | **Accepted risk** (no S-BLK, P3-D4). S-LRO and S-TAIL are the strong tests | Audit §3.6: Moran's I, LISA, residual variogram (range $\rho^*$), nested-basin check. Every S-ID metric is also reported on the **far** stratum (no training site within $\rho^*$) and the **not-nested** stratum. Label rule §3.6.7 |
| L4 | Target leakage through predictors | Measured discharge, `drain_area_va`, `chan_mat_cd`, HYDROSWOT velocity | Banned list in the feature schema; schema test rejects them | `VE_MA` excluded from M2/M5 inputs (redundant; used only in $A_J$) |
| L5 | Preprocessing leakage | Scalers, cutpoints, outlier thresholds | Split-then-scale (§3.4); QC is row-wise (§2.4) | Strata cutpoints use covariates only (DA), never targets |
| L6 | Selection leakage | Hyperparameters, early stopping, $\mathrm M^*$, thresholds | Nested inner grouped CV; $\mathrm M^*$ fixed on S-ID OOF **before** S-LRO, S-TAIL or S-IND are scored; H thresholds fixed in this protocol | Selecting $\mathrm M^*$ on S-ID favours the comparator, which is conservative for H1–H3 (§9.5) |
| L7 | Target-construction dependence | Mean-flow row chosen by closeness to $Q_E$, which is also an input | Within-site selection only (no cross-site information) | $F_{Qsel}$ flag; SA4; oracle-Q diagnostic decomposes the discharge mismatch (§7.3) |
| L8 | Independent-set overlap | eHydro locations on or next to HYDROSWOT training reaches; Zarrabi product trained at HYDROSWOT sites | Drop eHydro locations whose member COMIDs include a training COMID or lie within 1 km (EPSG:5070) of a training site. Zarrabi product compared **only** on S-IND | Report the number dropped, and (descriptively) the number of retained locations within $\rho^*$ of a training site or nested with one (§3.6.5). eHydro is large-river only (size-tail confound); report its DA distribution and AOA status (§3.6.6) |
| L9 | Temporal mismatch | HYDROSWOT dates vs EROM 1971–2000 vs StreamCat/NLCD years | Not a forecasting problem; predictors are time-invariant | Documented, not controlled. Report the distribution of `q_meas_dt` |
| **L10** | Modelled-covariate circularity | Abeshu D50 is an XGBoost product trained on observed D50 at 1,691 flowlines with catchment predictors that overlap ours | Ablation A4 (without D50 and AI) | H4 (b) $\rho_S(\hat n,D_{50})$ may partly reflect shared predictors; reported as a caveat |
| **L11** | Gage-adjusted discharge | `QE_MA` (`Q0001E`) is adjusted with gage data, so at gauged HYDROSWOT sites $Q_E$ is more accurate than at ungauged reaches | SA1: rerun M5 and $\mathrm M^*$ with `QC_MA` (`Q0001C`, no gage adjustment) | All headline errors are optimistic for ungauged use. M5 is more exposed because $Q_E$ enters Φ directly. SA1 quantifies the gap |

### 3.6 Spatial and network dependence audit (diagnostic only; P3-D4: it cannot change the splits)

**Purpose.** Dependence between training and test locations inflates the apparent skill of random or site-grouped CV [@Roberts_2017_BlockCV; @Ploton_2020_SpatialValidation]. P3-D4 excludes spatial-block splits, so the audit does not size blocks. It **quantifies** how optimistic S-ID may be, sets a **reporting label** (§3.6.7) and defines the far/not-nested strata used in L3. All audit settings are in `params.yaml` → `audit`.

#### 3.6.1 Inputs

- (a) Site-level targets $\ln W$, $\ln\bar d$ and $t$ (dependence in the response itself).
- (b) S-ID OOF log residuals of **M-null** and **M2-XGB**, averaged over the three repeats per site. These are the primary audit models; they are computed before any NN is trained, so the label is fixed before M5 results exist.
- (c) M5 residuals, reported once available. Comparing (b) and (c) shows whether the physics layer changes the residual spatial structure; this is descriptive.
- Coordinates: EPSG:5070. $N$ = number of post-QC sites.

#### 3.6.2 Global Moran's I [@Moran_1950_Autocorrelation]

$$
I=\frac{N}{S_0}\,\frac{\sum_{i}\sum_{j}w_{ij}(z_i-\bar z)(z_j-\bar z)}{\sum_i(z_i-\bar z)^2},\qquad S_0=\sum_{i,j}w_{ij},\qquad \mathbb E[I]=-\frac1{N-1}.
$$

- **Weights $\mathbf W_1$ (primary):** $k=8$ nearest neighbours, binary, row-standardized ($w_{ii}=0$).
- **Weights $\mathbf W_2$ (sensitivity):** inverse distance $1/d_{ij}$ for $d_{ij}\le100$ km, row-standardized. Sites with no neighbour within 100 km get a zero row and are reported as isolates.
- **Correlogram:** $I$ computed with binary distance-band weights for the classes (0, 10], (10, 25], (25, 50], (50, 100], (100, 250] and (250, 500] km.
- **Inference:** 999 random permutations of $z$ over sites. One-sided pseudo-p $=(1+\#\{I^*\ge I\})/1000$ for positive dependence. Report $I$, $\mathbb E[I]$, the permutation z-score and the pseudo-p.

#### 3.6.3 Local Moran's I (LISA) [@Anselin_1995_LISA]

- $I_i=\dfrac{(z_i-\bar z)}{m_2}\sum_j w_{ij}(z_j-\bar z)$, with $m_2=\sum_i(z_i-\bar z)^2/N$ and weights $\mathbf W_1$.
- 999 conditional permutations per site (site $i$ held fixed); significance at pseudo-p < 0.01.
- With about 4 k local tests, about 40 false positives are expected at this level, so LISA maps are **descriptive only** and enter no decision.
- Report counts of High–High, Low–Low, High–Low and Low–High sites by F&J division, and whether the clusters align with the S-LRO divisions.

#### 3.6.4 Residual semivariogram [@Matheron_1963_Geostatistics]

- **Estimator:** $\hat\gamma(h_k)=\frac1{2N_k}\sum_{(i,j)\in\mathcal P_k}(e_i-e_j)^2$, with 20 lag classes of equal pair count up to 500 km.
- **Models:** spherical, exponential and pure nugget, each fitted by least squares weighted by the pair counts $N_k$.
- **Selection:** lowest weighted SSE. If the pure-nugget model is within 5 % of the best weighted SSE, the residuals are taken as spatially unstructured and $\hat\rho:=0$.
- **Practical range $\hat\rho$:** the distance at which the fitted model reaches 95 % of its sill. Report $\hat\rho$ and the nugget/sill ratio.
- **Use of the range (replaces the S-BLK block size):** the **dependence radius** is $\rho^*=\min(\hat\rho_{\bar d}^{\,M2\text{-}XGB},\,250\ \text{km})$. It defines:
  1. the **far** stratum of S-ID test sites: no training site in that fold within $\rho^*$ (Euclidean, EPSG:5070);
  2. the descriptive eHydro proximity count (L8).

  No split, buffer or fold is modified.

#### 3.6.5 Nested-basin check (NHDPlusV2.1 topology)

- **Fields:** VAA `HydroSeq`, `UpHydroSeq`, `DnHydroSeq`, `LevelPathI`, the terminal-path identifier (⚠️ exact field name verified at acquisition), `PathLength` (km), `ArbolateSu` and `TotDASqKm`; `PlusFlow.FromComID` → `ToComID`.
- **Nested pair:** training site $i$ and test site $j$ are nested if one's COMID is reachable from the other's by upstream navigation of the PlusFlow graph (divergences included) within the same terminal path. One catchment then contains the other.
- **Strength:** $\psi=\mathrm{DA}_{\text{small}}/\mathrm{DA}_{\text{large}}$. **Strongly nested** if $\psi\ge0.5$. Also report $\lvert\Delta\texttt{PathLength}\rvert$ and whether the pair shares `LevelPathI` (mainstem).
- **Outputs per fold and family:** the share of test sites with at least one strongly nested training site, and every metric stratified as {strongly nested, weakly nested, not nested}.
- **S-LRO is checked too:** rivers cross division boundaries, so a held-out division can contain sites nested with training sites upstream or downstream in another division. Their count is reported.

#### 3.6.6 Area of applicability [@Meyer_2021_AreaOfApplicability]

- **Dissimilarity index DI:** for each test site, the distance in standardized, importance-weighted predictor space to the nearest training site, divided by the mean pairwise distance of the training set. Importance weights are M2-XGB permutation importances from the same outer training set.
- **AOA threshold:** the upper whisker (75th percentile + 1.5 IQR) of the training sites' cross-validated DI.
- **Reporting:** metrics inside vs outside the AOA for S-LRO, S-TAIL and S-IND. Descriptive only.

#### 3.6.7 Pre-registered reporting rule (replaces the "S-ID vs S-BLK reference" choice, which P3-D4 removed)

**S-ID remains the in-distribution reference in every case.** Two conditions are evaluated once, on M2-XGB $\bar d$ residuals, before any NN result is examined:

- **D (material spatial dependence):** (global Moran's I with $\mathbf W_1$ has pseudo-p < 0.01 **and** $I\ge0.10$) **or** $\hat\rho>\tilde d_{NN}$. Here $\tilde d_{NN}$ is the median distance from S-ID test sites to their nearest training site.
- **N (material nesting):** at least 20 % of S-ID test sites have a strongly nested training site.

| Audit result | Consequence (reporting only; H1–H4 decisions unchanged) |
|---|---|
| Neither D nor N | S-ID reported as is; strata in the supplement |
| D or N | Every S-ID headline number (H2 included) is labelled "optimistic under spatial/network dependence". The far and not-nested strata are reported alongside it, with their own paired-bootstrap CIs. The H2 decision is still taken on full S-ID $\mathcal Q_1$, as pre-registered. If a stratum CI has the opposite sign to the full-set CI, this is stated in the results summary |

The thresholds 0.10 and 20 % are new numbers fixed here (ADL-042).

---

## 4. Benchmark evaluation suite and metrics matrix

### 4.1 Model ladder

| Model | Definition in this study | Targets produced | Fallbacks |
|---|---|---|---|
| **M-null** (P3-D8) | Training-fold median of each log target per drainage-area quintile (cutpoints from the training fold) | $W,\bar d,d_{\max},A$ | – |
| **A_J** (P3-D8) | $A_J=Q_E/V_E$ (EROM; Jobson velocity). Training-free | $A$ only | – |
| **M0** | **Core (mean flow): refit** of the Bieger-form power law $\ln y=a_g+b_g\ln\mathrm{DA}$ per F&J division $g$ on the training fold [@Bieger_2015_RegionalCurves]. The published Bieger coefficients are **bankfull** curves, so they apply only to the secondary bankfull stage | $W,\bar d$; $A=W\bar d$ | Held-out division (S-LRO) or a division with < 30 training sites: pooled national curve from the training fold |
| **M1** | Zarrabi Tier-2 XGB [@Zarrabi_2025_ChannelGeometryCONUS] retrained on our folds. Feature set = Zarrabi Tier-2 list ($Q_E$, DA, $S$, stream order, $z_{\min}$, dams, population density, EVI ×4, clay/silt/sand, developed/forest/agriculture, D50, AI). The dams, population, EVI, soil and land-cover features come from the USGS release [@Wieczorek_2018_NHDPlusAttributes], as in Zarrabi (O-2 (b)); D50 and AI as for M2. Hyperparameter grid centred on the values read from their notebooks (text only) and re-tuned by inner CV. Target scale as in their notebooks (⚠️ verify in Phase 5); negative predictions floored at the 0.1th training percentile, with counts reported | $W,\bar d$; $A=W\bar d$ | – |
| **M2-MLP / M2-XGB** | Pure ML with **independent** log targets $\ln W,\ln\bar d,\ln d_{\max},\ln A$ (closure not enforced) on our full feature set. M2-MLP uses the shared backbone (§4.2) with a heteroscedastic Gaussian head per target | $W,\bar d,d_{\max},A$ | – |
| **M3** | Physics only (`2_mathematical_proofs.md` §7.9): $W$ regressor (shared backbone, $\mathcal L^{M3}=\sum\omega e_W^2$); $r=\bar r$ = training-fold median of $r_{obs}$ computed as $1/(e^{\mathrm{median}(t_{obs})}-1)$; $n_{emp}$ = Strickler ($S<S_J$) or Jarrett ($S\ge S_J$); depth via Φ (Jarrett branch via $\rho_J$, Proposition 7.5); $n$ projected onto $[n_{\min},n_{\max}]$ | all | – |
| **M4** (P3-09) | **Base M4 under $\mathcal I_{blind}$** (adopted, O-4): heads $(w,\ell,r,\nu)$, soft Manning residual, capped schedule $\lambda_M(\mathsf e)=\min(\lambda_0\gamma^{\mathsf e},\lambda_{\max})$ with $\lambda_0=1$, $\gamma=1.1$, $\lambda_{\max}\in\{10,10^2,10^3\}$ by inner CV; $\lambda_A$ not used (no M4⁺). Log $\Xi_{\mathsf e}$ every epoch (A7.3) | all | – |
| **M5** (primary) | Hard hybrid, full κ (P2-D3 (a)), heads $(w,r,\nu)$, $\mathcal L_{phys}\equiv0$, NLL per §9.1. $n$-head under $\mathcal I_{blind}$ (P3-11; O-4) | all | – |
| **M6** (P3-10) | **M6-E** (labels with $Q_E$; headline). **M6-NN primary** (shared backbone, heads $g_w,g_t,g_\nu$; $g_\nu$ under $\mathcal I_{blind}$ for parity with M5), M6-XGB secondary. Labels outside $[\nu_{\min},\nu_{\max}]$ are **retained** in training and projected by $\Pi$ at stage 3; their share is reported. Labels at $F_{tout}$ rows use $r_{obs}$ clipped to $[r_{\min},r_{\max}]$ for κ | all | – |
| **M5 ablations** | A1 κ ≡ 1; A2 $w_t=0$ (excluded from H4); A3-SP Strickler prior ($\tau_p=0.5\Rightarrow\lambda_p=2$); A3-SR logit residual ($\varpi=0.01$, $\lambda_\Delta=\lambda_p\Delta_n^2/16=2.865$, the Proposition 7.2 match at $\pi_S=\tfrac12$); A4 without D50 and AI (applied to M5 and $\mathrm M^*$); $w_A=1$ sensitivity | – | – |

**Information routing (P3-11; adopted, O-4).** Under $\mathcal I_{blind}$ the $n$-head receives every feature except $Q_E$, $S$, $F_S$ and $V_E$. The $W$- and $r$-heads receive the full feature set. Consequences (Propositions 6.5, 7.4): base M4 vs M5 is then a clean soft-vs-hard contrast, and M5 enforces the Manning elasticities in $Q$ and $S$ approximately (drainage area remains a $Q$ proxy). Under $\mathcal I_{full}$, base M4 degenerates and M4-S would be required.

**Strickler (P3-07; adopted, O-3).** $n_S=c_SD_{50}^{1/6}$ with $D_{50}$ in m and **$c_S=0.047$** [@Strickler_1923_Geschwindigkeitsformel]. ⚠️ The source has **no DOI** and the coefficient has not been verified from the original German report; it is recorded with the same ⚠️ status as McKay et al. (2012). No DOI-verified secondary source was found; WSP 2339 gives Limerinos ($d_{84}$-based), not Strickler. $\nu_S=\ln0.047+\tfrac16\ln D_{50}$.

**Jarrett (P3-08; verified from the full text).** [@Jarrett_1985_RoughnessColorado], eq. 6: $n=0.39\,S_f^{0.38}R^{-0.16}$ with $R$ in ft (worked example: $0.39(0.02)^{0.38}(2)^{-0.16}=0.079$). Valid for $0.002\le S\le0.04$ and $0.5\le R\le7$ ft; average standard error 28 %. "Higher-gradient" is defined there as $S>0.002$, so **$S_J=0.002$**. $a_J=0.38$, $b_J=-0.16<5/3$ ✓ (Proposition 7.5). SI form: $c_J'=0.39\cdot0.3048^{0.16}=0.3225$, so $n=0.3225\,S^{0.38}R_m^{-0.16}$. Rows outside the validity range are flagged and reported; nothing is clipped except by $\Pi$.

### 4.2 Shared NN configuration (P3-06, P3-12, P3-13, P3-14, P3-17)

| Item | Value |
|---|---|
| Backbone | MLP, 3 × 128, SiLU, **LayerNorm** (row-wise; no BatchNorm, so Proposition 3.4's scope holds) |
| Optimizer | AdamW, learning rate $10^{-3}$, cosine decay, batch 256, ≤ 500 epochs, early stopping with patience 50 on the member's grouped 10 % split |
| Inner grid | $\lambda_{wd}\in\{0,10^{-5},10^{-4},10^{-3}\}$ × learning rate $\{10^{-3},3\times10^{-4}\}$ (plus $\lambda_{\max}$ for M4) |
| Loss weights (P3-06) | $w_W=w_{\bar d}=w_t=1$, $w_A=0$ primary; $w_A=1$ sensitivity |
| Precision and device | **CPU, float64** for every model containing Φ or the Manning residual (M3, M4, M5, M6 stage 3), and for M2-MLP for parity. PyTorch MPS has no float64 (O-8) |
| Solver (P3-12) | $\tau_\ell=10^{-10}$ (log units), $K_{\max}=40$, per-row stopping (Proposition 3.4c); $\ln F$ via log1p of the integrated $F-1$ (Proposition 3.6 note) |
| Quadrature (P3-12) | Double-exponential (tanh-sinh) rule on the substituted integrand; node count fixed so that the Phase 7 test reproduces §10.2 E1–E5 to $\tau_Q=10^{-8}$ |
| UQ (P3-13) | Deep ensemble $M=10$; $\sigma_{\min}=0.01$; no correlation head ($\bar\varrho$ not used); β-NLL with $\beta_N=0.5$ [@Seitzer_2022_BetaNLL]; warm start of 20 epochs on MSE; $\alpha=0.10$ |
| Point predictor (P3-14) | Mixture median approximated by $\exp(\mu^*)$ with $\mu^*$ the ensemble mean of member means, everywhere (raw-space metrics included). The raw-space mean is not used |
| XGB grid | `max_depth` {4, 6, 8} × `eta` {0.03, 0.1} × `min_child_weight` {1, 5}; subsample 0.8; colsample 0.8; ≤ 2000 rounds, early stopping 100 |

### 4.3 Metrics: definitions and matrix

#### 4.3.1 Formal definitions

For an evaluation set with sites $j=1,\dots,J$ (one row per site, so site weighting is uniform; S-RND rows are weighted $1/\lvert I_j\rvert$ as in §9.5), with observation $o_j>0$, prediction $p_j>0$, log error $e_j=\ln p_j-\ln o_j$ and point prediction $p_j=\exp(\mu_j^*)$ (P3-14):

| Tier | Metric | Definition | Notes |
|---|---|---|---|
| **Primary** | log-RMSE | $\mathrm{RMSE}_{\log}=\bigl[\tfrac1J\sum_je_j^2\bigr]^{1/2}$ | For $W,\bar d,A,d_{\max}$. §9.5 statistic |
| **Primary** | log bias | $B_{\log}=\tfrac1J\sum_je_j$; multiplicative bias $e^{B}-1$ | Same targets. §9.5 statistic |
| Primary (shape) | $\mathrm{RMSE}_t$, $B_t$, $\rho_S(\hat r,r_{obs})$ | errors in $t=\ln(d_{\max}/\bar d)$ units | H4 (c) |
| Secondary | $R^2_{\log}\equiv\mathrm{NSE}_{\log}$ | $1-\sum_je_j^2\big/\sum_j(\ln o_j-\overline{\ln o})^2$ | |
| Secondary | $R^2_{raw}\equiv\mathrm{NSE}_{raw}$ | $1-\sum_j(p_j-o_j)^2\big/\sum_j(o_j-\bar o)^2$ | **Equivalence:** both are exactly the Nash–Sutcliffe efficiency [@Nash_1970_RiverFlowForecasting], with the evaluation-set mean as the benchmark. They can be negative. On OOD sets the benchmark is the test-set mean, which the model never sees |
| Secondary (comparability only) | $r^2$ | squared Pearson correlation of $(p_j,o_j)$, raw space | What Zarrabi et al. report as "R²" (`Performance_metrics` in their `utils.py`, commit `a30f1f5`). It ignores bias and scale, so it is **never used for ranking** |
| Secondary | $\mathrm{RMSE}_{raw}$ | $\bigl[\tfrac1J\sum_j(p_j-o_j)^2\bigr]^{1/2}$ | m, m², … |
| Secondary | MdAPE | $100\cdot\mathrm{median}_j\lvert p_j/o_j-1\rvert$ | |
| Secondary | APB % | $100\sum_j\lvert p_j-o_j\rvert\big/\sum_jo_j$ | Zarrabi's `abs_pbias` (verified in code). It is a normalized **absolute** error, not a signed bias, and is dominated by large rivers |
| Descriptive only | KGE | $1-\sqrt{(r-1)^2+(\alpha-1)^2+(\beta-1)^2}$, with $\alpha=\sigma_p/\sigma_o$ and $\beta=\mu_p/\mu_o$, raw space [@Gupta_2009_KGEDecomposition] | Reported because Zarrabi report it. Kept out of every decision: it was designed for hydrograph time series; across sites $\alpha$ penalizes the shrinkage that squared-loss estimators correctly apply; $\beta$ is dominated by large rivers; and in log space $\beta$ is undefined, because $\ln o$ can be ≤ 0 |

#### 4.3.2 Physical-consistency ("mass-balance") analogues

| Quantity | Definition | Applies to | Report |
|---|---|---|---|
| Observed closure $\epsilon_A$ | $\ln A_{obs}-\ln W_{obs}-\ln\bar d_{obs}$ | data (QC R8) | distribution; share removed |
| Predicted closure residual $\hat\epsilon_A$ | $\ln\hat A-\ln\hat W-\ln\hat{\bar d}$ | models with an independent $A$ head: **M2-MLP, M2-XGB**. Identically 0 for M0, M1, M3, M4, M5 and M6 ($A=W\bar d$ by construction) | median $\lvert\hat\epsilon_A\rvert$; share with $\lvert\hat\epsilon_A\rvert>\ln1.05$ |
| Shape-invariant violation | share with $\hat d_{\max}\le\hat{\bar d}$ (invariant 4) | M2 (independent heads); 0 by construction where $r$ is a head | share |
| Manning residual, M4 | $\hat R_M=\tfrac53\hat\ell+\tfrac23k(e^{\hat\ell-\hat w},\hat r)+\hat w+\tfrac12s-\hat\nu-q_E$ (§7.7), on test sites | M4 | RMS and distribution in $\ln Q$ units. Test-site residuals are not controlled by training (§7.7.1 remark) |
| Manning consistency, models without $n$ | back-calculated $\hat\nu_{back}$ from $(\hat W,\hat{\bar d},\hat r,Q_E,S)$ (Proposition 6.2), with $\hat r$ from $\hat d_{\max}/\hat{\bar d}$ (M2) or $\bar r$ (M0, M1); residual $\operatorname{dist}(\hat\nu_{back},[\nu_{\min},\nu_{\max}])$ (Proposition 7.4) | M0, M1, M2 | share with a positive residual: predictions that **no admissible $n$** reconciles with Manning at $Q_E$ |
| Manning, hard models | 0 up to $\tau_\ell$ | M3, M5, M6 (stage 3) | max solver residual |
| Continuity | $\hat V:=Q_E/\hat A$ for every model, so continuity holds by construction at evaluation and **no continuity residual exists** for any model | all | – |
| Implied-$V$ skill (diagnostic) | $\mathrm{RMSE}_{\log}$ and $B_{\log}$ of $\hat V$ vs $V_{obs}$ | all models producing $A$ | **$Q$ caveat:** $\ln\hat V-\ln V_{obs}=(q_E-q_{obs})-(\ln\hat A-\ln A_{obs})$ (§7.3), so the discharge mismatch enters directly. Also report the oracle $\hat V^{orc}=Q_{obs}/\hat A$ and the share of $V$-error variance from $q_E-q_{obs}$. Never a headline result |

#### 4.3.3 Matrix

Every cell is reported overall, by drainage-area quintile $\mathcal Q_1$–$\mathcal Q_5$ (cutpoints from post-QC `TotDASqKm`), by stream order and by division. S-ID cells are also reported on the far and not-nested strata (§3.6).

| Target ↓ / Split → | S-ID | S-LRO | S-TAIL-DA | S-TAIL-SO | S-IND (eHydro) | S-RND |
|---|---|---|---|---|---|---|
| $\bar d$ | P: RMSE$_{\log}$, $B_{\log}$. S: NSE$_{\log}$, NSE$_{raw}$, $r^2$, RMSE$_{raw}$, MdAPE, APB %. D: KGE | P (H1) | P (H3) | S | P (H1 corroboration) | S only |
| $W$ | P / S / D as above | P | P | S | P | S only |
| $A$ | P (H1, k = A) | P | P | S | P (A = W·d̄) | S only |
| $d_{\max}$ | P: RMSE$_{\log}$, $B_{\log}$ | S | S | S | – (not observed) | – |
| $t$ (shape) | P: RMSE$_t$, $B_t$, $\rho_S(\hat r,r_{obs})$ | S | S | S | – | – |
| $V$ (diagnostic) | implied-$V$ skill and oracle decomposition (§4.3.2) | S | S | S | – | – |
| Consistency | §4.3.2 table | S | S | S | – | – |
| UQ | Coverage $\hat C_{0.10}$, $\bar{\mathcal W}_{0.10}$, CRPS, PIT | descriptive | descriptive | descriptive | descriptive | – |

P = primary, S = secondary, D = descriptive only.

### 4.4 Pre-registered hypothesis procedures (P3-15, P3-16)

**Common settings.**
- **$\mathrm M^*$:** for each target $k\in\{\bar d,A\}$, the model among {M1, M2-MLP, M2-XGB} with the lowest S-ID OOF RMSE$_{\log,k}$ (mean over the three repeats). Fixed before any other split is scored.
- **Bootstrap:** paired cluster bootstrap over sites [@Field_2007_ClusteredBootstrap], $B=10{,}000$, percentile CI, $\alpha_b=0.05$, stratified by division (sites resampled within division). Seed 20261008.
- **S-ID repeats:** in each bootstrap draw the statistic is computed per repeat and averaged over the three repeats.
- **S-IND:** cluster bootstrap over the 25 **rivers** (locations on one river are not independent). With 25 clusters the CI is wide; this is expected and reported.

| H | Primary statistic (frozen definitions, §9.5) | Data | **Supported iff** | **Falsified iff** (`0_point_zero.md` §3, verbatim logic) |
|---|---|---|---|---|
| **H1** | $\Delta^{H1}_{\bar d}=\mathrm{RMSE}_{\bar d}^{(M5)}-\mathrm{RMSE}_{\bar d}^{(\mathrm M^*)}$; $k=A$ reported alongside | **Primary:** pooled S-LRO OOF. **Corroboration:** S-IND | S-LRO 95 % CI upper bound < 0 **and** the S-IND CI does not lie entirely above 0 | S-LRO 95 % CI includes 0 or is positive. An S-IND CI entirely above 0 is reported as a contradiction |
| **H2** | $\Delta^{H2}=\lvert B_{\bar d}^{(M5)}(\mathcal Q_1)\rvert-\lvert B_{\bar d}^{(\mathrm M^*)}(\mathcal Q_1)\rvert$; signed biases also reported | **Primary:** S-ID OOF, sites in $\mathcal Q_1$. Secondary: S-LRO $\mathcal Q_1$ | 95 % CI upper bound < 0 | CI includes 0 or is positive |
| **H3** | $\Delta^{H3}=L^{(M5)}-L^{(\mathrm M^*)}$, $L=\mathcal S(\mathcal T_{ID})-\mathcal S(\mathcal T_{OOD})$, **$\mathcal S=-\mathrm{RMSE}_{\log,\bar d}$** (not $R^2$). $\mathcal T_{OOD}$ = OOD predictions; $\mathcal T_{ID}$ = S-ID OOF predictions **for the same sites**. Two-sample bootstrap as in §9.5 | Separately: **H3-LRO** (all sites) and **H3-TAIL** (DA deciles 1 + 10). S-TAIL-SO secondary | Both H3-LRO and H3-TAIL have 95 % CI upper bound < 0 | Either CI includes 0 or is positive. A one-split pass is reported as "partial" |
| **H4** | (a) $\pi^\nu_\partial(\vartheta)$, $\pi^r_\partial(\vartheta)$ with **$\vartheta=0.05$** (n < 0.0064 or n > 0.47 for ν), and $\pi_{lit}$ on $\mathcal N_{lit}=[0.012,0.33]$. (b) $\rho_S(\hat n,D_{50})$ and $\rho_S(\hat n,S)$; **expected signs positive** for both. (c) $\rho_S(\hat r,r_{obs})$, RMSE$_t$, $B_t$. Site-level $\hat\nu$ = mean over repeats of the ensemble-mean $\nu$ | M5 on S-ID OOF (primary); S-LRO reported | none of the falsification conditions holds | $\pi^\nu_\partial>0.10$ or $\pi^r_\partial>0.10$ (0.10 is the share a uniform spread over the band would give), or $\pi_{lit}<0.80$, or the bootstrap 95 % CI of any of $\rho_S(\hat n,D_{50})$, $\rho_S(\hat n,S)$, $\rho_S(\hat r,r_{obs})$ includes 0 or lies below 0 |

**H4 caveats (reported, not thresholds).** C2: slope error alone induces a positive $n$–$S$ association, so passing (b) for slope is weak evidence; it is also reported stratified by $F_S$. L10: D50 is modelled. (d) diagnostics: S1 (§4.6) and $\hat n$ grouped by `chan_mat_cd`. Comparisons across κ treatments use Proposition 7.6.

### 4.5 Secondary contrasts S2 (P3-15)

- **Family (4 contrasts):** M5 vs M6-E and M4 vs M5, each on $\Delta$RMSE$_{\log,\bar d}$, on S-ID and S-LRO.
- **Multiplicity rule (adopted, O-7):** **Bonferroni** simultaneous percentile CIs at level $1-0.05/4$ (98.75 %) [@Dunn_1961_MultipleComparisons]. This fits the CI-only framework, needs no p-values, and avoids the unresolvable Holm (1979) DOI (doi.org returns 404 for 10.2307/4615733).
- Stratification by aspect ratio $a$ (quintiles of $\hat\kappa$) and by ID/OOD is **descriptive only**.
- All other comparisons (ablations, sensitivity analyses, M0/M3/A_J/M-null) are **descriptive** and labelled as such.

### 4.6 UQ targets and diagnostics (P3-13, P3-18)

| Item | Target |
|---|---|
| Nominal level | $\alpha=0.10$ (90 % central interval of the mixture, §9.3) |
| Acceptance band | $\hat C_{0.10}\in[0.85,0.95]$ on pooled S-ID OOF for $\bar d$ (primary), $W$ and $t$ (secondary). This is a quality target, **not** a hypothesis |
| OOD coverage | S-LRO, S-TAIL, S-IND: reported descriptively, with no guarantee claimed |
| Also reported | $\bar{\mathcal W}_{0.10}$, CRPS (closed form, §9.4), PIT histograms, per division and per quintile |
| Conformal calibration | **Not adopted** (P2-D4 (a)); P3-18 closed |
| S1 (P3-D9) | Diagnostic only: $\rho_S$ and log-RMSE between $\hat n$ and HyG at-a-station $n$ at matching gages. HyG $n$ is back-calculated from the same USGS measurements and the same NHDPlusV2 slopes, so S1 is a **consistency check, not independent validation** |

### 4.7 Outcome decision matrix (carried from the design note §8; pre-registered)

| Outcome | Claim | Venue (ADL-015) |
|---|---|---|
| H1 and H3 supported (M5 ≈ M2 in-distribution, better OOD and on S-IND) | "Physics constraints improve transferability of national channel geometry" | escalate to WRR / HESS permitted |
| H1 falsified, H4 supported, and **accuracy not worse**, defined as an S-LRO 95 % CI upper bound of $\Delta^{H1}_{\bar d}$ below +0.05 log units (≈ 5 %) | "Hydraulically consistent geometry, shape, roughness and velocity at no loss of accuracy" | ASCE primary |
| H2 supported only | "Physics constraints reduce small-stream depth bias" | ASCE / JAWRA |
| M5 worse (S-LRO CI lower bound > 0) | Diagnose with the **pre-registered** ablations (A1, A3, M4, SA1, SA5) and report. No post-hoc headline switch: any new variant is labelled exploratory | ASCE; fallback per ADL-014 |

> [!IMPORTANT]
> The "no loss of accuracy" margin (+0.05 log units on $\Delta^{H1}_{\bar d}$) is a new number. It makes row 2 operational and was adopted as O-10.

### 4.8 Compute budget (ADL-016)

| Item | Count |
|---|---|
| Sites (one row each) | ≈ 2.6–4.6 k (Zarrabi: 4,607 after QC, 2,626 after linking) |
| Outer fits per model | S-ID 15 + S-LRO ≈ 7 + S-TAIL 2 + S-IND 1 ≈ 25 |
| Primary NN models (M2-MLP, M4, M5, M6-NN) | 4 × 25 × (24 inner + 10 members) ≈ 3,400 fits |
| Ablations (A1, A2, A3-SP, A3-SR, A4, $w_A$; reuse M5 hyperparameters) | 6 × 25 × 10 ≈ 1,500 |
| Sensitivity analyses SA1–SA5 (M5 and the $\mathrm M^*$-NN) | ≈ 2,000–2,500 |
| **NN total** | **≈ 7,000–7,500 fits** |
| Cost per fit (CPU float64, ≈ 3.5 k rows, batch 256, early stopping) | ≈ 15–35 s (estimate; Φ adds ≤ 40 fixed-point iterations per row) |
| Wall time, ≈ 10 parallel workers on the M4 Max performance cores | **≈ 4–7 h** per full run |
| XGB fits, bootstrap ($B=10^4$ on ≤ 5 k sites), audits | minutes |
| **Verdict** | **GO, local** (ADL-016 stands). A Phase 6 timing pilot (one outer fold of M5) confirms the per-fit cost; if it exceeds 120 s, $K_{\max}$ and the inner grid are revisited by ADL |

### 4.9 Sensitivity analyses (pre-registered, descriptive)

SA1 `QC_MA` instead of `QE_MA` (L11) · SA2 `meas_q_va` as $Q_{obs}$ · SA3 include R8/R9 rows · SA4 drop $F_{Qsel}$ sites · SA5 drop $F_S$ sites · SA6 include $F_{DA}$ sites.

---

## 5. Closure of the Phase 3 register (P3-01 to P3-19) and carry-overs

| ID | Item | Resolution | Section | Status |
|---|---|---|---|---|
| P3-01 | $n$ bounds | Transform [0.005, 0.6]; $\mathcal N_{lit}$ = [0.012, 0.33] (WSP 2339) | §2.2 | ✅ |
| P3-02 | $r$ bounds | [0.5, 10] retained; $F_{tout}$ share reported | §2.2 | ✅ (O-9) |
| P3-03 | Slope floor | $10^{-5}$ confirmed by the user guide (A1.1); −9998 excluded; $F_S$ flag and feature | §2.2, §2.4 | ✅ |
| P3-04 | $Q_E\le0$ / missing | Excluded (R13) | §2.4 | ✅ |
| P3-05 | Field names, units, $\epsilon_A$, $d_{\max}>\bar d$, $t_{obs}$ share | HYDROSWOT fields from `FieldDefinitions.txt`; EROM `Q0001E`/`V0001E` (≡ `QE_MA`/`VE_MA` ⚠️ Phase 5 assertion); R6, R8, R9; $F_{tout}$ | §2.1, §2.3, §2.4, §4.3.2 | ✅ (D50/AI scale factors confirmed at acquisition). $\epsilon_A$ is also a consistency metric for models with an independent $A$ head (§4.3.2) |
| P3-06 | Loss weights | $w_W=w_{\bar d}=w_t=1$, $w_A=0$; $\lambda_{wd}$ by inner CV | §4.2 | ✅ |
| P3-07 | Strickler | $c_S=0.047$ (Strickler 1923, ⚠️ no DOI); $\tau_p=0.5$, $\lambda_p=2$, $\lambda_\Delta=2.865$, $\varpi=0.01$ | §4.1 | ✅ (O-3; ⚠️ source) |
| P3-08 | Jarrett, $S_J$, $\bar r$ | $c_J'=0.3225$ (SI), $a_J=0.38$, $b_J=-0.16$, $S_J=0.002$; $\bar r$ from the training-fold median of $t_{obs}$ | §4.1 | ✅ |
| P3-09 | M4 variant and schedule | Base M4, capped schedule ($\lambda_0=1$, $\gamma=1.1$, $\lambda_{\max}$ by inner CV), AdamW, $\Xi_{\mathsf e}$ logged | §4.1 | ✅ with P3-11 (O-4) |
| P3-10 | M6 | M6-E headline; M6-NN primary, M6-XGB secondary; labels retained, then $\Pi$ | §4.1 | ✅ |
| P3-11 | Routing | $\mathcal I_{blind}$ for the $n$-head and $g_\nu$ | §4.1 | ✅ (O-4) |
| P3-12 | Solver and quadrature | $\tau_\ell=10^{-10}$, $K_{\max}=40$, per-row stopping, tanh-sinh, $\tau_Q=10^{-8}$, log1p | §4.2 | ✅ |
| P3-13 | UQ | $M=10$, $\sigma_{\min}=0.01$, no correlation head, β-NLL $\beta_N=0.5$, warm start, $\alpha=0.10$, band [0.85, 0.95] | §4.2, §4.6 | ✅ |
| P3-14 | Point predictor | $\exp(\mu^*)$ (median) everywhere | §4.2 | ✅ |
| P3-15 | H1–H3 settings; S2 | $\mathrm M^*$ rule, $B=10^4$, $\alpha_b=0.05$, division-stratified, $\mathcal S=-\mathrm{RMSE}_{\log}$, quintile source, splits and seeds; S2 Bonferroni | §3.2, §4.3.1, §4.4, §4.5 | ✅ (O-7). Metric definitions fixed in §4.3.1; S-ID dependence label and far/not-nested strata per §3.6.7 (reporting only; H2 decided on full S-ID) |
| P3-16 | H4 | $\mathcal N_{lit}$, $\vartheta=0.05$, positive expected signs, S1 diagnostic only | §4.4, §4.6 | ✅ |
| P3-17 | Architecture | MLP 3 × 128, SiLU, LayerNorm (no batch coupling), AdamW | §4.2 | ✅ |
| P3-18 | Conformal | Not adopted | §4.6 | ✅ |
| P3-19 | Carry-overs | See below | – | ✅ (NHDPlus terms per O-6) |
| (middle prompt) | Dependence audit; DVC specification; metric definitions | §3.6 replaces S-BLK block sizing with a diagnostic audit and a pre-registered label rule (thresholds: ADL-042); §1.5 records acquisition as DVC stages tracked in `dvc.lock` (ADL-043); §4.3.1–4.3.2 define every metric and physical-consistency analogue; §2.3 adds raw-unit schemas (gate review) | §1.5, §2.3, §3.6, §4.3 | ✅ |

**Carry-overs.**

| Item | Resolution |
|---|---|
| McKay et al. (2012) author list | **Closed.** The user guide's recommended citation, verbatim: "McKay, L., Bondelid, T., Dewald, T., Johnston, J., Moore, R., and Rea, A., 'NHDPlus Version 2: User Guide', 2012". Note: the guide's team list names "Craig Johnston", so the initial "J." is the guide's own inconsistency. The bib entry uses the recommended citation as printed and records the discrepancy |
| NHDPlusV2.1 licence | **Closed (O-6).** No licence or terms statement exists in the guide or on the EPA download page (EPA disclaimers only). It is treated as a U.S. Government work and cited with attribution; the absence of a statement is recorded here |
| Zarrabi code licence | **Closed: MIT** (LICENSE file plus README), pinned at commit `a30f1f5` |
| Q3 field | **Closed:** `q_va` is $Q_{obs}$; `meas_q_va` is SA2 |

---

## 6. Resolved protocol decisions (PI, 2026-10-08)

| ID | Item | Decision |
|---|---|---|
| O-1 | Aridity source | **Global-AI v3.1** (Zomer et al. 2022; CC BY 4.0), the product family Zarrabi used (v2) |
| O-2 | Catchment predictors for M1 | **(b)** M1 uses the USGS release (Wieczorek et al. 2018; ⚠️ drifted DataCite record) for fidelity to Zarrabi; M2, M4, M5 and M6 use the StreamCat set (P3-D6) |
| O-3 | Strickler coefficient | $c_S=0.047$ (D in m), Strickler (1923), ⚠️ no DOI |
| O-4 | P3-11 routing | $\mathcal I_{blind}$ for the $n$-head and $g_\nu$; base M4 |
| O-5 | eHydro acquisition | **(c)** author request (with licence) plus re-derivation from USACE eHydro in Phase 5; re-derivation verifies the author table and is the fallback |
| O-6 | NHDPlusV2.1, StreamCat, physiography and USACE terms | Treated as U.S. Government works with attribution; absence of a licence statement recorded |
| O-7 | S2 multiplicity | Bonferroni (Dunn 1961) |
| O-8 | Precision and device | CPU float64 for all NN models |
| O-9 | $r_{\min}$ | 0.5 kept |
| O-10 | "No loss of accuracy" margin | +0.05 log units on the S-LRO CI upper bound of $\Delta^{H1}_{\bar d}$ |
| O-11 | M0 for mean flow | Refit per division on training folds |
| O-12 | L11 sensitivity | SA1 (`QC_MA`) adopted |
| O-13 | Drive and symlinks | Phase 5 dummy-file test before any data; moving the working tree out of Drive is the fallback |

---

## 7. ADL rows appended at the gate (final mapping, PI gate decision 2026-10-08)

The earlier decision-organised proposal (ADL-042 to 050) was merged into the topic structure of the PHASE_03 gate prompt. Every earlier row is carried as shown.

| ADL | Topic | Carries (earlier proposal) |
|---|---|---|
| ADL-042 | Data partitioning protocol: group key, split families, no S-BLK, leakage controls L1–L11, dependence-audit thresholds | earlier 043 and 049 |
| ADL-043 | DVC tracking scheme: stage-based acquisition, `dvc.lock` replaces `.dvc` pointers (amends ADL-009), cache/remote, `--local` config, Drive test | earlier 048 and the item-2 part of 047 |
| ADL-044 | Baseline suite and metrics matrix | item-3 part of 047 |
| ADL-045 | Provenance and licences; P3-19 closure; StreamCat citation correction; eHydro as sole independent set | earlier 044 and 045 |
| ADL-046 | Schemas, QC and bounds (P3-01 to P3-08), including the raw-unit schemas | – |
| ADL-047 | Pre-registered H1–H4 statistics, bootstrap, multiplicity, UQ targets, decision matrix (closes ADL-010) | – |
| ADL-048 | P3-D1 to P3-D9, O-1 to O-13 and the compute verdict (ADL-016) | earlier 042 and 046 |
| ADL-049 | Phase 3 gate PASSED, Phase 3 → 4, code unlock | earlier 050 |

## 8. Decision record (PI, 2026-10-08)

| ID | Decision |
|---|---|
| P3-D1 | (a) DAG, `.dvcignore` and commands live in this protocol, not executed; skeleton `dvc.yaml`/`.dvcignore` committed (`chore(ops)`) only after the phase flips to 4; `dvc init` in Phase 5 |
| P3-D2 | (a) Cache and local remote at `~/dvc_store/ANN_XS_Estimation/{cache,remote}`; Zenodo for Phase 11. Follow-up: `data/raw` and `data/processed` must not sync in Drive; link type decided in Phase 5 |
| P3-D3 | (a) Markdown schemas; Pydantic plus Pandera or Great Expectations in Phase 4/5 |
| P3-D4 | (i) only: site–COMID connected components; no S-BLK; autocorrelation audit kept as a diagnostic |
| P3-D5 | (a) Bieger (2015) physiographic divisions |
| P3-D6 | NHDPlusV2.1 VAA + EROM, StreamCat, D50 (Abeshu 2022), aridity index (Global-AI v3.1, O-1); plus the "without D50/AI" ablation |
| P3-D7 | eHydro (76 locations) is the only independent set |
| P3-D8 | (a) M-null plus $A_J=Q_E/V_E$ |
| P3-D9 | S4: literature-anchored $n$ bounds, pre-registered, never fitted to test targets. S1: diagnostic only |

## 9. References

Cited keys (existing): @Canova_2016_HYDROSWOT, @Zarrabi_2025_ChannelGeometryCONUS, @Zarrabi_2024_GeometryDataset, @Bieger_2015_RegionalCurves, @Heldmyer_2022_NWMChannelParameterization, @Enzminger_2024_HyGDataset, @Jarrett_1984_HighGradientStreams, @Seitzer_2022_BetaNLL; added in the middle-prompt revision: @Roberts_2017_BlockCV, @Ploton_2020_SpatialValidation, @Meyer_2021_AreaOfApplicability.

**New entries (to add identically to `master.bib` and `docs/references.bib` after approval):**

| Key | Reference | Identifier (verified 2026-10-08) |
|---|---|---|
| Hill_2016_StreamCat | Hill, R. A., Weber, M. H., Leibowitz, S. G., Olsen, A. R., & Thornbrugh, D. J. (2016). The Stream-Catchment (StreamCat) Dataset: A database of watershed metrics for the conterminous United States. *JAWRA*, 52(1), 120–128 | doi:10.1111/1752-1688.12372 (Crossref; online 2015) |
| Abeshu_2022_D50 | Abeshu, G. W., Li, H.-Y., Zhu, Z., Tan, Z., & Leung, L. R. (2022). Median bed-material sediment particle size across rivers in the contiguous US. *ESSD*, 14(2), 929–942 | doi:10.5194/essd-14-929-2022 (Crossref) |
| Li_2021_D50Map | Li, H.-Y., Abeshu, G., Zhu, Z., Tan, Z., & Leung, L. R. (2021). *A national map of riverine median bed-material particle size over CONUS* (v1.1) [Dataset]. Zenodo | doi:10.5281/zenodo.4921987 (DataCite; CC BY 4.0) |
| Zomer_2022_GlobalAridityV3 | Zomer, R. J., Xu, J., & Trabucco, A. (2022). Version 3 of the Global Aridity Index and Potential Evapotranspiration Database. *Scientific Data*, 9, 409 | doi:10.1038/s41597-022-01493-1 (Crossref) |
| Zomer_2025_GlobalAIv31 | Zomer, R., & Trabucco, A. (2025). *Global Aridity Index and Potential Evapotranspiration (ET0) Database: Version 3.1* [Dataset]. figshare | doi:10.6084/m9.figshare.7504448.v7 (DataCite; CC BY 4.0) |
| USGS_2023_PhysiographicDivisions | U.S. Geological Survey (2023). *Physiographic divisions of the conterminous U.S.* [Data release; after Fenneman & Johnson 1946] | doi:10.5066/P9B1S3K8 (DataCite) |
| Enzminger_2023_HyGData | Enzminger, T. L., Minear, J. T., & Livneh, B. (2023). *HyG: A hydraulic geometry dataset derived from historical stream gage measurements across the conterminous United States* [Dataset]. Zenodo | doi:10.5281/zenodo.7868764 (DataCite; CC BY 4.0) |
| Jarrett_1985_RoughnessColorado | Jarrett, R. D. (1985). *Determination of roughness coefficients for streams in Colorado*. USGS Water-Resources Investigations Report 85-4004 | doi:10.3133/wri854004 (resolves via doi.org; metadata from the USGS Publications Warehouse) |
| Arcement_1989_ManningSelection | Arcement, G. J., & Schneider, V. R. (1989). *Guide for selecting Manning's roughness coefficients for natural channels and flood plains*. USGS Water-Supply Paper 2339 | doi:10.3133/wsp2339 (Crossref title) |
| Moran_1950_Autocorrelation | Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1–2), 17–23 | doi:10.1093/biomet/37.1-2.17 (Crossref) |
| Field_2007_ClusteredBootstrap | Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *JRSS B*, 69(3), 369–390 | doi:10.1111/j.1467-9868.2007.00593.x (Crossref) |
| Dunn_1961_MultipleComparisons | Dunn, O. J. (1961). Multiple comparisons among means. *JASA*, 56, 52–64 | doi:10.1080/01621459.1961.10482090 (Crossref) |
| McKay_2012_NHDPlusV2 | McKay, L., Bondelid, T., Dewald, T., Johnston, J., Moore, R., & Rea, A. (2012). *NHDPlus Version 2: User Guide* (Data Model v2.1, 2019) | ⚠️ No DOI. https://www.epa.gov/system/files/documents/2023-04/NHDPlusV2_User_Guide.pdf |
| Zarrabi_2024_Code | Zarrabi, R., et al. *Bankfull-and-mean-flow-channel-geometry-for-CONUS* [Software], commit a30f1f5. MIT | ⚠️ No DOI. https://github.com/Reizrb/Bankfull-and-mean-flow-channel-geometry-for-CONUS |
| Wieczorek_2018_NHDPlusAttributes | Wieczorek, M. E., Jackson, S. E., Schwarz, G. E., Sekellick, A. J., & Staub, L. E. (2018). *Select attributes for NHDPlus Version 2.1 reach catchments and modified network routed upstream watersheds for the conterminous United States* (ver. 6.0, August 2026) [Data release]. U.S. Geological Survey | doi:10.5066/F7765D7V (citation from the parent ScienceBase item; ⚠️ the DataCite record and doi.org resolution point to a child item) |
| Strickler_1923_Geschwindigkeitsformel | Strickler, A. (1923). *Beiträge zur Frage der Geschwindigkeitsformel und der Rauhigkeitszahlen für Ströme, Kanäle und geschlossene Leitungen*. Mitteilungen des Eidgenössischen Amtes für Wasserwirtschaft, Bern | ⚠️ No DOI; bibliographic details from secondary sources; not verified from the original |

**New entries from the middle-prompt revision:**

| Key | Reference | Identifier (verified 2026-10-08) |
|---|---|---|
| Anselin_1995_LISA | Anselin, L. (1995). Local indicators of spatial association—LISA. *Geographical Analysis*, 27(2), 93–115 | doi:10.1111/j.1538-4632.1995.tb00338.x (Crossref) |
| Matheron_1963_Geostatistics | Matheron, G. (1963). Principles of geostatistics. *Economic Geology*, 58(8), 1246–1266 | doi:10.2113/gsecongeo.58.8.1246 (Crossref) |
| Nash_1970_RiverFlowForecasting | Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I — A discussion of principles. *Journal of Hydrology*, 10(3), 282–290 | doi:10.1016/0022-1694(70)90255-6 (Crossref) |
| Gupta_2009_KGEDecomposition | Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1–2), 80–91 | doi:10.1016/j.jhydrol.2009.08.003 (Crossref) |

Not added: Holm (1979), Valavi et al. (2018), Ferguson (2010); verified this session but not cited.

## 10. Phase 5 amendments (append-only; P5-D10)

Appended at the Phase 5 gate (2026-10-08). Sections 1–9 are unchanged from `a98feff`. Each item cites the ADL row that records it in `.agents/.research_state.md`.

### 10.1 Frozen artefacts (§1.2 item 3, §3.3) [ADL-069, ADL-075]

| Artefact | SHA-256 |
|---|---|
| `data/processed/site_table.parquet` (4,879 sites; 13,289 rows: primary 4,205, sa2 4,205, sa3 4,879) | `8d186b07aae165eddff23119b9bc67223705c8dff5cf50d7950711d2566d5470` |
| `data/processed/splits/fold_assignment.parquet` (126,854 rows; 26 blocks) | `1bf628c83fd658c179243499af5d533738fefaeb15b63a71c476530c40dc872c` |
| `data/processed/splits/srnd_assignment.parquet` (38,693 rows) | `b3c07ba313f5e3b834d85dbeb7ec1c9b17c31e7f5b2db0f1ef775a52013808a1` |
| `dvc.lock` | `b9835a193895217b4f04f8c0076c34517887912fac7522ce987767e20b3fba6d` |

The artefacts were written under `params_sha256` `03dc55ea51b8feebe3aceb57ea50ac6d5db8735f0dff552bb2c3b80a549c3241`. The current `params.yaml` (`5d9e972a6c516632747958dfa23172ed291a3aba71775322e9a8988ab2af8de0`) only adds the `transforms` group (§10.9), which no stage reads.

### 10.2 Erratum to §1.5 (`dvc.yaml` listing) [ADL-058; commit `82580d7`]

In the `acquire` stage, entries containing `${item}` must be quoted YAML strings: `outs: ["data/raw/${item}"]`. Parameters are tracked per unit: `params: ["acquire.${item}"]`.

### 10.3 Acquisition manifest clarifications (§1.1, §1.2) [ADL-058, ADL-061, ADL-074]

- **NHDPlusV2.1.** Per-VPU `.7z` archives from the EPA S3 bucket (21 VPUs). Their multipart ETag is not an md5, so the SHA-256 at acquisition is the integrity anchor.
  - Catchment polygons are added for the aridity zonal mean (P5-D3).
  - GageLoc comes from the national archive.
- **StreamCat.** The bulk files have been withdrawn, so the data come from the StreamCat API: 21 regional `Ws` queries with the approved 20-metric list, and row counts checked against `countOnly`. A re-acquisition can differ byte-wise, so the DVC cache is the reproducibility anchor.
- **HYDROSWOT.** No publisher checksum; the SHA-256 at acquisition is the anchor.
- **d50, hyg, zarrabi_product.** These servers send no ETag or Last-Modified header; the anchor is the publisher md5 plus SHA-256.
- **usgs_attrs.** Five ScienceBase child items (the M1 columns).
- **ehydro (T5) and ehydro_author.** Not acquired at the Phase 5 gate. `ind_prepare` / S-IND passes under an explicit PI waiver (P5-D8) and is carried into Phase 6 [ADL-078].

### 10.4 Delivered field names and scale factors (§2.1, §2.3 ⚠️ items) [ADL-061, ADL-063]

- HYDROSWOT delivers `xsec_area_va` (§2.3 lists `xsecond_area_va`). Likewise `q2_xsec_area_X_mean_vel_va` and `q4_xsec_area_D_stream_width`.
- NHDPlusV2.1 field-name case differs across tables and VPUs, so fields are matched case-insensitively. The VAA carries `TerminalPa` (10-character DBF truncation).
- `Q_E` is read as `EROM_MA0001.Q0001E` (≡ `QE_MA`, §2.1) and `Q_C` as `Q0001C`.
- The D50 field is `D50[mm]`, multiplied by 1e-3 to give metres.
- Global-AI v3.1 is a uint16 raster with **no declared nodata**. The values 0 and 65535 are excluded, and the scale is 1e-4.

### 10.5 QC rule interpretations (§2.4) [ADL-064]

- **R2:** invalid physiography polygons are repaired with `make_valid`. A point with no single division is dropped.
- **R3:** a measurement is dropped whole if any of its rows has `q_meas_chan_nu ≥ 2`; channel 0 is kept.
- **R7:** all three discharge pairs must agree within `q_tol`.
- **R10:** `data_aging_cd = W` counts as not approved (flag only).
- **R11:** among duplicates, the lowest `row_id` is kept.

### 10.6 Count deviation (§2.4, §4.8) [ADL-064, ADL-065, ADL-075]

- `hydroswot_init` has 39,927 rows at 5,141 sites, against ≈ 38 k rows at ≈ 4.6 k sites expected in §2.4.
- Linked sites: 4,889 (4,569 GageLoc, 320 snapped; `F_DA` 261).
- Site-table union U: 4,879 sites in 4,851 groups (22 pairs, 3 triples).
- Primary analysis frame: 3,988 sites, inside the §4.8 range of 2.6–4.6 k. SA3 has 4,620 and SA6 4,205.
- The Phase 6 timing pilot uses these counts.

### 10.7 Aridity zonal mean (amends P5-D4; §2.3) [ADL-066]

Pixel centres are used (`all_touched=False`). A catchment with no pixel centre inside falls back to its touched pixels and is flagged `F_AI_touched` (368 COMIDs, 370 sites).

### 10.8 Fold table (amends §3.3 columns) [ADL-068]

- One row per (family, repeat, outer_fold, site_no), with the added columns `is_test` and `lro_division`.
- On test rows, `inner_fold` = −1 and every `es_member_m` = False.
- Inner folds are grouped 3-fold, with the DA quintiles recomputed inside each outer training set.
- ES is a seeded 10 % of training groups per member, from `SeedSequence([seed, family_idx, repeat, outer_fold, m])`.
- S-RND is a separate table (P5-D7).

### 10.9 Feature transforms (amends the P5-D5 placement; §3.4, §4.2) [ADL-072]

- The per-feature transforms live in a separate `params.yaml` group `transforms`, so the frozen artefacts keep their hashes.
- `log`: DA, S, Q_E, popden, precip, runoff, elev, perm, D50, AI.
- `identity`: stream order, z_min, clay, sand, tmean, BFI, wetness index, land-cover sums.
- `log_offset`: `damdensws`, with c = 0.5 × the smallest positive training value.
- `passthrough`: F_S.

### 10.10 Scaler artefact (§3.4, L5) [ADL-076]

- No global scaler exists. Each fit saves one `NormStats` JSON (`float.hex`, no pickle), fitted on the rows of its training part only.
- The JSON records the analysis, the fold block, the part and index, and the SHA-256 of the site and fold tables.
- Saving fails if this provenance is missing or names a non-training part.

## 11. Phase 6 amendments (append-only)

### 11.1 M1 target scale and grid (§4.1, §4.2) [ADL-085]

- Target scale: raw metres for W and d̄, as in the Zarrabi notebooks at commit `a30f1f5` (`mean_depth_va_mf` after `HYDRoSWOTUnitsConvertor`; read as text only). This resolves the §4.1 ⚠️ item.
- Features: the 19 "Model 2" predictors of the §4.1 list. "Model 1", which uses measured discharge, is excluded.
- Grid: `model.baselines.m1_max_depth` = {4, 6, 8}, the shared §4.2 values, × the shared `eta` and `min_child_weight`.
  - The notebooks load pickled boosters (never opened), so no hyperparameter can be read from their text.
  - The value 12 in the Phase 5 carry-over could not be verified, so the §4.1 rule "centred on the values read from their notebooks" falls back to the shared grid.
- Non-positive predictions are floored at the 0.1th training percentile; counts are reported.

### 11.2 M6 stage-3 shape projection (§4.1 M6 row) [ADL-084]

- At stage 3, ĝ_t is clamped to [t(r_max), t(r_min)] = [ln 1.1, ln 3] before r̂ = r(t̂). This mirrors the label rule for F_tout rows.
- The projected t̂ is reported, so d_max = d̄(1 + 1/r̂) holds exactly. The share of projected rows is reported.
