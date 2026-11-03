# CONUS River Cross-Section (XS) WebGIS Explorer (`v1.0.0`)

A zero-build, single-page static WebGIS application showcasing the **Physics-Informed Neural Networks and Tree-Based Surrogates for Continental-Scale River Cross-Section Estimation** benchmark (`Event 1–3` geometry + `OPS-1 v1–v3` operational 1D flood-stage routing and `CCHE1D` solver readiness).

Designed for direct hosting on **GitHub Pages** or any static HTTP server with zero Node/Webpack build steps required at deploy time.

---

## 1. Interactive Views & Capabilities

1. **CONUS Map & XS Inspector (`#view-map`)**
   - **MapLibre GL JS v4** canvas displaying all **3,988 CONUS USGS gauges** (`HYDROSWOT` + `NHDPlusV2.1`) and **131 independent USACE `eHydro` locations** (`18` rivers, `33` single-COMID UQ sites), overlaid on the **8 USGS Fenneman & Johnson physiographic divisions** (`physio_divisions.geojson`).
   - Dynamic map styling by **Winning Model Family (`M1`–`M6`)**, **Reach Log-Space Mean RMSE**, **Covariate Tier Completeness**, **Physiographic Division**, **NHDPlus Stream Order**, or **Log10 Drainage Area ($A_d$)**.
   - **Analytical Cross-Section ($\text{XS}$) Profile Drawer:** clicking any reach renders its observed Dingman bed profile $z(x) = -d_{\max}\left(1 - |2x/W|^r\right)$ (`HYDROSWOT` median or `USACE eHydro` survey) against predicted profiles from all six models (`M1` Power-Law, `M2` Tabular MLP, `M3` LightGBM, `M4` PINN-MLP, `M5` Physics-Guided ResNet, `M6` Gated Residual Network) under both **In-Distribution (`S-ID`)** and **Leave-Region-Out (`S-LRO`)** regimes.
   - **90% Deep-Ensemble UQ Bands (`M5` $K = 5$)** and **USGS 3DEP 1/3 arc-second (~10 m) Lidar Valley Insets** (±5 bankfull widths) where available (`3,632` CONUS gauges).
   - **Live Stage–Discharge Rating Simulator:** interactive stage ratio slider ($y / \bar{d} \in [0.25, 2.00]$) solving the coupled Dingman–Manning system $Q(y) = \frac{1}{n} A(y) R(y)^{2/3} S_0^{1/2}$ in real time with dynamic water-surface rendering.

2. **Model Benchmarks (`#view-benchmarks`)**
   - **Multi-Regime Matrix:** interactive comparison across `S-ID` ($n = 599$ test gauges), `S-LRO` ($n = 3,988$ across 8 divisions), `S-LSO` ($n = 3,893$ stream-order holdout), and `S-IND` (`USACE eHydro`, $n = 131$ locations) for `W`, `d_mean`, `A`, `d_max`, `r`, and `n`.
   - **Physiographic Division Breakdown:** division-by-division winners under `S-ID` vs. `S-LRO` with 10,000-replicate paired bootstrap confidence intervals (`reports/operational/best_model_map/`).
   - **Operational Flood-Stage Routing (`OPS-1 v1–v3`):** compound-channel normal-depth stage RMSE ($\text{m}$), normalized stage error ($\text{NSE}_h$), water-surface top-width error, 90% UQ empirical coverage (`95.0%` CONUS / `100.0%` `S-IND`), and 5-tier `CCHE1D` 1D unsteady St. Venant solver certification (`99.0%` Grade A on `S-ID`, `100.0%` Grade A on `S-IND`).

3. **Data Availability & QA/QC (`#view-data`)**
   - **4-Stage `HYDROSWOT` Harmonization Cascade:** interactive breakdown from `246,729` raw field measurements (`2,621` stations) through `F_Q`, `F_CONT`, `F_MANNING`, and `F_STAGE` to `155,705` harmonized records (`3,988` primary site medians).
   - **Covariate Tier Matrix (`T1`–`T6`):** coverage and imputation flags across `NHDPlusV2.1`, `StreamCat`, `Abeshu et al. (2022) D50`, `Global-AI` aridity, and `USGS 3DEP` lidar valley transects (`91.07%` full `T1–T6` coverage).
   - **Searchable 4,119-Reach Virtualized Table:** instant filtering across all `3,988` CONUS gauges and `131` USACE `eHydro` locations with **one-click CSV export**.

4. **XS Gallery & Interactive Simulator (`#view-gallery`)**
   - Curated canonical river cross-sections across all 8 CONUS physiographic divisions plus a **Live Custom Dingman–Manning Cross-Section Sandbox** ($W$, $\bar{d}$, $r$, $n$, $S_0$, $y/\bar{d}$, and $z_{\text{bank}}$).

---

## 2. Directory Structure

```text
webgis/
├── index.html                   # Semantic HTML5 single-page shell (WCAG AA, dark/light themes)
├── styles.css                   # Responsive CSS Grid/Flexbox tokens, tables, and drawer styling
├── app.js                       # Vanilla ES2022 module (MapLibre GL JS v4, SVG XS engine, simulator)
├── README.md                    # Documentation and deployment instructions
└── data/                        # Deterministic static JSON/GeoJSON bundle (~5.5 MB total, ~1.1 MB gzipped)
    ├── manifest.json            # SHA-256 checksums and canonical record counts
    ├── benchmarks.json          # Event 1–3 + OPS-1 v1–v3 + CCHE1D + physiographic division metrics
    ├── physio_divisions.geojson # 8 simplified USGS physiographic division polygons (EPSG:4326)
    ├── sites_conus.json         # 3,988 CONUS stations with covariates, S-ID/S-LRO predictions, UQ & 3DEP
    └── sites_ehydro.json        # 131 USACE eHydro S-IND locations with survey & predicted geometry
```

---

## 3. Local Preview

Because ES modules and `fetch()` require an HTTP origin (`http://` or `https://` rather than `file://`), serve the `webgis/` directory using Python's built-in HTTP server from the repository root:

```bash
python3 -m http.server -d webgis 8080
```

Then open [http://localhost:8080](http://localhost:8080) in any modern browser.

---

## 4. Regenerating the Static Data Bundle

All JSON and GeoJSON payloads in `webgis/data/` are deterministically compiled from the repository's frozen `reports/` and `data/` artifacts via [`scripts/build_webgis_data.py`](../scripts/build_webgis_data.py):

```bash
source scripts/dev_env.sh
uv run python scripts/build_webgis_data.py
```

And verified against physical and statistical invariants via Pytest:

```bash
source scripts/dev_env.sh
uv run pytest tests/unit/test_webgis_bundle.py -v
```

---

## 5. Deploying to GitHub Pages

### Option A: GitHub Actions Static Workflow (Recommended for `/webgis` subfolder)
Create `.github/workflows/pages.yml` in your GitHub repository:

```yaml
name: Deploy WebGIS to GitHub Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      - name: Configure GitHub Pages
        uses: actions/configure-pages@v5
      - name: Upload WebGIS static artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: "./webgis"
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### Option B: Branch / Folder Publishing
In your GitHub repository settings (**Settings → Pages**), set **Source** to **GitHub Actions** (with the workflow above) or publish the `webgis/` directory directly to your `gh-pages` branch:

```bash
git subtree push --prefix webgis origin gh-pages
```
