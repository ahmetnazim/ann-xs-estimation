# CONUS River Cross-Section (XS) WebGIS Explorer

A zero-build, single-page static WebGIS application for exploring continental-scale river cross-section geometry predictions, 3DEP Lidar valley profiles, stage–discharge rating curves, and 1D hydraulic solver readiness (`CCHE1D`).

Designed for direct hosting on **GitHub Pages** or any static HTTP server with zero Node/Webpack build steps required at deploy time.

---

## 1. Interactive Views & Capabilities

1. **WebGIS & XS Explorer (`#view-map`)**
   - **MapLibre GL JS v4** canvas displaying all **3,988 CONUS USGS gauges** (`HYDROSWOT` + `NHDPlusV2.1`) and **131 independent USACE `eHydro` survey reaches** (`18` rivers, `33` single-COMID UQ reaches), overlaid on the **8 USGS physiographic divisions** (`physio_divisions.geojson`).
   - Dynamic map styling by **Best Model (Bankfull Geometry, Flood Stage with Dynamic Roughness, or 3DEP Floodplain)**, **Single-Model Reach Error**, **NHDPlus Slope Regime ($S = 10^{-5}$ vs. $S > 10^{-5}$)**, **Data Availability Tier**, or **Predicted Manning's $\hat{n}$ and Dingman Shape $\hat{r}$**.
   - **Analytical Cross-Section ($\text{XS}$) Profile Inspector:** clicking any reach renders its observed Dingman bed profile $z(x) = -d_{\max}\left(1 - |2x/W|^r\right)$ against predicted profiles from all 10 benchmark models (`M2-XGB`, `M2-MLP`, `M1`, `M5`, `M4`, `M6-NN`, `M6-XGB`, `M3`, `M0`, `M-null`) under both **5-Fold Grouped CV** and **Regional Holdout** splits.
   - **90% Deep-Ensemble UQ Bands** (`M5` and `M2-MLP`) and docked **USGS 3DEP 10 m Lidar Valley Transects** across `3,350` valid CONUS gauges.
   - **Interactive Stage–Discharge Rating Curve Simulator:** flow ratio slider ($Q / Q_a \in [0.4, 10.0]$) comparing constant roughness ($n = n_a$) against flow-dependent dynamic roughness ($n(Q) = n_a (Q / Q_a)^{\gamma}$).

2. **Channel Geometry (`#view-geometry`)**
   - **Multi-Split Accuracy & Continuity Matrix:** interactive comparison across all 10 models, 6 validation splits (**Regional Holdout**, **5-Fold Grouped CV**, **USACE eHydro Surveys**, **Drainage Area Extremes**, **Large Rivers: Stream Order $\ge 7$**, and **Random 75:25 Split**), and 5 cross-section targets ($W$, $\bar{d}$, $d_{\max}$, $A$, and thalweg excess $t = d_{\max} - \bar{d}$).
   - **Key Findings & Input Sensitivity:** regional bootstrap comparisons ($B = 10,000$) and sensitivity of physics-constrained depth (`M5`) to NHDPlus slope clamping ($S = 10^{-5}$) and mean-discharge ($Q_E$) uncertainty.

3. **Stage Routing & 1D Hydraulics (`#view-operational`)**
   - **High-Flow Stage Error ($Q_i / Q_a \ge 2$):** comparison of in-channel power-law routing, 3DEP Lidar floodplain strip conveyance (`3,282` reaches), unclamped vs. clamped reach slopes, and dynamic high-flow roughness across `75,343` stage observations.
   - **1D Hydraulic Solver Readiness (`CCHE1D`) & Multi-Discharge Geometry:** evaluation of hydraulic radius $R_0 = A_0 / P_0$, continuity-consistent area $\hat{A}_{\text{cons}} = \hat{W}\hat{\bar{d}}$, 99th-percentile thalweg depth error $P_{99}(|\Delta d_{\max}|)$, and cross-section geometry across `31,367` multi-discharge field surveys.

4. **Data & Quality Control (`#view-provenance`)**
   - **Quality Control & Reach Selection Pipeline:** step-by-step filtering from `223,022` raw USGS HYDROSWOT ADCP records to `64,956` quality-controlled surveys, `3,988` primary CONUS benchmark gauges, `3,350` 3DEP Lidar valley profiles, and `131` USACE `eHydro` bathymetric reaches.
   - **Gauge Network Spatial & Basin Clustering:** nearest-neighbor distance, spatial variogram range, and nested-basin overlap metrics motivating basin-grouped cross-validation and physiographic regional holdouts.
   - **Public Hydrographic & Watershed Data Sources:** persistent DOIs, sample coverage, and open-access licenses (`CC0-1.0` / `CC-BY-4.0`) for all input datasets.

---

## 2. Directory Structure

```text
webgis/
├── index.html                   # Semantic HTML5 single-page shell (WCAG AA, dark/light themes)
├── styles.css                   # Responsive CSS Grid/Flexbox tokens, tables, and inspector styling
├── app.js                       # Vanilla ES2022 module (MapLibre GL JS v4, SVG XS engine, simulator)
├── README.md                    # Documentation and deployment instructions
└── data/                        # Deterministic static JSON/GeoJSON bundle
    ├── manifest.json            # SHA-256 checksums and canonical record counts
    ├── benchmarks.json          # Multi-split geometry, stage routing, CCHE1D & division metrics
    ├── physio_divisions.geojson # 8 simplified USGS physiographic division polygons (EPSG:4326)
    ├── sites_conus.json         # 3,988 CONUS stations with covariates, CV/regional predictions, UQ & 3DEP
    └── sites_ehydro.json        # 131 USACE eHydro locations with survey & predicted geometry
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
