# `ann-xs-estimation`: Hard Manning + Continuity Hybrid vs. Unconstrained Machine Learning for Reach-Scale River Cross-Section Geometry Across the Continental United States

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Data Capsule DOI](https://img.shields.io/badge/Data%20%26%20Run%20Capsule-10.5281%2Fzenodo.YYYYYYY-blue)](https://doi.org/10.5281/zenodo.YYYYYYY)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-0072B2.svg)](LICENSE)
[![Python 3.12.13](https://img.shields.io/badge/Python-3.12.13-3776AB.svg)](pyproject.toml)
[![S-SDLC Master Protocol](https://img.shields.io/badge/S--SDLC-Phases%200--11%20Complete-009E73.svg)](CITATION.cff)

---

## 1. Overview & Scientific Scope

Continental-scale hydrodynamic routing frameworks (including the NOAA National Water Model, USGS National Hydrologic Model, and satellite discharge inversions) require reach-scale river cross-sectional geometry—wetted top width $W$ ($\text{m}$), mean depth $\bar{d} \equiv A / W$ ($\text{m}$), maximum depth $d_{\max}$ ($\text{m}$), cross-sectional flow area $A$ ($\text{m}^2$), and Dingman shape exponent $r$ ($\text{dimensionless}$)—across millions of ungaged river reaches where direct bathymetric surveys are unavailable.

`ann-xs-estimation` is an end-to-end reproducible Scientific Software Development Life Cycle (S-SDLC) benchmark evaluating whether embedding **Dingman (2007) analytical channel shape geometry** and **Manning–Gauckler–Strickler open-channel flow conservation** via a hard differentiable Picard projection layer (`M5`, $\Phi$) improves reach-scale cross-section geometry estimation over unconstrained machine learning (`M2-XGB`, `M2-MLP`) and empirical power-law baselines (`M0`, `M1`, `A_J`) across the conterminous United States (CONUS).

### Physical Parameterization & Differentiable Projection Layer ($\Phi$)

Following Dingman (2007), symmetric power-law cross-sections on normalized lateral coordinate $\xi = 2x/W \in [-1, 1]$ with bed profile $z(\xi) = d_{\max} |\xi|^r$ satisfy the exact analytical cross-sectional area and mean-depth identities:

$$A(W, d_{\max}, r) = \frac{r}{r + 1} W d_{\max}, \qquad \bar{d} \equiv \frac{A}{W} = \frac{r}{r + 1} d_{\max}, \qquad r \in [0.50, 10.0]$$

Wetted perimeter $P(W, \bar{d}, r)$ and hydraulic radius $R = A / P = W \bar{d} \, \kappa(\bar{d}/W, r)$ are evaluated via a `513`-node **`tanh-sinh` (double-exponential) quadrature** rule (`quad_level = 6`, `quad_trunc = 4.0`) with custom first-order analytical autograd (`_LogKappa` and `ManningPhi` in `src/physics/`):

$$P(W, \bar{d}, r) = W \int_0^1 \sqrt{1 + \left(\frac{2 (r + 1) \bar{d}}{W}\right)^2 u^{2(r-1)}} \, du, \qquad Q_E = \frac{1}{n} W \bar{d}^{5/3} S^{1/2} \kappa(\bar{d}/W, r)^{2/3}$$

Given predicted width $\hat{W}$, shape exponent $\hat{r} \in [0.50, 10.0]$, and Manning roughness $\hat{n} \in [0.005, 0.60]$ (with literature plausibility range $n_{\text{lit}} = [0.012, 0.33]$), the hard projection operator $\Phi(\hat{W}, \hat{r}, \hat{n}; Q_E, S)$ solves the nonlinear Manning–Dingman closure for log mean depth $\ell = \ln \hat{\bar{d}}$ to Picard tolerance $\tau_\ell = 10^{-10}$ ($k_{\max} = 40$) and sets $\hat{A} \equiv \hat{W}\hat{\bar{d}}$ and $\hat{d}_{\max} \equiv \frac{\hat{r}+1}{\hat{r}}\hat{\bar{d}}$, guaranteeing exact continuity closure ($|\varepsilon_A| \equiv |\ln(\hat{A}/(\hat{W}\hat{\bar{d}}))| = 0$).

---

## 2. Model Ladder (`M-null`, `M0`, `M1`, `M2-MLP`, `M2-XGB`, `M3`, `M4`, `M5`, `M6-NN`, `M6-XGB`, `A_J`)

All neural models (`M2-MLP`, `M4`, `M5`, `M6-NN`) use $K = 5$ independent seed ensemble members (`float64` CPU execution, $L = 3$ hidden layers $\times$ $H = 128$ SiLU units, Gaussian-mixture predictive distribution) trained under identical AdamW schedules:

| Model ID | Architecture / Physical Inductive Bias | Input Feature Set | Continuity Closure ($\|\varepsilon_A\|$) | Manning Closure |
| :--- | :--- | :--- | :--- | :--- |
| **`M-null`** | Physiographic division mean baseline | Division indicator | Unconstrained | Unconstrained |
| **`M0`** | At-a-station / global power-law OLS hydraulic geometry | $\ln Q_E$ | Unconstrained | Unconstrained |
| **`M1`** | Multivariable regional log-OLS hydraulic geometry | $\mathcal{I}_{\text{full}}$ (28 predictors) | Unconstrained | Unconstrained |
| **`A_J`** | Jobson (2001) empirical area baseline | $(Q_E, S, DA)$ | Target $A$ only | Unconstrained |
| **`M2-MLP`** | Unconstrained multi-output deep ensemble MLP ($K = 5$ seeds) | $\mathcal{I}_{\text{full}}$ | Unconstrained | Unconstrained |
| **`M2-XGB`** | Unconstrained gradient-boosted trees (`XGBoost`, selected $M^*$) | $\mathcal{I}_{\text{full}}$ | Unconstrained | Unconstrained |
| **`M3`** | Post-hoc physics projection ($\Phi$ applied at inference to `M2` outputs) | $\mathcal{I}_{\text{full}}$ | **Exact ($0.0000$)** | Post-hoc $\Phi$ |
| **`M4`** | Soft-constrained PINN (Manning residual loss penalty $\lambda_{\text{phys}} \mathcal{R}_M$) | $\mathcal{I}_{\text{full}}$ | Soft penalty | Soft penalty ($\mathcal{R}_M$) |
| **`M5`** | Hard-constrained hybrid ($\mathcal{I}_{\text{blind}}$ roughness head + differentiable $\Phi$) | $\mathcal{I}_{\text{blind}}$ for $\hat{n}$ (excludes $Q_E, S, F_S, V_E$) | **Exact ($0.0000$)** | **Exact ($\le 10^{-10}$)** |
| **`M6-NN` / `M6-XGB`** | Full-input ($\mathcal{I}_{\text{full}}$) hard-projected neural / tree baselines | $\mathcal{I}_{\text{full}}$ for $(\hat{W}, \hat{r}, \hat{n})$ + $\Phi$ | **Exact ($0.0000$)** | **Exact ($\le 10^{-10}$)** |

---

## 3. Headline Empirical Results & Transparent Negative-Result Disclosure

Evaluation follows a strictly sealed, cryptographically hash-chained three-event scoring ledger (`reports/results/scoring_ledger.json`, 7 records):
- **`Event 1` (`audit`):** Spatial and stream-network dependence audit of the `S-ID` $3 \times 5$ grouped cross-validation split across $n = 3,988$ `HYDROSWOT` sites (`d_nn,med = 15.01 km`, geostatistical range $\hat{\rho} = \rho^* = 169.23\text{ km}$, nested strong $\Psi \ge 0.5$ share `0.4213`, weak share `0.4879`; mandatory audit label: **`optimistic under spatial/network dependence`**).
- **`Event 2` (`select_mstar`):** Pre-registered selection of the strongest unconstrained baseline $M^*$ among `{M1, M2-MLP, M2-XGB}` on `S-ID` out-of-fold $\text{RMSE}_{\log}$, locking **`M* = M2-XGB`** on both primary targets ($\text{RMSE}_{\log}(\bar{d}) = 0.4396$, $\text{RMSE}_{\log}(A) = 0.6254$).
- **`Event 3` (`evaluate_event3`):** Confirmatory evaluation across `S-LRO` (`8` held-out Fenneman & Johnson physiographic divisions, $n = 3,988$ sites), `S-IND` (`131` independent USACE eHydro surveyed locations across `18` rivers; UQ evaluated on `33` single-COMID locations), `S-TAIL-DA` ($n = 678$), `S-TAIL-SO` ($n = 339$), and `S-RND` ($n = 9,673$ records), with $B = 10,000$ cluster-bootstrap replicates (`reports/results/event3.json`, `docs/tables/benchmark_results.md`).

### 3.1 Primary Benchmark Metrics Across `S-ID`, `S-LRO`, and `S-IND` (`Table T2`)

| Split | Model | $\text{RMSE}_{\log}(\bar{d})$ | $\text{MdAPE}(\bar{d})$ (%) | $\text{RMSE}_{\log}(A)$ | $\text{MdAPE}(A)$ (%) | $\text{RMSE}_{\log}(W)$ | $\text{MdAPE}(W)$ (%) | $\|\varepsilon_A\|$ (median) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`S-ID`** ($n = 3,988$ sites, 3 repeats; *optimistic*) | **`M2-XGB` ($M^*$)** | **0.4396** | **28.47** | **0.6254** | **35.66** | **0.3667** | **21.45** | 0.0566 |
| | **`M2-MLP`** | 0.4520 | 29.56 | 0.6494 | 36.89 | 0.3866 | 23.07 | 0.0195 |
| | **`M1`** | 0.4745 | 32.98 | 0.7031 | 45.67 | 0.4172 | 26.04 | 0.0558 |
| | **`M4`** | 0.6805 | 38.09 | 0.9888 | 50.96 | 0.4715 | 25.16 | 0.1309 |
| | **`M5`** | 0.7036 | 39.93 | 0.9816 | 51.21 | 0.4419 | 25.71 | **0.0000** |
| | **`M6-NN`** | 0.8032 | 44.34 | 0.9229 | 48.88 | 0.3962 | 22.84 | **0.0000** |
| **`S-LRO`** ($n = 3,988$ sites, 8 held-out divisions) | **`M2-XGB` ($M^*$)** | **0.4782** | **29.67** | **0.6931** | **37.40** | **0.4258** | **25.23** | 0.0607 |
| | **`M2-MLP`** | 0.4866 | 30.93 | 0.7247 | 40.22 | 0.4465 | 26.67 | 0.0253 |
| | **`M1`** | 0.5016 | 34.95 | 0.7438 | 48.59 | 0.4565 | 29.59 | 0.0574 |
| | **`M4`** | 0.6864 | 39.55 | 1.0013 | 51.18 | 0.5104 | 28.17 | 0.1370 |
| | **`M5`** | 0.7284 | 42.16 | 1.0025 | 51.93 | 0.4743 | 28.13 | **0.0000** |
| | **`M6-NN`** | 0.8300 | 46.11 | 0.9556 | 50.63 | 0.4436 | 26.50 | **0.0000** |
| **`S-IND`** ($n = 131$ locations / 18 USACE eHydro rivers) | **`M1`** | **0.5450** | **27.48** | 0.9590 | 74.43 | 0.7380 | 54.24 | 0.0442 |
| | **`M4`** | 0.6078 | 36.21 | 0.9852 | 55.00 | 0.6682 | **34.47** | 0.2518 |
| | **`M2-XGB` ($M^*$)** | 0.6729 | 33.21 | 0.9839 | 62.71 | 0.6827 | 50.61 | 0.0470 |
| | **`M2-MLP`** | 0.7610 | 52.57 | 1.0993 | 65.66 | 0.6784 | 39.61 | 0.0176 |
| | **`M5`** | 0.8318 | 67.08 | **0.9495** | **44.32** | 0.8666 | 46.92 | **0.0000** |

### 3.2 Pre-Registered §4.7 Row-4 Outcome & Confirmatory Hypothesis Verdicts (`H1`–`H4`)

Per `docs/3_experimental_protocol.md` §4.7, the benchmark realized **Row 4 (`M5 worse (S-LRO CI lower bound > 0)`)**, triggering the mandatory pre-registered diagnostic protocol (`row 4: "Diagnose with the pre-registered ablations (A1, A3, M4, SA1, SA5) and report. No post-hoc headline switch: any new variant is labelled exploratory"`):

1. **`H1` (Paired Accuracy Contrast $\Delta_{\text{H1}} = \text{RMSE}_{\log}(\text{M5}) - \text{RMSE}_{\log}(M^*)$) — FALSIFIED:** On `S-LRO`, `M5` exhibits higher log-RMSE than `M2-XGB` ($M^*$) by $\Delta_{\text{H1}}(\bar{d}) = +0.2502$ ($95\%\text{ CI } [+0.2262, +0.2765]$) and $\Delta_{\text{H1}}(A) = +0.3094$ ($[+0.2728, +0.3520]$). On `S-IND` (`131 locations / 18 rivers`), $\Delta_{\text{H1}}(\bar{d}) = +0.1589$ ($[-0.2778, +0.3907]$) and $\Delta_{\text{H1}}(A) = -0.0343$ ($[-0.1588, +0.0746]$).
2. **`H2` (Headwater `Q1` Bias Reduction $|B_{\bar{d}}(\text{M5})(Q_1)| - |B_{\bar{d}}(M^*)(Q_1)|$) — FALSIFIED:** In the smallest drainage-area quintile $Q_1$ ($n = 798$ sites), $h_2 = +0.0299$ ($[-0.0335, +0.0740]$) on `S-ID` and $+0.0753$ ($[+0.0435, +0.1057]$) on `S-LRO`.
3. **`H3` (Out-of-Distribution Skill-Loss Reduction $\Delta_{\text{H3}} = L_{\text{OOD}}(\text{M5}) - L_{\text{OOD}}(M^*)$) — FALSIFIED:** While point skill loss $L_{\text{OOD}} = \text{RMSE}_{\text{OOD}} - \text{RMSE}_{\text{ID}}$ is smaller for `M5` than `M2-XGB` across `S-LRO` ($\Delta_{\text{H3}} = -0.0138$ $[-0.0494, +0.0209]$), `S-TAIL-DA` ($-0.0485$ $[-0.1750, +0.0759]$), and `S-TAIL-SO` ($-0.2088$ $[-0.4357, +0.0151]$), all 95% cluster-bootstrap CIs straddle zero.
4. **`H4` (Latent Roughness & Shape Parameter Plausibility in `M5`) — FALSIFIED:** While `M5` achieves literature-plausible Manning roughness on $92.65\%$ of `S-ID` sites ($\pi_{\text{lit}} = 0.9265 \ge 0.80$; `S-LRO` $\pi_{\text{lit}} = 0.9519$), negligible boundary saturation ($\pi_\nu = 0.0013 \le 0.10$, $\pi_r = 0.0000 \le 0.10$), and positive slope correlation ($\rho_S(\hat{n}, S) = 0.4431$ $[0.4181, 0.4674]$), the grain-size correlation fails ($\rho_S(\hat{n}, D_{50}) = -0.0203$ $[-0.0500, +0.0083]$).
5. **Pre-Registered Row-4 Mechanism Diagnosis (`Fig. 5`):** Wide-channel Manning elasticity analysis ($\partial \ln \bar{d} / \partial \ln S = -3/10$, $\partial \ln \bar{d} / \partial \ln W = -3/5$, $\partial \ln \bar{d} / \partial \ln n = +3/5$) and the pre-registered ablations demonstrate that forcing exact Manning inversion through map-derived NHDPlusV2.1 slopes (`S`, floored at $10^{-5}$ on $20.6\%$ of sites in `F_S`) and EROM mean-annual discharge (`Q_E`) propagates slope and discharge errors directly into $\hat{\bar{d}}$: replacing NHDPlus slope in `SA5` reduces `M5` `S-LRO` $\text{RMSE}_{\log}(\bar{d})$ from `0.7284` to `0.6067`, while the structural $\mathcal{I}_{\text{blind}}$ restriction in `M5` improves `S-LRO` $\text{RMSE}_{\log}(\bar{d})$ by $0.1016$ ($98.75\%\text{ Bonferroni CI } [-0.1110, -0.0930]$) over full-input `M6-NN` (`0.8300`).
6. **Physical Consistency & Deep-Ensemble UQ Calibration (`Fig. 3`–`Fig. 4`):** Unconstrained `M2-XGB` violates continuity closure ($|\varepsilon_A|$ median `0.0607` on `S-LRO`) and produces $5.52\%$ of predictions with no physically admissible back-calculated Manning roughness $n_{\text{back}} \in [0.005, 0.60]$, whereas `M5` guarantees exact continuity ($|\varepsilon_A| = 0.0000$), zero inadmissible roughness ($0.0000$), and Picard solver residual $\le 6.65 \times 10^{-11} \le \tau_\ell = 10^{-10}$. Empirical 90% deep-ensemble Gaussian-mixture intervals (`params.uq.conformal = false`) fall squarely inside the pre-registered **`[0.85, 0.95]`** calibration band on both `S-ID` (`M5 = 0.8997`, `M2-MLP = 0.8961`) and `S-LRO` (`M5 = 0.9025`, `M2-MLP = 0.8965`), and achieve `0.8182` on the `33` single-COMID `S-IND` locations.

---

## 4. Repository Architecture

```text
.
├── CITATION.cff                     # Citation File Format (v1.2.0) metadata
├── codemeta.json                    # CodeMeta v2.0 JSON-LD machine-readable metadata
├── LICENSE                          # BSD-3-Clause open-source software license
├── README.md                        # Comprehensive documentation & reproduction guide
├── Dockerfile                       # Multi-stage CPU-only container (base -> runtime -> check -> capsule)
├── pyproject.toml / uv.lock         # Locked Python 3.12 dependency specification (150 packages)
├── params.yaml                      # Single source of truth for scientific & training hyperparameters
├── dvc.yaml / dvc.lock              # End-to-end 11-stage DVC pipeline DAG & cryptographic lockfile
├── src/
│   ├── config.py                    # Strict Pydantic v2 configuration schemas & physical constants
│   ├── data/                        # ETL, QA/QC (R1-R10), NHDPlusV2.1/StreamCat/Abeshu linking, splits, scalers
│   ├── physics/                     # Differentiable Dingman shape, 513-node tanh-sinh quadrature, ManningPhi
│   ├── models/                      # M-null, M0, M1, M2-XGB, M2-MLP..M6-NN PyTorch architectures, orchestrator
│   └── evaluation/                  # Metrics, cluster bootstrap, spatial LISA/variograms, UQ, hash ledger
├── figures/                         # Isolated publication figure (Fig 1-5, S1-S5) & table (T1-T9) subproject
├── scripts/
│   ├── dev_env.sh                   # Redirects venv/caches outside cloud-synced folders; sets deterministic env
│   ├── ci_local.sh                  # Local mirror of 7-stage CI gate (ruff, mypy, pytest, coverage gates)
│   └── container_check.sh           # Builds & verifies rootless, offline Docker capsule reproducibility
├── reports/
│   ├── results/                     # Committed Event 1-3 JSON/Parquet outputs & 7-event scoring_ledger.json
│   ├── training/                    # Training summary, learning curves, and M4 Xi_e penalty trajectories
│   ├── run_manifest.json            # Cryptographic Merkle root (51,508 leaves) of the Phase 8 training grid
│   ├── container_check.json         # Container capsule verification digest & parity report
│   └── archival_manifest.json       # Phase 11 FAIR archival manifest (SHA-256 inventory & run-root tar digest)
└── tests/
    ├── unit/                        # Unit, schema, CI, container, Phase 10 audit, and Phase 11 archival tests
    ├── physics/                     # Property-based (Hypothesis) & mpmath analytical physical invariant tests
    └── integration/                 # End-to-end pipeline, training loop, and ledger integration tests
```

---

## 5. Hardware & Software Reproduction Guide

### 5.1 Reference Compute Environment
- **Reference Host:** Apple M3 Max (`arm64`, 16 CPU cores, 128 GB unified RAM, macOS) and Linux (`x86_64` / `aarch64` container runtime).
- **Execution Mode:** 100% deterministic CPU `float64` execution (`torch.use_deterministic_algorithms(True)`, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`). No discrete GPU is required.

### 5.2 Option A — Local Reproduction (`uv` + `DVC`)

```bash
# 1. Configure deterministic environment & externalize virtualenvs/caches
source scripts/dev_env.sh

# 2. Materialize locked Python 3.12 environment (150 packages)
uv sync --frozen --all-groups

# 3. Execute the full local CI gate (ruff, mypy, 1,010+ unit/physics tests, per-package coverage gates)
bash scripts/ci_local.sh

# 4. Run integration tests and publication figure/table tests (36 tests)
uv run pytest -m integration -q
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/ANN_XS_Estimation_figures" \
  uv run --project figures --frozen pytest figures/tests -q

# 5. Verify DVC pipeline lock status or reproduce stages from cached processed data
uv run dvc status
# Optional full pipeline reproduction:
# uv run dvc repro

# 6. Regenerate all 10 publication figures (Fig 1-5, Fig S1-S5 in PDF/SVG/600-DPI PNG) and Tables T1-T9
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/ANN_XS_Estimation_figures" \
  uv run --project figures --frozen python -m annxs_figures.render_all
```

### 5.3 Option B — Hermetic Container Capsule (`Docker`)

The multi-stage [`Dockerfile`](Dockerfile) (`python:3.12.13-slim-bookworm` pinned by SHA-256 manifest digest) builds four targets: `base`, `runtime`, `check`, and `capsule` (non-root user `appuser`, `uid=10001`).

```bash
# Build runtime and self-contained evaluation capsule images
docker build --target runtime -t ann-xs:runtime .
docker build --target capsule -t ann-xs:capsule .

# Run end-to-end hermetic container verification (--network=none, --read-only, uid=10001)
bash scripts/container_check.sh
```

---

## 6. Third-Party Data Provenance & License Matrix

All primary hydrographic, topographic, substrate, and climatic inputs derive from open scientific and U.S. Federal repositories:

| Dataset | Agency / Source | Persistent Identifier / URL | Spatial / Temporal Scope | License / Terms | DVC Stage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HYDRoSWOT** | USGS (Canova et al., 2016) | `https://doi.org/10.5066/F7D798H6` | CONUS ADCP & current-meter survey database (`n = 3,988` sites, `n = 9,673` `S-RND` records) | `CC0-1.0` / U.S. Public Domain (`CC-BY-4.0` attribution requested) | `acquire`, `clean` |
| **NHDPlusV2.1** | U.S. EPA & USGS (McKay et al., 2012; Wieczorek et al., 2018) | `https://doi.org/10.5066/F7P84B24` | 1:100,000 national hydrography (`COMID`, `QE_MA`, `VE_MA`, `SLOPE`, `TOTDASQKM`) | `CC0-1.0` (U.S. Government Public Domain) | `acquire`, `link` |
| **EPA StreamCat** | U.S. EPA (Hill et al., 2016) | `https://doi.org/10.1111/1752-1688.12372` | Catchment & watershed soil, lithology, climate, land cover, and dam regulation metrics | `CC0-1.0` (U.S. Government Public Domain) | `acquire`, `features` |
| **CONUS $D_{50}$** | Abeshu et al. (2022) | `https://doi.org/10.5281/zenodo.5550700` | Reach-scale median bed-material grain size $D_{50}$ ($\text{mm}$) across CONUS NHDPlusV2 reaches | `CC-BY-4.0` | `acquire`, `features` |
| **Global Aridity Index v3** | Zomer et al. (2022) | `https://doi.org/10.6084/m9.figshare.7504448.v5` | High-resolution global aridity index (`AI`) & potential evapotranspiration (`PET`) | `CC-BY-4.0` | `acquire`, `features` |
| **USGS Physiographic Divisions** | Fenneman & Johnson (1946) / USGS | `https://water.usgs.gov/GIS/metadata/usgswrd/XML/physio.xml` | `8` CONUS physiographic divisions defining `S-LRO` regional holdout blocks | `CC0-1.0` (U.S. Government Public Domain) | `acquire`, `split` |
| **HyG / US-SWD (`S1`)** | Enzminger et al. (2023, 2024) | `https://doi.org/10.5281/zenodo.10035344` | Derived Manning roughness $n$ consistency benchmark (`2,887` matched sites) | `CC-BY-4.0` | `acquire`, `evaluate_event3` |
| **USACE eHydro (`S-IND`)** | U.S. Army Corps of Engineers | `https://navigation.usace.army.mil/Survey/Hydro` | `131` surveyed navigation channel locations across `18` rivers (`33` single-COMID UQ rows) | `CC0-1.0` (U.S. Government Public Domain) | `acquire`, `ind_prepare` |

---

## 7. ASCE Data, Models, and Code Availability Statement

> **Data, Models, and Code Availability Statement (ASCE *Journal of Hydrologic Engineering*):**  
> All source code, physics-informed neural network and baseline implementations (`M-null`, `M0`, `M1`, `M2-MLP`, `M2-XGB`, `M3`, `M4`, `M5`, `M6-NN`, `M6-XGB`, and `A_J`), DVC pipeline specifications (`dvc.yaml`, `dvc.lock`, `params.yaml`), container build definitions (`Dockerfile`), publication figure/table rendering scripts (`figures/`), and the cryptographically hash-chained evaluation ledger (`reports/results/scoring_ledger.json`) are openly available under the BSD-3-Clause license at `https://github.com/ansahin/ann-xs-estimation` and permanently archived on Zenodo at `https://doi.org/10.5281/zenodo.XXXXXXX` (tag `v1.0.0`). Processed analysis-ready datasets (`data/processed/site_table.parquet`, `fold_assignment.parquet`, `srnd_assignment.parquet`, `ehydro_ind.parquet`), out-of-fold prediction matrices (`reports/results/event3_matrix.parquet`), and the deterministic POSIX `ustar` archive of all trained model checkpoints (`p8-da9b7303cb84.tar`, SHA-256 `a6a88f772aa864ed24cadcc380aeb85831c02dd0e28473143ce9905e450db4b8`, Merkle root `bb97203df74599de568bee5316d4b147d1b7dea770220d7917da464b937a1fb2`) are archived at `https://doi.org/10.5281/zenodo.YYYYYYY`. All raw observational inputs derive from publicly accessible repositories (USGS HYDRoSWOT `https://doi.org/10.5066/F7D798H6`, NHDPlusV2.1 `https://doi.org/10.5066/F7P84B24`, EPA StreamCat, Abeshu et al. $D_{50}$ `https://doi.org/10.5281/zenodo.5550700`, Global Aridity Index v3 `https://doi.org/10.6084/m9.figshare.7504448.v5`, Enzminger et al. HyG `https://doi.org/10.5281/zenodo.10035344`, and USACE eHydro).

---

## 8. How to Cite

### 8.1 Companion Journal Manuscript (Preferred Citation)

**APA (7th Edition):**
> Sahin, A. N., Yasarer, H., Ozeren, Y., Langendoen, E. J., Al-Hamdan, M. Z., & D'Alessio, M. (2026). *Physics-informed versus unconstrained machine learning for reach-scale river cross-sectional geometry across the continental United States: A pre-registered multi-regime benchmark*. Journal of Hydrologic Engineering (ASCE), in preparation.

**BibTeX:**
```bibtex
@article{Sahin2026_JHE_ANN_XS,
  author  = {Sahin, Ahmet Nazim and Yasarer, Hakan and Ozeren, Yavuz and Langendoen, Eddy J. and Al-Hamdan, Mohammad Z. and D'Alessio, Matteo},
  title   = {Physics-Informed Versus Unconstrained Machine Learning for Reach-Scale River Cross-Sectional Geometry Across the Continental {United States}: A Pre-Registered Multi-Regime Benchmark},
  journal = {Journal of Hydrologic Engineering},
  year    = {2026},
  note    = {American Society of Civil Engineers (ASCE), in preparation. Supported by USDA-ARS Cooperative Agreement No. 6060-13000-030-00D}
}
```

### 8.2 Software Release Citation

**APA (7th Edition):**
> Sahin, A. N., Yasarer, H., Ozeren, Y., Langendoen, E. J., Al-Hamdan, M. Z., & D'Alessio, M. (2026). *ann-xs-estimation: Hard Manning + continuity hybrid versus unconstrained machine learning for reach-scale river cross-section geometry across the continental United States* (Version 0.1.0 / Tag v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

**BibTeX:**
```bibtex
@software{Sahin2026_Zenodo_ANN_XS_Software,
  author    = {Sahin, Ahmet Nazim and Yasarer, Hakan and Ozeren, Yavuz and Langendoen, Eddy J. and Al-Hamdan, Mohammad Z. and D'Alessio, Matteo},
  title     = {{ann-xs-estimation}: Hard Manning + Continuity Hybrid Versus Unconstrained Machine Learning for Reach-Scale River Cross-Section Geometry Across the Continental {United States}},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX},
  license   = {BSD-3-Clause}
}
```

---

## 9. Acknowledgements & Funding

This research was supported by the **United States Department of Agriculture – Agricultural Research Service (USDA-ARS)** under **Cooperative Agreement No. `6060-13000-030-00D`** between the **USDA-ARS National Sedimentation Laboratory (Oxford, MS)** and the **National Center for Computational Hydroscience and Engineering (NCCHE), University of Mississippi**. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the official views or policies of the USDA-ARS or the University of Mississippi.
