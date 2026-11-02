# OPS-1 v3 Operational Diagnostics, Dynamic-Resistance & CCHE1D Readiness Benchmark — results

- git commit: `9d821f091e0179ce4503452d895aa5b500970f9a` (dirty: False)
- params_operational_v3.yaml SHA-256: `3c2cdd15a8cc80e3255fc234d36c6d4d234eb93b2e3f20c592e67594de5e1fda`
- inherited params_operational_v2.yaml SHA-256: `4923cd32d6a613ae09d591343fc3d2b92cdd75a5c52b8f214023ec62b6945e68`
- inherited params_operational.yaml SHA-256: `839237d3d385bc9a54353bac8a9485cfd48118e80ffc33b8c86aec6ef30fc205`

## 1. S-ID All-Model OPS-1 v2 Diagnostics (total = 2,842, not_F_S = 2,264, F_S = 578)

| Variant | Regime | Bin | Model | Sites | RMSE Δh (m) | Bias Δh (m) | Median Site RMSE (m) | Skill vs M0 |
|---|---|---|---|---:|---:|---:|---:|---:|
| control | all | high | M-null | 2391 | 1.632 | +0.773 | 0.795 | +0.129 |
| control | all | high | M0 | 2391 | 1.874 | +0.948 | 0.887 | +0.000 |
| control | all | high | M1 | 2391 | 2.118 | +1.215 | 1.061 | -0.130 |
| control | all | high | M2-MLP | 2391 | 1.779 | +0.945 | 0.824 | +0.051 |
| control | all | high | M2-XGB | 2391 | 1.760 | +0.917 | 0.807 | +0.061 |
| control | all | high | M3 | 2391 | 1.503 | +0.011 | 0.547 | +0.198 |
| control | all | high | M4 | 2391 | 2.935 | +1.394 | 0.880 | -0.566 |
| control | all | high | M5 | 2391 | 3.062 | +1.398 | 0.885 | -0.634 |
| control | all | high | M6-NN | 2391 | 3.804 | +1.554 | 0.878 | -1.029 |
| control | all | high | M6-XGB | 2391 | 3.488 | +1.498 | 0.813 | -0.861 |
| control | all | all_flows | M-null | 2842 | 1.211 | +0.299 | 0.479 | +0.094 |
| control | all | all_flows | M0 | 2842 | 1.337 | +0.331 | 0.513 | +0.000 |
| control | all | all_flows | M1 | 2842 | 1.501 | +0.368 | 0.594 | -0.122 |
| control | all | all_flows | M2-MLP | 2842 | 1.274 | +0.290 | 0.494 | +0.047 |
| control | all | all_flows | M2-XGB | 2842 | 1.265 | +0.286 | 0.480 | +0.054 |
| control | all | all_flows | M3 | 2842 | 1.077 | +0.005 | 0.351 | +0.194 |
| control | all | all_flows | M4 | 2842 | 1.980 | +0.387 | 0.523 | -0.481 |
| control | all | all_flows | M5 | 2842 | 2.067 | +0.363 | 0.538 | -0.546 |
| control | all | all_flows | M6-NN | 2842 | 2.570 | +0.384 | 0.528 | -0.922 |
| control | all | all_flows | M6-XGB | 2842 | 2.312 | +0.389 | 0.496 | -0.729 |
| floodplain | all | high | M-null | 2391 | 1.407 | +0.566 | 0.640 | +0.045 |
| floodplain | all | high | M0 | 2391 | 1.473 | +0.688 | 0.702 | +0.000 |
| floodplain | all | high | M1 | 2391 | 1.591 | +0.905 | 0.823 | -0.080 |
| floodplain | all | high | M2-MLP | 2391 | 1.422 | +0.704 | 0.657 | +0.034 |
| floodplain | all | high | M2-XGB | 2391 | 1.407 | +0.691 | 0.647 | +0.045 |
| floodplain | all | high | M3 | 2391 | 1.383 | +0.052 | 0.536 | +0.061 |
| floodplain | all | high | M4 | 2391 | 2.004 | +0.983 | 0.730 | -0.361 |
| floodplain | all | high | M5 | 2391 | 2.020 | +0.975 | 0.734 | -0.372 |
| floodplain | all | high | M6-NN | 2391 | 2.119 | +0.967 | 0.719 | -0.439 |
| floodplain | all | high | M6-XGB | 2391 | 2.123 | +0.975 | 0.696 | -0.441 |
| floodplain | all | all_flows | M-null | 2842 | 1.050 | +0.209 | 0.421 | +0.036 |
| floodplain | all | all_flows | M0 | 2842 | 1.089 | +0.225 | 0.451 | +0.000 |
| floodplain | all | all_flows | M1 | 2842 | 1.176 | +0.240 | 0.522 | -0.080 |
| floodplain | all | all_flows | M2-MLP | 2842 | 1.064 | +0.191 | 0.432 | +0.022 |
| floodplain | all | all_flows | M2-XGB | 2842 | 1.052 | +0.190 | 0.429 | +0.034 |
| floodplain | all | all_flows | M3 | 2842 | 1.028 | +0.024 | 0.352 | +0.056 |
| floodplain | all | all_flows | M4 | 2842 | 1.413 | +0.230 | 0.466 | -0.297 |
| floodplain | all | all_flows | M5 | 2842 | 1.494 | +0.207 | 0.472 | -0.372 |
| floodplain | all | all_flows | M6-NN | 2842 | 1.713 | +0.174 | 0.464 | -0.573 |
| floodplain | all | all_flows | M6-XGB | 2842 | 1.533 | +0.195 | 0.448 | -0.408 |
| floodplain | not_F_S | high | M-null | 1892 | 1.393 | +0.642 | 0.629 | +0.051 |
| floodplain | not_F_S | high | M0 | 1892 | 1.467 | +0.752 | 0.705 | +0.000 |
| floodplain | not_F_S | high | M1 | 1892 | 1.572 | +0.915 | 0.813 | -0.071 |
| floodplain | not_F_S | high | M2-MLP | 1892 | 1.420 | +0.729 | 0.646 | +0.032 |
| floodplain | not_F_S | high | M2-XGB | 1892 | 1.400 | +0.712 | 0.634 | +0.046 |
| floodplain | not_F_S | high | M3 | 1892 | 1.209 | -0.204 | 0.459 | +0.176 |
| floodplain | not_F_S | high | M4 | 1892 | 1.321 | +0.541 | 0.561 | +0.100 |
| floodplain | not_F_S | high | M5 | 1892 | 1.333 | +0.530 | 0.567 | +0.092 |
| floodplain | not_F_S | high | M6-NN | 1892 | 1.347 | +0.464 | 0.540 | +0.082 |
| floodplain | not_F_S | high | M6-XGB | 1892 | 1.344 | +0.469 | 0.534 | +0.084 |
| floodplain | not_F_S | all_flows | M-null | 2264 | 1.064 | +0.230 | 0.412 | +0.041 |
| floodplain | not_F_S | all_flows | M0 | 2264 | 1.110 | +0.249 | 0.441 | +0.000 |
| floodplain | not_F_S | all_flows | M1 | 2264 | 1.180 | +0.262 | 0.509 | -0.064 |
| floodplain | not_F_S | all_flows | M2-MLP | 2264 | 1.080 | +0.212 | 0.417 | +0.026 |
| floodplain | not_F_S | all_flows | M2-XGB | 2264 | 1.066 | +0.210 | 0.410 | +0.039 |
| floodplain | not_F_S | all_flows | M3 | 2264 | 0.933 | +0.000 | 0.287 | +0.159 |
| floodplain | not_F_S | all_flows | M4 | 2264 | 1.015 | +0.190 | 0.372 | +0.086 |
| floodplain | not_F_S | all_flows | M5 | 2264 | 1.040 | +0.182 | 0.361 | +0.062 |
| floodplain | not_F_S | all_flows | M6-NN | 2264 | 1.091 | +0.165 | 0.345 | +0.017 |
| floodplain | not_F_S | all_flows | M6-XGB | 2264 | 1.033 | +0.171 | 0.334 | +0.069 |
| floodplain | F_S | high | M-null | 499 | 1.459 | +0.279 | 0.680 | +0.023 |
| floodplain | F_S | high | M0 | 499 | 1.493 | +0.445 | 0.689 | +0.000 |
| floodplain | F_S | high | M1 | 499 | 1.664 | +0.868 | 0.876 | -0.114 |
| floodplain | F_S | high | M2-MLP | 499 | 1.430 | +0.609 | 0.702 | +0.042 |
| floodplain | F_S | high | M2-XGB | 499 | 1.431 | +0.612 | 0.690 | +0.042 |
| floodplain | F_S | high | M3 | 499 | 1.903 | +1.026 | 1.026 | -0.274 |
| floodplain | F_S | high | M4 | 499 | 3.553 | +2.659 | 2.365 | -1.379 |
| floodplain | F_S | high | M5 | 499 | 3.580 | +2.661 | 2.256 | -1.397 |
| floodplain | F_S | high | M6-NN | 499 | 3.825 | +2.870 | 2.555 | -1.562 |
| floodplain | F_S | high | M6-XGB | 499 | 3.840 | +2.895 | 2.672 | -1.571 |
| floodplain | F_S | all_flows | M-null | 578 | 0.992 | +0.126 | 0.457 | +0.011 |
| floodplain | F_S | all_flows | M0 | 578 | 1.003 | +0.129 | 0.477 | +0.000 |
| floodplain | F_S | all_flows | M1 | 578 | 1.161 | +0.155 | 0.593 | -0.157 |
| floodplain | F_S | all_flows | M2-MLP | 578 | 1.000 | +0.106 | 0.502 | +0.003 |
| floodplain | F_S | all_flows | M2-XGB | 578 | 0.993 | +0.110 | 0.478 | +0.010 |
| floodplain | F_S | all_flows | M3 | 578 | 1.336 | +0.116 | 0.732 | -0.332 |
| floodplain | F_S | all_flows | M4 | 578 | 2.404 | +0.389 | 1.640 | -1.397 |
| floodplain | F_S | all_flows | M5 | 578 | 2.596 | +0.305 | 1.614 | -1.588 |
| floodplain | F_S | all_flows | M6-NN | 578 | 3.126 | +0.210 | 1.789 | -2.116 |
| floodplain | F_S | all_flows | M6-XGB | 578 | 2.716 | +0.287 | 1.830 | -1.707 |

Statistical equivalence cliques (95% paired cluster bootstrap):

- `floodplain / all / high`: leader `M3` (1.383 m), top clique `[M3, M2-XGB, M-null, M2-MLP]`
- `floodplain / all / all_flows`: leader `M3` (1.028 m), top clique `[M3, M-null, M2-XGB, M2-MLP, M0]`
- `floodplain / all / [0,0.5)`: leader `M0` (0.805 m), top clique `[M0, M-null]`
- `floodplain / all / [0.5,2)`: leader `M0` (0.292 m), top clique `[M0, M2-XGB, M-null, M3]`
- `floodplain / not_F_S / high`: leader `M3` (1.209 m), top clique `[M3]`
- `floodplain / not_F_S / all_flows`: leader `M3` (0.933 m), top clique `[M3]`
- `floodplain / not_F_S / [0,0.5)`: leader `M3` (0.803 m), top clique `[M3, M4, M-null, M0, M2-MLP]`
- `floodplain / not_F_S / [0.5,2)`: leader `M3` (0.229 m), top clique `[M3]`
- `floodplain / F_S / high`: leader `M2-MLP` (1.430 m), top clique `[M2-MLP, M2-XGB, M-null, M0]`
- `floodplain / F_S / all_flows`: leader `M-null` (0.992 m), top clique `[M-null, M2-XGB, M2-MLP, M0]`
- `floodplain / F_S / [0,0.5)`: leader `M0` (0.728 m), top clique `[M0, M-null]`
- `floodplain / F_S / [0.5,2)`: leader `M0` (0.383 m), top clique `[M0, M2-XGB, M-null, M2-MLP]`

- **H-V3-1 (S-ID)**: supported (superior) — `not_F_S` M5 - M2-XGB = -0.0676 m [-0.1120, -0.0256]; `F_S` M5 - M2-XGB = +2.1487 m [+1.9338, +2.3751]

## 1. S-LRO All-Model OPS-1 v2 Diagnostics (total = 2,843, not_F_S = 2,264, F_S = 579)

| Variant | Regime | Bin | Model | Sites | RMSE Δh (m) | Bias Δh (m) | Median Site RMSE (m) | Skill vs M0 |
|---|---|---|---|---:|---:|---:|---:|---:|
| control | all | high | M-null | 2392 | 1.640 | +0.771 | 0.798 | +0.109 |
| control | all | high | M0 | 2392 | 1.841 | +0.885 | 0.872 | +0.000 |
| control | all | high | M1 | 2392 | 2.143 | +1.229 | 1.056 | -0.164 |
| control | all | high | M2-MLP | 2392 | 1.937 | +0.951 | 0.830 | -0.052 |
| control | all | high | M2-XGB | 2392 | 1.755 | +0.851 | 0.781 | +0.047 |
| control | all | high | M3 | 2392 | 1.526 | +0.047 | 0.543 | +0.171 |
| control | all | high | M4 | 2392 | 2.954 | +1.428 | 0.924 | -0.604 |
| control | all | high | M5 | 2392 | 3.164 | +1.427 | 0.918 | -0.718 |
| control | all | high | M6-NN | 2392 | 4.270 | +1.829 | 0.970 | -1.320 |
| control | all | high | M6-XGB | 2392 | 3.756 | +1.673 | 0.883 | -1.040 |
| control | all | all_flows | M-null | 2843 | 1.221 | +0.301 | 0.477 | +0.096 |
| control | all | all_flows | M0 | 2843 | 1.351 | +0.340 | 0.508 | +0.000 |
| control | all | all_flows | M1 | 2843 | 1.500 | +0.363 | 0.597 | -0.111 |
| control | all | all_flows | M2-MLP | 2843 | 1.329 | +0.277 | 0.493 | +0.016 |
| control | all | all_flows | M2-XGB | 2843 | 1.229 | +0.259 | 0.459 | +0.090 |
| control | all | all_flows | M3 | 2843 | 1.088 | +0.019 | 0.350 | +0.195 |
| control | all | all_flows | M4 | 2843 | 1.967 | +0.386 | 0.550 | -0.456 |
| control | all | all_flows | M5 | 2843 | 2.119 | +0.357 | 0.547 | -0.568 |
| control | all | all_flows | M6-NN | 2843 | 2.828 | +0.425 | 0.573 | -1.093 |
| control | all | all_flows | M6-XGB | 2843 | 2.487 | +0.402 | 0.535 | -0.841 |
| floodplain | all | high | M-null | 2392 | 1.412 | +0.559 | 0.639 | +0.020 |
| floodplain | all | high | M0 | 2392 | 1.441 | +0.620 | 0.680 | +0.000 |
| floodplain | all | high | M1 | 2392 | 1.604 | +0.906 | 0.817 | -0.113 |
| floodplain | all | high | M2-MLP | 2392 | 1.501 | +0.685 | 0.643 | -0.042 |
| floodplain | all | high | M2-XGB | 2392 | 1.380 | +0.623 | 0.624 | +0.042 |
| floodplain | all | high | M3 | 2392 | 1.382 | +0.068 | 0.528 | +0.041 |
| floodplain | all | high | M4 | 2392 | 1.990 | +0.982 | 0.731 | -0.381 |
| floodplain | all | high | M5 | 2392 | 2.063 | +0.979 | 0.748 | -0.432 |
| floodplain | all | high | M6-NN | 2392 | 2.292 | +1.101 | 0.775 | -0.591 |
| floodplain | all | high | M6-XGB | 2392 | 2.186 | +1.046 | 0.716 | -0.517 |
| floodplain | all | all_flows | M-null | 2843 | 1.055 | +0.209 | 0.419 | +0.019 |
| floodplain | all | all_flows | M0 | 2843 | 1.075 | +0.230 | 0.434 | +0.000 |
| floodplain | all | all_flows | M1 | 2843 | 1.184 | +0.234 | 0.521 | -0.101 |
| floodplain | all | all_flows | M2-MLP | 2843 | 1.102 | +0.177 | 0.422 | -0.025 |
| floodplain | all | all_flows | M2-XGB | 2843 | 1.031 | +0.169 | 0.415 | +0.042 |
| floodplain | all | all_flows | M3 | 2843 | 1.030 | +0.031 | 0.348 | +0.042 |
| floodplain | all | all_flows | M4 | 2843 | 1.403 | +0.223 | 0.475 | -0.305 |
| floodplain | all | all_flows | M5 | 2843 | 1.528 | +0.198 | 0.481 | -0.421 |
| floodplain | all | all_flows | M6-NN | 2843 | 1.824 | +0.177 | 0.504 | -0.696 |
| floodplain | all | all_flows | M6-XGB | 2843 | 1.608 | +0.178 | 0.474 | -0.495 |
| floodplain | not_F_S | high | M-null | 1892 | 1.397 | +0.633 | 0.631 | +0.031 |
| floodplain | not_F_S | high | M0 | 1892 | 1.442 | +0.696 | 0.675 | +0.000 |
| floodplain | not_F_S | high | M1 | 1892 | 1.577 | +0.915 | 0.801 | -0.094 |
| floodplain | not_F_S | high | M2-MLP | 1892 | 1.495 | +0.720 | 0.628 | -0.037 |
| floodplain | not_F_S | high | M2-XGB | 1892 | 1.364 | +0.648 | 0.601 | +0.055 |
| floodplain | not_F_S | high | M3 | 1892 | 1.208 | -0.187 | 0.453 | +0.163 |
| floodplain | not_F_S | high | M4 | 1892 | 1.334 | +0.556 | 0.565 | +0.075 |
| floodplain | not_F_S | high | M5 | 1892 | 1.358 | +0.531 | 0.575 | +0.058 |
| floodplain | not_F_S | high | M6-NN | 1892 | 1.411 | +0.557 | 0.571 | +0.021 |
| floodplain | not_F_S | high | M6-XGB | 1892 | 1.353 | +0.514 | 0.536 | +0.062 |
| floodplain | not_F_S | all_flows | M-null | 2264 | 1.068 | +0.229 | 0.408 | +0.029 |
| floodplain | not_F_S | all_flows | M0 | 2264 | 1.100 | +0.251 | 0.426 | +0.000 |
| floodplain | not_F_S | all_flows | M1 | 2264 | 1.184 | +0.256 | 0.501 | -0.076 |
| floodplain | not_F_S | all_flows | M2-MLP | 2264 | 1.113 | +0.194 | 0.410 | -0.011 |
| floodplain | not_F_S | all_flows | M2-XGB | 2264 | 1.039 | +0.186 | 0.395 | +0.055 |
| floodplain | not_F_S | all_flows | M3 | 2264 | 0.935 | +0.006 | 0.290 | +0.150 |
| floodplain | not_F_S | all_flows | M4 | 2264 | 1.020 | +0.185 | 0.362 | +0.073 |
| floodplain | not_F_S | all_flows | M5 | 2264 | 1.053 | +0.173 | 0.369 | +0.043 |
| floodplain | not_F_S | all_flows | M6-NN | 2264 | 1.116 | +0.170 | 0.371 | -0.015 |
| floodplain | not_F_S | all_flows | M6-XGB | 2264 | 1.056 | +0.164 | 0.356 | +0.040 |
| floodplain | F_S | high | M-null | 500 | 1.470 | +0.281 | 0.668 | -0.023 |
| floodplain | F_S | high | M0 | 500 | 1.436 | +0.335 | 0.690 | +0.000 |
| floodplain | F_S | high | M1 | 500 | 1.702 | +0.874 | 0.873 | -0.185 |
| floodplain | F_S | high | M2-MLP | 500 | 1.524 | +0.555 | 0.700 | -0.061 |
| floodplain | F_S | high | M2-XGB | 500 | 1.440 | +0.529 | 0.708 | -0.002 |
| floodplain | F_S | high | M3 | 500 | 1.900 | +1.034 | 1.051 | -0.323 |
| floodplain | F_S | high | M4 | 500 | 3.496 | +2.595 | 2.309 | -1.434 |
| floodplain | F_S | high | M5 | 500 | 3.658 | +2.678 | 2.255 | -1.547 |
| floodplain | F_S | high | M6-NN | 500 | 4.195 | +3.157 | 2.590 | -1.921 |
| floodplain | F_S | high | M6-XGB | 500 | 3.993 | +3.057 | 2.646 | -1.780 |
| floodplain | F_S | all_flows | M-null | 579 | 1.003 | +0.127 | 0.466 | -0.033 |
| floodplain | F_S | all_flows | M0 | 579 | 0.971 | +0.149 | 0.476 | +0.000 |
| floodplain | F_S | all_flows | M1 | 579 | 1.185 | +0.151 | 0.604 | -0.220 |
| floodplain | F_S | all_flows | M2-MLP | 579 | 1.058 | +0.108 | 0.476 | -0.089 |
| floodplain | F_S | all_flows | M2-XGB | 579 | 0.996 | +0.102 | 0.478 | -0.025 |
| floodplain | F_S | all_flows | M3 | 579 | 1.339 | +0.125 | 0.711 | -0.379 |
| floodplain | F_S | all_flows | M4 | 579 | 2.368 | +0.372 | 1.573 | -1.437 |
| floodplain | F_S | all_flows | M5 | 579 | 2.669 | +0.295 | 1.546 | -1.748 |
| floodplain | F_S | all_flows | M6-NN | 579 | 3.385 | +0.203 | 1.978 | -2.484 |
| floodplain | F_S | all_flows | M6-XGB | 579 | 2.887 | +0.230 | 1.992 | -1.972 |

Statistical equivalence cliques (95% paired cluster bootstrap):

- `floodplain / all / high`: leader `M2-XGB` (1.380 m), top clique `[M2-XGB, M3, M-null]`
- `floodplain / all / all_flows`: leader `M3` (1.030 m), top clique `[M3, M2-XGB, M-null, M0]`
- `floodplain / all / [0,0.5)`: leader `M0` (0.797 m), top clique `[M0]`
- `floodplain / all / [0.5,2)`: leader `M0` (0.295 m), top clique `[M0, M2-XGB, M3]`
- `floodplain / not_F_S / high`: leader `M3` (1.208 m), top clique `[M3]`
- `floodplain / not_F_S / all_flows`: leader `M3` (0.935 m), top clique `[M3]`
- `floodplain / not_F_S / [0,0.5)`: leader `M3` (0.809 m), top clique `[M3, M0, M-null, M4]`
- `floodplain / not_F_S / [0.5,2)`: leader `M3` (0.231 m), top clique `[M3]`
- `floodplain / F_S / high`: leader `M0` (1.436 m), top clique `[M0, M2-XGB, M-null, M2-MLP]`
- `floodplain / F_S / all_flows`: leader `M0` (0.971 m), top clique `[M0, M2-XGB, M-null, M2-MLP]`
- `floodplain / F_S / [0,0.5)`: leader `M0` (0.753 m), top clique `[M0, M-null, M2-XGB]`
- `floodplain / F_S / [0.5,2)`: leader `M0` (0.396 m), top clique `[M0, M2-XGB, M-null]`

- **H-V3-1 (S-LRO)**: supported (non-inferior) — `not_F_S` M5 - M2-XGB = -0.0053 m [-0.0440, +0.0330]; `F_S` M5 - M2-XGB = +2.2184 m [+1.9833, +2.4562]

## 2. Leopold-Maddock Dynamic Resistance & Oracle Benchmark (S-ID high flow)

Off-anchor empirical exponents (n = 29,174 rows with |ln(Q_i/Q_a)| >= 0.2): width `b = +0.1557`, mean depth `f = +0.3172`, max depth `f_max = +0.2975`, area `b+f = +0.4724`, velocity `m = +0.5269`, roughness `gamma = -0.3246`.

| Model | Mean Anchor dmax0 (m) | RMSE Const-n (m) | Bias Const-n (m) | RMSE Dyn-n LODO (m) | Bias Dyn-n LODO (m) | ΔRMSE (Dyn - Const) |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 2.132 | 2.384 | +1.011 | 1.388 | -0.065 | -0.996 |
| M-null | 1.865 | 1.926 | +0.827 | 1.315 | -0.174 | -0.612 |
| M0 | 1.949 | 2.193 | +1.022 | 1.278 | -0.073 | -0.915 |
| M1 | 2.239 | 2.459 | +1.311 | 1.263 | +0.090 | -1.197 |
| M2-MLP | 2.052 | 2.136 | +1.035 | 1.246 | -0.055 | -0.890 |
| M2-XGB | 2.037 | 2.169 | +1.020 | 1.248 | -0.066 | -0.920 |
| M3 | 1.314 | 1.859 | +0.063 | 1.513 | -0.584 | -0.346 |
| M4 | 2.453 | 3.360 | +1.515 | 1.640 | +0.210 | -1.719 |
| M5 | 2.541 | 3.509 | +1.536 | 1.740 | +0.227 | -1.769 |
| M6-NN | 2.741 | 4.562 | +1.757 | 2.272 | +0.348 | -2.291 |
| M6-XGB | 2.618 | 4.116 | +1.676 | 1.986 | +0.299 | -2.130 |
| M5_A1 | 2.548 | 3.538 | +1.555 | 1.747 | +0.237 | -1.791 |
| M5_wA1 | 2.693 | 4.096 | +1.711 | 2.005 | +0.323 | -2.090 |
| M5_A2 | 2.255 | 3.732 | +1.592 | 1.741 | +0.214 | -1.991 |
| M5_A3-SP | 1.343 | 1.851 | +0.063 | 1.500 | -0.582 | -0.351 |
| M5_A3-SR | 1.321 | 1.824 | +0.039 | 1.500 | -0.596 | -0.324 |
| M5_A4 | 2.557 | 3.558 | +1.554 | 1.761 | +0.237 | -1.797 |

- **H-V3-2**: supported — Oracle bias const-n `+1.011 m` -> dyn-n `-0.065 m`; M2-XGB - M3 const-n = `+0.3098 m` [+0.1727, +0.4587] -> dyn-n = `-0.2648 m` [-0.3567, -0.1864]

## 3. Off-Anchor Absolute Hydraulic States (n = 31,367 obs, high flow n = 8,267)

| Model | RMSE ln dbar | RMSE ln dmax | RMSE ln W | RMSE ln A | High RMSE ln dmax | High Raw RMSE dmax (m) | Anchor Bias (m) | Flood Bias (m) | Stage-Rise Bias (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 0.3667 | 0.3804 | 0.2935 | 0.4923 | 0.3865 | 2.820 | -0.000 | +0.952 | +1.043 |
| M-null | 0.6470 | 0.6442 | 0.6576 | 1.1334 | 0.5487 | 3.203 | -0.342 | +0.406 | +0.839 |
| M0 | 0.5553 | 0.5621 | 0.5048 | 0.8745 | 0.5184 | 2.760 | -0.257 | +0.625 | +0.974 |
| M1 | 0.4966 | 0.5126 | 0.4047 | 0.7217 | 0.5515 | 2.747 | +0.129 | +1.366 | +1.329 |
| M2-MLP | 0.4875 | 0.4986 | 0.3995 | 0.6987 | 0.4880 | 2.422 | -0.110 | +0.821 | +1.023 |
| M2-XGB | 0.4770 | 0.4888 | 0.3713 | 0.6661 | 0.4740 | 2.261 | -0.119 | +0.794 | +1.004 |
| M3 | 1.0539 | 1.0471 | 0.3817 | 1.1097 | 0.9516 | 3.105 | -0.802 | -0.787 | +0.106 |
| M4 | 0.6990 | 0.7079 | 0.4773 | 1.0150 | 0.7007 | 4.662 | +0.323 | +1.748 | +1.516 |
| M5 | 0.7148 | 0.7226 | 0.4545 | 1.0045 | 0.7233 | 5.759 | +0.547 | +2.053 | +1.597 |
| M6-NN | 0.8288 | 0.8339 | 0.4122 | 0.9506 | 0.8187 | 8.740 | +0.896 | +2.716 | +1.912 |
| M6-XGB | 0.8121 | 0.8187 | 0.3704 | 0.9431 | 0.8155 | 6.184 | +0.591 | +2.218 | +1.719 |

- **H-V3-3**: supported — high-flow site-weighted RMSE(ln dmax(Q_i)): M2-XGB - M3 = `-0.4776` [-0.5228, -0.4366]; M2-XGB - M5 = `-0.2493` [-0.2878, -0.2168]

## 4. CCHE1D 1D Hydraulic & Sediment-Transport Readiness Benchmark

| Split | Model | Sites | RMSE_log A_dir | RMSE_log A_cons | Cont Viol Dir | RMSE_log R0 | RMSE_raw R0 (m) | RMSE_log dmax0 | P99 |err dmax0| (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S-ID | M-null | 3988 | 0.9768 | 0.9768 | 0.229 | 0.5549 | 1.332 | 0.5577 | 7.745 |
| S-ID | M0 | 3988 | 0.8303 | 0.8303 | 0.000 | 0.5084 | 1.152 | 0.5116 | 6.874 |
| S-ID | M1 | 3988 | 0.7032 | 0.7032 | 0.000 | 0.4663 | 0.857 | 0.4710 | 5.250 |
| S-ID | M2-MLP | 3988 | 0.6494 | 0.6467 | 0.067 | 0.4409 | 0.912 | 0.4520 | 5.606 |
| S-ID | M2-XGB | 3988 | 0.6254 | 0.6221 | 0.496 | 0.4289 | 0.863 | 0.4399 | 5.561 |
| S-ID | M3 | 3988 | 1.2499 | 1.2499 | 0.000 | 1.1483 | 1.251 | 1.1804 | 7.390 |
| S-ID | M4 | 3988 | 0.9888 | 0.9888 | 0.000 | 0.6675 | 1.332 | 0.6794 | 6.738 |
| S-ID | M5 | 3988 | 0.9816 | 0.9816 | 0.000 | 0.6891 | 1.507 | 0.7045 | 7.739 |
| S-ID | M6-NN | 3988 | 0.9229 | 0.9229 | 0.000 | 0.7751 | 1.979 | 0.8041 | 10.762 |
| S-ID | M6-XGB | 3988 | 0.9121 | 0.9121 | 0.000 | 0.7713 | 1.693 | 0.7996 | 8.807 |
| S-ID | CCHE1D-Hybrid | 3988 | 0.6254 | 0.6221 | 0.496 | 0.4289 | 0.863 | 0.4399 | 5.561 |
| S-LRO | M-null | 3988 | 1.0055 | 1.0025 | 0.260 | 0.5663 | 1.345 | 0.5683 | 7.746 |
| S-LRO | M0 | 3988 | 0.9348 | 0.9348 | 0.000 | 0.5516 | 1.233 | 0.5526 | 7.381 |
| S-LRO | M1 | 3988 | 0.7438 | 0.7438 | 0.000 | 0.4925 | 0.932 | 0.4951 | 5.529 |
| S-LRO | M2-MLP | 3988 | 0.7247 | 0.7182 | 0.196 | 0.4745 | 1.012 | 0.4837 | 6.050 |
| S-LRO | M2-XGB | 3988 | 0.6931 | 0.6918 | 0.586 | 0.4658 | 0.950 | 0.4793 | 5.901 |
| S-LRO | M3 | 3988 | 1.2697 | 1.2697 | 0.000 | 1.1291 | 1.244 | 1.1593 | 7.302 |
| S-LRO | M4 | 3988 | 1.0013 | 1.0013 | 0.000 | 0.6726 | 1.364 | 0.6837 | 6.825 |
| S-LRO | M5 | 3988 | 1.0025 | 1.0025 | 0.000 | 0.7127 | 1.562 | 0.7276 | 7.995 |
| S-LRO | M6-NN | 3988 | 0.9556 | 0.9556 | 0.000 | 0.7995 | 2.117 | 0.8294 | 11.275 |
| S-LRO | M6-XGB | 3988 | 0.9658 | 0.9658 | 0.000 | 0.7937 | 1.842 | 0.8245 | 9.965 |
| S-LRO | CCHE1D-Hybrid | 3988 | 0.6931 | 0.6918 | 0.586 | 0.4658 | 0.950 | 0.4793 | 5.901 |
| S-TAIL-DA | M-null | 678 | 1.5472 | 1.5541 | 0.000 | 0.7779 | 2.864 | 0.7604 | 19.619 |
| S-TAIL-DA | M0 | 678 | 1.0828 | 1.0828 | 0.000 | 0.6518 | 2.345 | 0.6420 | 14.091 |
| S-TAIL-DA | M1 | 678 | 0.9986 | 0.9986 | 0.000 | 0.5512 | 2.149 | 0.5447 | 16.417 |
| S-TAIL-DA | M2-MLP | 678 | 1.0549 | 1.0245 | 0.298 | 0.5538 | 2.345 | 0.5585 | 16.867 |
| S-TAIL-DA | M2-XGB | 678 | 1.0342 | 1.0319 | 0.544 | 0.6021 | 2.533 | 0.6022 | 16.554 |
| S-TAIL-DA | M3 | 678 | 1.4731 | 1.4731 | 0.000 | 1.3633 | 2.302 | 1.3790 | 18.158 |
| S-TAIL-DA | M4 | 678 | 1.2690 | 1.2690 | 0.000 | 0.7586 | 2.543 | 0.7512 | 17.241 |
| S-TAIL-DA | M5 | 678 | 1.0921 | 1.0921 | 0.000 | 0.8541 | 3.840 | 0.8956 | 50.570 |
| S-TAIL-DA | M6-NN | 678 | 1.0371 | 1.0371 | 0.000 | 0.9226 | 4.066 | 0.9981 | 93.861 |
| S-TAIL-DA | M6-XGB | 678 | 1.0344 | 1.0344 | 0.000 | 0.9529 | 4.216 | 0.9949 | 67.698 |
| S-TAIL-DA | CCHE1D-Hybrid | 678 | 0.9986 | 0.9986 | 0.000 | 0.5512 | 2.149 | 0.5447 | 16.417 |
| S-TAIL-SO | M-null | 339 | 1.8898 | 1.8936 | 0.000 | 0.9578 | 3.890 | 0.9131 | 21.614 |
| S-TAIL-SO | M0 | 339 | 1.2638 | 1.2638 | 0.000 | 0.7838 | 3.340 | 0.7465 | 18.657 |
| S-TAIL-SO | M1 | 339 | 0.9157 | 0.9157 | 0.000 | 0.5685 | 2.908 | 0.5363 | 18.288 |
| S-TAIL-SO | M2-MLP | 339 | 1.1125 | 1.0544 | 0.572 | 0.5808 | 3.026 | 0.5622 | 18.706 |
| S-TAIL-SO | M2-XGB | 339 | 1.0968 | 1.1594 | 0.717 | 0.6777 | 3.358 | 0.6396 | 20.473 |
| S-TAIL-SO | M3 | 339 | 1.1559 | 1.1559 | 0.000 | 0.9288 | 2.815 | 0.9370 | 17.381 |
| S-TAIL-SO | M4 | 339 | 1.4616 | 1.4616 | 0.000 | 0.9134 | 3.389 | 0.8890 | 19.803 |
| S-TAIL-SO | M5 | 339 | 1.2212 | 1.2212 | 0.000 | 0.8920 | 4.413 | 0.9264 | 37.731 |
| S-TAIL-SO | M6-NN | 339 | 1.1448 | 1.1448 | 0.000 | 1.0019 | 5.217 | 1.0593 | 59.551 |
| S-TAIL-SO | M6-XGB | 339 | 1.1256 | 1.1256 | 0.000 | 1.0140 | 5.598 | 1.0735 | 84.345 |
| S-TAIL-SO | CCHE1D-Hybrid | 339 | 0.9157 | 0.9157 | 0.000 | 0.5685 | 2.908 | 0.5363 | 18.288 |

- **H-V3-4**: supported — M2-XGB S-ID continuity violation `0.496` -> `0.000` (RMSE_log A `0.6254` -> `0.6221`); tail R0 contrasts (CCHE1D-Hybrid - M2-XGB): S-TAIL-DA `-0.0508` [-0.0787, -0.0225], S-TAIL-SO `-0.1092` [-0.1452, -0.0732]

