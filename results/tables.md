
## Clean data

### Accuracy vs clean ground truth (total variance w; iv_rmse in vol units)

| Regime | Model | n | rmse_all | rmse_heldout | iv_rmse_all |
|---|---|---|---|---|---|
| normal | SVI (per slice) | 5 | 0.0150 ± 0.0012 | 0.0154 ± 0.0020 | 0.0469 ± 0.0077 |
| normal | SSVI | 5 | 0.0074 ± 0.0006 | 0.0072 ± 0.0005 | 0.0253 ± 0.0057 |
| normal | Soft-penalty MLP | 5 | 0.0092 ± 0.0023 | 0.0089 ± 0.0023 | 0.0272 ± 0.0116 |
| normal | Hard-constrained (ours) | 5 | 0.0099 ± 0.0017 | 0.0094 ± 0.0020 | 0.0306 ± 0.0104 |
| normal | Hard, no init calibration | 5 | 0.1378 ± 0.0016 | 0.1411 ± 0.0027 | 0.1785 ± 0.0033 |
| normal | Dugas-style (naive init) | 5 | 111.7751 ± 5.1044 | 112.5833 ± 5.5493 | 14.0908 ± 0.4400 |
| high_skew | SVI (per slice) | 5 | 0.0154 ± 0.0007 | 0.0161 ± 0.0008 | 0.0495 ± 0.0068 |
| high_skew | SSVI | 5 | 0.0096 ± 0.0018 | 0.0094 ± 0.0017 | 0.0374 ± 0.0067 |
| high_skew | Soft-penalty MLP | 5 | 0.0112 ± 0.0017 | 0.0109 ± 0.0017 | 0.0368 ± 0.0098 |
| high_skew | Hard-constrained (ours) | 5 | 0.0124 ± 0.0015 | 0.0123 ± 0.0017 | 0.0403 ± 0.0106 |
| high_skew | Hard, no init calibration | 5 | 0.1368 ± 0.0015 | 0.1399 ± 0.0023 | 0.1884 ± 0.0096 |
| high_skew | Dugas-style (naive init) | 5 | 111.7690 ± 5.1040 | 112.5775 ± 5.5484 | 14.0651 ± 0.4381 |
| stressed | SVI (per slice) | 5 | 0.0160 ± 0.0018 | 0.0169 ± 0.0021 | 0.0341 ± 0.0055 |
| stressed | SSVI | 5 | 0.0135 ± 0.0019 | 0.0133 ± 0.0019 | 0.0336 ± 0.0053 |
| stressed | Soft-penalty MLP | 5 | 0.0154 ± 0.0065 | 0.0156 ± 0.0071 | 0.0328 ± 0.0206 |
| stressed | Hard-constrained (ours) | 5 | 0.0176 ± 0.0029 | 0.0182 ± 0.0033 | 0.0388 ± 0.0101 |
| stressed | Hard, no init calibration | 5 | 0.1104 ± 0.0023 | 0.1134 ± 0.0041 | 0.1749 ± 0.0115 |
| stressed | Dugas-style (naive init) | 5 | 111.7147 ± 5.1029 | 112.5252 ± 5.5470 | 13.8278 ± 0.4448 |

### No-arbitrage diagnostics at the 5 maturities (fine k-grid; same finite-difference checker for all models)

| Regime | Model | n | butterfly (convexity) | Durrleman g(k) | calendar | max calendar viol. (w units) |
|---|---|---|---|---|---|---|
| normal | SVI (per slice) | 5 | 0.00% ± 0.00% | 2.69% ± 2.73% | 31.02% ± 2.63% | 0.04136 ± 0.00954 |
| normal | SSVI | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| normal | Soft-penalty MLP | 5 | 0.02% ± 0.04% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| normal | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| normal | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| normal | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 72.56% ± 43.64% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| high_skew | SVI (per slice) | 5 | 0.00% ± 0.00% | 3.20% ± 3.08% | 28.13% ± 4.60% | 0.04164 ± 0.01192 |
| high_skew | SSVI | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| high_skew | Soft-penalty MLP | 5 | 0.00% ± 0.00% | 0.30% ± 0.62% | 0.02% ± 0.06% | 0.00006 ± 0.00014 |
| high_skew | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| high_skew | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| high_skew | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 73.13% ± 43.50% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | SVI (per slice) | 5 | 0.00% ± 0.00% | 3.88% ± 5.39% | 17.86% ± 4.60% | 0.03501 ± 0.01884 |
| stressed | SSVI | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | Soft-penalty MLP | 5 | 0.04% ± 0.06% | 0.84% ± 1.48% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 74.01% ± 43.36% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |

### Continuous-in-T models: violations on a dense T grid between maturities

| Regime | Model | n | butterfly (convexity) | Durrleman g(k) | calendar |
|---|---|---|---|---|---|
| normal | Soft-penalty MLP | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| normal | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| normal | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| normal | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 77.79% ± 43.75% | 0.00% ± 0.00% |
| high_skew | Soft-penalty MLP | 5 | 0.00% ± 0.00% | 0.09% ± 0.19% | 0.00% ± 0.00% |
| high_skew | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| high_skew | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| high_skew | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 77.99% ± 43.81% | 0.00% ± 0.00% |
| stressed | Soft-penalty MLP | 5 | 0.00% ± 0.00% | 0.35% ± 0.55% | 0.01% ± 0.03% |
| stressed | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| stressed | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| stressed | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 78.32% ± 43.93% | 0.00% ± 0.00% |

### Paired per-seed comparison on held-out RMSE (negative = hard-constrained better)

| Regime | vs model | mean(hard_constrained − model) | hard_constrained lower in |
|---|---|---|---|
| normal | SVI (per slice) | -0.00598 | 5/5 seeds |
| normal | SSVI | +0.00217 | 1/5 seeds |
| normal | Soft-penalty MLP | +0.00047 | 2/5 seeds |
| normal | Hard, no init calibration | -0.13168 | 5/5 seeds |
| normal | Dugas-style (naive init) | -112.57396 | 5/5 seeds |
| high_skew | SVI (per slice) | -0.00376 | 5/5 seeds |
| high_skew | SSVI | +0.00288 | 0/5 seeds |
| high_skew | Soft-penalty MLP | +0.00142 | 0/5 seeds |
| high_skew | Hard, no init calibration | -0.12759 | 5/5 seeds |
| high_skew | Dugas-style (naive init) | -112.56520 | 5/5 seeds |
| stressed | SVI (per slice) | +0.00123 | 3/5 seeds |
| stressed | SSVI | +0.00482 | 0/5 seeds |
| stressed | Soft-penalty MLP | +0.00257 | 1/5 seeds |
| stressed | Hard, no init calibration | -0.09525 | 5/5 seeds |
| stressed | Dugas-style (naive init) | -112.50706 | 5/5 seeds |

### Optimisation diagnostics (final training MSE)

| Regime | Model | n | initial train MSE | final train MSE |
|---|---|---|---|---|
| normal | Soft-penalty MLP | 5 | 0.027252 ± 0.052096 | 0.000269 ± 0.000046 |
| normal | Hard-constrained (ours) | 5 | 0.001221 ± 0.001118 | 0.000268 ± 0.000059 |
| normal | Hard, no init calibration | 5 | 68.575842 ± 0.552629 | 0.017625 ± 0.000514 |
| normal | Dugas-style (naive init) | 5 | 138068240.000000 ± 7522710.345139 | 12185.205156 ± 1038.468847 |
| high_skew | Soft-penalty MLP | 5 | 0.033412 ± 0.057218 | 0.000277 ± 0.000053 |
| high_skew | Hard-constrained (ours) | 5 | 0.003526 ± 0.002787 | 0.000311 ± 0.000062 |
| high_skew | Hard, no init calibration | 5 | 68.378613 ± 0.551016 | 0.017396 ± 0.000402 |
| high_skew | Dugas-style (naive init) | 5 | 138067964.800000 ± 7522725.404279 | 12183.792246 ± 1038.421622 |
| stressed | Soft-penalty MLP | 5 | 0.087342 ± 0.088599 | 0.000340 ± 0.000215 |
| stressed | Hard-constrained (ours) | 5 | 0.012865 ± 0.010130 | 0.000415 ± 0.000068 |
| stressed | Hard, no init calibration | 5 | 66.443752 ± 0.583627 | 0.011334 ± 0.000805 |
| stressed | Dugas-style (naive init) | 5 | 138065134.400000 ± 7522677.172842 | 12171.019551 ± 1037.806991 |

## Calendar-inversion stress test

### Accuracy vs clean ground truth (total variance w; iv_rmse in vol units)

| Regime | Model | n | rmse_all | rmse_heldout | iv_rmse_all | rmse_distorted_vs_truth | rmse_distorted_vs_target |
|---|---|---|---|---|---|---|---|
| normal | SVI (per slice) | 5 | 0.0208 ± 0.0032 | 0.0218 ± 0.0040 | 0.0709 ± 0.0161 | 0.0434 ± 0.0154 | 0.0218 ± 0.0091 |
| normal | SSVI | 5 | 0.0111 ± 0.0028 | 0.0112 ± 0.0032 | 0.0375 ± 0.0148 | 0.0185 ± 0.0079 | 0.0222 ± 0.0057 |
| normal | Soft-penalty MLP | 5 | 0.0119 ± 0.0012 | 0.0120 ± 0.0012 | 0.0377 ± 0.0112 | 0.0163 ± 0.0064 | 0.0257 ± 0.0079 |
| normal | Hard-constrained (ours) | 5 | 0.0115 ± 0.0011 | 0.0112 ± 0.0006 | 0.0317 ± 0.0054 | 0.0151 ± 0.0052 | 0.0265 ± 0.0047 |
| normal | Hard, no init calibration | 5 | 0.1365 ± 0.0014 | 0.1398 ± 0.0025 | 0.1770 ± 0.0032 | 0.1260 ± 0.0010 | 0.1655 ± 0.0035 |
| normal | Dugas-style (naive init) | 5 | 111.7736 ± 5.1045 | 112.5819 ± 5.5493 | 14.0906 ± 0.4400 | 70.8896 ± 4.1710 | 70.9292 ± 4.1694 |
| high_skew | SVI (per slice) | 5 | 0.0215 ± 0.0031 | 0.0228 ± 0.0043 | 0.0710 ± 0.0125 | 0.0461 ± 0.0140 | 0.0239 ± 0.0109 |
| high_skew | SSVI | 5 | 0.0131 ± 0.0025 | 0.0130 ± 0.0030 | 0.0476 ± 0.0144 | 0.0192 ± 0.0085 | 0.0238 ± 0.0074 |
| high_skew | Soft-penalty MLP | 5 | 0.0127 ± 0.0015 | 0.0127 ± 0.0018 | 0.0387 ± 0.0084 | 0.0171 ± 0.0066 | 0.0263 ± 0.0079 |
| high_skew | Hard-constrained (ours) | 5 | 0.0139 ± 0.0013 | 0.0138 ± 0.0014 | 0.0383 ± 0.0064 | 0.0167 ± 0.0040 | 0.0303 ± 0.0049 |
| high_skew | Hard, no init calibration | 5 | 0.1355 ± 0.0014 | 0.1385 ± 0.0022 | 0.1872 ± 0.0097 | 0.1230 ± 0.0021 | 0.1643 ± 0.0022 |
| high_skew | Dugas-style (naive init) | 5 | 111.7675 ± 5.1039 | 112.5760 ± 5.5483 | 14.0650 ± 0.4381 | 70.8841 ± 4.1695 | 70.9257 ± 4.1705 |
| stressed | SVI (per slice) | 5 | 0.0230 ± 0.0048 | 0.0246 ± 0.0058 | 0.0463 ± 0.0054 | 0.0488 ± 0.0191 | 0.0222 ± 0.0122 |
| stressed | SSVI | 5 | 0.0175 ± 0.0020 | 0.0177 ± 0.0026 | 0.0434 ± 0.0108 | 0.0263 ± 0.0094 | 0.0230 ± 0.0076 |
| stressed | Soft-penalty MLP | 5 | 0.0167 ± 0.0034 | 0.0169 ± 0.0029 | 0.0316 ± 0.0078 | 0.0249 ± 0.0096 | 0.0248 ± 0.0065 |
| stressed | Hard-constrained (ours) | 5 | 0.0193 ± 0.0015 | 0.0198 ± 0.0015 | 0.0370 ± 0.0056 | 0.0248 ± 0.0058 | 0.0320 ± 0.0076 |
| stressed | Hard, no init calibration | 5 | 0.1090 ± 0.0024 | 0.1119 ± 0.0041 | 0.1756 ± 0.0112 | 0.0921 ± 0.0057 | 0.1357 ± 0.0022 |
| stressed | Dugas-style (naive init) | 5 | 111.7131 ± 5.1031 | 112.5235 ± 5.5471 | 13.8276 ± 0.4448 | 70.8393 ± 4.1714 | 70.8849 ± 4.1685 |

### No-arbitrage diagnostics at the 5 maturities (fine k-grid; same finite-difference checker for all models)

| Regime | Model | n | butterfly (convexity) | Durrleman g(k) | calendar | max calendar viol. (w units) |
|---|---|---|---|---|---|---|
| normal | SVI (per slice) | 5 | 0.00% ± 0.00% | 5.91% ± 4.25% | 33.23% ± 5.64% | 0.07245 ± 0.02760 |
| normal | SSVI | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| normal | Soft-penalty MLP | 5 | 0.14% ± 0.31% | 0.00% ± 0.00% | 0.45% ± 0.61% | 0.00035 ± 0.00072 |
| normal | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| normal | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| normal | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 72.56% ± 43.64% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| high_skew | SVI (per slice) | 5 | 0.00% ± 0.00% | 8.68% ± 4.33% | 32.94% ± 5.74% | 0.07851 ± 0.02713 |
| high_skew | SSVI | 5 | 0.00% ± 0.00% | 0.82% ± 1.84% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| high_skew | Soft-penalty MLP | 5 | 0.02% ± 0.04% | 0.08% ± 0.18% | 0.15% ± 0.22% | 0.00064 ± 0.00097 |
| high_skew | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| high_skew | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| high_skew | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 73.11% ± 43.51% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | SVI (per slice) | 5 | 0.00% ± 0.00% | 9.05% ± 2.46% | 23.41% ± 4.79% | 0.07961 ± 0.03466 |
| stressed | SSVI | 5 | 0.00% ± 0.00% | 1.19% ± 2.65% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | Soft-penalty MLP | 5 | 0.00% ± 0.00% | 2.01% ± 1.76% | 0.00% ± 0.00% | 0.00002 ± 0.00004 |
| stressed | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |
| stressed | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 73.99% ± 43.36% | 0.00% ± 0.00% | 0.00000 ± 0.00000 |

### Continuous-in-T models: violations on a dense T grid between maturities

| Regime | Model | n | butterfly (convexity) | Durrleman g(k) | calendar |
|---|---|---|---|---|---|
| normal | Soft-penalty MLP | 5 | 0.01% ± 0.02% | 0.00% ± 0.00% | 0.01% ± 0.03% |
| normal | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| normal | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| normal | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 77.79% ± 43.75% | 0.00% ± 0.00% |
| high_skew | Soft-penalty MLP | 5 | 0.00% ± 0.01% | 0.02% ± 0.04% | 0.00% ± 0.00% |
| high_skew | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| high_skew | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| high_skew | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 77.99% ± 43.81% | 0.00% ± 0.00% |
| stressed | Soft-penalty MLP | 5 | 0.00% ± 0.00% | 1.87% ± 1.25% | 0.00% ± 0.00% |
| stressed | Hard-constrained (ours) | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| stressed | Hard, no init calibration | 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| stressed | Dugas-style (naive init) | 5 | 0.00% ± 0.00% | 78.32% ± 43.93% | 0.00% ± 0.00% |

### Paired per-seed comparison on held-out RMSE (negative = hard-constrained better)

| Regime | vs model | mean(hard_constrained − model) | hard_constrained lower in |
|---|---|---|---|
| normal | SVI (per slice) | -0.01062 | 5/5 seeds |
| normal | SSVI | +0.00007 | 2/5 seeds |
| normal | Soft-penalty MLP | -0.00078 | 4/5 seeds |
| normal | Hard, no init calibration | -0.12854 | 5/5 seeds |
| normal | Dugas-style (naive init) | -112.57066 | 5/5 seeds |
| high_skew | SVI (per slice) | -0.00903 | 5/5 seeds |
| high_skew | SSVI | +0.00076 | 2/5 seeds |
| high_skew | Soft-penalty MLP | +0.00107 | 1/5 seeds |
| high_skew | Hard, no init calibration | -0.12473 | 5/5 seeds |
| high_skew | Dugas-style (naive init) | -112.56218 | 5/5 seeds |
| stressed | SVI (per slice) | -0.00477 | 4/5 seeds |
| stressed | SSVI | +0.00206 | 2/5 seeds |
| stressed | Soft-penalty MLP | +0.00286 | 0/5 seeds |
| stressed | Hard, no init calibration | -0.09212 | 5/5 seeds |
| stressed | Dugas-style (naive init) | -112.50376 | 5/5 seeds |

### Optimisation diagnostics (final training MSE)

| Regime | Model | n | initial train MSE | final train MSE |
|---|---|---|---|---|
| normal | Soft-penalty MLP | 5 | 0.026945 ± 0.051362 | 0.000361 ± 0.000068 |
| normal | Hard-constrained (ours) | 5 | 0.001414 ± 0.001094 | 0.000377 ± 0.000107 |
| normal | Hard, no init calibration | 5 | 68.623286 ± 0.556848 | 0.018306 ± 0.000526 |
| normal | Dugas-style (naive init) | 5 | 138068300.800000 ± 7522703.776696 | 12185.381074 ± 1038.459250 |
| high_skew | Soft-penalty MLP | 5 | 0.032985 ± 0.056466 | 0.000368 ± 0.000077 |
| high_skew | Hard-constrained (ours) | 5 | 0.003744 ± 0.002825 | 0.000434 ± 0.000106 |
| high_skew | Hard, no init calibration | 5 | 68.427975 ± 0.549270 | 0.018085 ± 0.000450 |
| high_skew | Dugas-style (naive init) | 5 | 138068035.200000 ± 7522726.614721 | 12183.975400 ± 1038.433931 |
| stressed | Soft-penalty MLP | 5 | 0.086023 ± 0.087740 | 0.000498 ± 0.000232 |
| stressed | Hard-constrained (ours) | 5 | 0.013152 ± 0.010324 | 0.000553 ± 0.000114 |
| stressed | Hard, no init calibration | 5 | 66.497223 ± 0.591690 | 0.011935 ± 0.000800 |
| stressed | Dugas-style (naive init) | 5 | 138065208.000000 ± 7522667.802001 | 12171.221113 ± 1037.793110 |