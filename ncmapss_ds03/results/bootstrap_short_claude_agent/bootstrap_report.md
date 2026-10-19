# DS03 engine-bootstrap prognostics

The independent sampling unit is the engine. At each cutoff, the five held-out short-haul engine errors are resampled with replacement, and RUL RMSE is recomputed for 20,000 bootstrap replicates. Flight rows are not resampled because observations from one run-to-failure engine are serially dependent.

| Observed life | Claude LIP-EM | Gemini LIP-EM | Uniform EM | Global pool | Target only |
|---:|---:|---:|---:|---:|---:|
| 10% | 11.38 [5.07, 16.11] | 11.92 [4.60, 16.32] | 13.35 [5.84, 18.04] | 19.03 [11.07, 24.53] | 19.03 [11.07, 24.53] |
| 20% | 8.89 [3.96, 12.08] | 10.94 [3.93, 14.97] | 16.05 [7.39, 21.47] | 26.54 [19.60, 32.01] | 26.54 [19.60, 32.01] |
| 30% | 10.45 [7.16, 12.74] | 11.56 [5.95, 15.31] | 15.60 [7.11, 20.88] | 34.21 [28.20, 39.31] | 30.65 [18.25, 39.31] |
| 40% | 9.84 [6.45, 12.92] | 10.86 [4.88, 15.04] | 10.36 [5.40, 13.62] | 38.19 [30.29, 44.69] | 11.18 [7.87, 13.72] |
| 50% | 10.68 [7.02, 14.22] | 11.02 [8.17, 13.09] | 10.10 [4.62, 13.60] | 40.22 [31.26, 47.52] | 9.08 [2.43, 14.43] |
| 60% | 9.43 [6.71, 10.94] | 9.50 [5.00, 12.47] | 9.52 [3.67, 12.95] | 40.95 [31.57, 48.56] | 39.36 [18.26, 52.58] |
| 70% | 5.50 [4.06, 6.85] | 5.55 [3.42, 7.30] | 5.41 [3.41, 7.08] | 38.78 [26.41, 48.07] | 23.01 [14.00, 30.39] |
| 80% | 3.28 [1.76, 4.29] | 3.04 [1.68, 3.93] | 2.71 [1.15, 3.83] | 36.38 [25.37, 44.75] | 20.81 [10.42, 31.41] |
| 90% | 6.19 [2.96, 8.57] | 6.21 [3.01, 8.57] | 6.30 [3.19, 8.57] | 39.76 [35.62, 43.50] | 8.97 [5.51, 11.43] |

Intervals are percentile 95% engine-bootstrap intervals. With only five engines, they quantify finite-fleet variability but will be wide and should not be interpreted as high-powered hypothesis tests.

## Paired comparison with uniform EM

The same resampled target engines are used for each LIP-EM versus uniform-EM comparison. Negative differences favor LIP-EM.

| Observed life | LIP | RMSE difference [95% CI] | Bootstrap P(LIP better) | Engine W/T/L |
|---:|---|---:|---:|---:|
| 10% | Claude | -1.97 [-3.36, 0.11] | 0.974 | 4/0/1 |
| 10% | Gemini | -1.43 [-2.01, -0.74] | 0.998 | 4/0/1 |
| 20% | Claude | -7.16 [-9.72, -3.10] | 0.995 | 4/0/1 |
| 20% | Gemini | -5.12 [-6.67, -2.81] | 0.998 | 4/0/1 |
| 30% | Claude | -5.14 [-9.22, 1.72] | 0.939 | 3/0/2 |
| 30% | Gemini | -4.04 [-6.12, -0.32] | 0.979 | 4/0/1 |
| 40% | Claude | -0.52 [-3.31, 2.59] | 0.657 | 2/0/3 |
| 40% | Gemini | 0.50 [-2.71, 2.57] | 0.344 | 2/0/3 |
| 50% | Claude | 0.57 [-4.34, 7.18] | 0.442 | 3/0/2 |
| 50% | Gemini | 0.92 [-1.24, 4.80] | 0.228 | 3/0/2 |
| 60% | Claude | -0.09 [-3.08, 4.80] | 0.556 | 2/0/3 |
| 60% | Gemini | -0.02 [-0.95, 1.67] | 0.539 | 3/0/2 |
| 70% | Claude | 0.09 [-0.24, 0.65] | 0.324 | 2/0/3 |
| 70% | Gemini | 0.14 [-0.43, 0.56] | 0.271 | 2/0/3 |
| 80% | Claude | 0.57 [-0.08, 1.79] | 0.318 | 2/2/1 |
| 80% | Gemini | 0.34 [-0.08, 1.20] | 0.318 | 2/2/1 |
| 90% | Claude | -0.11 [-0.40, 0.00] | 0.671 | 1/4/0 |
| 90% | Gemini | -0.09 [-0.32, 0.00] | 0.671 | 1/4/0 |

## Complete-trajectory paired comparison

This summary pools all 45 engine/cutoff predictions. Each bootstrap replicate resamples complete nine-cutoff engine trajectories, preserving within-engine dependence. Negative differences favor the first method.

| First method | Second method | First RMSE | Second RMSE | Difference [95% CI] | P(first better) |
|---|---|---:|---:|---:|---:|
| Claude LIP-EM | Uniform EM | 8.800 | 10.829 | -2.029 [-4.316, 1.722] | 0.901 |
| Gemini LIP-EM | Uniform EM | 9.449 | 10.829 | -1.380 [-2.572, 0.492] | 0.945 |
| Claude LIP-EM | Gemini LIP-EM | 8.800 | 9.449 | -0.649 [-1.925, 1.447] | 0.772 |

## RUL-cap diagnostics

A cap hit means the fitted path did not cross the source-learned failure threshold within the maximum source RUL horizon. These are valid capped predictions, but they explain the abrupt RMSE changes at some cutoffs.

| Observed life | Claude LIP-EM | Gemini LIP-EM | Uniform EM | Global pool | Target only |
|---:|---:|---:|---:|---:|---:|
| 10% | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| 20% | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| 30% | 0/5 | 0/5 | 0/5 | 3/5 | 3/5 |
| 40% | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 |
| 50% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 60% | 0/5 | 0/5 | 0/5 | 0/5 | 2/5 |
| 70% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 80% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 90% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
