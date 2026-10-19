# DS03 engine-bootstrap prognostics

The independent sampling unit is the engine. At each cutoff, the five held-out long-haul engine errors are resampled with replacement, and RUL RMSE is recomputed for 20,000 bootstrap replicates. Flight rows are not resampled because observations from one run-to-failure engine are serially dependent.

| Observed life | Claude LIP-EM | Gemini LIP-EM | Uniform EM | Global pool | Target only |
|---:|---:|---:|---:|---:|---:|
| 10% | 5.92 [3.57, 7.58] | 5.79 [3.24, 7.62] | 8.23 [4.21, 11.94] | 32.73 [27.29, 37.25] | 32.73 [27.29, 37.25] |
| 20% | 6.71 [2.75, 9.47] | 6.49 [2.37, 9.29] | 8.46 [4.91, 11.51] | 39.10 [34.48, 42.84] | 39.10 [34.48, 42.84] |
| 30% | 7.34 [4.01, 10.34] | 7.49 [3.80, 10.63] | 8.00 [4.32, 11.13] | 45.59 [41.75, 48.90] | 45.59 [41.75, 48.90] |
| 40% | 6.97 [3.90, 9.73] | 7.17 [3.73, 10.06] | 7.89 [4.29, 10.93] | 49.12 [44.98, 53.43] | 38.27 [14.81, 52.06] |
| 50% | 6.48 [3.81, 8.96] | 6.61 [3.13, 9.18] | 6.60 [3.66, 9.04] | 49.15 [44.99, 53.47] | 7.54 [4.88, 9.48] |
| 60% | 4.96 [2.90, 6.83] | 5.24 [2.40, 7.37] | 3.56 [1.46, 5.30] | 49.15 [45.06, 53.46] | 5.00 [4.17, 5.66] |
| 70% | 2.45 [0.81, 3.50] | 2.85 [0.96, 3.91] | 1.15 [0.34, 1.81] | 48.90 [45.01, 53.11] | 7.76 [2.08, 12.89] |
| 80% | 3.40 [0.96, 5.21] | 3.47 [0.75, 5.26] | 3.33 [0.36, 5.15] | 48.27 [44.48, 52.23] | 4.11 [1.82, 5.87] |
| 90% | 1.61 [0.58, 2.24] | 1.63 [0.64, 2.25] | 1.74 [0.98, 2.29] | 48.37 [44.37, 52.13] | 1.57 [1.00, 2.00] |

Intervals are percentile 95% engine-bootstrap intervals. With only five engines, they quantify finite-fleet variability but will be wide and should not be interpreted as high-powered hypothesis tests.

## Paired comparison with uniform EM

The same resampled target engines are used for each LIP-EM versus uniform-EM comparison. Negative differences favor LIP-EM.

| Observed life | LIP | RMSE difference [95% CI] | Bootstrap P(LIP better) | Engine W/T/L |
|---:|---|---:|---:|---:|
| 10% | Claude | -2.31 [-5.73, 2.23] | 0.851 | 4/0/1 |
| 10% | Gemini | -2.44 [-5.80, 2.17] | 0.853 | 4/0/1 |
| 20% | Claude | -1.76 [-3.59, 0.34] | 0.957 | 3/0/2 |
| 20% | Gemini | -1.97 [-4.33, 0.39] | 0.957 | 3/0/2 |
| 30% | Claude | -0.66 [-5.04, 4.03] | 0.599 | 3/0/2 |
| 30% | Gemini | -0.51 [-5.72, 4.78] | 0.586 | 3/0/2 |
| 40% | Claude | -0.92 [-4.94, 3.25] | 0.683 | 3/0/2 |
| 40% | Gemini | -0.72 [-5.52, 4.11] | 0.624 | 3/0/2 |
| 50% | Claude | -0.13 [-3.56, 3.27] | 0.536 | 3/0/2 |
| 50% | Gemini | 0.01 [-4.24, 3.93] | 0.483 | 3/0/2 |
| 60% | Claude | 1.40 [0.64, 2.24] | 0.000 | 1/0/4 |
| 60% | Gemini | 1.68 [0.35, 2.84] | 0.000 | 0/0/5 |
| 70% | Claude | 1.30 [0.23, 2.02] | 0.006 | 1/0/4 |
| 70% | Gemini | 1.70 [0.38, 2.75] | 0.002 | 1/0/4 |
| 80% | Claude | 0.07 [-0.10, 0.70] | 0.177 | 1/0/4 |
| 80% | Gemini | 0.14 [0.07, 0.52] | 0.000 | 0/0/5 |
| 90% | Claude | -0.13 [-0.49, 0.04] | 0.938 | 4/0/1 |
| 90% | Gemini | -0.10 [-0.42, 0.05] | 0.889 | 4/0/1 |

## Complete-trajectory paired comparison

This summary pools all 45 engine/cutoff predictions. Each bootstrap replicate resamples complete nine-cutoff engine trajectories, preserving within-engine dependence. Negative differences favor the first method.

| First method | Second method | First RMSE | Second RMSE | Difference [95% CI] | P(first better) |
|---|---|---:|---:|---:|---:|
| Claude LIP-EM | Uniform EM | 5.469 | 6.122 | -0.653 [-3.017, 2.049] | 0.684 |
| Gemini LIP-EM | Uniform EM | 5.549 | 6.122 | -0.573 [-3.369, 2.401] | 0.651 |
| Claude LIP-EM | Gemini LIP-EM | 5.469 | 5.549 | -0.080 [-0.649, 0.575] | 0.602 |

## RUL-cap diagnostics

A cap hit means the fitted path did not cross the source-learned failure threshold within the maximum source RUL horizon. These are valid capped predictions, but they explain the abrupt RMSE changes at some cutoffs.

| Observed life | Claude LIP-EM | Gemini LIP-EM | Uniform EM | Global pool | Target only |
|---:|---:|---:|---:|---:|---:|
| 10% | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| 20% | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| 30% | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| 40% | 0/5 | 0/5 | 0/5 | 1/5 | 2/5 |
| 50% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 60% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 70% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 80% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 90% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
