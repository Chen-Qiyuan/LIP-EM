# DS03 engine-bootstrap prognostics

The independent sampling unit is the engine. At each cutoff, the five held-out medium-haul engine errors are resampled with replacement, and RUL RMSE is recomputed for 20,000 bootstrap replicates. Flight rows are not resampled because observations from one run-to-failure engine are serially dependent.

| Observed life | Claude LIP-EM | Gemini LIP-EM | Uniform EM | Global pool | Target only |
|---:|---:|---:|---:|---:|---:|
| 10% | 7.85 [3.49, 10.89] | 7.22 [3.27, 9.67] | 8.05 [3.95, 11.62] | 30.65 [24.44, 35.54] | 30.65 [24.44, 35.54] |
| 20% | 7.29 [3.25, 9.89] | 7.21 [3.21, 9.81] | 7.18 [3.45, 9.70] | 37.20 [31.74, 41.19] | 37.20 [31.74, 41.19] |
| 30% | 7.22 [3.05, 10.28] | 6.98 [3.05, 9.72] | 7.38 [3.53, 10.24] | 43.73 [39.20, 47.53] | 44.07 [39.20, 47.64] |
| 40% | 6.93 [2.85, 9.81] | 6.65 [2.98, 9.05] | 6.50 [3.24, 9.55] | 46.69 [43.02, 51.59] | 42.03 [28.47, 52.18] |
| 50% | 6.61 [2.95, 8.99] | 4.55 [2.38, 6.67] | 5.79 [2.81, 8.28] | 47.18 [43.20, 52.68] | 6.78 [3.47, 9.96] |
| 60% | 5.66 [3.07, 7.65] | 5.48 [3.00, 7.40] | 5.54 [3.03, 7.22] | 47.32 [43.27, 52.79] | 1.90 [1.07, 2.54] |
| 70% | 4.36 [1.99, 6.06] | 4.04 [1.90, 6.22] | 3.92 [1.41, 5.36] | 47.32 [43.41, 52.67] | 5.93 [1.82, 9.35] |
| 80% | 3.23 [1.83, 4.34] | 3.86 [2.71, 4.97] | 3.22 [1.75, 4.24] | 46.76 [42.98, 51.93] | 4.89 [2.99, 6.53] |
| 90% | 1.62 [0.48, 2.67] | 2.32 [0.36, 3.93] | 1.88 [0.50, 3.10] | 47.12 [43.15, 52.13] | 2.57 [1.23, 3.89] |

Intervals are percentile 95% engine-bootstrap intervals. With only five engines, they quantify finite-fleet variability but will be wide and should not be interpreted as high-powered hypothesis tests.

## Paired comparison with uniform EM

The same resampled target engines are used for each LIP-EM versus uniform-EM comparison. Negative differences favor LIP-EM.

| Observed life | LIP | RMSE difference [95% CI] | Bootstrap P(LIP better) | Engine W/T/L |
|---:|---|---:|---:|---:|
| 10% | Claude | -0.20 [-1.53, 1.76] | 0.570 | 2/0/3 |
| 10% | Gemini | -0.83 [-2.78, 1.89] | 0.723 | 3/0/2 |
| 20% | Claude | 0.11 [-0.46, 0.41] | 0.314 | 2/0/3 |
| 20% | Gemini | 0.03 [-0.63, 0.71] | 0.497 | 3/0/2 |
| 30% | Claude | -0.16 [-2.38, 1.83] | 0.551 | 2/0/3 |
| 30% | Gemini | -0.40 [-2.38, 1.59] | 0.614 | 3/0/2 |
| 40% | Claude | 0.43 [-2.22, 3.87] | 0.403 | 2/0/3 |
| 40% | Gemini | 0.14 [-2.35, 3.31] | 0.441 | 3/0/2 |
| 50% | Claude | 0.82 [-1.37, 4.48] | 0.318 | 3/0/2 |
| 50% | Gemini | -1.24 [-2.07, 0.04] | 0.964 | 3/0/2 |
| 60% | Claude | 0.13 [-0.65, 0.84] | 0.391 | 3/0/2 |
| 60% | Gemini | -0.05 [-1.45, 0.90] | 0.531 | 2/0/3 |
| 70% | Claude | 0.44 [-0.56, 1.25] | 0.200 | 2/0/3 |
| 70% | Gemini | 0.12 [-2.55, 1.47] | 0.394 | 1/0/4 |
| 80% | Claude | 0.01 [-0.37, 0.30] | 0.432 | 1/0/4 |
| 80% | Gemini | 0.64 [-0.22, 1.59] | 0.075 | 2/0/3 |
| 90% | Claude | -0.26 [-0.45, 0.05] | 0.941 | 4/0/1 |
| 90% | Gemini | 0.44 [-0.27, 0.82] | 0.289 | 3/0/2 |

## Complete-trajectory paired comparison

This summary pools all 45 engine/cutoff predictions. Each bootstrap replicate resamples complete nine-cutoff engine trajectories, preserving within-engine dependence. Negative differences favor the first method.

| First method | Second method | First RMSE | Second RMSE | Difference [95% CI] | P(first better) |
|---|---|---:|---:|---:|---:|
| Claude LIP-EM | Uniform EM | 5.989 | 5.837 | 0.153 [-1.107, 1.693] | 0.461 |
| Gemini LIP-EM | Uniform EM | 5.620 | 5.837 | -0.216 [-0.957, 0.699] | 0.679 |
| Claude LIP-EM | Gemini LIP-EM | 5.989 | 5.620 | 0.369 [-0.378, 1.120] | 0.204 |

## RUL-cap diagnostics

A cap hit means the fitted path did not cross the source-learned failure threshold within the maximum source RUL horizon. These are valid capped predictions, but they explain the abrupt RMSE changes at some cutoffs.

| Observed life | Claude LIP-EM | Gemini LIP-EM | Uniform EM | Global pool | Target only |
|---:|---:|---:|---:|---:|---:|
| 10% | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| 20% | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 |
| 30% | 0/5 | 0/5 | 0/5 | 4/5 | 5/5 |
| 40% | 0/5 | 0/5 | 0/5 | 2/5 | 3/5 |
| 50% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 60% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 70% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 80% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 90% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
