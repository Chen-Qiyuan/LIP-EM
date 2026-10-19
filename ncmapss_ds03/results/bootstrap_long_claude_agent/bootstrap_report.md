# DS03 engine-bootstrap prognostics

The independent sampling unit is the engine. At each cutoff, the five held-out long-haul engine errors are resampled with replacement, and RUL RMSE is recomputed for 20,000 bootstrap replicates. Flight rows are not resampled because observations from one run-to-failure engine are serially dependent.

| Observed life | Claude LIP-EM | Gemini LIP-EM | Uniform EM | Global pool | Target only |
|---:|---:|---:|---:|---:|---:|
| 10% | 7.12 [3.64, 9.59] | 5.79 [3.24, 7.62] | 8.23 [4.21, 11.94] | 32.73 [27.29, 37.25] | 32.73 [27.29, 37.25] |
| 20% | 7.28 [2.86, 10.44] | 6.49 [2.37, 9.29] | 8.46 [4.91, 11.51] | 39.10 [34.48, 42.84] | 39.10 [34.48, 42.84] |
| 30% | 7.86 [3.10, 10.67] | 7.49 [3.80, 10.63] | 8.00 [4.32, 11.13] | 45.59 [41.75, 48.90] | 45.59 [41.75, 48.90] |
| 40% | 7.75 [2.94, 10.56] | 7.17 [3.73, 10.06] | 7.89 [4.29, 10.93] | 49.12 [44.98, 53.43] | 38.27 [14.81, 52.06] |
| 50% | 6.97 [2.19, 9.62] | 6.61 [3.13, 9.18] | 6.60 [3.66, 9.04] | 49.15 [44.99, 53.47] | 7.54 [4.88, 9.48] |
| 60% | 6.04 [2.27, 8.24] | 5.24 [2.40, 7.37] | 3.56 [1.46, 5.30] | 49.15 [45.06, 53.46] | 5.00 [4.17, 5.66] |
| 70% | 3.29 [1.39, 4.44] | 2.85 [0.96, 3.91] | 1.15 [0.34, 1.81] | 48.90 [45.01, 53.11] | 7.76 [2.08, 12.89] |
| 80% | 3.35 [1.58, 5.10] | 3.47 [0.75, 5.26] | 3.33 [0.36, 5.15] | 48.27 [44.48, 52.23] | 4.11 [1.82, 5.87] |
| 90% | 1.54 [0.32, 2.19] | 1.63 [0.64, 2.25] | 1.74 [0.98, 2.29] | 48.37 [44.37, 52.13] | 1.57 [1.00, 2.00] |

Intervals are percentile 95% engine-bootstrap intervals. With only five engines, they quantify finite-fleet variability but will be wide and should not be interpreted as high-powered hypothesis tests.

## Paired comparison with uniform EM

The same resampled target engines are used for each LIP-EM versus uniform-EM comparison. Negative differences favor LIP-EM.

| Observed life | LIP | RMSE difference [95% CI] | Bootstrap P(LIP better) | Engine W/T/L |
|---:|---|---:|---:|---:|
| 10% | Claude | -1.11 [-3.12, 2.42] | 0.768 | 4/0/1 |
| 10% | Gemini | -2.44 [-5.80, 2.17] | 0.853 | 4/0/1 |
| 20% | Claude | -1.18 [-3.61, 0.95] | 0.893 | 3/0/2 |
| 20% | Gemini | -1.97 [-4.33, 0.39] | 0.957 | 3/0/2 |
| 30% | Claude | -0.14 [-2.21, 3.14] | 0.580 | 3/0/2 |
| 30% | Gemini | -0.51 [-5.72, 4.78] | 0.586 | 3/0/2 |
| 40% | Claude | -0.14 [-2.07, 2.81] | 0.580 | 4/0/1 |
| 40% | Gemini | -0.72 [-5.52, 4.11] | 0.624 | 3/0/2 |
| 50% | Claude | 0.37 [-1.92, 2.64] | 0.359 | 4/0/1 |
| 50% | Gemini | 0.01 [-4.24, 3.93] | 0.483 | 3/0/2 |
| 60% | Claude | 2.48 [0.48, 4.10] | 0.011 | 1/0/4 |
| 60% | Gemini | 1.68 [0.35, 2.84] | 0.000 | 0/0/5 |
| 70% | Claude | 2.14 [0.72, 3.21] | 0.011 | 2/0/3 |
| 70% | Gemini | 1.70 [0.38, 2.75] | 0.002 | 1/0/4 |
| 80% | Claude | 0.02 [-0.77, 1.41] | 0.416 | 1/0/4 |
| 80% | Gemini | 0.14 [0.07, 0.52] | 0.000 | 0/0/5 |
| 90% | Claude | -0.19 [-0.76, -0.02] | 0.981 | 4/0/1 |
| 90% | Gemini | -0.10 [-0.42, 0.05] | 0.889 | 4/0/1 |

## Complete-trajectory paired comparison

This summary pools all 45 engine/cutoff predictions. Each bootstrap replicate resamples complete nine-cutoff engine trajectories, preserving within-engine dependence. Negative differences favor the first method.

| First method | Second method | First RMSE | Second RMSE | Difference [95% CI] | P(first better) |
|---|---|---:|---:|---:|---:|
| Claude LIP-EM | Uniform EM | 6.102 | 6.122 | -0.020 [-1.555, 1.845] | 0.589 |
| Gemini LIP-EM | Uniform EM | 5.549 | 6.122 | -0.573 [-3.369, 2.401] | 0.651 |
| Claude LIP-EM | Gemini LIP-EM | 6.102 | 5.549 | 0.553 [-1.498, 2.278] | 0.298 |

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
