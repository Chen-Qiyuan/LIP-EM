# N-CMAPSS DS03 joint-sensor prognostics

LIP: Gemini LIP (30 independent queries) (`lip_priors/short/gemini_lip_priors.json`).

Each source engine has one shared relevance weight. The Gaussian backbone jointly regresses the 14 signed, operating-adjusted sensor trajectories on the raw-cycle natural-spline basis. The source-only fusion map is applied to the predicted sensor trajectories, after which RUL is the first crossing of the source-terminal threshold.

Exact Gaussian EM uses `tau=0.001`, `nu=0.05`, and counts convergence patience from `t >= 14`.
The uniform-EM baseline initializes every source-membership probability at `p0=0.01`.
As in the C-MAPSS experiment, Target Only and Global Pool use ridge `lambda=10000` with an unpenalized intercept.

| Observed target life | Target only | Global pool | Uniform EM | LIP-EM |
|---:|---:|---:|---:|---:|
| 10% | 19.032 | 19.032 | 13.354 | 11.922 |
| 20% | 26.541 | 26.541 | 16.054 | 10.937 |
| 30% | 30.646 | 34.208 | 15.596 | 11.556 |
| 40% | 11.184 | 38.186 | 10.359 | 10.859 |
| 50% | 9.080 | 40.223 | 10.102 | 11.025 |
| 60% | 39.360 | 40.954 | 9.520 | 9.498 |
| 70% | 23.011 | 38.783 | 5.410 | 5.554 |
| 80% | 20.805 | 36.376 | 2.707 | 3.044 |
| 90% | 8.975 | 39.762 | 6.296 | 6.206 |

Entries are RUL RMSE over the 5 held-out short-haul target engines.

## Guardrails

- No target RUL or future target sensor value is used for fitting.
- Time is raw flight cycle and is not normalized by engine life.
- The health-index weights, failure threshold, and RUL cap are estimated from source engines only.
- The target flight-class label appears only in the LIP prompt. Source flight-class labels are used only for scoring source recovery; neither is passed to EM or the prognostics model.
