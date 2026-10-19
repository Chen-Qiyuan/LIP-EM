# N-CMAPSS DS03 joint-sensor prognostics

LIP: Gemini LIP (30 independent queries) (`lip_priors/long/gemini_lip_priors.json`).

Each source engine has one shared relevance weight. The Gaussian backbone jointly regresses the 14 signed, operating-adjusted sensor trajectories on the raw-cycle natural-spline basis. The source-only fusion map is applied to the predicted sensor trajectories, after which RUL is the first crossing of the source-terminal threshold.

Exact Gaussian EM uses `tau=0.001`, `nu=0.05`, and counts convergence patience from `t >= 14`.
The uniform-EM baseline initializes every source-membership probability at `p0=0.01`.
As in the C-MAPSS experiment, Target Only and Global Pool use ridge `lambda=10000` with an unpenalized intercept.

| Observed target life | Target only | Global pool | Uniform EM | LIP-EM |
|---:|---:|---:|---:|---:|
| 10% | 32.726 | 32.726 | 8.230 | 5.794 |
| 20% | 39.105 | 39.105 | 8.465 | 6.491 |
| 30% | 45.587 | 45.587 | 8.000 | 7.485 |
| 40% | 38.272 | 49.117 | 7.890 | 7.170 |
| 50% | 7.541 | 49.145 | 6.604 | 6.611 |
| 60% | 5.000 | 49.151 | 3.558 | 5.240 |
| 70% | 7.760 | 48.900 | 1.147 | 2.847 |
| 80% | 4.111 | 48.266 | 3.325 | 3.470 |
| 90% | 1.572 | 48.371 | 1.737 | 1.634 |

Entries are RUL RMSE over the 5 held-out long-haul target engines.

## Guardrails

- No target RUL or future target sensor value is used for fitting.
- Time is raw flight cycle and is not normalized by engine life.
- The health-index weights, failure threshold, and RUL cap are estimated from source engines only.
- The target flight-class label appears only in the LIP prompt. Source flight-class labels are used only for scoring source recovery; neither is passed to EM or the prognostics model.
