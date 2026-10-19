# N-CMAPSS DS03 joint-sensor prognostics

LIP: Claude LIP (coding-agent session, 30 queries) (`lip_priors/medium/claude_agent_lip_priors.json`).

Each source engine has one shared relevance weight. The Gaussian backbone jointly regresses the 14 signed, operating-adjusted sensor trajectories on the raw-cycle natural-spline basis. The source-only fusion map is applied to the predicted sensor trajectories, after which RUL is the first crossing of the source-terminal threshold.

Exact Gaussian EM uses `tau=0.001`, `nu=0.05`, and counts convergence patience from `t >= 14`.
The uniform-EM baseline initializes every source-membership probability at `p0=0.01`.
As in the C-MAPSS experiment, Target Only and Global Pool use ridge `lambda=10000` with an unpenalized intercept.

| Observed target life | Target only | Global pool | Uniform EM | LIP-EM |
|---:|---:|---:|---:|---:|
| 10% | 30.653 | 30.653 | 8.052 | 7.854 |
| 20% | 37.197 | 37.197 | 7.182 | 7.290 |
| 30% | 44.066 | 43.725 | 7.378 | 7.222 |
| 40% | 42.033 | 46.690 | 6.503 | 6.930 |
| 50% | 6.783 | 47.175 | 5.788 | 6.606 |
| 60% | 1.898 | 47.323 | 5.537 | 5.665 |
| 70% | 5.930 | 47.321 | 3.921 | 4.361 |
| 80% | 4.886 | 46.761 | 3.222 | 3.231 |
| 90% | 2.572 | 47.115 | 1.881 | 1.623 |

Entries are RUL RMSE over the 5 held-out medium-haul target engines.

## Guardrails

- No target RUL or future target sensor value is used for fitting.
- Time is raw flight cycle and is not normalized by engine life.
- The health-index weights, failure threshold, and RUL cap are estimated from source engines only.
- The target flight-class label appears only in the LIP prompt. Source flight-class labels are used only for scoring source recovery; neither is passed to EM or the prognostics model.
