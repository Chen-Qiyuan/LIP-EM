# LIP query-complexity simulation

Reproduces the appendix simulation on how many LLM subgroup queries `N_M` the
Language-Induced Prior needs as the number of sources `K` grows (scaling to
`K >= 1000` sources). There is no EM in this experiment: it only exercises the
LIP fit (paper eq. `eq:LIP_opt`) and the choice model (eq. `eq:lip_choice`).

## What it shows

The conditional logit has `K + 1` worths (one per source plus the null
option), and a query of size `s` informs `s` of them, so each source is seen
`E[n_k] = N_M * s / K` times. The simulation checks that accuracy is a
function of `N_M / K`: at matched `N_M / K` the `K = 100` and `K = 1000`
results agree within seed noise, so holding accuracy fixed needs `N_M`
linear in `K`. The C-MAPSS case study runs at `N_M / K = 200 / 99 = 2.02`
(about 8 appearances per source), so `K = 1000` would need about 2,000 queries.

## Design

* **Ground truth.** True worths are resampled with replacement from the
  fitted C-MAPSS LIP, `alpha_k - alpha_0` for target machine 80 from the
  cached Claude responses (`n = 99`, mean 1.40, sd 1.93, range
  [-0.90, 6.26]). The null worth is fixed at 0.
* **Queries.** Subgroups are drawn uniformly at random. Choices are sampled
  from the conditional logit with an outside option.
* **Fit.** The objective of `lip.fit_lip` (L-BFGS, strong Wolfe, 200
  iterations, `p_0 = 0.01`), vectorized so that `K = 1000` with 20,000
  queries runs in seconds. Each run first checks the vectorized fit against
  `lip.fit_lip` on the cached responses (`max |pi_ref - pi_fast| = 8.9e-16`).
* **Regularization.** `eps` (`lambda_LIP`) scales with votes per source,
  `eps = 0.1 * (N_M / K) / (200 / 99)`, which gives the paper's `lambda_LIP = 0.1` at
  the C-MAPSS setting.
* **Metrics.** The choice likelihood does not change under a common shift of
  all worths, so the coefficient error is the RMSE of the centred worth
  vector, also divided by the sd of the true worths (`nRMSE`). Ranking is
  measured by Spearman correlation and by recall of the true top 10% of
  sources. Each cell has 5 seeds.
* **Sweeps.**
  1. `K in {10, 100, 1000}` with `s = 5` and `N_M / K in {0.5, 1, 2, 5, 10, 20}`.
  2. `K = 100` with `s in {2, 5, 10}` and appearances per source
     `N_M * s / K in {2, 5, 10, 20, 50}`.

## Data

There is nothing to download and no LLM API is called. The only input is the
cached C-MAPSS subgroup responses, shipped with the C-MAPSS case study at
`../cmapss/data/llm_subgroup_queries_claude.json` (200 single-choice queries over
the 100-machine pool, and the 11 queries that contain machine 80 are dropped when
the file is loaded, which leaves 189).

## Run

From this folder:

```bash
pip install -r requirements.txt
python lip_scaling_simulation.py
```

Options: `--responses PATH` reads the cached responses from another location,
and `--out-dir DIR` writes the outputs somewhere other than `results/`.

The run takes about 5 s on a CPU (4 threads) with a peak memory of about
320 MB. It is deterministic, and repeating it reproduces the shipped
`results/` byte for byte.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `tables.md` | The tables of both sweeps (nRMSE, Spearman, Recall@10% averaged over seeds), the reference-worth statistics, the simulated vs empirical null-choice rate (4.72% vs 4.76%) and the fit check. The paper's appendix table (`tab:lip_scaling`) is the Spearman column of sweep 1 for `N_M / K` ≥ 1. |
| `summary.json` | Configuration, the fit check, and per-cell means and standard errors of every metric for both sweeps (`sweep_k`, `sweep_s`). |
| `runs.csv` | One row per run (165 runs). |

Columns in `runs.csv` and `summary.json`:

* `appearances`: `N_M * s / K`.
* `null_rate`, `n_null_choices`: how often the simulated choice is the null option.
* `coef_rmse_centred`, `coef_nrmse`: error of the centred worths (the headline metric).
* `coef_rmse`: error of the raw contrast `alpha_k - alpha_0`.
* `spearman`, `recall_top10pct`, `recall_top5`: ranking metrics.
* `alpha0_identified` (`summary.json` only): fraction of seeds whose responses contain at least one null choice.

`alpha_0` is left unregularized, as in the paper. When a simulated response
set contains no null choice, `alpha_0` is not identified and drifts to about
-25. `coef_rmse` then becomes large, but the centred and ranking metrics are
unaffected. This happens in the cells whose `alpha_0 identified` entry in
`tables.md` is below 1:

* sweep 1: every `K = 10` cell except `N_M = 50`, and 1 of 5 seeds at
  `K = 100, N_M = 50`.
* sweep 2: `s = 10` at all budgets except `N_M = 500` (a larger subgroup
  makes a null choice less likely).

Every `K = 1000` cell is identified in all seeds.
