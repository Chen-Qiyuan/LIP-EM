# LIP-EM on C-MAPSS

Code and data for the C-MAPSS case study (paper §5.2 and the C-MAPSS appendix). This folder reproduces:

| Paper artifact | Script | Output |
|---|---|---|
| Table 1 (`tab:cmapss`, C-MAPSS core-speed RMSE), the appendix table with all 9 cutoffs (`tab:cmapss_full`) and the paired tests | `paper_table.py` | `results/runs.json`, `results/table1.json`, `results/table1_rows.tex` |
| Figure 4 (`fig:selected_sources`), core speed of the 100 engines with the 10 targets | `plot_core_speed.py` | `figures/Core_Speed_Highlights.{pdf,png}` |
| Figure 5 (`fig:engine80`), engine 80 at 30% observed life (RUL 70%) | `plot_engine.py` | `figures/engine_80_rul_70.{pdf,png}` |

Each engine of C-MAPSS FD001 is one domain. For each of 10 target engines and each RUL level (90%, 80%, …, 10%), the first (1 − RUL) of the target's cycles are observed and the rest of its physical core speed (sensor 9) is predicted. The backbone is a GLM on a natural cubic spline in the cycle index. The methods are:

- **LIP-C / LIP-G**: LIP-EM with the LIP fitted to Claude / Gemini responses.
- **Uniform**: the same EM with π_k = p₀ for every source.
- **Pooled / Target-Only**: ridge regression on all the data, or on the target prefix only.

## Setup

```
pip install -r requirements.txt
```

The code runs on CPU, and the CPU build of torch is enough (`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`). The shared LIP code is imported from `../lip/lip.py`, so run the scripts from inside this folder of the full repository.

## Reproduce

Run these from this folder. Times were measured on a desktop CPU. The scripts are effectively single-threaded.

```
python paper_table.py                                # Table 1: 450 fits, ~25 s
python paper_table.py --from-runs results/runs.json  # rebuild the table from the shipped runs, ~1 s
python plot_engine.py                                # engine-80 figure, ~3 s
python plot_core_speed.py                            # selected-sources figure, ~2 s
```

`paper_table.py` prints the table for all 9 RUL levels, EM diagnostics, the paired tests, and the LaTeX rows for the paper's five rows. The paper labels each cutoff by its observed life, 1 − RUL, so its rows at 10/30/50/70/90% observed life are the RUL levels 90/70/50/30/10% printed here. Use `--targets 4,9 --out-dir results_subset` to run a subset of targets without overwriting the shipped results (paired tests need at least 3 targets).

### Expected output

Mean (SE) of the future RMSE over the 10 targets. SE is std/√10, using the sample std (ddof = 1) as the paper does. `*` marks the best mean in each row and every method not distinguishable from it (two-sided paired t-test, p ≥ 0.05).

| RUL | LIP-C | LIP-G | Uniform | Pooled | Target-Only |
|---|---|---|---|---|---|
| 90% | 14.3 (2.1)* | 15.0 (3.5)* | 21.4 (3.2)* | 33.9 (2.8) | 43.7 (2.9) |
| 80% | 14.2 (1.6)* | 15.2 (2.5)* | 23.5 (3.0) | 35.8 (2.9) | 45.3 (3.1) |
| 70% | 15.8 (2.1)* | 15.2 (3.8)* | 35.0 (3.2) | 38.1 (3.1) | 58.7 (11.4) |
| 60% | 16.8 (2.7)* | 16.0 (4.5)* | 33.4 (3.5) | 41.1 (3.4) | 90.1 (16.0) |
| 50% | 17.7 (3.7)* | 17.9 (6.5)* | 24.8 (4.6) | 44.7 (3.7) | 36.6 (5.3) |
| 40% | 21.0 (3.9)* | 19.7 (5.1)* | 25.5 (4.4)* | 49.6 (4.1) | 30.8 (11.7)* |
| 30% | 20.1 (3.4)* | 16.4 (3.4)* | 15.2 (2.3)* | 56.5 (4.6) | 27.0 (6.1)* |
| 20% | 15.3 (2.4)* | 13.4 (1.9)* | 13.1 (2.2)* | 66.7 (5.5) | 18.1 (2.7)* |
| 10% | 11.5 (1.7) | 9.0 (1.4)* | 7.3 (0.8)* | 82.9 (6.9) | 10.1 (2.8)* |

**Paired tests at 90/70/50% RUL** (10/30/50% observed life, LIP-C and LIP-G against each baseline, 18 comparisons): LIP is significantly better in 14 of 18 comparisons (two-sided paired t-test, p ≤ 0.029). The exceptions are LIP-C and LIP-G against Uniform at 90% (differences −7.1 and −6.3, p = 0.072 and 0.144), LIP-G against Uniform at 50% (−6.9, p = 0.193) and LIP-G against Target-Only at 50% (−18.7, p = 0.052). Exact Wilcoxon p-values are also printed.

**EM diagnostics.** At every RUL level except 70%, most EM runs (LIP-C, LIP-G and Uniform) end with their weights away from the prior (max_k |w_k − π_k| ≥ 0.02). At 70%, 26 of 30 runs end at the prior after the minimum 19 iterations (t₀ = 14 plus 5 patience iterations). The counts at every RUL level are printed by `paper_table.py`. No run reaches the 1000-iteration cap.

`plot_engine.py` also prints the future RMSEs of the engine-80 figure: LIP-G 6.41, Uniform EM 30.07, Pooled 29.13, Target Only 42.38.

## Method settings

`em.py` implements Algorithm 1 of the paper (closed-form M-step, no approximation). The defaults in `EMConfig` and `experiment.py` are the settings used in the paper:

| | |
|---|---|
| Basis | natural cubic spline, 5 interior knots at cycles 50, 100, …, 250 (window [0, 300]) |
| θ̂ | maximum-likelihood estimates (OLS, σ² = RSS/N) |
| τ (prior std of θ_k around θ⁽ᵗ⁾) | 1e-3, exact τ²-corrected E-step (eq. rel_prox) |
| Null | empirical Bayes mixture of the other sources' fits, weighted by the normalised (1 − w_j), self excluded (eq. eb-null) |
| Tempering | β_k(t) = (1 − e^{−νt})/ε_k with ε_k² = Tr(H₀⁺ H_k) for the pseudo-inverse H₀⁺, ν = 0.05 |
| Stopping | ‖w⁽ᵗ⁾ − w⁽ᵗ⁻¹⁾‖_∞ ≤ 1e-3 for 5 consecutive iterations, counted only from t ≥ t₀ = ⌈ln 2/ν⌉ = 14. At most 1000 iterations |
| LIP | conditional logit with an outside option (eq. lip_choice), p₀ = 0.01, λ_LIP = 0.1, L-BFGS |
| Pooled / Target-Only | ridge regression, λ = 1e4, intercept not penalised |
| Targets | engines 4, 9, 18, 26, 27, 48, 51, 53, 55, 80 (10 of 12 candidates with a fast-growing core speed, sampled with seed 42) |

The pseudo-inverse H₀⁺ enters only the tempering scale. It matters when H₀ is singular, for example when the observed target prefix ends before some of the spline knots.

## Data

Everything needed to reproduce the paper ships in `data/` (about 1 MB):

| | |
|---|---|
| `Physical_Core_Speed_Sequences/` | 100 single-engine CSVs (column `Physical_core_speed_rpm`) = sensor 9 of C-MAPSS FD001 (training set) |
| `Damage_Propagation_Modeling.pdf` | the NASA report (Saxena et al., 2008) that is uploaded to the LLM as context |
| `target_description.txt` | the target description given to the LLM (the desert / HPC-degradation paragraph) |
| `llm_subgroup_queries_gemini.json` | 200 single-choice responses from `gemini-3-flash-preview` (LIP-G) |
| `llm_subgroup_queries_claude.json` | 200 single-choice responses from the Claude Code agent, Opus 4.7 (LIP-C, with the protocol in the paper's agent appendix) |

Both response files use the same 200 subgroups: seed 42, subgroups of 3–5 of the 100 engines. For each target, `lip.load_responses(..., target_basename=...)` drops the queries that contain the target and re-indexes the remaining 99 sources.

**Raw data (optional).** To rebuild the CSVs, download the C-MAPSS archive `CMAPSSData.zip` ("Turbofan Engine Degradation Simulation" data set) from the NASA Prognostics Center of Excellence data set repository (https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) or the NASA Open Data Portal (data.nasa.gov). The archive we used has SHA-256 `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`, and its `train_FD001.txt` has SHA-256 `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`. Then run:

```
python prepare_data.py --raw path/to/train_FD001.txt
```

This writes `data/Physical_Core_Speed_Sequences/` and reproduces the shipped CSVs byte for byte.

**Re-collecting the Gemini LIP (optional, with calls to the Gemini API).**

```
export GEMINI_API_KEY=...
python collect_responses.py --out llm_subgroup_queries.json
```

The defaults are those of the paper's collection: `gemini-3-flash-preview`, temperature 0, 200 queries, subgroup size 3–5, seed 42, 10 parallel workers. The prompt names the sources by anonymous letters (A, B, …) in attachment order, and the attachments themselves are passed without text labels. The subgroups are deterministic, but the LLM's answers can differ from the cached ones. To use a new file, point `LIP_FILES` in `experiment.py` at it.

## Files

| | |
|---|---|
| `paper_table.py` | runs every method at every (target, RUL) and writes Table 1, the paired tests and the LaTeX rows |
| `plot_engine.py` | per-engine prediction figure (default: engine 80, 70% RUL) |
| `plot_core_speed.py` | trajectories of all 100 engines with the 10 targets highlighted |
| `collect_responses.py` | re-collects the Gemini subgroup responses (optional) |
| `prepare_data.py` | rebuilds the per-engine CSVs from the raw NASA file (optional) |
| `experiment.py` | shared setup: targets, RUL levels, hyperparameters, LIP files, data loading and method fitting |
| `em.py` | LIP-EM for a Gaussian GLM (Algorithm 1) |
| `dataset.py` | CSV loader and natural cubic spline basis |
| `../lip/lip.py` | shared LIP code: LLM elicitation, response I/O, LIP fit (eq. LIP_opt) |
| `results/` | outputs of `paper_table.py` for the paper's run (the per-run results rebuild the table without refitting) |
| `figures/` | outputs of the two plotting scripts |

## Reproducibility notes

- With the versions in `requirements.txt`, `results/runs.json` is reproduced exactly. With numpy 2.5.4, the RMSEs differ by at most 6e-11.
- Matplotlib 3.10.9 reproduces the paper's figure layout. Matplotlib 3.11 moves the axes by less than 1 pt, but the plotted values are the same.
