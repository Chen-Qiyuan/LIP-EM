# LIP-aided EM on the C-MAPSS dataset

Code and data for the C-MAPSS case study (paper §5.2). Reproduces the paper-style RUL-prediction table by:

1. Eliciting an LLM-derived prior over which historical engines best match a target description (the "Language-Induced Prior", LIP).
2. Running EM with closed-form M-step on a Generalized Linear backbone (natural cubic spline regression on cycle index → physical core speed).
3. Comparing **LIP-EM** (Gemini & Claude variants) against **Uniform-EM**, **Pooled OLS**, and **Target-Only OLS**.

## Environment

| | |
|---|---|
| Python | 3.14.2 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| torch | 2.11.0 |
| matplotlib | 3.10.8 |
| google-genai | 1.75.0 (only needed to re-run elicitation; cached responses ship with the repo) |

CPU is fine — the EM is closed-form GLM, no GPU required. `pip install -r requirements.txt` installs everything.

## Quick start

```
pip install -r requirements.txt          # numpy, pandas, torch, matplotlib, google-genai
python paper_table.py                    # generate the RUL table (text + LaTeX)
python plot_core_speed.py                # generate the cluster-trajectory figure
python plot_engine.py                    # generate the per-engine prediction figure
```

To re-collect the LIP from a Gemini judge (cached responses are bundled, so this is optional):

```
GEMINI_API_KEY=... python collect_responses.py \
    --n-queries 200 --n-parallel 10 \
    --out llm_subgroup_queries_gemini_v3.json
```

## Files

### Code (drivers)

| | |
|---|---|
| `paper_table.py` | Generates the headline RUL table (text + LaTeX) |
| `plot_core_speed.py` | Generates the cluster-trajectory figure (`Core_Speed_Highlights.{pdf,png}`) |
| `plot_engine.py` | Generates the per-engine prediction figure (`engine_<id>_rul_<level>.{pdf,png}`) |
| `collect_responses.py` | Re-collects LIP queries from a Gemini judge (optional; responses are cached) |

### Code (modules)

| | |
|---|---|
| `em.py` | LIP-aided EM with closed-form M-step (paper main text §3.1.2 / §3.2 / §3.3) |
| `lip.py` | LLM elicitation, JSON I/O, LIP fitting (paper eq:LIP_opt) |
| `data.py` | CSV loader, natural-cubic-spline basis (`basis`, `N_KNOTS`, `X_MAX`) |

### Data

| | |
|---|---|
| `Physical_Core_Speed_Sequences/` | 100 single-machine CSVs (column: `Physical_core_speed_rpm`) |
| `Damage_Propagation_Modeling.pdf` | NASA C-MAPSS data-generation paper (uploaded to the LLM as context) |
| `target_description.txt` | Target-domain description (`"desert HPC degradation"`) given to the LLM |
| `llm_subgroup_queries_gemini_v2.json` | 200 Gemini-3-flash single-choice responses (cached) |
| `llm_subgroup_queries_claude.json` | 200 Claude single-choice responses (cached) |

### Outputs (already generated, included for verification)

| | |
|---|---|
| `Core_Speed_Highlights.{pdf,png}` | Trajectory figure produced by `plot_core_speed.py` |
| `engine_80_rul_70.{pdf,png}` | Per-engine prediction figure produced by `plot_engine.py` |

## Final config

Set as defaults in `paper_table.py` and the `EMConfig` dataclass:

| | |
|---|---|
| Basis | natural cubic spline, K=5 knots, x_max=300 |
| τ (prior std on θ_k around θ^(t)) | 1e-3 |
| ν (tempering rate) | 0.05 |
| p_0 (LIP uniform default) | 0.01 |
| ε (LIP L2 regularizer) | 0.1 |
| Convergence | ‖Δw‖_∞ ≤ 1e-3 for 5 consecutive iters |
| Target-only & Pooled | ridge regression, λ=1e4, intercept un-penalized |

## Likelihood model

The LIP fit uses the paper's **conditional logit with outside option** (`eq:lip_choice`):

```
P(k_m | S_m) = exp(α_{k_m}) / (exp(α_0) + Σ_{j ∈ S_m} exp(α_j))
```

with `k_m ∈ S_m ∪ {0}`. The null worth `α_0` is unregularized; each `α_k` has L2 regularization toward `log(p_0/(1-p_0))`. Returns `π_k = sigmoid(α_k)`.

The EM uses paper main-text closed-form M-step (`eq:m_step_update`, exact `M_k = (I + τ²H_k)⁻¹H_k`) and exact E-step τ² correction (`eq:rel_prox`). The §3.3.1 small-tau approximation is available via `EMConfig.approximation=True` for reproducing the appendix table.

## Reference numbers

`paper_table.py` produces (10 randomly-sampled targets, seed=42):

| RUL | LIP-EM (Gemini) | LIP-EM (Claude) | Uniform | Pooled | Target-Only |
|---|---|---|---|---|---|
| 90% | 15.0 (10.5) | **14.3 (6.2)** | 21.4 | 33.9 | 43.7 |
| 80% | 15.2 (7.6) | **14.2 (4.7)** | 23.5 | 35.8 | 45.3 |
| 70% | **15.2 (11.3)** | 15.9 (6.2) | 35.2 | 38.1 | 58.7 |
| 60% | **16.1 (13.5)** | 16.8 (8.0) | 33.8 | 41.1 | 90.1 |
| 50% | 17.9 (19.4) | **17.7 (11.1)** | 24.9 | 44.7 | 36.6 |
| 40% | **19.7 (15.2)** | 21.0 (11.6) | 25.8 | 49.6 | 30.8 |
| 30% | 16.5 (10.3) | 20.1 (10.1) | **15.2 (6.9)** | 56.5 | 27.0 |
| 20% | 13.4 (5.6) | 15.3 (7.2) | **13.1 (6.5)** | 66.7 | 18.1 |
| 10% | 9.0 (4.2) | 11.5 (5.0) | **7.3 (2.5)** | 82.9 | 10.1 |

LIP-EM beats Uniform / Pooled / Target-Only at every cold-start level (RUL ≥ 40%).