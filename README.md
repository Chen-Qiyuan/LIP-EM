# Language-Induced Priors for Domain Adaptation

Code, cached LLM responses and per-run results for the paper

> **Language-Induced Priors for Domain Adaptation**<br>
> Qiyuan Chen, Jiayu Zhou, Raed Al Kontar (University of Michigan)<br>
> *Advances in Neural Information Processing Systems (NeurIPS) 2026*

Every figure and table in the experiments section and every experimental study in the appendix can be rebuilt from this repository. Table 2 (Hopper) is rebuilt from the shipped per-run results. A full Hopper rerun needs the replay buffers (see [Data](#data)). No LLM access is needed: the LLM responses used in the paper are cached here.

## Method

In multi-source domain adaptation, a small target sample D₀ is combined with K source data sets D₁, …, D_K, only some of which are relevant to the target. LIP-EM treats the relevance of source k as a latent binary variable c_k with prior P(c_k = 1) = π_k. The prior π, the **Language-Induced Prior (LIP)**, comes from an LLM. For random subgroups of sources, the LLM is given a text description of the target and the raw source data and names the best-matching source, or none. π is fitted to these choices with a conditional logit with an outside option (paper eq. `LIP_opt`). EM then alternates two steps. The E-step computes the weights w_k = σ(β_k(t)·log LR_k + logit π_k), where the log-likelihood ratio log LR_k compares how well the current target parameter explains D_k with a null likelihood (an empirical Bayes mixture of the other sources). The M-step maximises the target log-likelihood plus the w-weighted source log-likelihoods. The tempering β_k(t) = (1 − e^{−νt})/ε_k starts EM at the prior and lets the data take over as it accumulates. With a correct LIP, EM borrows from the relevant sources from the start (cold start). With a wrong LIP, the data still correct the weights once they are informative enough.

## Paper artifacts

Run every command from the folder given in the second column. Runtimes were measured on a desktop CPU with `OMP_NUM_THREADS=4`, unless a GPU is stated.

| Paper artifact | Folder | Command | Runtime |
|---|---|---|---|
| Figure 3, Gaussian estimation (1-d and 2-d panels) | `gaussian/` | `python gaussian.py plot --dim 1`<br>`python gaussian.py plot --dim 2` | 2 s each |
| Table 1, C-MAPSS core-speed RMSE, and the appendix table with all nine cutoffs (`tab:cmapss_full`) | `cmapss/` | `python paper_table.py` (450 fits)<br>`python paper_table.py --from-runs results/runs.json` (from the shipped runs) | 25 s / 1 s |
| Figure 4, core speed of the 100 engines with the 10 targets | `cmapss/` | `python plot_core_speed.py` | 2 s |
| Figure 5, engine 80 at 30% observed life | `cmapss/` | `python plot_engine.py` | 3 s |
| Table 2, Hopper return | `hopper/` | `python src/make_table.py --latex` (from the shipped per-run results)<br>`RESULTS=results_rerun bash scripts/run_all.sh` (full rerun) | 1 s / about 27 h on a GPU |
| App. "LIP Construction via Subgroup Preference Elicitation", paragraph "Query complexity" (`tab:lip_scaling`) | `lip_scaling/` | `python lip_scaling_simulation.py` | 5 s |
| App. "Sensitivity to the Null Likelihood (C-MAPSS)" (`tab:null_sweep`) | `cmapss_ablations/null_sensitivity/` | `python run.py`<br>`python analyze.py` | 40 s |
| App. "Shuffled LIP Control (C-MAPSS)" (`tab:shuffled`) | `cmapss_ablations/shuffled_lip/` | `python run.py`<br>`python run.py --shuffles 500 --seed 314159 --rul 0.9,0.7,0.5` (additional check, not in the paper)<br>`python analyze.py` | 4.5 min |
| App. "Non-Gaussian Likelihoods and Controlled Prior Corruption" (`tab:nongauss`, `tab:corruption`) | `cmapss_ablations/nongaussian_prior/` | `python run.py`<br>`python analyze.py` | 5 min |
| App. "Sensitivity to the Target Description (C-MAPSS)" (`tab:prompt`) | `cmapss_ablations/prompt_ablation/` | `python run.py`<br>`python analyze.py` | 2.5 min |
| App. "Direct RUL Prediction on N-CMAPSS DS03" (`tab:ds03`, plus per-fleet and pooled results not reported in the paper) | `ncmapss_ds03/` | `./run_all.sh`<br>`diff -r results reproduced` | 6 min |

Every `analyze.py` and table script also works on the shipped `results/` without running the experiment first. The README in each folder gives the expected output, the paper numbers it reproduces, the method settings and the options of every script. The appendix "Detailed Experimental Setup" lists the same settings.

## Repository layout

```
LIP-EM/
├── lip/                 # shared LIP code: LLM subgroup queries, response I/O, LIP fit (eq. LIP_opt)
├── gaussian/            # 1-d and 2-d Gaussian estimation (Algorithm 1)
├── cmapss/              # C-MAPSS case study: Table 1, Figures 4 and 5 (Algorithm 1)
├── cmapss_ablations/    # null sweep, shuffled LIPs, Bernoulli/Poisson study, prompt ablation
├── ncmapss_ds03/        # N-CMAPSS DS03 direct-RUL study (Algorithm 1, multi-output)
├── hopper/              # MuJoCo Hopper with a gravity shift: Table 2 (Algorithm 3, neural-network dynamics)
├── lip_scaling/         # query-complexity simulation for the LIP fit (no EM)
├── LICENSE
└── README.md
```

The shared code in `lip/` is not installed as a package. Scripts that use the shared LIP code add the repository root to `sys.path` and import `lip.lip`, so run the scripts from their folder inside the cloned repository. `cmapss_ablations/` also imports `cmapss/em.py`, `cmapss/experiment.py` and `cmapss/dataset.py` in this way.

## Method settings

Every code path uses the method of the paper.

| | |
|---|---|
| E-step | w_k = σ(β_k(t)·log LR_k + logit π_k) |
| Tempering | β_k(t) = (1 − e^{−νt})/ε_k. Algorithm 1: ε_k² = Tr(H₀⁺ H_k) with the pseudo-inverse H₀⁺. Algorithms 2 and 3: ε_k² = d·N_k/N₀ |
| ν | 0.05 (Gaussian, C-MAPSS, N-CMAPSS, Bernoulli/Poisson), 0.1 (Hopper) |
| Stopping | ‖w⁽ᵗ⁾ − w⁽ᵗ⁻¹⁾‖_∞ ≤ 10⁻³ for 5 consecutive iterations, counted only from t ≥ t₀ = ⌈ln 2/ν⌉ (t₀ = 14 for ν = 0.05, 7 for ν = 0.1). At most 1000 iterations (100 for Hopper). The Bernoulli/Poisson study uses tolerance 10⁻⁸ and at most 500 iterations |
| Null | empirical Bayes mixture of the other sources' fits with normalised weights (1 − w_j)/Σ_{i≠k}(1 − w_i) (paper eq. `eb-null`). Hopper uses the pooled dynamics model as the null, and the Bernoulli/Poisson study the true null model |
| θ̂ | maximum-likelihood estimates (a Gaussian covariance divides by n) |
| LIP | conditional logit with an outside option, p₀ = 0.01 |

## Installation

Each folder has its own `requirements.txt`, which pins the package versions that were used to verify that folder's outputs. Use a separate virtual environment per folder, for example:

```bash
git clone https://github.com/Chen-Qiyuan/LIP-EM.git
cd LIP-EM/cmapss
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| Folder | Python | Main packages | Hardware |
|---|---|---|---|
| `gaussian/` | 3.14 | numpy, scipy, matplotlib 3.10.9, torch (CPU) | CPU |
| `cmapss/`, `cmapss_ablations/` | 3.14 | numpy, scipy, pandas, torch (CPU), matplotlib 3.10.9 | CPU |
| `ncmapss_ds03/` | 3.14 (3.12 also tested) | numpy, scipy (h5py only to rebuild the features) | CPU |
| `lip_scaling/` | 3.12 | numpy, torch (CPU) | CPU |
| `hopper/` | 3.14 | torch with CUDA 12.8, gymnasium, mujoco, h5py, pandas (numpy and scipy are enough to rebuild Table 2) | GPU for EM, IQL and data collection |

The CPU build of torch is enough everywhere except `hopper/`. Matplotlib 3.10.9 reproduces the layout of the paper figures exactly. Other versions give the same numbers.

## Data

| Data | Shipped | How to obtain it |
|---|---|---|
| C-MAPSS FD001 physical core speed (sensor 9), 100 engines | yes, `cmapss/data/` (1 MB) | Optional rebuild from the NASA `CMAPSSData.zip` with `cmapss/prepare_data.py`. See `cmapss/README.md` (download page and SHA-256) |
| N-CMAPSS DS03 flight-level features, 15 engines | yes, `ncmapss_ds03/data/` (0.4 MB) | Optional rebuild from `N-CMAPSS_DS03-012.h5` (3.7 GB, NASA PCoE data set repository, item 17) with `ncmapss_ds03/build_features.py`. See `ncmapss_ds03/README.md` |
| Hopper SAC replay buffers, 11 × 10⁶ transitions | no (1.15 GB) | Needed only to rerun EM and IQL. Regenerate with `hopper/src/collect_target_sac.py`. `hopper/README.md` gives the commands and the SHA-256 of the files we used. GPU SAC is not bitwise reproducible |
| Hopper pooled dynamics model θ_pool | yes, `hopper/models/dyn_pool/dynamics.pt` (3.2 MB) | Retrain with `hopper/src/train_dyn_pool.py` (about 14 h on a GPU) |
| Gaussian, Bernoulli/Poisson and query-scaling studies | not needed | synthetic, drawn from fixed seeds |

The per-run results behind every table ship in the `results/` folders, so the tables can be rebuilt without the raw data.

## LLM responses and API keys

The LLM responses used in the paper are cached:

| Experiment | Cached responses | LLM |
|---|---|---|
| Gaussian | `gaussian/data/lip_responses_{1,2}d.json` | gemini-3-flash-preview |
| C-MAPSS, LIP-G and LIP-C | `cmapss/data/llm_subgroup_queries_{gemini,claude}.json` | gemini-3-flash-preview and Claude Code agent (Claude Opus 4.7) |
| C-MAPSS prompt ablation | `cmapss_ablations/prompt_ablation/responses/` | Claude Code agent sessions |
| N-CMAPSS DS03 | `ncmapss_ds03/llm_responses/` | Claude Code agent and independent Claude Code requests (Claude Opus 4.7), and gemini-3-flash-preview |
| Hopper, LIP-G | `hopper/elicitation/gemini_responses.json`, fitted prior `hopper/lips/lip_gemini.json` | gemini-3-flash-preview |
| Hopper, LIP-C | `hopper/elicitation/claude_responses.json` (with the agent's prompt and scripts in `hopper/elicitation/claude_agent/`) | Claude Opus 4.7 local agent |

The response files also hold the LLM's reasoning for every answer, except the C-MAPSS LIP-C file (`cmapss/data/llm_subgroup_queries_claude.json`), which records only the choices. For Hopper LIP-C, the reasoning strings are the per-query summaries written by the agent's `run_queries.py`. To collect new responses, the Gemini scripts read the key from the `GEMINI_API_KEY` environment variable and nowhere else:

```bash
export GEMINI_API_KEY=...   # your own key; do not write it into any file of the repository
```

The Claude responses come from Claude Code agent sessions (paper appendix "Using a Local Agent for Relevance Assessment"). `cmapss_ablations/prompt_ablation/elicit.py` and `ncmapss_ds03/prepare_sessions.py` prepare such sessions, which need the `claude` command-line interface to run. LLM answers change between model versions and sessions, so new responses can give different priors and numbers.

## Hardware

The CPU experiments ran on a workstation with an AMD Ryzen 9 9950X3D CPU and 64 GB of RAM. For Hopper, the replay buffers were collected and the pooled dynamics model was trained on an NVIDIA RTX 5090. The EM and IQL runs behind Table 2 ran on an NVIDIA RTX PRO 6000 Blackwell shared with other jobs.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{chen2026language,
  title     = {Language-Induced Priors for Domain Adaptation},
  author    = {Chen, Qiyuan and Zhou, Jiayu and Al Kontar, Raed},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026}
}
```

## License

The code, the cached LLM responses, the fitted LIPs and the result files are released under the MIT License (see [LICENSE](LICENSE)).

Third-party material keeps its original terms and is not covered by the MIT License:

- `cmapss/data/Damage_Propagation_Modeling.pdf`: Saxena et al. (2008), the C-MAPSS dataset paper, given to the LLM as context.
- `cmapss/data/Physical_Core_Speed_Sequences/`: derived from the NASA C-MAPSS data set (FD001).
- `ncmapss_ds03/data/cycle_features.csv`: derived from the NASA N-CMAPSS data set (DS03).

The NASA data sets are distributed by the NASA Prognostics Center of Excellence. Please cite the original sources (see the paper's references).
