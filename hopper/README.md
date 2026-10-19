# MuJoCo Hopper with a gravity shift (paper Table 2)

This folder reproduces **Table 2** of the paper (MuJoCo Hopper return) and the Hopper statistics in the text and appendix (which source EM selects in each run). `src/make_table.py` also prints seed-paired tests, which are not reported in the paper.

The agent is deployed at Venus gravity (g = 8.87 m/s²). It has a small target sample D₀ of N₀ transitions and K = 10 source replay buffers collected at g = 1, …, 10 m/s². For each cell, LIP-EM learns source weights w_k with a neural-network dynamics model. Weighted IQL then trains a policy on the target sample plus the sources, each source sampled in proportion to w_k·N_k. The table reports the policy's mean return over 200 evaluation episodes at g = 8.87.

| Column | `--method` | Weights used by IQL |
|---|---|---|
| LIP-C | `claude` | EM with the LIP elicited by the Claude Opus 4.7 local agent (`lips/lip_claude.json`, peak at g = 9) |
| LIP-G | `gemini` | EM with the LIP from Gemini 3 Flash (`lips/lip_gemini.json`, misspecified, argmax at g = 7) |
| Uniform EM | `uniform` | EM with π_k = 0.01 for every source |
| Pooled | `pool` | w_k = 1 for every source (no EM) |
| Target Only | `target_only` | w_k = 0 for every source (no EM) |

**Design.** We use 10 seeds, s = 42, …, 51, and N₀ ∈ {64, 128, 256, 512, 1024, 2048, 4096}. In each cell, the seed fixes:
- The target sample D₀: the first N₀ rows of a seeded permutation of the 1,000,000 target rows.
- The EM minibatches.
- The IQL training seed.

EM and IQL use the **same** D₀. EM fits the dynamics on the non-terminal rows of D₀, because the dynamics are undefined at the end of an episode. IQL uses all N₀ rows. `run_em.py` records the SHA-256 of the D₀ row indices, and `weighted_iql.py` stops with an error if its own draw does not match that hash. Each cell in the table is the mean (SE) over the 10 seeds.

## Layout

```
hopper/
├── README.md, requirements.txt
├── src/
│   ├── common.py              # Hopper env, policy / Q / dynamics networks, data loading, D0 permutation
│   ├── em.py                  # LIP-EM with NN dynamics (E-step, M-step, stopping rule)
│   ├── run_em.py              # one EM cell (or the pool / target_only weights)
│   ├── weighted_iql.py        # one weighted-IQL cell on the same D0 + 200-episode evaluation
│   ├── make_table.py          # Table 2 (Markdown report or LaTeX rows) from results/
│   ├── train_dyn_pool.py      # pool dynamics theta_pool (EM initialization and null model)
│   ├── collect_target_sac.py  # SAC replay buffers (target and sources)
│   ├── setup_elicitation.py   # anonymize sources as source_NN.csv for the LLM
│   ├── lip_query_gemini.py    # Gemini subgroup queries (LIP-G)
│   └── fit_lip.py             # fit the LIP from the responses (shared lip/lip.py)
├── scripts/run_all.sh         # all 350 cells (EM, then IQL); resumable
├── elicitation/               # technical_report.md, target.md, gemini_responses.json, claude_responses.json (50 queries each), claude_agent/
├── lips/                      # lip_claude.json (LIP-C), lip_gemini.json (LIP-G)
├── models/dyn_pool/           # dynamics.pt (pretrained theta_pool, 3.2 MB) + training log
└── results/
    ├── seed<s>/em/<method>_N<N0>/result.json   # EM weights, iterations, D0 hash (350 files)
    ├── seed<s>/iql/<method>_N<N0>.json         # IQL training curve + final 200-episode eval (350 files)
    ├── table2.md                               # output of src/make_table.py
    └── table2_rows.tex                         # output of src/make_table.py --latex
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The runs used Python 3.14.4, PyTorch 2.11.0 with CUDA 12.8, gymnasium 1.2.3 and MuJoCo 3.8.0. To rebuild the table you only need `numpy` and `scipy`.

## 1. Rebuild Table 2 from the shipped results (CPU, < 5 s)

Run from `hopper/`:

```bash
python src/make_table.py > results/table2.md      # Markdown report
python src/make_table.py --latex                  # LaTeX rows of the paper table
```

`results/table2.md` contains:
- The mean ± SE [min–max] over the seeds for every cell.
- The source that EM concentrates on in each run (the source with weight > 0.5, otherwise "prior"), with the number of EM iterations.
- The seed-paired comparisons: difference, paired t-test p, exact sign-flip p, and wins.
- The returns grouped by the selected source.

## 2. Data (not shipped, 1.15 GB)

The scripts expect this layout:

```
hopper/data/
├── sources_replay/g1.hdf5 … g10.hdf5      # 10 source buffers, 1,000,000 transitions each (105 MB each)
└── venus_sac_replay_seed100.hdf5           # target buffer, 1,000,000 transitions (1,522 terminals)
```

Each file stores one complete SAC replay buffer in the D4RL layout: `observations`, `actions`, `next_observations`, `rewards`, `terminals`, plus a `metadata` group with `g`, `seed`, `steps` and `policy = vanilla_sac_replay`. To store the data elsewhere, pass `--target-data` and `--source-dir` to the scripts.

**Regenerate.** All 11 buffers come from 1M-step runs of `src/collect_target_sac.py`. Each run takes several hours on one GPU.

```bash
python src/collect_target_sac.py --g 8.87 --steps 1000000 --seed 100 --out data/venus_sac_replay_seed100.hdf5
for g in 1 2 3 4 5 6 7 8 9 10; do
  python src/collect_target_sac.py --g $g --steps 1000000 --seed 42 --out data/sources_replay/g$g.hdf5
done
```

SAC on a GPU is not bitwise reproducible, so regenerated buffers will differ from ours. The shipped results were produced from files with these SHA-256 checksums:

```
e4aa74a6648c1097630a788b4b4e8c36e9d8b3cbbe18d288dd9bba524ecc545a  venus_sac_replay_seed100.hdf5
6c76a779f39c347f35c9377b9391f20315fc6add0d6069799df3b7911424a3a9  sources_replay/g1.hdf5
3a75ea8e6515f90348607ea936f9bbb8988cc2cf90d8d9aa343924cba6f3283c  sources_replay/g2.hdf5
940d0f6ac268917169577776c6b779f5257d51bc074a4c324099deb25f199b3e  sources_replay/g3.hdf5
4eb014b25154ce4a1c2acc8b287d53f242f3fb361f354e92b38438151bad7141  sources_replay/g4.hdf5
9ad024357e00beeb04b5008996410f01b6694f3b4012c951ff91ff4601dae48b  sources_replay/g5.hdf5
9e912167c70031b5fc0b4a4482cfd0ce572b2e555f7b65b4b63be039c75c99f4  sources_replay/g6.hdf5
ace1d469065c010f4b5cc5b3d74cb6c76a6172c3a26d88c18445e4814bcc1dbd  sources_replay/g7.hdf5
d5de16307738b5bea733ffdee484a65e4e30de2080e1ec512f4c731765620f03  sources_replay/g8.hdf5
1648d21cd15f2b7bda2210640c8b0de848973a439d447c5e8c0455c426f0864e  sources_replay/g9.hdf5
c6913c36a8675e25e692b0805f983c55353d5f94a1bb8a2271aceb5290008961  sources_replay/g10.hdf5
```

## 3. Re-run the experiment (GPU)

Run one cell from `hopper/`:

```bash
python src/run_em.py --method claude --N0 128 --seed 42 --out-dir results_rerun/seed42/em/claude_N128
python src/weighted_iql.py --em-result-dir results_rerun/seed42/em/claude_N128 \
    --out results_rerun/seed42/iql/claude_N128.pt --seed 42
```

To run all 350 cells (10 seeds × 7 N₀ × 5 methods), use the driver script:

```bash
RESULTS=results_rerun NPAR=2 nohup bash scripts/run_all.sh > run_all.log 2>&1 &
python src/make_table.py --results results_rerun
```

The driver:
- Skips every cell whose IQL result already exists, so it can be resumed.
- Makes a second pass to retry failed cells.
- Starts a cell only when at least 10 GB of RAM and 4,000 MiB of GPU memory are free (settings `MIN_RAM_GB` and `GPU_FREE_MIB`).
- Accepts `SEEDS`, `N0LIST` and `METHODS` to select a subset.

Logs go to `logs/`. On a shared machine you can cap the memory and make the run the first target of the OOM killer:

```bash
RESULTS=results_rerun systemd-run --user --scope -p MemoryMax=10G choom -n 1000 -- bash scripts/run_all.sh
```

**Runtime.** Our runs used one RTX PRO 6000 Blackwell GPU (shared with other jobs) and 6 cells in parallel:
- One EM cell takes 15 s to 6 minutes, with a median of 80 s (about 50 s for `claude`, N₀ = 128, alone on the GPU).
- One IQL cell (200,000 steps plus 200 evaluation episodes) takes about 15 minutes.
- The full grid took about 27 hours.

**Device.** D₀ is drawn with a generator on the run device. The shipped results use CUDA draws, so a CPU run draws a different (equally valid) target sample.

**Outputs.**
- `run_em.py` writes `result.json` with these fields: `method`, `N0`, `split_seed`, `train_idx_sha`, `N0_em` (non-terminal rows used by EM), `pi`, `t0`, `final_weights`, `final_argmax_g`, `n_iter`, `test_nll_trajectory` (held-out target NLL, diagnostics only), `theta_init_test_nll` and `em_seconds`.
- `weighted_iql.py` writes the policy `.pt` and a JSON file with these fields: the weights, the sampling probabilities, the training curve `history`, and `final_eval[<g>] = {mean, std}` over 200 episodes.

## Method configuration (paper App. "MuJoCo Hopper")

| | |
|---|---|
| E-step | w_k = σ(β_k(t)·[log p(D_k \| θ^t) − log p(D_k \| θ_pool)] + logit π_k), with β_k(t) = (1 − e^{−νt}) / √(d_eff·N_k/N₀), ν = 0.1, and d_eff = 811,030 (the parameter count of θ_pool) |
| Null | pool null log p(D_k \| θ_pool), computed once before EM |
| M-step | 100 AdamW steps per iteration, batch 1024, constant lr 1e-5 (the pool's final lr), weight decay 1e-4, gradient clipping 1.0, with θ initialized at θ_pool |
| Stopping | stop when ‖w^t − w^{t−1}‖_∞ ≤ 1e-3 for 5 consecutive iterations, counted only for t ≥ t₀ = ⌈ln 2/ν⌉ = 7 (derived from ν in `em.stop_start_iter`). At most 100 iterations |
| Dynamics model | Gaussian MLP, hidden 512, 4 layers (Linear + LayerNorm + SiLU), input/output normalization |
| Pool θ_pool | 1000 epochs on the union of the 10 sources, batch 1024, AdamW lr 3e-4 with cosine decay to 1e-5, 20,000 held-out transitions per source |
| IQL | 200,000 steps, batch 256, hidden 256 with LayerNorm, Adam lr 3e-4 with cosine decay, γ = 0.99, τ = 0.005, expectile 0.7, AWR β = 3, clamp 100, and a final evaluation over 200 episodes |
| LIP fit | conditional logit with an outside option (`lip/lip.py`), p₀ = 0.01, λ_LIP = 1.0 (`--eps` of `src/fit_lip.py`) |

At N₀ = 64, 9 of the 20 LIP runs (3 with `claude`, 6 with `gemini`) stop at the 100-iteration cap. In those runs the weights have already settled at 0.96–0.97, but minibatch noise keeps ‖Δw‖ near 1e-3.

## LIPs and elicitation

- **LIP-G** (`lips/lip_gemini.json`) is fitted from the 50 cached Gemini 3 Flash responses in `elicitation/gemini_responses.json`. Running `python src/fit_lip.py` refits it with the shared `lip/lip.py` and prints the difference to the shipped prior (7e-8). To collect new responses, set `GEMINI_API_KEY` in the environment and run:

  ```bash
  python src/setup_elicitation.py --seed 42        # elicitation/data/source_NN.csv (needs data/)
  python src/lip_query_gemini.py --out elicitation/my_responses.json --max-concurrent 3 --fire-interval 120
  python src/fit_lip.py --responses elicitation/my_responses.json --out lips/my_lip.json
  ```

  The seed-42 anonymization reproduces the `source_NN` → g map that is hard-coded in `fit_lip.py`. The query script draws the same 50 subgroups as the cached file.
- **LIP-C** (`lips/lip_claude.json`) was fitted with the same p₀ and λ_LIP from the 50 subgroup answers of the Claude Opus 4.7 local agent (paper appendix "Using a Local Agent for Relevance Assessment"). The agent's prompt, its gravity-estimation scripts and its 50 answers (`elicitation/claude_responses.json`, the agent's original output, see `elicitation/claude_agent/README.md`) are shipped, and `python src/fit_lip.py --responses elicitation/claude_responses.json --compare lips/lip_claude.json` reproduces the prior to 4e-7.

## Pool dynamics

`models/dyn_pool/dynamics.pt` is the pretrained θ_pool used in every EM cell. To retrain it, run `python src/train_dyn_pool.py`. The run takes about 14 hours on one GPU and overwrites `models/dyn_pool/`.

## Checks of this release

- Running `src/make_table.py` on the shipped results reproduces the report of the original runs byte for byte. It also reproduces the LaTeX rows of Table 2 in the paper.
- We re-ran one cell (`claude`, N₀ = 128, seed 42) on the same GPU, with the original data and the shipped pool model:
  - `run_em.py` reproduced `results/seed42/em/claude_N128/result.json` exactly: the same D₀ hash, `final_weights`, `n_iter` = 46 and test-NLL trajectory.
  - `weighted_iql.py` drew the same D₀ and reproduced `results/seed42/iql/claude_N128.json` exactly: the same training curve and a final return of 2693.0 ± 745.2.
