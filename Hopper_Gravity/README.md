# LIP-aided EM correction for negative transfer (Hopper, gravity)

Code and trained artifacts to reproduce the IQL returns table:

| N₀ | pool | target only | correct LIP | uniform LIP | weak LIP |
|---|---|---|---|---|---|
| 128  | 1610 (48) |  19 (0)  | **2670 (57)** | 1283 (34) | 1627 (69) |
| 256  | 1298 (32) |  24 (4)  | **2539 (61)** | 1578 (51) | 1647 (70) |
| 512  | 1364 (38) | 223 (3)  | **2586 (55)** | 2491 (57) | 1760 (76) |
| 1024 | 1336 (33) | 179 (9)  | 2599 (57) | 2604 (55) | **2636 (56)** |
| 2048 | 1496 (42) | 164 (13) | 2607 (56) | **2666 (56)** | 2607 (56) |
| 4096 | 1389 (49) | 420 (30) | 2329 (61) | 2443 (62) | 2443 (62) |

Cells are **mean (SEM) episode return** across 200 evaluation rollouts of the trained IQL policy in Hopper-v5 at venus gravity (g = 8.87 m/s²). Bold marks the column-best entry per row. SEM = std / √200.

Each row reflects a different number of target transitions N₀ used in EM. Each column reflects what mixture the IQL policy is trained on:
- `pool`: source mix + target, weights = [1]·10 (no correction)
- `target only`: just N₀ target transitions
- `correct LIP`: weights from EM with the LIP fit from sharp LLM responses (peak at g = 9, the closest source to venus)
- `uniform LIP`: weights from EM with a flat LIP at p₀ = 0.01
- `weak LIP`: weights from EM with a wrong-pointing LIP — fit at p₀ = 0.01, eps = 1.0 from a less-decisive set of LLM responses (g = 7 ≈ g = 9 in mass; ablation showing the LIP needs to be both right-pointing AND confident)

The qualitative story (with the final EM config — see "Environment" and "Final published configuration" below):

| | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| Cold start (only sharp LIP wins) | ✓ | ✓ |   |   |   |   |
| Mid (correct ≈ uniform > weak) |   |   | ✓ |   |   |   |
| Asymptote (all LIPs converge) |   |   |   | ✓ | ✓ | ✓ |

---

## Environment

Pinned dependencies are in `requirements.txt`. The published results were produced with:

| | |
|---|---|
| OS              | Windows 11 |
| GPU             | NVIDIA GeForce RTX 5090 (CUDA 12.8) |
| Python          | 3.14.2 |
| PyTorch         | 2.11.0 + cu128 |
| NumPy           | 2.4.4 |
| h5py            | 3.16.0 |
| pandas          | 3.0.2 |
| gymnasium       | 1.2.3 (with `mujoco==3.8.0` for the Hopper-v5 env) |
| tqdm            | 4.67.3 |
| google-genai    | 1.75.0 (only needed to re-run LLM elicitation; the cached `elicitation/results.json` is already provided) |

Suggested setup:
```bash
python -m venv .venv && source .venv/bin/activate   # or PowerShell equivalent
pip install -r requirements.txt
```

The pipeline assumes a single CUDA-capable GPU; all scripts default to `cuda` if available and fall back to CPU otherwise (training the pool dynamics on CPU is impractical — would take days).

API key for re-running elicitation:
```bash
export GEMINI_API_KEY=...
```

### Expected wall time on a single RTX 5090

| Stage | Wall time |
|---|---|
| Stage 2 (pool dynamics, 1000 epochs)             | ~14 hours |
| Stage 3 (LLM elicitation, 50 queries @ 2-min throttle) | ~2 hours |
| Stage 4 (target SAC, 200 K steps)                | ~30 min |
| EM sweep (18 cells, 2-parallel)                  | ~10–15 min |
| IQL sweep (30 cells, 2-parallel, 200 K steps each) | ~5 hours  |

---

## Repository layout

```
publish/
├── README.md
├── requirements.txt                  # Pinned dependencies
├── src/
│   ├── common.py                     # Hopper env, GaussianPolicy, DynamicsModel, dataset loader
│   ├── em.py                         # E-step / M-step / em_train (pool null, constant LR)
│   ├── run_em.py                     # CLI: one EM cell
│   ├── weighted_iql.py               # CLI: one weighted-IQL cell
│   ├── train_dyn_pool.py             # Train the pool dynamics theta_pool
│   ├── collect_target_sac.py         # Generate target HDF5 via SAC
│   ├── lip_query_gemini.py           # Query Gemini for LIP elicitation (50 queries)
│   ├── fit_lip.py                    # Fit LIP from LLM responses (paper's MLE)
│   ├── setup_elicitation.py          # Anonymize source HDF5s into elicitation CSVs
│   ├── make_baseline_em_results.py   # Synth EM-result.json for {pool, target_only}
│   └── aggregate_results.py          # Build the final mean (SEM) table (Markdown + LaTeX)
├── scripts/
│   ├── run_em_sweep.sh               # 18 EM cells  (6 N₀ × {correct, uniform, weak})
│   └── run_iql_sweep.sh              # 30 IQL cells (6 N₀ × {pool, target_only, correct, uniform, weak})
├── elicitation/
│   ├── technical_report.md           # Hopper environment description (LLM context)
│   ├── target.md                     # Target environment specification
│   ├── results.json                  # Cached LLM responses (50 queries, gemini-3-flash-preview)
│   └── data/                         # source_NN.csv (regenerate with src/setup_elicitation.py)
├── data/                             # Offline RL datasets used to produce the table
│   ├── sources_replay/g{1..10}.hdf5  # 10 SAC medium-replay buffers, ~105 MB each (~1 GB total)
│   └── venus_sac_replay_seed100.hdf5 # Target offline data, ~105 MB
└── results/
    ├── lip.json                      # Fitted "correct" LIP (p₀=0.01, eps=1.0; sharp, peak at g=9)
    ├── lip_weak.json                 # Fitted "weak" LIP from less-decisive responses (ablation; argmax at g=7)
    ├── dyn_pool/                     # Pretrained pool dynamics (h=512, n_layers=4, 1000 epochs)
    │   ├── dynamics.pt               # Final checkpoint
    │   ├── dynamics_e{100..1000}.pt  # Per-100-epoch checkpoints
    │   └── result.json               # Training curves + per-source test log-lik
    ├── em/venus_<lip>_N<N0>/         # 18 EM cells: 6 N₀ × {correct, uniform, weak}
    │   ├── result.json               # Final weights, argmax, test-NLL trajectory
    │   └── theta_em.pt               # Trained dynamics theta after EM
    ├── em_baselines/<kind>_N<N0>/    # 12 synthetic EM-result.json for {pool, target_only} baselines
    │   └── result.json               # weights = [1]·10 (pool) or [0]·10 (target_only)
    └── iql/venus_<lip>_N<N0>.{pt,json,log}  # 30 trained IQL policies + final eval
```

Total package size: ~1.2 GB (1.1 GB is the offline RL data).

---

## Pipeline overview

The published table is produced by 5 stages:

1. **Source data** (one-time): collect K = 10 SAC medium-replay buffers, one per integer gravity g = 1..10.
2. **Pool dynamics** (one-time, ~14 hours): train `theta_pool` on the union of all 10 source datasets.
3. **LIP elicitation** (one-time): anonymize the 10 sources as `source_NN.csv`, send 50 paired-comparison queries to an LLM, fit the conditional-logit LIP from the responses.
4. **Target data** (one-time per target gravity): SAC training in the target environment, save the entire replay buffer.
5. **EM × IQL sweeps**: for each N₀ × LIP-mode combination, run EM correction → save weights → train weighted IQL → evaluate.

Stages 1–4 produce the cached artifacts in `results/` and `elicitation/`. Stage 5 is what `scripts/run_em_sweep.sh` and `scripts/run_iql_sweep.sh` execute.

---

## Data setup

The offline RL datasets used to produce the published table are bundled in `publish/data/`:

```
data/
├── sources_replay/
│   ├── g1.hdf5            # 105 MB each, ~1M transitions
│   ├── g2.hdf5
│   ...
│   └── g10.hdf5
└── venus_sac_replay_seed100.hdf5   # target, ~1M transitions
```

Each HDF5 has standard D4RL keys: `observations`, `actions`, `next_observations`, `rewards`, `terminals`, plus a `metadata` group with `g` (gravity), `seed`, `steps`, `policy`.

To regenerate the **target data** from scratch (e.g. for a different gravity or seed):
```bash
py src/collect_target_sac.py --g 8.87 \
    --steps 200000 --seed 100 \
    --out data/venus_sac_replay_seed100.hdf5
```
(~30 min on a single GPU.)

Each of the 10 source HDF5s was produced by an analogous SAC run at the corresponding integer gravity.

---

## End-to-end reproduction

The package bundles **all** stage outputs (data, dynamics, LIPs, EM thetas, IQL policies), so you have two options:

### Option A — just rebuild the table

If you trust the bundled results, regenerate the headline table directly:
```bash
py src/aggregate_results.py --results-dir results/iql \
    --target-g 8.87 --tag venus
```
(<1 second.)

### Option B — re-run from any stage

The sweep scripts are idempotent — they skip any cell whose result file already exists. To re-run a specific cell, delete its file(s) first.

From `publish/`:

```bash
# 1. Generate baseline EM-result.json files for {pool, target_only} (already
#    present at results/em_baselines/; safe to re-run).
py src/make_baseline_em_results.py --target-g 8.87 \
    --out-dir results/em_baselines

# 2. EM sweep (18 cells; ~10–15 min wall at 2-parallel; bundled results in
#    results/em/ will all be skipped — delete any cell to force re-run).
nohup bash scripts/run_em_sweep.sh > results/em_master.log 2>&1 &

# 3. IQL sweep (30 cells; ~5 hr wall at 2-parallel, 200K steps each; bundled
#    results in results/iql/ will all be skipped).
nohup bash scripts/run_iql_sweep.sh > results/iql_master.log 2>&1 &

# 4. Rebuild the final table.
py src/aggregate_results.py --results-dir results/iql \
    --target-g 8.87 --tag venus
```

Both sweep scripts run at most 2 cells in parallel and skip any cell that already has its result file, so they are idempotent and resumable.

---

## Re-running stages 1–4 from scratch

### Stage 2 — pool dynamics
```bash
py src/train_dyn_pool.py --out-dir results/dyn_pool
```
Trains 1000 epochs over the 10-source pool (~14 hours on a single GPU). Default config: hidden = 512, n_layers = 4, batch = 1024, lr = 3e-4 with cosine decay to **eta_min = 1e-5**. The terminal 1e-5 matches the EM M-step's constant LR — the M-step picks up where pool training left off and never re-peaks.

### Stage 3 — LIP elicitation
```bash
# (a) Anonymize sources as elicitation/data/source_NN.csv (mapping saved to
#     elicitation/_truth.json; SEED-DEPENDENT — see warning below).
py src/setup_elicitation.py --seed 42

# (b) Query the LLM (set $GEMINI_API_KEY first; ~$0.50 worth of API calls).
py src/lip_query_gemini.py --n-queries 50 --max-concurrent 3 --fire-interval 120

# (c) Fit the LIP (default p₀ = 0.01, eps = 1.0; outputs results/lip.json).
py src/fit_lip.py
```

> **Warning.** `setup_elicitation.py` chooses a random source_NN ↔ g mapping seeded by `--seed`. The fitting script `src/fit_lip.py` contains a hardcoded `SOURCE_TO_G` reflecting the mapping used to produce the cached `elicitation/results.json`. If you re-run `setup_elicitation.py` with a different seed or input order, you must update `SOURCE_TO_G` in `fit_lip.py` to match the new `elicitation/_truth.json` before fitting.

The cached `elicitation/results.json` is the 50-query response set used for the published "correct" LIP (`results/lip.json`); with that file present you can skip steps (a) and (b) and go straight to `fit_lip.py`. The "weak" LIP (`results/lip_weak.json`) is fit identically (p₀ = 0.01, eps = 1.0) but on a separate set of less-decisive responses; it is included as an ablation showing what happens when the LLM cannot disambiguate the closest source.

---

## Final published configuration

All hyperparameters in this package are set to the values used to produce the table. The CLI tools accept only the inputs that vary between cells.

### Pool dynamics (`train_dyn_pool.py`)
| | |
|---|---|
| Architecture | hidden = 512, n_layers = 4, LayerNorm + SiLU |
| Optimizer    | AdamW, lr = 3e-4, weight_decay = 1e-4 |
| Schedule     | CosineAnnealingLR, **eta_min = 1e-5**, T_max = **1000 epochs** |
| Batch size   | 1024 |
| Held-out test | 20 000 transitions per source |

### LIP fit (`fit_lip.py`)
| | |
|---|---|
| Form | conditional logit with null option (paper's MLE) |
| p₀   | 0.01 |
| eps  | 1.0  (L2 reg on α₁..α_K, not on α₀) |
| Optimizer | LBFGS with strong-Wolfe line search |
| n_queries | 50 |

### EM (`run_em.py` → `em.py`)
| | |
|---|---|
| Outer loop      | up to **100 iterations**, weight-conv stop at ‖Δw‖_∞ ≤ 1e-3 for 5 consecutive iters |
| M-step          | 100 grad steps per iter, batch = **1024** |
| Optimizer       | AdamW, weight_decay = 1e-4 |
| Learning rate   | **constant 1e-5** (matches pool's terminal training LR) |
| Tempering (β)   | β = (1 - exp(-νt)) / √(d_eff · N_k / N₀), **ν = 0.1** |
| d_eff           | NN parameter count from theta_pool |
| Null model      | log p(D_k \| theta_pool) — pool-baseline, computed once |
| theta init      | theta_pool |
| Network         | hidden = 512, n_layers = 4 |

The constant-LR choice matters: with a well-trained pool, peaking the M-step LR back to 3e-4 each iter yanks theta off the pool basin and corrupts the diff = log L(D_k\|θ^t) − log L(D_k\|θ_pool) signal. Holding theta near the pool — by training with the same LR the pool finished at — preserves the data signal.

### Weighted IQL (`weighted_iql.py`)
| | |
|---|---|
| Steps         | 200 000 |
| Optimizer     | Adam, lr = 3e-4, cosine decay over training |
| Batch size    | 256 |
| Hidden        | 256, LayerNorm in Q and V networks |
| γ, τ          | 0.99, 0.005 (target soft-update) |
| Expectile     | 0.7 |
| AWR β / clamp | 3.0 / 100 |
| Final eval    | 200 episodes |

### Target data (`collect_target_sac.py`)
Vanilla SAC for 200 000 env steps in Hopper-v5 with custom gravity. Replay buffer is saved as the target offline dataset (no policy filtering). Default seed = 100 for the published `venus_sac_replay_seed100.hdf5`.

---

## Citation
Citation details will be added once the paper is on arXiv.
