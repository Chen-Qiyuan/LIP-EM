"""
Run one cell of LIP-aided EM correction.

Required inputs:
  --target-data      HDF5 of target environment offline data
  --target-g         Target gravity (m/s^2)  [for the env, not training]
  --pool-dyn-path    Pretrained pool dynamics model
  --N0               Number of target transitions to use
  --lip-mode         {from_file, uniform}

  Either:  --lip-pi-file  (when --lip-mode=from_file)
  Or:      --lip-p0       (when --lip-mode=uniform)

Optional:
  --out-dir          Where to save result.json and theta_em.pt
  --seed             RNG seed (default 42)

The remaining EM hyperparameters are fixed in the published configuration:
  K=10 sources at integer gravities g=1..10
  N-test = 20 000 (held-out target slice for diagnostics)
  EM:    1000 iters, weight-conv thresh 1e-3, patience 5
  M-step: 100 steps, batch 256, lr 3e-4 with cosine_per_iter decay
  E-step: nu = 0.05, beta uses paper formula with d_eff = pool param count,
          pool null mode (D_k log-lik baseline = log p(D_k | theta_pool))
  Model: hidden 512, n_layers 4
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from common import S, A, DynamicsModel, load_dataset
from em import EMConfig, em_train, total_log_lik

# ---- Fixed configuration ----
GRAVITIES   = list(range(1, 11))    # K=10 sources, g=1..10
DATA_DIR    = "data/sources_replay"
N_TEST      = 20_000
HIDDEN      = 512
N_LAYERS    = 4
EM_ITERS    = 100
M_STEPS     = 100
BATCH_SIZE  = 1024
LR          = 1e-5            # constant (no schedule); matches pool's terminal LR
NU          = 0.1
WCONV_THRESH   = 1e-3
WCONV_PATIENCE = 5


def load_target(path: Path, device: str) -> dict:
    with h5py.File(path, "r") as f:
        s  = np.array(f["observations"], dtype=np.float32)
        a  = np.array(f["actions"], dtype=np.float32)
        ns = np.array(f["next_observations"], dtype=np.float32)
        t  = np.array(f["terminals"], dtype=bool)
    mask = ~t
    s, a, ns = s[mask], a[mask], ns[mask]
    d = ns - s
    dev = torch.device(device)
    return dict(states=torch.tensor(s, device=dev),
                actions=torch.tensor(a, device=dev),
                deltas=torch.tensor(d, device=dev))


def build_lip(mode: str, p0: float, K: int, pi_file: str | None) -> np.ndarray:
    """Returns the LIP pi (length K)."""
    if mode == "uniform":
        return np.full(K, p0, dtype=np.float64)
    if mode == "from_file":
        if pi_file is None:
            raise ValueError("--lip-mode=from_file requires --lip-pi-file")
        with open(pi_file) as f:
            data = json.load(f)
        pi = np.array(data["pi"], dtype=np.float64)
        if len(pi) != K:
            raise ValueError(f"pi has length {len(pi)} but K={K}")
        return pi
    raise ValueError(f"Unknown lip-mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-g", type=float, required=True,
                   help="Target gravity (m/s^2). Recorded in result.json.")
    p.add_argument("--target-data", type=str, required=True,
                   help="HDF5 of target offline data (states, actions, next_obs, terminals).")
    p.add_argument("--pool-dyn-path", type=str, required=True,
                   help="Path to pretrained pool dynamics state-dict.")
    p.add_argument("--data-dir", type=str, default=DATA_DIR,
                   help=f"Directory with source HDF5s named g{{1..10}}.hdf5. Default: {DATA_DIR}")
    p.add_argument("--N0", type=int, required=True,
                   help="Number of target transitions to use for EM.")
    p.add_argument("--lip-mode", type=str, required=True,
                   choices=["from_file", "uniform"])
    p.add_argument("--lip-pi-file", type=str, default=None,
                   help="Required when --lip-mode=from_file. JSON with key 'pi' (length 10).")
    p.add_argument("--lip-p0", type=float, default=0.01,
                   help="Uniform pi value (used when --lip-mode=uniform). Default 0.01.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    K = len(GRAVITIES)

    # --- Target ---
    print(f"Loading target ({args.target_data}, g={args.target_g})...")
    a_tgt = load_target(Path(args.target_data), dev)
    N = a_tgt["states"].shape[0]
    g_tgt = torch.Generator(device=dev).manual_seed(7)
    perm = torch.randperm(N, generator=g_tgt, device=dev)
    n_test = min(N_TEST, N - args.N0)
    target_test  = {k: v[perm[-n_test:]]   for k, v in a_tgt.items()}
    target_train = {k: v[perm[:args.N0]]   for k, v in a_tgt.items()}
    print(f"  target_train: {target_train['states'].shape[0]:,}  "
          f"target_test: {target_test['states'].shape[0]:,}")

    # --- Sources ---
    print(f"\nLoading {K} sources at g={GRAVITIES}...")
    sources = [load_dataset(str(Path(args.data_dir) / f"g{g}.hdf5"), dev)
               for g in GRAVITIES]

    # --- Pool dynamics (theta_init AND null model) ---
    print(f"\nLoading pool dynamics from {args.pool_dyn_path}...")
    pool_model = DynamicsModel(state_dim=S, action_dim=A,
                                hidden=HIDDEN, n_layers=N_LAYERS).to(dev)
    pool_model.load_state_dict(torch.load(args.pool_dyn_path,
                                            map_location=dev,
                                            weights_only=False))
    pool_model.eval()
    target_only_nll = -total_log_lik(pool_model, target_test) / target_test["states"].shape[0]
    print(f"  theta_init test NLL (pool, no EM) = {target_only_nll:.3f}")

    # --- LIP ---
    pi = build_lip(args.lip_mode, args.lip_p0, K, args.lip_pi_file)
    print(f"\nLIP mode = {args.lip_mode}")
    for g, pi_g in zip(GRAVITIES, pi):
        print(f"  pi[g={g:>2}] = {pi_g:.4f}")

    # --- d_eff = NN parameter count ---
    d_eff = sum(p.numel() for p in pool_model.parameters() if p.requires_grad)
    print(f"\nd_eff (pool params) = {d_eff:,}")

    cfg = EMConfig(
        n_em_iter=EM_ITERS,
        m_steps_per_iter=M_STEPS,
        batch_size=BATCH_SIZE,
        lr=LR,
        nu=NU,
        d_eff=d_eff,
        pool_model_path=args.pool_dyn_path,
        weight_convergence_thresh=WCONV_THRESH,
        weight_convergence_patience=WCONV_PATIENCE,
        verbose=True,
    )

    print(f"\n[Run EM] iters<={EM_ITERS}, m_steps={M_STEPS}, "
          f"batch={BATCH_SIZE}, lr={LR}, nu={NU}, lip_mode={args.lip_mode}")
    t0 = time.time()
    theta_em, hist = em_train(
        target_train, sources, pi, cfg,
        theta_init=pool_model, target_test=target_test)
    em_time = time.time() - t0
    print(f"\nEM done in {em_time:.1f}s")

    torch.save(theta_em.state_dict(), out_dir / "theta_em.pt")

    final_w = hist.weights[-1] if hist.weights else None
    final_argmax_idx = int(np.argmax(final_w)) if final_w is not None else -1
    final_argmax_g = GRAVITIES[final_argmax_idx] if final_argmax_idx >= 0 else None

    result = {
        "target_g": args.target_g,
        "gravities": GRAVITIES,
        "lip_mode": args.lip_mode,
        "lip_pi_file": args.lip_pi_file,
        "N0": args.N0,
        "pi": pi.tolist(),
        "final_weights": final_w.tolist() if final_w is not None else None,
        "final_argmax": final_argmax_idx,
        "final_argmax_g": final_argmax_g,
        "test_nll_trajectory": [float(x) for x in hist.test_nll],
        "final_test_nll": float(hist.test_nll[-1]) if hist.test_nll else None,
        "em_seconds": float(em_time),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFinal argmax = g={final_argmax_g} (target_g = {args.target_g})")
    print(f"Saved -> {out_dir}/{{result.json, theta_em.pt}}")


if __name__ == "__main__":
    main()
