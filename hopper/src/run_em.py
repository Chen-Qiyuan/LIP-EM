"""
Run one Hopper EM cell (one seed, one N0, one prior) of Table 2.

The seed s of a cell controls every random draw:
  * the target sample D0 = perm[:N0], where perm is a seeded permutation of all
    rows of the target buffer (terminals included; see common.target_permutation).
    EM fits the dynamics on the non-terminal rows of D0 (dynamics are undefined
    at episode ends); weighted_iql.py trains on all N0 rows of the same D0;
  * the held-out target slice used only to monitor EM: non-terminal rows of
    perm[-20000:];
  * the M-step minibatches (torch.manual_seed(s)).

Methods (--method):
  claude       LIP-C: EM with the LIP elicited by the Claude agent (lips/lip_claude.json)
  gemini       LIP-G: EM with the LIP from Gemini 3 Flash (lips/lip_gemini.json; argmax g = 7)
  uniform      Uniform EM: pi_k = 0.01 for every source
  pool         Pooled: no EM, weights 1 (IQL on the target plus all sources)
  target_only  Target only: no EM, weights 0 (IQL on the target sample alone)

Writes <out-dir>/result.json with final_weights, N0, split_seed and the SHA-256
of the D0 row indices (weighted_iql.py checks it).

Fixed configuration (paper App. "MuJoCo Hopper"):
  K = 10 sources at g = 1..10, target g = 8.87
  EM:     <= 100 iterations, stop when ||dw||_inf <= 1e-3 for 5 consecutive
          iterations t >= ceil(ln 2 / nu) = 7
  M-step: 100 AdamW steps per iteration, batch 1024, constant lr 1e-5
  E-step: nu = 0.1, eps_k^2 = d_eff * N_k / N0 with d_eff = pool parameter count,
          pool null log p(D_k | theta_pool)
  Model:  hidden 512, n_layers 4, initialized at theta_pool
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from common import (S, A, GRAVITIES, TARGET_G, SOURCE_DIR, TARGET_DATA, POOL_DYN,
                    LIP_DIR, DynamicsModel, load_dataset, target_permutation,
                    index_sha)
from em import EMConfig, em_train, stop_start_iter, total_log_lik

# ---- Fixed configuration ----
N_TEST   = 20_000
HIDDEN   = 512
N_LAYERS = 4
UNIFORM_P0 = 0.01
LIP_FILES = {"claude": "lip_claude.json", "gemini": "lip_gemini.json"}
METHODS = ["claude", "gemini", "uniform", "pool", "target_only"]


def load_target(path: Path, device: str) -> tuple[dict, torch.Tensor]:
    """All rows of the target buffer as {states, actions, deltas}, plus the terminal mask."""
    with h5py.File(path, "r") as f:
        s  = np.array(f["observations"], dtype=np.float32)
        a  = np.array(f["actions"], dtype=np.float32)
        ns = np.array(f["next_observations"], dtype=np.float32)
        t  = np.array(f["terminals"], dtype=bool)
    dev = torch.device(device)
    data = dict(states=torch.tensor(s, device=dev),
                actions=torch.tensor(a, device=dev),
                deltas=torch.tensor(ns - s, device=dev))
    return data, torch.tensor(t, device=dev)


def load_prior(method: str, lip_dir: Path) -> np.ndarray:
    if method == "uniform":
        return np.full(len(GRAVITIES), UNIFORM_P0)
    with open(lip_dir / LIP_FILES[method]) as f:
        pi = np.array(json.load(f)["pi"], dtype=np.float64)
    if len(pi) != len(GRAVITIES):
        raise ValueError(f"pi has length {len(pi)} but K={len(GRAVITIES)}")
    return pi


def dump_atomic(obj: dict, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--N0", type=int, required=True,
                   help="Target sample size (rows of the target buffer).")
    p.add_argument("--seed", type=int, required=True,
                   help="Cell seed: target sample, EM minibatches.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--target-data", type=str, default=str(TARGET_DATA))
    p.add_argument("--source-dir", type=str, default=str(SOURCE_DIR),
                   help="Directory with source HDF5s named g{1..10}.hdf5.")
    p.add_argument("--pool-dyn", type=str, default=str(POOL_DYN))
    p.add_argument("--lip-dir", type=str, default=str(LIP_DIR))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = args.device
    if dev != "cuda":
        print(f"[warning] device={dev}: the target sample D0 is drawn with a {dev} generator and will differ "
              "from the paper's runs, which used CUDA.", flush=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # --- Target sample D0 ---
    target, terminal = load_target(Path(args.target_data), dev)
    N = target["states"].shape[0]
    perm = target_permutation(N, args.seed, dev)
    d0 = perm[:args.N0]
    base = dict(method=args.method, target_g=TARGET_G, gravities=GRAVITIES,
                N0=args.N0, split_seed=args.seed, train_idx_sha=index_sha(d0),
                N0_terminals=int(terminal[d0].sum()))

    if args.method in ("pool", "target_only"):
        w = [1.0 if args.method == "pool" else 0.0] * len(GRAVITIES)
        dump_atomic(dict(base, final_weights=w, final_argmax_g=-1),
                    out_dir / "result.json")
        print(f"DONE {args.method} N0={args.N0} seed={args.seed}")
        return

    n_test = min(N_TEST, N - args.N0)
    test_idx = perm[-n_test:]
    test_idx = test_idx[~terminal[test_idx]]
    train_idx = d0[~terminal[d0]]
    target_train = {k: v[train_idx] for k, v in target.items()}
    target_test = {k: v[test_idx] for k, v in target.items()}
    print(f"target: {N:,} rows  D0: {args.N0} rows ({train_idx.numel()} non-terminal)  "
          f"test: {test_idx.numel():,}")

    # --- Sources ---
    sources = [load_dataset(Path(args.source_dir) / f"g{g}.hdf5", dev)
               for g in GRAVITIES]

    # --- Pool dynamics (theta_init AND null model) ---
    pool = DynamicsModel(S, A, HIDDEN, N_LAYERS).to(dev)
    pool.load_state_dict(torch.load(args.pool_dyn, map_location=dev,
                                    weights_only=True))
    pool.eval()
    theta_init_test_nll = -total_log_lik(pool, target_test) / test_idx.numel()

    # --- Prior and EM ---
    pi = load_prior(args.method, Path(args.lip_dir))
    d_eff = sum(q.numel() for q in pool.parameters() if q.requires_grad)
    cfg = EMConfig(d_eff=d_eff)
    print(f"method={args.method}  pi={np.round(pi, 4)}  d_eff={d_eff:,}")

    t_start = time.time()
    _, hist = em_train(target_train, sources, pi, cfg,
                       theta_init=pool, target_test=target_test)
    w = np.asarray(hist.weights[-1], dtype=np.float64)
    result = dict(base,
                  N0_em=int(train_idx.numel()),
                  pi=pi.tolist(),
                  t0=stop_start_iter(cfg.nu),
                  final_weights=w.tolist(),
                  final_argmax_g=GRAVITIES[int(np.argmax(w))],
                  n_iter=len(hist.weights),
                  test_nll_trajectory=[float(x) for x in hist.test_nll],
                  theta_init_test_nll=float(theta_init_test_nll),
                  em_seconds=time.time() - t_start)
    dump_atomic(result, out_dir / "result.json")
    print(f"DONE n_iter={result['n_iter']} argmax_g={result['final_argmax_g']} "
          f"wmax={w.max():.6f} seconds={result['em_seconds']:.1f}")


if __name__ == "__main__":
    main()
