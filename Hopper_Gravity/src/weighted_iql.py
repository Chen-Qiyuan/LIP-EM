"""
Weighted IQL: train an IQL policy on a weighted mixture of K source datasets
plus the target slice. The mixture weights match the EM-step's:

    p(sample from target)   ∝ N_0
    p(sample from source k) ∝ w_k * N_k

where {w_k} are read from an EM result.json's `final_weights` field.

Required inputs:
  --em-result-dir   Directory containing result.json with N0 + final_weights
                    (this file may be a real EM result OR a synthetic baseline
                    constructed by make_baseline_em_results.py).
  --target-data     HDF5 of target offline data (must match the EM run).
  --out             Path to save the trained policy state dict (.pt).

Optional:
  --eval-g-list     Comma-separated extra gravities for final eval (default: target_g only)
  --seed            RNG seed (default 42)

The remaining IQL hyperparameters are fixed:
  K=10 sources at integer gravities g=1..10, source data dir = data/sources_replay
  Steps:        200 000
  Batch size:   256
  Hidden:       256
  LR:           3e-4 with cosine decay over training
  IQL:          gamma=0.99, tau=0.005 (target soft-update),
                expectile=0.7, awr_beta=3.0, awr_clamp=100
  Final eval:   200 episodes
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from common import GaussianPolicy, make_hopper

S, A = 11, 3

# ---- Fixed configuration ----
GRAVITIES = list(range(1, 11))
DATA_DIR  = "data/sources_replay"
STEPS       = 200_000
BATCH_SIZE  = 256
HIDDEN      = 256
LR          = 3e-4
GAMMA       = 0.99
TAU         = 0.005
EXPECTILE   = 0.7
AWR_BETA    = 3.0
AWR_CLAMP   = 100.0
EVAL_EVERY  = 20_000
N_EVAL_EPS_INTRAIN = 20
N_EVAL_EPS_FINAL   = 200


class QNet(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S + A, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class VNet(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s):
        return self.net(s).squeeze(-1)


def load_hdf5(path: Path, device: str) -> dict:
    with h5py.File(path, "r") as f:
        s  = np.array(f["observations"], dtype=np.float32)
        a  = np.array(f["actions"], dtype=np.float32)
        r  = np.array(f["rewards"], dtype=np.float32)
        ns = np.array(f["next_observations"], dtype=np.float32)
        d  = np.array(f["terminals"], dtype=np.float32)
    dev = torch.device(device)
    return dict(s=torch.tensor(s, device=dev), a=torch.tensor(a, device=dev),
                r=torch.tensor(r, device=dev), ns=torch.tensor(ns, device=dev),
                d=torch.tensor(d, device=dev))


def evaluate_policy(policy, gravity, n_episodes, seed, device):
    dev = torch.device(device)
    env = make_hopper(gravity)
    rng = np.random.default_rng(seed + 999)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False; trunc = False; ep_ret = 0.0
        while not (done or trunc):
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                a, _ = policy.sample(s)
                a = a.squeeze(0).cpu().numpy()
            obs, r, done, trunc, _ = env.step(a)
            ep_ret += r
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def soft_update(tgt, src, tau=TAU):
    for tp, sp in zip(tgt.parameters(), src.parameters()):
        tp.data.mul_(1 - tau).add_(sp.data * tau)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--em-result-dir", type=str, required=True,
                   help="Path to EM result dir (has result.json with N0 and final_weights).")
    p.add_argument("--target-data", type=str, required=True)
    p.add_argument("--data-dir", type=str, default=DATA_DIR)
    p.add_argument("--eval-g-list", type=str, default=None,
                   help="Comma-separated extra gravities for final eval.")
    p.add_argument("--out", type=str, required=True,
                   help="Path to save trained policy (.pt). Result JSON written to .json next to it.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    K = len(GRAVITIES)

    # --- Load EM result ---
    em_result = json.load(open(Path(args.em_result_dir) / "result.json"))
    final_w = np.array(em_result["final_weights"], dtype=np.float64)
    N_0 = int(em_result["N0"])
    target_g = float(em_result["target_g"])
    print(f"EM result: target_g={target_g}  N0={N_0}  weights={final_w}")
    if len(final_w) != K:
        raise ValueError(f"weights length {len(final_w)} != K={K}")

    # --- Target slice (matches EM's permutation: seed-7 randperm, take first N0) ---
    print(f"Loading target ({args.target_data}, N0={N_0})...")
    target = load_hdf5(Path(args.target_data), dev)
    N_target = target["s"].shape[0]
    g_perm = torch.Generator(device=dev).manual_seed(7)
    perm = torch.randperm(N_target, generator=g_perm, device=dev)
    target_train = {k: v[perm[:N_0]] for k, v in target.items()}
    print(f"  target_train: {target_train['s'].shape[0]:,}")

    # --- Sources ---
    print(f"\nLoading {K} sources at g={GRAVITIES}...")
    sources = []
    for g in GRAVITIES:
        d = load_hdf5(Path(args.data_dir) / f"g{g}.hdf5", dev)
        sources.append(d)
        print(f"  g={g}: {d['s'].shape[0]:,}")

    # --- Sampling probabilities ---
    N_k = np.array([s["s"].shape[0] for s in sources], dtype=np.float64)
    weights_total = final_w * N_k
    Z = N_0 + weights_total.sum()
    p_target = N_0 / Z
    p_source = weights_total / Z
    print(f"\nSampling probabilities (target + {K} sources):")
    print(f"  p_target = {p_target:.6f}  (N_0={N_0})")
    for i, g in enumerate(GRAVITIES):
        print(f"  p_g={g} = {p_source[i]:.6f}  (w={final_w[i]:.4f}, N={int(N_k[i]):,})")

    # --- Networks ---
    pi = GaussianPolicy(state_dim=S, action_dim=A, hidden=HIDDEN).to(dev)
    Q1 = QNet().to(dev); Q2 = QNet().to(dev)
    Q1_t = QNet().to(dev); Q1_t.load_state_dict(Q1.state_dict())
    Q2_t = QNet().to(dev); Q2_t.load_state_dict(Q2.state_dict())
    V    = VNet().to(dev)

    pi_opt = torch.optim.Adam(pi.parameters(), lr=LR)
    q_opt  = torch.optim.Adam(list(Q1.parameters()) + list(Q2.parameters()), lr=LR)
    v_opt  = torch.optim.Adam(V.parameters(), lr=LR)
    pi_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pi_opt, T_max=STEPS, eta_min=0.0)
    q_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(q_opt,  T_max=STEPS, eta_min=0.0)
    v_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(v_opt,  T_max=STEPS, eta_min=0.0)

    # --- Sampler: precompute concatenated buffers + (target, sources) sampling probs ---
    src_choice_probs = torch.tensor(np.concatenate(
        [[p_target], p_source]), dtype=torch.float64, device=dev)
    src_choice_probs = src_choice_probs / src_choice_probs.sum()

    all_datasets = [target_train] + sources
    all_sizes = [d["s"].shape[0] for d in all_datasets]
    cat_s  = torch.cat([d["s"]  for d in all_datasets], dim=0)
    cat_a  = torch.cat([d["a"]  for d in all_datasets], dim=0)
    cat_r  = torch.cat([d["r"]  for d in all_datasets], dim=0)
    cat_ns = torch.cat([d["ns"] for d in all_datasets], dim=0)
    cat_d  = torch.cat([d["d"]  for d in all_datasets], dim=0)
    offsets = torch.tensor([0] + list(np.cumsum(all_sizes)[:-1]),
                            device=dev, dtype=torch.long)
    sizes_t = torch.tensor(all_sizes, device=dev, dtype=torch.long)

    print(f"\nIQL training: {STEPS:,} steps, batch={BATCH_SIZE}, "
          f"in-training eval at g={target_g}")
    history = []
    t0 = time.time()

    for step in range(1, STEPS + 1):
        # Sampling: pick (target | source-k) per row, then a uniform offset within that buffer.
        choices = torch.multinomial(src_choice_probs, BATCH_SIZE, replacement=True)
        max_size = int(sizes_t.max().item())
        rnd = torch.randint(0, max_size, (BATCH_SIZE,), device=dev)
        local_idx = rnd % sizes_t[choices]
        global_idx = offsets[choices] + local_idx
        s  = cat_s[global_idx];  a = cat_a[global_idx];  r = cat_r[global_idx]
        ns = cat_ns[global_idx]; d = cat_d[global_idx]

        # V: expectile regression on min(Q1, Q2)
        with torch.no_grad():
            q_min = torch.minimum(Q1_t(s, a), Q2_t(s, a))
        v = V(s)
        diff = q_min - v
        ww = torch.where(diff > 0, EXPECTILE, 1.0 - EXPECTILE)
        v_loss = (ww * diff.pow(2)).mean()
        v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        # Q: TD with V(s')
        with torch.no_grad():
            v_next = V(ns)
            q_target = r + GAMMA * (1.0 - d) * v_next
        q1 = Q1(s, a); q2 = Q2(s, a)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        q_opt.zero_grad(); q_loss.backward(); q_opt.step()

        # Pi: AWR
        with torch.no_grad():
            adv = q_min - V(s)
            wa = (AWR_BETA * adv).exp().clamp(max=AWR_CLAMP)
        logp = pi.log_prob(s, a)
        pi_loss = -(wa * logp).mean()
        pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()

        pi_sched.step(); q_sched.step(); v_sched.step()
        soft_update(Q1_t, Q1); soft_update(Q2_t, Q2)

        if step % EVAL_EVERY == 0 or step == STEPS:
            m_ret, s_ret = evaluate_policy(pi, target_g, N_EVAL_EPS_INTRAIN,
                                            args.seed + 200 + step, dev)
            elapsed = time.time() - t0
            print(f"[step {step:>7,}/{STEPS:,}]  v_loss={v_loss.item():.3f}  "
                  f"q_loss={q_loss.item():.3f}  pi_loss={pi_loss.item():.3f}  "
                  f"eval@g={target_g}={m_ret:.1f}±{s_ret:.1f}  ({elapsed/60:.1f}min)")
            history.append(dict(step=step, eval_ret=m_ret, eval_std=s_ret))

    # --- Save policy ---
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pi.state_dict(), args.out)
    print(f"\nSaved -> {args.out}")

    # --- Final 200-episode eval at target_g (and any --eval-g-list extras) ---
    eval_gs = [target_g]
    if args.eval_g_list:
        for tok in args.eval_g_list.split(","):
            g = float(tok)
            if g not in eval_gs:
                eval_gs.append(g)
    print(f"\nFinal {N_EVAL_EPS_FINAL}-episode eval at g in {eval_gs}...")
    final_eval = {}
    for g in eval_gs:
        m, s = evaluate_policy(pi, g, N_EVAL_EPS_FINAL,
                                args.seed + 9999, dev)
        final_eval[str(g)] = {"mean": m, "std": s}
        print(f"  g={g}: {m:.1f} +/- {s:.1f}")

    result = {
        "em_result_dir": str(args.em_result_dir),
        "target_g": target_g,
        "N0": N_0,
        "weights": final_w.tolist(),
        "p_target": float(p_target),
        "p_source": [float(x) for x in p_source],
        "steps": STEPS,
        "history": history,
        "final_eval": final_eval,
    }
    with open(Path(args.out).with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
