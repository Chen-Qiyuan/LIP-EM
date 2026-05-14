"""
Vanilla SAC training, then save the entire replay buffer as the target dataset.

No target-return matching, no archive selection — just train SAC for N steps
in a Hopper-v5 environment with custom gravity, and dump every transition
to an HDF5 file.

Usage:
    py collect_target_sac.py --g 8.87 --steps 200000 \
        --out data/venus_sac_replay_seed100.hdf5 --seed 100
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from common import (S as STATE_DIM, A as ACTION_DIM, make_hopper, WidePolicy,
                    QNetwork)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g", type=float, required=True,
                   help="Target gravity (m/s^2).")
    p.add_argument("--steps", type=int, default=200_000,
                   help="SAC env steps (= replay buffer size).")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--out", type=str, required=True,
                   help="Output HDF5 path.")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)

    env = make_hopper(args.g)
    act_high_np = env.action_space.high
    act_high = torch.tensor(act_high_np, dtype=torch.float32, device=dev)
    print(f"[SAC] g={args.g}  steps={args.steps:,}  seed={args.seed}")

    # Networks
    policy = WidePolicy().to(dev)
    q1 = QNetwork().to(dev); q2 = QNetwork().to(dev)
    q1_tgt = QNetwork().to(dev); q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt = QNetwork().to(dev); q2_tgt.load_state_dict(q2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=dev)
    target_entropy = -float(ACTION_DIM)

    p_opt     = optim.Adam(policy.parameters(), lr=3e-4)
    q1_opt    = optim.Adam(q1.parameters(),     lr=3e-4)
    q2_opt    = optim.Adam(q2.parameters(),     lr=3e-4)
    alpha_opt = optim.Adam([log_alpha],         lr=3e-4)

    # Replay buffer (rewards/dones stored as (B, 1) so they broadcast correctly
    # with Q outputs of shape (B, 1) — shape (B,) silently produces (B, B)).
    buf_size = args.steps
    buf_s  = np.zeros((buf_size, STATE_DIM),  dtype=np.float32)
    buf_a  = np.zeros((buf_size, ACTION_DIM), dtype=np.float32)
    buf_r  = np.zeros((buf_size, 1),          dtype=np.float32)
    buf_ns = np.zeros((buf_size, STATE_DIM),  dtype=np.float32)
    buf_d  = np.zeros((buf_size, 1),          dtype=np.float32)
    buf_idx = 0

    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    ep_ret = 0.0; ep_len = 0; n_eps = 0
    recent_rets: list[float] = []
    t0 = time.time()

    for step in range(args.steps):
        # Action: warmup random, else SAC sample
        if step < args.warmup:
            a_norm = rng.uniform(-1.0, 1.0, size=(ACTION_DIM,)).astype(np.float32)
        else:
            with torch.no_grad():
                s_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                a_t, _ = policy.sample(s_t)
                a_norm = a_t.squeeze(0).cpu().numpy()
        a_env = a_norm * act_high_np
        next_obs, rew, term, trunc, _ = env.step(a_env)
        buf_s[buf_idx]      = obs
        buf_a[buf_idx]      = a_norm
        buf_r[buf_idx, 0]   = rew
        buf_ns[buf_idx]     = next_obs
        buf_d[buf_idx, 0]   = float(term)
        buf_idx += 1
        ep_ret += float(rew); ep_len += 1

        if term or trunc:
            recent_rets.append(ep_ret)
            if len(recent_rets) > 50:
                recent_rets = recent_rets[-50:]
            n_eps += 1
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            ep_ret = 0.0; ep_len = 0
            if n_eps % 100 == 0:
                avg50 = float(np.mean(recent_rets))
                print(f"  step={step:,} ep={n_eps} avg50={avg50:.0f} "
                      f"alpha={log_alpha.exp().item():.3f}", flush=True)
        else:
            obs = next_obs

        # SAC updates after warmup
        if buf_idx >= 1024 and step >= args.warmup:
            bi = rng.integers(0, buf_idx, size=256)
            bs  = torch.tensor(buf_s[bi],  device=dev)
            ba  = torch.tensor(buf_a[bi],  device=dev)
            br  = torch.tensor(buf_r[bi],  device=dev)
            bns = torch.tensor(buf_ns[bi], device=dev)
            bd  = torch.tensor(buf_d[bi],  device=dev)
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                na, nlogp = policy.sample(bns)
                q_tgt = torch.min(q1_tgt(bns, na), q2_tgt(bns, na))
                y = br + 0.99 * (1 - bd) * (q_tgt - alpha * nlogp.unsqueeze(-1))
                y = y.clamp(-500, 1500)

            for q, qo in [(q1, q1_opt), (q2, q2_opt)]:
                loss = (q(bs, ba) - y).pow(2).mean()
                qo.zero_grad(); loss.backward(); qo.step()

            pa, plogp = policy.sample(bs)
            q_pi = torch.min(q1(bs, pa), q2(bs, pa))
            p_loss = (alpha * plogp.unsqueeze(-1) - q_pi).mean()
            p_opt.zero_grad(); p_loss.backward(); p_opt.step()

            alpha_loss = -(log_alpha * (plogp.detach() + target_entropy).unsqueeze(-1)).mean()
            alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

            for p, pt in zip(q1.parameters(), q1_tgt.parameters()):
                pt.data.mul_(0.995).add_(p.data * 0.005)
            for p, pt in zip(q2.parameters(), q2_tgt.parameters()):
                pt.data.mul_(0.995).add_(p.data * 0.005)

    env.close()
    elapsed = time.time() - t0
    print(f"\n[SAC] Done. {n_eps} eps, {buf_idx:,} transitions, "
          f"avg50={np.mean(recent_rets) if recent_rets else 0:.0f}, "
          f"elapsed={elapsed/60:.1f} min")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f["observations"]      = buf_s[:buf_idx]
        f["actions"]           = buf_a[:buf_idx]
        f["next_observations"] = buf_ns[:buf_idx]
        f["rewards"]           = buf_r[:buf_idx, 0]
        f["terminals"]         = buf_d[:buf_idx, 0].astype(bool)
        f.create_group("metadata")
        f["metadata/seed"]   = args.seed
        f["metadata/g"]      = args.g
        f["metadata/steps"]  = args.steps
        f["metadata/policy"] = "vanilla_sac_replay"
    print(f"[saved] {out_path} ({buf_idx:,} transitions)")


if __name__ == "__main__":
    main()
