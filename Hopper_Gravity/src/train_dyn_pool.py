"""
Train the pooled dynamics model on the K=10 source replay buffers
(g=1..10, ~1M transitions each).

This is the theta_pool used by EM as both:
  1. The initialization theta_init.
  2. The null model: log p(D_k | theta_pool) is the EM null baseline.

Final published configuration:
  hidden = 512, n_layers = 4
  epochs = 500, batch = 1024
  lr = 3e-4 with cosine decay to lr/20
  AdamW, weight_decay = 1e-4, grad-clip = 1.0
  Per-source held-out test split: 20 000 transitions per source

Saves a checkpoint every 100 epochs and the final dynamics.pt.

Note: training takes ~8 hours on a single GPU (10 × 1M transitions × 500 epochs).
The published run produced results/dyn_pool/dynamics.pt and is included.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from common import S, A, DynamicsModel, load_dataset
from em import log_lik_per_sample

# ---- Fixed configuration ----
GRAVITIES        = list(range(1, 11))
N_TEST_PER_SRC   = 20_000
HIDDEN           = 512
N_LAYERS         = 4
EPOCHS           = 1000
BATCH_SIZE       = 1024
LR               = 3e-4
LR_MIN           = 1e-5         # cosine eta_min; matches EM M-step's constant LR
WEIGHT_DECAY     = 1e-4
CKPT_EVERY       = 100
VAL_FRAC         = 0.05


def cat_buffers(bs):
    return {k: torch.cat([b[k] for b in bs], dim=0)
            for k in ("states", "actions", "deltas")}


def eval_per_source_log_lik(model, test_bufs, gravities):
    model.eval()
    res = {}
    with torch.no_grad():
        for g in gravities:
            td = test_bufs[g]
            res[g] = float(log_lik_per_sample(
                model, td["states"], td["actions"], td["deltas"]).mean().item())
    model.train()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/sources_replay")
    p.add_argument("--out-dir", type=str, default="results/dyn_pool")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # --- Load sources, split off held-out test slice per source ---
    print(f"Loading {len(GRAVITIES)} sources at gravities {GRAVITIES}...")
    bufs, test_bufs = {}, {}
    for g in GRAVITIES:
        d = load_dataset(str(data_dir / f"g{g}.hdf5"), dev)
        N = d["states"].shape[0]
        perm = torch.randperm(
            N, generator=torch.Generator(device=dev).manual_seed(args.seed),
            device=dev,
        )
        ti = perm[:N_TEST_PER_SRC]
        tri = perm[N_TEST_PER_SRC:]
        test_bufs[g] = {k: v[ti] for k, v in d.items()}
        bufs[g] = {k: v[tri] for k, v in d.items()}
        print(f"  g={g}: N={N:,}  train={bufs[g]['states'].shape[0]:,}  "
              f"test={test_bufs[g]['states'].shape[0]:,}")

    pool_data = cat_buffers(list(bufs.values()))
    N_pool = pool_data["states"].shape[0]
    print(f"\n[pool] {N_pool:,} train transitions")

    # --- Build model + normalization ---
    model = DynamicsModel(state_dim=S, action_dim=A,
                          hidden=HIDDEN, n_layers=N_LAYERS).to(dev)
    model.set_normalization(pool_data["states"], pool_data["actions"],
                             pool_data["deltas"])

    # Train/val split
    perm = torch.randperm(N_pool, device=dev)
    n_val = int(N_pool * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    in_all = (torch.cat([pool_data["states"], pool_data["actions"]], -1)
              - model.in_mean) / model.in_std
    out_all = (pool_data["deltas"] - model.out_mean) / model.out_std

    print(f"\nTraining DynamicsModel on POOL "
          f"(h={HIDDEN}, n_layers={N_LAYERS}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}, lr={LR}, cosine schedule, "
          f"ckpt_every={CKPT_EVERY})...")
    print(f"  train={len(train_idx):,}  val={n_val:,}")

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    epoch_log = []
    t0 = time.time()
    for epoch in tqdm(range(EPOCHS), desc="Pool"):
        loader = DataLoader(
            TensorDataset(in_all[train_idx], out_all[train_idx]),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        ml = 0.0
        for xb, yb in loader:
            mean_norm, lv = model.forward_normalized(xb)
            loss = DynamicsModel.gaussian_nll(mean_norm, lv, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ml += loss.item()
        sched.step()
        avg_loss = ml / len(loader)

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                mn, lv = model.forward_normalized(in_all[val_idx])
                val = DynamicsModel.gaussian_nll(mn, lv, out_all[val_idx]).item()
                model.train()
            print(f"  epoch {epoch+1}/{EPOCHS}  train={avg_loss:.4f}  "
                  f"val={val:.4f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            epoch_log.append({"epoch": epoch + 1, "train": avg_loss, "val": val})

        if (epoch + 1) % CKPT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt = out_dir / f"dynamics_e{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  [ckpt e={epoch + 1}] saved -> {ckpt}")

    train_time = time.time() - t0
    model.eval()

    print(f"\nFinal per-source test log-lik:")
    per_source = eval_per_source_log_lik(model, test_bufs, GRAVITIES)
    for g in GRAVITIES:
        print(f"  g={g}: log_p = {per_source[g]:.3f}")
    print(f"\nTrained in {train_time:.1f}s ({train_time/60:.1f} min)")

    torch.save(model.state_dict(), out_dir / "dynamics.pt")
    with open(out_dir / "result.json", "w") as f:
        json.dump({
            "gravities": GRAVITIES,
            "hidden": HIDDEN, "n_layers": N_LAYERS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR,
            "pool_size": int(N_pool),
            "per_source_test_log_lik": per_source,
            "epoch_log": epoch_log,
            "dyn_train_seconds": float(train_time),
        }, f, indent=2)
    print(f"Saved -> {out_dir / 'dynamics.pt'} and {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
