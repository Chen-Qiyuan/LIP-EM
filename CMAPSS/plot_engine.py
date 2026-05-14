"""Per-engine prediction figure: target data points + four method curves +
true-future dotted curve + cutoff line.

Plot styling matches paper Figure (5.cmapss-no-prox companion):
  * Target data points (observed prefix) — black crosses
  * Target-Only OLS (ridge)             — grey curve
  * Pooled OLS (ridge)                  — brown curve
  * Uniform-EM                          — yellow curve
  * LIP-EM (Gemini)                     — purple curve
  * True future                         — black dotted curve
  * Vertical line at the cutoff cycle

Defaults to engine 80 at 50% RUL, but `--machine` and `--rul` are configurable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from data import load_machine
from em import EMConfig, em_train, predict
from lip import fit_lip, _filter_for_target, multipick_to_singlepick


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Physical_Core_Speed_Sequences")

# Match paper_table.py
TAU = 1e-3
P_0 = 0.01
EPS_LIP = 0.1
RIDGE_LAMBDA = 1e4
LIP_FILE = "llm_subgroup_queries_gemini_v2.json"


def ridge_fit(X, y, lam):
    if lam == 0:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    p = X.shape[1]
    P = np.eye(p)
    P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * P, X.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", type=int, default=80,
                    help="target machine number")
    ap.add_argument("--rul", type=float, default=0.5,
                    help="remaining-useful-life fraction (e.g. 0.5 = "
                         "observe first 50%% of cycles, predict last 50%%)")
    ap.add_argument("--out", type=str, default=None,
                    help="output PDF path (default: engine_<m>_rul_<r>.pdf)")
    args = ap.parse_args()

    machine = args.machine
    target_name = f"{machine}.csv"
    target_file = os.path.join(DATA_DIR, target_name)
    if not os.path.exists(target_file):
        raise FileNotFoundError(target_file)

    # ---- Source data ----
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    source_files = [f for f in all_files if os.path.basename(f) != target_name]
    K = len(source_files)
    X_sources, y_sources = [], []
    for f in source_files:
        _, X_s, y_s = load_machine(f)
        X_sources.append(X_s); y_sources.append(y_s)

    # ---- Target full trajectory and observed prefix ----
    x_full, X_grid, y_full = load_machine(target_file)
    target_len = len(y_full)
    cutoff = int(target_len * (1.0 - args.rul))
    _, X_t, y_t = load_machine(target_file, cutoff=cutoff)

    # ---- LIP for this target ----
    with open(os.path.join(HERE, LIP_FILE)) as f:
        lip_payload = json.load(f)
    responses = [(list(r["S"]), list(r["C"]))
                 for r in lip_payload["responses"]]
    basenames = lip_payload["source_basenames"]
    target_idx = basenames.index(target_name)
    filt = _filter_for_target(responses, target_idx)
    if any(len(c) > 1 for _, c in filt):
        filt = multipick_to_singlepick(filt)
    pi_lip = fit_lip(filt, K=K, p0=P_0, eps=EPS_LIP)
    pi_uni = np.full(K, P_0)

    # ---- Fit each method ----
    cfg = EMConfig(max_iter=1000, tau=TAU, convergence_tol=0.1*P_0,
                   verbose=False)
    beta_lip, _ = em_train(X_t, y_t, X_sources, y_sources, pi=pi_lip, cfg=cfg)
    beta_uni, _ = em_train(X_t, y_t, X_sources, y_sources, pi=pi_uni, cfg=cfg)
    X_pool = np.vstack([X_t] + X_sources)
    y_pool = np.concatenate([y_t] + y_sources)
    beta_pool = ridge_fit(X_pool, y_pool, RIDGE_LAMBDA)
    beta_t = ridge_fit(X_t, y_t, RIDGE_LAMBDA)

    pred_lip = predict(X_grid, beta_lip)
    pred_uni = predict(X_grid, beta_uni)
    pred_pool = predict(X_grid, beta_pool)
    pred_tonly = predict(X_grid, beta_t)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    # 1) Observed target points
    ax.scatter(x_full[:cutoff], y_full[:cutoff],
               color="black", marker="x", s=42, linewidths=1.0,
               label="Observed (target)", zorder=3)
    # 5) True future (dotted black, only after cutoff)
    ax.plot(x_full[cutoff:], y_full[cutoff:],
            color="black", linestyle=":", linewidth=2.0,
            label="True future", zorder=3)
    # 6) Target-Only OLS (grey)
    ax.plot(x_full, pred_tonly,
            color="grey", linewidth=2.0, linestyle="-",
            label="Target-Only OLS")
    # 4) Pooled (brown)
    ax.plot(x_full, pred_pool,
            color="saddlebrown", linewidth=2.0, linestyle="-.",
            label="Pooled OLS")
    # 3) Uniform-EM (orange)
    ax.plot(x_full, pred_uni,
            color="orange", linewidth=2.0, linestyle="--",
            label="Uniform EM")
    # 2) LIP-EM (purple)
    ax.plot(x_full, pred_lip,
            color="purple", linewidth=2.5, linestyle="-",
            label="LIP-EM")
    # 7) Vertical line at cutoff
    ax.axvline(x=cutoff, color="red", linestyle="-.", alpha=0.5,
               linewidth=1.5, label=f"Cutoff (RUL {int(args.rul*100)}%)")

    ax.set_xlabel("Time (Cycle)", fontsize=12)
    ax.set_ylabel("Core Speed (RPM)", fontsize=12)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = args.out or os.path.join(
        HERE, f"engine_{machine}_rul_{int(args.rul*100):d}.pdf")
    plt.savefig(out)
    out_png = out.replace(".pdf", ".png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved -> {out}")
    print(f"Saved -> {out_png}")
    print()
    print(f"Engine {machine}, target_len={target_len}, cutoff={cutoff} "
          f"(observed {cutoff}/{target_len} = {cutoff/target_len:.1%})")
    y_fut = y_full[cutoff:]
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"  LIP-EM     future-RMSE: {rmse(y_fut, pred_lip[cutoff:]):>6.2f}")
    print(f"  Uniform EM future-RMSE: {rmse(y_fut, pred_uni[cutoff:]):>6.2f}")
    print(f"  Pooled     future-RMSE: {rmse(y_fut, pred_pool[cutoff:]):>6.2f}")
    print(f"  Target-Only future-RMSE: {rmse(y_fut, pred_tonly[cutoff:]):>6.2f}")


if __name__ == "__main__":
    main()
