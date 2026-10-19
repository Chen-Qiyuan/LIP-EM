"""Per-engine prediction figure (paper Fig. fig:engine80, engine 80 at 30% observed life, i.e. 70% RUL).

Plot styling:
  * Observed target prefix  — black crosses
  * True future             — black dotted curve
  * Target Only (ridge)     — grey curve
  * Pooled (ridge)          — brown dash-dot curve
  * Uniform EM              — orange dashed curve
  * LIP-G (LIP-EM, Gemini LIP) — purple curve
  * Vertical line at the cutoff cycle

Usage:
    python plot_engine.py                       # figures/engine_80_rul_70.{pdf,png}
    python plot_engine.py --machine 4 --rul 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3 (conference PDF requirement)
import numpy as np

import experiment as E
from dataset import load_machine
from em import em_train, predict


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--machine", type=int, default=80,
                    help="target engine number")
    ap.add_argument("--rul", type=float, default=0.7,
                    help="remaining-useful-life fraction (0.7 = observe the "
                         "first 30%% of cycles, predict the last 70%%)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PDF path (default: "
                         "figures/engine_<machine>_rul_<pct>.pdf); a PNG is "
                         "written next to it")
    args = ap.parse_args()

    machine = args.machine
    target_file = E.SEQ_DIR / f"{machine}.csv"
    if not target_file.exists():
        raise FileNotFoundError(target_file)

    # ---- Sources, target trajectory and observed prefix ----
    X_sources, y_sources = E.load_sources(machine)
    x_full, X_grid, y_full = load_machine(target_file)
    cutoff = E.cutoff_for(len(y_full), args.rul)
    _, X_t, y_t = load_machine(target_file, cutoff=cutoff)

    # ---- Fit each method (the figure shows the Gemini LIP) ----
    pi_lip = E.lip_prior(E.LIP_FILES["LIP-G"], machine)
    pi_uni = np.full(len(pi_lip), E.P_0)
    beta_lip, _ = em_train(X_t, y_t, X_sources, y_sources, pi=pi_lip,
                           cfg=E.EM_CONFIG)
    beta_uni, _ = em_train(X_t, y_t, X_sources, y_sources, pi=pi_uni,
                           cfg=E.EM_CONFIG)
    X_pool = np.vstack([X_t] + X_sources)
    y_pool = np.concatenate([y_t] + y_sources)
    beta_pool = E.ridge_fit(X_pool, y_pool, E.RIDGE_LAMBDA)
    beta_t = E.ridge_fit(X_t, y_t, E.RIDGE_LAMBDA)

    pred_lip = predict(X_grid, beta_lip)
    pred_uni = predict(X_grid, beta_uni)
    pred_pool = predict(X_grid, beta_pool)
    pred_tonly = predict(X_grid, beta_t)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_full[:cutoff], y_full[:cutoff],
               color="black", marker="x", s=42, linewidths=1.0,
               label="Observed (target)", zorder=3)
    ax.plot(x_full[cutoff:], y_full[cutoff:],
            color="black", linestyle=":", linewidth=2.0,
            label="True future", zorder=3)
    ax.plot(x_full, pred_tonly,
            color="grey", linewidth=2.0, linestyle="-",
            label="Target Only")
    ax.plot(x_full, pred_pool,
            color="saddlebrown", linewidth=2.0, linestyle="-.",
            label="Pooled")
    ax.plot(x_full, pred_uni,
            color="orange", linewidth=2.0, linestyle="--",
            label="Uniform EM")
    ax.plot(x_full, pred_lip,
            color="purple", linewidth=2.5, linestyle="-",
            label="LIP-G")
    rul_pct = int(round(args.rul * 100))
    ax.axvline(x=cutoff, color="red", linestyle="-.", alpha=0.5,
               linewidth=1.5, label=f"Cutoff ({100 - rul_pct}% observed)")

    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("Physical Core Speed (rpm)", fontsize=12)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = args.out or E.FIGURES_DIR / f"engine_{machine}_rul_{rul_pct}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, metadata={'CreationDate': None})
    plt.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved -> {out}")
    print(f"Saved -> {out.with_suffix('.png')}")
    print()
    print(f"Engine {machine}, target_len={len(y_full)}, cutoff={cutoff} "
          f"(observed {cutoff}/{len(y_full)} = {cutoff / len(y_full):.1%})")
    y_fut = y_full[cutoff:]
    for name, pred in (("LIP-G", pred_lip), ("Uniform EM", pred_uni),
                       ("Pooled", pred_pool), ("Target Only", pred_tonly)):
        print(f"  {name + ' future-RMSE:':<25s}{E.rmse(y_fut, pred[cutoff:]):>6.2f}")


if __name__ == "__main__":
    main()
