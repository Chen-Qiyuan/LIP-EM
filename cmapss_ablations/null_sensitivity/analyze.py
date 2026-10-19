"""Summarise the null-likelihood sweep (paper Table tab:null_sweep and the text
of App. "Sensitivity to the Null Likelihood").

Usage:
    python analyze.py                       # reads results/runs.json
    python analyze.py --runs other.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
LIPS = ["LIP-C", "LIP-G"]
EM_METHODS = LIPS + ["Uniform"]
# A run "stays at the prior" if max_k |w_k - pi_k| < AT_PRIOR_TOL at the end.
AT_PRIOR_TOL = 0.02


def pct(rul: float) -> int:
    return int(round(rul * 100))


def fmt_c(c: float) -> str:
    return "0" if c == 0 else f"{c:+g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE / "results" / "runs.json")
    args = ap.parse_args()
    with open(args.runs) as f:
        payload = json.load(f)
    runs, baselines = payload["runs"], payload["baselines"]
    settings = payload["settings"]
    ruls, levels = settings["rul_levels"], settings["null_levels"]
    n_targets = len(payload["targets"])
    rows = [("eb", c) for c in levels] + [("pooled", 0.0)]

    cell = defaultdict(list)          # (method, rul, null, c) -> runs
    for r in runs:
        cell[(r["method"], r["rul"], r["null"], r["c"])].append(r)
    base = defaultdict(list)          # (method, rul) -> RMSEs
    for r in baselines:
        base[(r["method"], r["rul"])].append(r["rmse"])

    def mean(key, field="rmse"):
        return float(np.mean([r[field] for r in cell[key]]))

    def label(null, c):
        return "pooled null" if null == "pooled" else f"c = {fmt_c(c)}"

    # ---- Table ----
    print("=" * 92)
    print(f"Mean future RMSE over {n_targets} targets when the null likelihood "
          f"is multiplied by e^c per observation")
    print("[mean sum_k w_k in brackets]; c = 0 is the EB null of Table 1")
    print("=" * 92)
    for rul in ruls:
        print(f"\nRUL {pct(rul)}%   [Pooled {np.mean(base[('Pooled', rul)]):.1f}"
              f" | Target-Only {np.mean(base[('Target-Only', rul)]):.1f}]")
        print(f"{'':>12s}" + "".join(f"{m:>18s}" for m in EM_METHODS))
        for null, c in rows:
            cells = [f"{mean((m, rul, null, c)):.1f} "
                     f"[{mean((m, rul, null, c), 'sum_w'):.2f}]" for m in EM_METHODS]
            print(f"{label(null, c):>12s}" + "".join(f"{x:>18s}" for x in cells))

    print("\nLaTeX rows (columns: LIP-C, LIP-G, Uniform EM; each at "
          + ", ".join(f"{100 - pct(r)}%" for r in ruls) + " observed life):")
    for null, c in rows:
        name = "Pooled null" if null == "pooled" else f"${fmt_c(c)}$"
        vals = [f"{mean((m, rul, null, c)):.1f}" for m in EM_METHODS for rul in ruls]
        print(f"    {name:<11s} & " + " & ".join(vals) + r" \\")

    # ---- Headline numbers ----
    print("\nHeadline numbers")
    n_cells = n_base = n_unif = 0
    lip_means = []
    for m in LIPS:
        for rul in ruls:
            for c in levels:
                a = mean((m, rul, "eb", c))
                n_cells += 1
                n_base += (a < np.mean(base[("Pooled", rul)])
                           and a < np.mean(base[("Target-Only", rul)]))
                n_unif += a < mean(("Uniform", rul, "eb", c))
                lip_means.append(a)
            lip_means.append(mean((m, rul, "pooled", 0.0)))
    print(f"  LIP-EM below Pooled and Target-Only: {n_base} of {n_cells} "
          f"cells; below Uniform EM: {n_unif} of {n_cells}")
    print(f"  LIP RMSE over all null levels and the pooled null: "
          f"[{min(lip_means):.1f}, {max(lip_means):.1f}]")

    eb0 = [r for r in runs if r["null"] == "eb" and r["c"] == 0.0]
    print(f"  per-observation log-LR at c = 0 (EB null, last E-step): "
          f"lowest 5th percentile {min(r['llr_p5'] for r in eb0):.2f}, "
          f"largest value {max(r['llr_max'] for r in eb0):.2f} nats")
    for rul in ruls:
        inv_eps = np.median([r["inv_eps_median"] for r in runs if r["rul"] == rul])
        sum_w = {m: np.mean([r["sum_w"] for r in runs
                             if r["rul"] == rul and r["method"] == m])
                 for m in EM_METHODS}
        print(f"  RUL {pct(rul)}%: median 1/eps_k {inv_eps:.1e}; mean sum_k w_k "
              "over all null settings " + ", ".join(f"{m} {v:.2f}"
                                                    for m, v in sum_w.items()))

    print(f"\nRuns whose final weights leave the prior (max_k |w_k - pi_k| >= "
          f"{AT_PRIOR_TOL}):")
    for rul in ruls:
        moved = [r for r in runs if r["rul"] == rul
                 and r["max_abs_w_minus_pi"] >= AT_PRIOR_TOL]
        n_rul = sum(r["rul"] == rul for r in runs)
        print(f"  RUL {pct(rul)}%: {len(moved)} of {n_rul}; targets "
              f"{sorted({r['target'] for r in moved}) or '-'}")
        for null, c in rows:
            for m in EM_METHODS:
                t = sorted(r["target"] for r in moved if r["method"] == m
                           and r["null"] == null and r["c"] == c)
                if t:
                    print(f"      {label(null, c):>12s}  {m:<8s} targets {t}")
    iters = [r["n_iter"] for r in runs]
    print(f"\nIterations: median {np.median(iters):.0f}, max {max(iters)}; "
          f"{sum(not r['converged'] for r in runs)} of {len(runs)} runs "
          f"reach the {settings['max_iter']}-iteration cap")


if __name__ == "__main__":
    main()
