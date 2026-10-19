"""Summarise the Bernoulli/Poisson study (paper Tables tab:nongauss and
tab:corruption and the findings of App. "Non-Gaussian Likelihoods and
Controlled Prior Corruption"). Excess NLLs are reported x 10^3, mean (SE)
over the replications.

Usage:
    python analyze.py          # reads results/summary.csv and results/replicates.csv.gz
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SCALE = 1e3
FAMILIES = ["bernoulli", "poisson"]
TABLE_N0 = [2, 8, 64]
METHODS = [("target_only", "Target-Only"), ("pooled", "Pooled"),
           ("uniform", "Uniform EM"), ("aligned", "Aligned prior"),
           ("shuffled", "Shuffled prior"), ("oracle", "Oracle prior")]
CORRUPTION = [("oracle", 0), ("corrupt_20", 20), ("corrupt_40", 40),
              ("corrupt_60", 60), ("corrupt_80", 80), ("corrupt_100", 100)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    args = ap.parse_args()
    summary = pd.read_csv(args.results / "summary.csv")
    reps = pd.read_csv(args.results / "replicates.csv.gz")
    s = summary.set_index(["family", "target_size", "method"])
    n_rep = int(summary.replicates.iloc[0])
    sizes = sorted(summary.target_size.unique())

    def cell(family, n0, method, sep=" "):
        r = s.loc[(family, n0, method)]
        return f"{SCALE * r.excess_nll:.1f}{sep}({SCALE * r.excess_nll_se:.1f})"

    print("=" * 104)
    print(f"Excess test NLL x 10^3, mean (SE) over {n_rep} replications")
    print("=" * 104)
    for family in FAMILIES:
        print(f"\n{family.capitalize()}")
        print(f"{'N_0':>4}" + "".join(f"{label:>17s}" for _, label in METHODS))
        for n0 in sizes:
            print(f"{n0:>4d}" + "".join(f"{cell(family, n0, m):>17s}"
                                        for m, _ in METHODS))

    print("\nLaTeX rows (paper Table tab:nongauss; columns "
          + ", ".join(label for _, label in METHODS[:5]) + "):")
    for family in FAMILIES:
        for i, n0 in enumerate(TABLE_N0):
            name = family.capitalize() if i == 0 else ""
            print(f"    {name:<9s} & {n0:<2d} & " + " & ".join(
                cell(family, n0, m, "\\,") for m, _ in METHODS[:5]) + r" \\")

    print("\nControlled corruption of the oracle prior at N_0 = 2 (x 10^3); "
          "Pooled: " + ", ".join(f"{f} {SCALE * s.loc[(f, 2, 'pooled')].excess_nll:.1f}"
                                  for f in FAMILIES))
    print(f"{'corrupted':>10s}" + "".join(f"{f:>18s}" for f in FAMILIES))
    for method, level in CORRUPTION:
        print(f"{level:>9d}%" + "".join(f"{cell(f, 2, method):>18s}" for f in FAMILIES))
    max_se = max(SCALE * s.loc[(f, 2, m)].excess_nll_se
                 for f in FAMILIES for m, _ in CORRUPTION)
    print(f"  largest SE {max_se:.1f}")
    print("LaTeX rows (paper Table tab:corruption):")
    for f in FAMILIES:
        print(f"    {f.capitalize():<9s} & " + " & ".join(
            f"{SCALE * s.loc[(f, 2, m)].excess_nll:.1f}" for m, _ in CORRUPTION)
            + r" \\")
    print("Fully corrupted oracle by N_0 (x 10^3): " + "; ".join(
        f + " " + ", ".join(f"{n0}: {SCALE * s.loc[(f, n0, 'corrupt_100')].excess_nll:.1f}"
                            for n0 in sizes) for f in FAMILIES))

    print("\nSource recovery at N_0 = 2: final-weight AUC (relevant vs irrelevant) / Brier")
    for f in FAMILIES:
        print(f"  {f:<10s}" + "  ".join(
            f"{label} {s.loc[(f, 2, m)].source_auc:.4f} / {s.loc[(f, 2, m)].source_brier:.4f}"
            for m, label in METHODS[2:]))

    print("\nPaired differences from the aligned prior, x 10^3: mean (paired SE), z")
    em = reps.pivot_table(index=["family", "target_size", "replicate"],
                          columns="method", values="excess_nll")
    print(f"{'':>10s}{'N_0':>5s}{'Shuffled - Aligned':>30s}{'Uniform - Aligned':>30s}")
    for f in FAMILIES:
        for n0 in sizes:
            sub = em.loc[(f, n0)]
            cells = []
            for other in ("shuffled", "uniform"):
                d = SCALE * (sub[other] - sub["aligned"])
                se = d.std(ddof=1) / math.sqrt(len(d))
                cells.append(f"{d.mean():+.2f} ({se:.2f}), z = {d.mean() / se:5.1f}")
            print(f"{f:>10s}{n0:>5d}" + "".join(f"{c:>30s}" for c in cells))

    fits = reps[reps.n_iter.notna()]
    cap = json.loads((args.results / "settings.json").read_text())["max_iter"]
    print(f"\nEM: {len(fits)} fits; iterations median {fits.n_iter.median():.0f}, "
          f"min {int(fits.n_iter.min())}, max {int(fits.n_iter.max())}; "
          f"{int((fits.n_iter >= cap).sum())} reach the {cap}-iteration cap")


if __name__ == "__main__":
    main()
