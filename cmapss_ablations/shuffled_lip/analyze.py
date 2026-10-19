"""Summarise the shuffled-LIP control (paper Table tab:shuffled and the text of
App. "Shuffled LIP Control (C-MAPSS)").

Cells are mean (SE) over the target engines, SE = std / sqrt(n) with the
sample std (ddof = 1), as in Table 1. A shuffled entry first averages the
permutations within each engine. The permutation test compares the real LIP's
mean RMSE over targets with the mean RMSE of each permutation:
p = (1 + #{permutations with mean <= real}) / (1 + #permutations).

Usage:
    python analyze.py                          # every results/shuffles*/runs.csv
    python analyze.py results/shuffles100      # one run directory
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COLUMNS = ["LIP-C", "Shuffled LIP-C", "LIP-G", "Shuffled LIP-G", "Uniform",
           "Pooled", "Target-Only"]
PAPER_RUL_LEVELS = [0.9, 0.7, 0.5, 0.3, 0.1]
PAPER_COLUMNS = ["LIP-C", "Shuffled LIP-C", "LIP-G", "Shuffled LIP-G", "Uniform"]
AT_PRIOR_TOL = 0.02


def pct(rul: float) -> int:
    return int(round(rul * 100))


def per_target(runs: pd.DataFrame, method: str) -> pd.Series:
    """Mean RMSE per target (averaging the permutations of a shuffled method)."""
    return runs[runs.method == method].groupby("target").rmse.mean()


def mean_se(v: pd.Series) -> tuple[float, float]:
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def permutation_means(runs: pd.DataFrame, lip: str) -> pd.Series:
    """Mean RMSE over targets of each permutation."""
    return runs[runs.method == f"Shuffled {lip}"].groupby("shuffle").rmse.mean()


def report(run_dir: Path) -> None:
    runs = pd.read_csv(run_dir / "runs.csv")
    settings = json.loads((run_dir / "settings.json").read_text())
    n = settings["shuffles"]
    ruls = sorted(runs.rul.unique(), reverse=True)
    print("=" * 100)
    print(f"{run_dir.name}: {n} permutations per target (seed {settings['seed']}), "
          f"{len(settings['targets'])} targets")
    print("=" * 100)

    print("\nMean (SE) future RMSE; shuffled = per-target mean over the permutations")
    print(f"{'RUL':>4}  " + "".join(f"{m:>16s}" for m in COLUMNS))
    for rul in ruls:
        r = runs[runs.rul == rul]
        cells = ["%.1f (%.1f)" % mean_se(per_target(r, m)) for m in COLUMNS]
        print(f"{pct(rul):>3d}%  " + "".join(f"{c:>16s}" for c in cells))

    if set(PAPER_RUL_LEVELS) <= set(ruls):
        print("\nLaTeX rows (paper Table tab:shuffled, first column = observed life = 1 - RUL; columns "
              + ", ".join(PAPER_COLUMNS) + "):")
        for rul in PAPER_RUL_LEVELS:
            r = runs[runs.rul == rul]
            cells = ["%.1f\\,(%.1f)" % mean_se(per_target(r, m))
                     for m in PAPER_COLUMNS]
            print(f"    {100 - pct(rul)}\\% & " + " & ".join(cells) + r" \\")

    print(f"\nPermutation test: real LIP vs the {n} permutation means "
          "[min, 5%, median, 95%, max]; #permutations <= real; p; "
          "#permutations worse than Uniform")
    for rul in ruls:
        r = runs[runs.rul == rul]
        uniform = r[r.method == "Uniform"].rmse.mean()
        for lip in ("LIP-C", "LIP-G"):
            real = r[r.method == lip].rmse.mean()
            perm = permutation_means(r, lip)
            assert len(perm) == n
            q = np.quantile(perm, [0.0, 0.05, 0.5, 0.95, 1.0])
            n_le = int((perm <= real).sum())
            print(f"  {pct(rul):>3d}% {lip}: {real:6.2f} vs {perm.mean():6.2f} "
                  f"(SD {perm.std(ddof=1):.2f}) [" + ", ".join(f"{x:.1f}" for x in q)
                  + f"];  {n_le:>3d}/{n} <= real;  p = {(1 + n_le) / (n + 1):.4f};"
                  f"  worse than Uniform ({uniform:.1f}): {int((perm > uniform).sum())}/{n}")

    print(f"\nEM diagnostics: runs whose final weights stay within {AT_PRIOR_TOL} "
          "of the prior / runs, median iterations")
    groups = {"LIP-C": ["LIP-C"], "LIP-G": ["LIP-G"], "Uniform": ["Uniform"],
              "Shuffled": ["Shuffled LIP-C", "Shuffled LIP-G"]}
    print(f"{'RUL':>4}  " + "".join(f"{g:>22s}" for g in groups))
    for rul in ruls:
        r = runs[runs.rul == rul]
        cells = []
        for members in groups.values():
            g = r[r.method.isin(members)]
            at_prior = int((g.max_abs_w_minus_pi < AT_PRIOR_TOL).sum())
            cells.append(f"{at_prior}/{len(g)}, it {g.n_iter.median():.0f}")
        print(f"{pct(rul):>3d}%  " + "".join(f"{c:>22s}" for c in cells))
    em = runs[runs.n_iter.notna()]
    print(f"  {int((~em.converged.astype(bool)).sum())} of {len(em)} EM runs "
          f"reach the {settings['max_iter']}-iteration cap; iterations "
          f"{int(em.n_iter.min())}-{int(em.n_iter.max())}")
    check = settings["check_against_em_train"]
    print(f"  unshuffled rows vs em.em_train: max |w diff| {check['max_w_diff']:.1e}, "
          f"max |RMSE diff| {check['max_rmse_diff']:.1e}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", type=Path)
    args = ap.parse_args()
    run_dirs = args.run_dirs or sorted(
        p.parent for p in (HERE / "results").glob("shuffles*/runs.csv"))
    for d in run_dirs:
        report(d)


if __name__ == "__main__":
    main()
