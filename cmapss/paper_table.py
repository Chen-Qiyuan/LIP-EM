"""Reproduce paper Table 1 (tab:cmapss): C-MAPSS core-speed RMSE.

For each of the 10 target engines and 9 RUL levels (90%, 80%, ..., 10%), the
first (1 - RUL) of the target's cycles are observed and the rest is predicted.
The paper labels each cutoff by this observed share (observed life + RUL = 100%).
The methods are:

  * LIP-C / LIP-G : LIP-EM with the Claude / Gemini LIP
  * Uniform       : the same EM with pi_k = p_0 for every source
  * Pooled        : ridge regression on the target prefix plus all 99 sources
  * Target-Only   : ridge regression on the target prefix

Cells are mean (SE) of the future RMSE over the 10 targets, with
SE = std / sqrt(10) (sample std, ddof = 1, as in the paper). Paired tests over the
10 targets compare LIP-C and LIP-G with each baseline at 90/70/50% RUL.

Usage:
    python paper_table.py                                # run all fits, then summarise
    python paper_table.py --from-runs results/runs.json  # summarise saved runs only

Outputs:
    results/runs.json        one record per (target, RUL, method)
    results/table1.json      mean / SE per cell, EM diagnostics, paired tests
    results/table1_rows.tex  LaTeX rows of the paper table (10/30/50/70/90% observed life)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

import experiment as E
from dataset import N_KNOTS, X_MAX, load_machine
from em import predict


PAPER_RUL_LEVELS = [0.9, 0.7, 0.5, 0.3, 0.1]   # rows printed in the paper
TEST_RUL_LEVELS = [0.9, 0.7, 0.5]              # cold-start rows with paired tests
BASELINES = ["Uniform", "Pooled", "Target-Only"]
ALPHA = 0.05
# A run "stays at the prior" if max_k |w_k - pi_k| < AT_PRIOR_TOL at the end.
AT_PRIOR_TOL = 0.02
MIN_TARGETS_FOR_TESTS = 3


def pct(rul: float) -> int:
    return int(round(rul * 100))


# ============================================================================
# Experiments
# ============================================================================

def run_all(targets: list[int]) -> list[dict]:
    runs = []
    for machine in targets:
        t_start = time.time()
        pis = E.priors_for(machine)
        X_sources, y_sources = E.load_sources(machine)
        target_file = E.SEQ_DIR / f"{machine}.csv"
        _, X_grid, y_full = load_machine(target_file)

        for rul in E.RUL_LEVELS:
            co = E.cutoff_for(len(y_full), rul)
            _, X_t, y_t = load_machine(target_file, cutoff=co)
            fits = E.fit_all(X_t, y_t, X_sources, y_sources, pis)
            for name in E.METHODS:
                beta, hist = fits[name]
                rec = dict(machine=machine, rul=rul, method=name,
                           rmse=E.rmse(y_full[co:], predict(X_grid, beta)[co:]))
                if hist is not None:
                    rec.update(n_iter=hist.n_iter,
                               converged=hist.converged_at is not None,
                               max_abs_w_minus_prior=float(
                                   np.max(np.abs(hist.weights[-1] - pis[name]))))
                runs.append(rec)
        print(f"  engine {machine:>3d}: done in {time.time() - t_start:.1f}s",
              flush=True)
    return runs


# ============================================================================
# Summary
# ============================================================================

def rmse_vector(runs, method, rul, targets) -> np.ndarray:
    by_key = {(r["method"], r["rul"], r["machine"]): r["rmse"] for r in runs}
    return np.array([by_key[(method, rul, m)] for m in targets])


def mean_se(v: np.ndarray) -> tuple[float, float]:
    return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided paired t-test and exact Wilcoxon signed-rank test of a vs b."""
    d = a - b
    return dict(mean_diff=float(d.mean()),
                se_diff=float(d.std(ddof=1) / np.sqrt(len(d))),
                t_pvalue=float(stats.ttest_rel(a, b).pvalue),
                wilcoxon_pvalue=float(stats.wilcoxon(a, b).pvalue),
                n_a_better=int((d < 0).sum()), n=len(d))


def bold_set(vectors: dict[str, np.ndarray], test: bool) -> list[str]:
    """Best mean in the row, plus every method not distinguishable from it
    (two-sided paired t-test, p >= ALPHA) when `test` is set."""
    best = min(vectors, key=lambda m: vectors[m].mean())
    return [m for m in vectors
            if m == best
            or (test and stats.ttest_rel(vectors[m], vectors[best]).pvalue >= ALPHA)]


def summarize(runs: list[dict], targets: list[int]) -> dict:
    out = {"targets": targets, "methods": E.METHODS,
           "rul_levels": E.RUL_LEVELS, "rows": [], "paired_tests": []}
    test = len(targets) >= MIN_TARGETS_FOR_TESTS
    for rul in E.RUL_LEVELS:
        vecs = {m: rmse_vector(runs, m, rul, targets) for m in E.METHODS}
        em_runs = [r for r in runs if r["rul"] == rul and "n_iter" in r]
        row = {"rul": rul,
               "mean": {m: mean_se(v)[0] for m, v in vecs.items()},
               "se": {m: mean_se(v)[1] for m, v in vecs.items()},
               "bold": bold_set(vecs, test),
               "em_runs_at_prior": sum(r["max_abs_w_minus_prior"] < AT_PRIOR_TOL
                                       for r in em_runs),
               "em_runs": len(em_runs),
               "em_median_iter": float(np.median([r["n_iter"] for r in em_runs])),
               "em_max_iter": max(r["n_iter"] for r in em_runs),
               "em_not_converged": sum(not r["converged"] for r in em_runs)}
        out["rows"].append(row)
        if test and rul in TEST_RUL_LEVELS:
            for lip in ("LIP-C", "LIP-G"):
                for base in BASELINES:
                    out["paired_tests"].append(
                        dict(rul=rul, lip=lip, baseline=base,
                             **paired_test(vecs[lip], vecs[base])))
    return out


def cell(mean: float, se: float, sep: str = " ") -> str:
    return f"{mean:.1f}{sep}({se:.1f})"


def latex_rows(summary: dict) -> list[str]:
    lines = []
    for row in summary["rows"]:
        if row["rul"] not in PAPER_RUL_LEVELS:
            continue
        cells = [cell(row["mean"][m], row["se"][m], sep="\\,") for m in E.METHODS]
        lines.append(f"    {100 - pct(row['rul'])}\\% & " + " & ".join(cells) + " \\\\")
    return lines


def print_report(summary: dict) -> None:
    n = len(summary["targets"])
    print("=" * 96)
    print(f"Table 1 (all cutoffs): mean (SE) future RMSE over {n} targets; "
          f"* = best or not distinguishable from best (paired t, p >= {ALPHA})")
    print("=" * 96)
    print(f"{'RUL':>4}  " + "".join(f"{m:>16s}" for m in E.METHODS))
    for row in summary["rows"]:
        cells = [cell(row["mean"][m], row["se"][m])
                 + ("*" if m in row["bold"] else " ") for m in E.METHODS]
        print(f"{pct(row['rul']):>3d}%  " + "".join(f"{c:>16s}" for c in cells))

    print("\nEM diagnostics per cutoff (LIP-C, LIP-G, Uniform runs):")
    print(f"{'RUL':>4}  {'final w within ' + str(AT_PRIOR_TOL) + ' of prior':>28s}"
          f"  {'median iters':>12s}  {'max iters':>9s}  {'hit cap':>7s}")
    for row in summary["rows"]:
        print(f"{pct(row['rul']):>3d}%  "
              f"{row['em_runs_at_prior']:>22d} / {row['em_runs']:<3d}"
              f"  {row['em_median_iter']:>12.0f}  {row['em_max_iter']:>9d}"
              f"  {row['em_not_converged']:>7d}")

    print(f"\nPaired tests over {n} targets (difference = LIP - baseline; "
          f"negative favours LIP):")
    print(f"{'RUL':>4}  {'comparison':<24s} {'diff':>7s} {'SE':>5s} "
          f"{'t-test p':>9s} {'Wilcoxon p':>10s} {'LIP better':>10s}")
    tests = summary["paired_tests"]
    if not tests:
        print(f"  (skipped: needs at least {MIN_TARGETS_FOR_TESTS} targets)")
    for t in tests:
        print(f"{pct(t['rul']):>3d}%  {t['lip'] + ' vs ' + t['baseline']:<24s} "
              f"{t['mean_diff']:>+7.1f} {t['se_diff']:>5.1f} "
              f"{t['t_pvalue']:>9.4f} {t['wilcoxon_pvalue']:>10.4f} "
              f"{t['n_a_better']:>7d}/{t['n']}")
    sig = [t for t in tests if t["mean_diff"] < 0 and t["t_pvalue"] < ALPHA]
    if tests:
        print(f"LIP significantly better (paired t-test, p < {ALPHA}) in "
              f"{len(sig)} of {len(tests)} comparisons"
              + (f"; largest significant p = {max(t['t_pvalue'] for t in sig):.3f}"
                 if sig else ""))
    for t in tests:
        if t not in sig:
            print(f"  not significant: {pct(t['rul'])}% {t['lip']} vs "
                  f"{t['baseline']} (diff {t['mean_diff']:+.1f}, "
                  f"p = {t['t_pvalue']:.3f})")

    print("\nLaTeX rows (paper Table 1, first column = observed life = 1 - RUL; columns "
          + ", ".join(E.METHODS) + "):")
    print("\n".join(latex_rows(summary)))


# ============================================================================
# Main
# ============================================================================

def settings() -> dict:
    cfg = E.EM_CONFIG
    return dict(basis=f"natural cubic spline, {N_KNOTS} knots, x_max={X_MAX:g}",
                tau=cfg.tau, nu=cfg.nu,
                convergence_tol=cfg.convergence_tol,
                convergence_patience=cfg.convergence_patience,
                stop_count_start=cfg.stop_count_start, max_iter=cfg.max_iter,
                p_0=E.P_0, eps_lip=E.EPS_LIP, ridge_lambda=E.RIDGE_LAMBDA,
                lip_files={k: v.name for k, v in E.LIP_FILES.items()})


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--targets", type=str, default=",".join(map(str, E.TARGETS)),
                    help="comma-separated target engine numbers")
    ap.add_argument("--from-runs", type=Path, default=None,
                    help="summarise this runs.json instead of running the fits")
    ap.add_argument("--out-dir", type=Path, default=E.RESULTS_DIR)
    args = ap.parse_args()

    if args.from_runs is not None:
        with open(args.from_runs) as f:
            payload = json.load(f)
        runs, targets = payload["runs"], payload["targets"]
        print(f"Loaded {len(runs)} runs from {args.from_runs}")
    else:
        targets = [int(s) for s in args.targets.split(",")]
        for name, path in E.LIP_FILES.items():
            print(f"{name}: {path.name}")
        print(f"Running {len(targets)} targets x {len(E.RUL_LEVELS)} cutoffs "
              f"x {len(E.METHODS)} methods ...")
        t0 = time.time()
        runs = run_all(targets)
        print(f"{len(runs)} runs in {time.time() - t0:.0f}s\n")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.out_dir / "runs.json", "w") as f:
            json.dump(dict(settings=settings(), targets=targets,
                           rul_levels=E.RUL_LEVELS, methods=E.METHODS,
                           runs=runs), f, indent=1)

    summary = summarize(runs, targets)
    print_report(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "table1.json", "w") as f:
        json.dump(dict(settings=settings(), **summary), f, indent=1)
    with open(args.out_dir / "table1_rows.tex", "w") as f:
        f.write("\n".join(latex_rows(summary)) + "\n")
    print(f"\nWrote {args.out_dir / 'table1.json'} and "
          f"{args.out_dir / 'table1_rows.tex'}")


if __name__ == "__main__":
    main()
