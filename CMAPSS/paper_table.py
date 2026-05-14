"""Reproduce paper Table 5.2 (tab:cmapss-no-prox) format.

Mean (std) RMSE over 10 target machines, at 9 RUL levels (10% to 90%),
for two LIP-EM variants (Gemini-3-flash, Claude) plus Uniform EM /
Pooled / Target-Only baselines.

Final config:
  * Basis: natural cubic spline, K=5 knots, x_max=300 (see data.py)
  * tau=1e-3, p_0=0.01, eps=0.1, nu=0.05
  * LIP: paper eq:lip_choice (conditional logit with outside option)
  * 10 randomly-sampled targets (paper + new cluster, seed=42)
  * Target-Only and Pooled use ridge regression (lambda=1e4, intercept
    un-penalized) so the truncated-power natural cubic basis doesn't
    diverge at intermediate cutoffs

Usage:
    python paper_table.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from data import basis, load_machine, N_KNOTS, X_MAX
from em import EMConfig, em_train, predict
from lip import fit_lip, _filter_for_target, multipick_to_singlepick


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Physical_Core_Speed_Sequences")

# 10 target machines randomly sampled (seed=42) from a 12-machine candidate
# pool of fast-degrading engines (paper cluster + new cluster). See
# `plot_core_speed.py` for the visualization.
TARGETS = [4, 9, 18, 26, 27, 48, 51, 53, 55, 80]
RUL_LEVELS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# EM hyperparameters
TAU = 1e-3
P_0 = 0.01
EPS_LIP = 0.1

# Ridge regularization for target-only and pooled (intercept un-penalized).
RIDGE_LAMBDA = 1e4

TARGET_ONLY_CAP = 100.0

# Default LIPs to compare side-by-side.
DEFAULT_LIP_FILES = [
    ("Gemini",  "llm_subgroup_queries_gemini_v2.json"),
    ("Claude",  "llm_subgroup_queries_claude.json"),
]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge OLS, intercept (column 0) un-penalized."""
    if lam == 0:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    p = X.shape[1]
    P = np.eye(p)
    P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * P, X.T @ y)


def fmt_cell(mean: float, se: float, cap: float) -> str:
    """Format `mean (SE)` cell with cap on Target-Only RMSE for readability."""
    if not np.isfinite(mean):
        return "-"
    if mean > cap:
        return f">{cap:g}"
    return f"{mean:.1f} ({se:.1f})"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--targets", type=str,
                    default=",".join(map(str, TARGETS)),
                    help="comma-separated machine numbers")
    ap.add_argument("--out", type=str, default=None,
                    help="if given, write the LaTeX table to this file")
    args = ap.parse_args()
    targets = [int(s) for s in args.targets.split(",")]

    # Load each LIP file once
    lip_data = {}
    for lip_name, path in DEFAULT_LIP_FILES:
        full = os.path.join(HERE, path)
        with open(full) as f:
            data = json.load(f)
        responses = [(list(r["S"]), list(r["C"])) for r in data["responses"]]
        basenames = list(data["source_basenames"])
        lip_data[lip_name] = (responses, basenames)
        print(f"LIP[{lip_name}] = {path}  ({len(responses)} responses)")
    print()

    cfg = EMConfig(max_iter=1000, tau=TAU, convergence_tol=0.1*P_0,
                   verbose=False)

    methods = ([f"lip_em_{name}" for name, _ in DEFAULT_LIP_FILES]
               + ["uniform_em", "pooled_ols", "target_only"])
    results = {m: {rul: [] for rul in RUL_LEVELS} for m in methods}

    for machine in targets:
        target_name = f"{machine}.csv"
        target_file = os.path.join(DATA_DIR, target_name)
        all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
        source_files = [f for f in all_files
                        if os.path.basename(f) != target_name]
        K = len(source_files)

        # Fit one LIP per LLM source for this target
        pi_per_lip = {}
        for lip_name, _ in DEFAULT_LIP_FILES:
            responses, basenames = lip_data[lip_name]
            target_idx = basenames.index(target_name)
            filt = _filter_for_target(responses, target_idx)
            if any(len(c) > 1 for _, c in filt):
                filt = multipick_to_singlepick(filt)
            pi_per_lip[lip_name] = fit_lip(filt, K=K, p0=P_0, eps=EPS_LIP)
        pi_uni = np.full(K, P_0)

        X_sources, y_sources = [], []
        for f in source_files:
            _, X_s, y_s = load_machine(f)
            X_sources.append(X_s); y_sources.append(y_s)
        _, X_grid, y_full = load_machine(target_file)
        target_len = len(y_full)
        p = X_sources[0].shape[1]

        for rul in RUL_LEVELS:
            cutoff_p = 1.0 - rul
            co = int(target_len * cutoff_p)
            if co < p + 1:
                for m in methods:
                    results[m][rul].append(np.nan)
                continue

            _, X_t, y_t = load_machine(target_file, cutoff=co)

            for lip_name, _ in DEFAULT_LIP_FILES:
                beta_lip, _ = em_train(X_t, y_t, X_sources, y_sources,
                                        pi=pi_per_lip[lip_name], cfg=cfg)
                results[f"lip_em_{lip_name}"][rul].append(
                    rmse(y_full[co:], predict(X_grid, beta_lip)[co:]))
            beta_uni, _ = em_train(X_t, y_t, X_sources, y_sources,
                                    pi=pi_uni, cfg=cfg)
            X_pool = np.vstack([X_t] + X_sources)
            y_pool = np.concatenate([y_t] + y_sources)
            beta_pool = ridge_fit(X_pool, y_pool, RIDGE_LAMBDA)
            beta_t = ridge_fit(X_t, y_t, RIDGE_LAMBDA)

            y_fut = y_full[co:]
            results["uniform_em"][rul].append(
                rmse(y_fut, predict(X_grid, beta_uni)[co:]))
            results["pooled_ols"][rul].append(
                rmse(y_fut, predict(X_grid, beta_pool)[co:]))
            results["target_only"][rul].append(
                rmse(y_fut, predict(X_grid, beta_t)[co:]))

    # Aggregate: mean and standard error of the mean (SE = std / sqrt(N))
    # over the target machines for each (method, RUL) cell.
    means, ses = {}, {}
    for m in methods:
        means[m] = np.array([np.nanmean(results[m][rul]) for rul in RUL_LEVELS])
        ses[m] = np.array([
            np.nanstd(results[m][rul])
            / max(np.sqrt(np.sum(np.isfinite(results[m][rul]))), 1.0)
            for rul in RUL_LEVELS
        ])

    titles = ([(f"lip_em_{name}", f"LIP-EM ({name})")
                for name, _ in DEFAULT_LIP_FILES]
              + [("uniform_em", "Uniform EM"), ("pooled_ols", "Pooled"),
                 ("target_only", "Target-Only")])

    # ---- Plain text table ----
    print("=" * 100)
    print(f"PAPER TABLE — natcubic u{N_KNOTS} (x_max={X_MAX:g}), "
          f"eps={EPS_LIP}, tau={TAU}, ridge lam={RIDGE_LAMBDA:g}  "
          f"({len(targets)} machines)")
    print("=" * 100)
    print(f"{'RUL':>5}  " + "  ".join(f"{n:>15s}" for _, n in titles))
    print("-" * 100)
    for i, rul in enumerate(RUL_LEVELS):
        cells = [fmt_cell(means[k][i], ses[k][i], TARGET_ONLY_CAP)
                 for k, _ in titles]
        print(f"{int(rul*100):>3d}%  "
              + "  ".join(f"{c:>15s}" for c in cells))

    # ---- LaTeX ----
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(titles) + "}")
    latex_lines.append("\\toprule")
    latex_lines.append("RUL & " + " & ".join(name for _, name in titles) + " \\\\")
    latex_lines.append("\\midrule")
    for i, rul in enumerate(RUL_LEVELS):
        finite = {k: means[k][i] for k, _ in titles
                  if np.isfinite(means[k][i]) and means[k][i] <= TARGET_ONLY_CAP}
        best = min(finite, key=finite.get) if finite else None
        cells = []
        for k, _ in titles:
            cell = fmt_cell(means[k][i], ses[k][i], TARGET_ONLY_CAP)
            if k == best:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        latex_lines.append(f"{int(rul*100)}\\% & " + " & ".join(cells) + " \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(
        f"\\caption{{C-MAPSS Core Speed Prediction. "
        f"Basis: natural cubic spline, $K{{=}}{N_KNOTS}$ knots, "
        f"$x_{{\\max}}{{=}}{X_MAX:g}$, $\\epsilon{{=}}{EPS_LIP}$, "
        f"$\\tau{{=}}{TAU}$. Target-Only and Pooled use ridge regression "
        f"($\\lambda{{=}}{RIDGE_LAMBDA:g}$, intercept un-penalized). "
        f"{len(targets)} target machines; cells are mean (SE) RMSE.}}")
    latex_lines.append("\\label{tab:cmapss-no-prox}")
    latex_lines.append("\\end{table}")
    latex = "\n".join(latex_lines)

    print("\n" + "=" * 100)
    print("LaTeX:")
    print("=" * 100)
    print(latex)

    if args.out is not None:
        with open(args.out, "w") as f:
            f.write(latex + "\n")
        print(f"\nLaTeX written to {args.out}")


if __name__ == "__main__":
    main()
