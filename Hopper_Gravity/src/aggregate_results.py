"""
Aggregate weighted_iql results across N0 x {pool, target_only, correct, uniform}
into a Markdown + LaTeX table of mean (SEM) returns.

SEM = std / sqrt(N_eval), where N_eval is the eval-episode count from the
weighted_iql.py final_eval (200 in the published configuration).

Usage:
    py src/aggregate_results.py \
        --results-dir results/iql \
        --target-g 8.87 \
        --tag venus \
        --N0-list 125,250,500,1000,2000,4000 \
        --columns pool,target_only,correct,uniform
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def get_eval(p: Path):
    """Returns (mean, std, n_eval) from a weighted_iql result.json, or None."""
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    fe = j.get("final_eval", {})
    if not fe:
        return None
    k = list(fe.keys())[0]
    mean = fe[k]["mean"]; std = fe[k]["std"]
    return mean, std, 200  # weighted_iql.py: N_EVAL_EPS_FINAL = 200


def fmt_cell(r):
    if r is None:
        return "--"
    m, s, n = r
    sem = s / math.sqrt(n)
    return f"{m:.0f} ({sem:.0f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, required=True,
                   help="Dir with files <tag>_<col>_N<N0>.json from weighted_iql.py")
    p.add_argument("--tag", type=str, default="venus",
                   help="Filename prefix (e.g. 'venus' for 'venus_correct_N125.json').")
    p.add_argument("--target-g", type=float, default=None,
                   help="Optional, included in the caption.")
    p.add_argument("--N0-list", type=str,
                   default="128,256,512,1024,2048,4096")
    p.add_argument("--columns", type=str,
                   default="pool,target_only,correct,uniform,weak",
                   help="Comma-separated list of columns to include.")
    args = p.parse_args()

    base = Path(args.results_dir)
    n0s = [int(x) for x in args.N0_list.split(",")]
    cols = args.columns.split(",")

    rows = {n: {c: get_eval(base / f"{args.tag}_{c}_N{n}.json") for c in cols}
            for n in n0s}

    # ---- Markdown ----
    title_g = f" at g={args.target_g}" if args.target_g is not None else ""
    print(f"# IQL returns for tag={args.tag!r}{title_g}, mean (SEM) over 200 episodes\n")
    header = "| N0 | " + " | ".join(cols) + " |"
    sep    = "|---|" + "|".join(["---"] * len(cols)) + "|"
    print(header); print(sep)
    for n in n0s:
        cells = [str(n)] + [fmt_cell(rows[n][c]) for c in cols]
        print("| " + " | ".join(cells) + " |")

    # ---- LaTeX ----
    print(f"\n% LaTeX")
    print(r"\begin{tabular}{r" + "c" * len(cols) + "}")
    print(r"\toprule")
    print("$N_0$ & " + " & ".join(c.replace('_', r'\_') for c in cols) + r" \\")
    print(r"\midrule")
    for n in n0s:
        cells = [str(n)] + [fmt_cell(rows[n][c]).replace('(', r'\,(') for c in cols]
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
