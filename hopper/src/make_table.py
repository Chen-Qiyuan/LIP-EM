"""
Table 2 (MuJoCo Hopper): IQL returns at g = 8.87 over seeds 42-51.

Each seed sets the target sample D0, the EM minibatches and the IQL seed of a
cell. Score per run = final 200-episode mean return. Cells: mean over seeds
+- SE across seeds [min-max]. Also: which source EM concentrates on in each
replicate, and seed-paired t-tests / exact sign-flip tests.

Usage (from hopper/):
    python src/make_table.py > results/table2.md   # Markdown report on stdout
    python src/make_table.py --latex          # LaTeX rows of the paper table
"""
import argparse, itertools, json, math
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = list(range(42, 52))
N0S = [64, 128, 256, 512, 1024, 2048, 4096]
METH = [("claude", "LIP-C"), ("gemini", "LIP-G"), ("uniform", "Uniform EM"),
        ("pool", "Pooled"), ("target_only", "Target Only")]
PAIRS = [("claude", "uniform"), ("claude", "pool"), ("claude", "target_only"),
         ("claude", "gemini"), ("gemini", "uniform")]


def signflip_p(d):
    """Exact two-sided sign-flip p-value of the mean paired difference."""
    obs = abs(d.mean())
    allm = [abs(np.mean(d * np.array(sg)))
            for sg in itertools.product([1, -1], repeat=len(d))]
    return np.mean([m >= obs - 1e-12 for m in allm])


def pick(r):
    """Source EM concentrates on (weight > 0.5), else 'prior'."""
    w = np.array(r["final_weights"])
    return f"g{r['final_argmax_g']}" if w.max() > 0.5 else "prior"


def se(x):
    return x.std(ddof=1) / math.sqrt(len(x))


def markdown(V, em):
    print(f"## Hopper, seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)}), mean +- SE across seeds [min-max]\n")
    print("| N0 | " + " | ".join(n for _, n in METH) + " |")
    print("|---|" + "---|" * len(METH))
    for N in N0S:
        print(f"| {N} | " + " | ".join(
            f"{V[L, N].mean():.0f} ± {se(V[L, N]):.0f} [{V[L, N].min():.0f}–{V[L, N].max():.0f}]"
            for L, _ in METH) + " |")

    print("\n## EM outcome per replicate (source with weight > 0.5, else 'prior'); n_iter in parentheses\n")
    for L, name in METH[:3]:
        print(f"### {name}")
        print("| N0 | " + " | ".join(f"seed {s}" for s in SEEDS) + " | tally |")
        print("|---|" + "---|" * (len(SEEDS) + 1))
        for N in N0S:
            rs = [em(s, L, N) for s in SEEDS]
            ps = [pick(r) for r in rs]
            tally = ", ".join(f"{k}×{ps.count(k)}" for k in sorted(set(ps)))
            print(f"| {N} | " + " | ".join(f"{p} ({r['n_iter']})" for p, r in zip(ps, rs))
                  + f" | {tally} |")
        print()

    print("## Seed-paired comparisons (diff = mean of A - B; t-test p; exact sign-flip p; wins of A)\n")
    print("| N0 | A − B | diff | SE | p (t) | p (sign-flip) | wins |")
    print("|---|---|---|---|---|---|---|")
    nm = dict(METH)
    for N in N0S:
        for A, B in PAIRS:
            d = V[A, N] - V[B, N]
            print(f"| {N} | {nm[A]} − {nm[B]} | {d.mean():+.0f} | {se(d):.0f} | "
                  f"{stats.ttest_rel(V[A, N], V[B, N]).pvalue:.4f} | {signflip_p(d):.4f} | "
                  f"{int((d > 0).sum())}/{len(d)} |")

    print("\n## Returns grouped by the source EM picked (all methods, all N0, all seeds)\n")
    groups = {}
    for L, _ in METH[:3]:
        for N in N0S:
            for i, s in enumerate(SEEDS):
                groups.setdefault(pick(em(s, L, N)), []).append(V[L, N][i])
    for k, v in sorted(groups.items()):
        v = np.array(v)
        print(f"- {k}: n={len(v)}, mean {v.mean():.0f}, SD {v.std(ddof=1) if len(v) > 1 else 0:.0f}, "
              f"range {v.min():.0f}–{v.max():.0f}")


def latex(V):
    """Rows of the paper table: mean (SE across seeds)."""
    for N in N0S:
        cells = [f"{V[L, N].mean():.0f}\\,({se(V[L, N]):.0f})" for L, _ in METH]
        print(f"    {N:<4} & " + " & ".join(cells) + r" \\")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "results"),
                   help="Directory with seed<s>/{em,iql}/ results.")
    p.add_argument("--latex", action="store_true",
                   help="Print the LaTeX rows of the paper table instead.")
    args = p.parse_args()
    R = Path(args.results)

    def ret(s, L, N):
        fe = json.load(open(R / f"seed{s}" / "iql" / f"{L}_N{N}.json"))["final_eval"]
        return fe[list(fe)[0]]["mean"]

    def em(s, L, N):
        return json.load(open(R / f"seed{s}" / "em" / f"{L}_N{N}" / "result.json"))

    V = {(L, N): np.array([ret(s, L, N) for s in SEEDS]) for L, _ in METH for N in N0S}
    if args.latex:
        latex(V)
    else:
        markdown(V, em)


if __name__ == "__main__":
    main()
