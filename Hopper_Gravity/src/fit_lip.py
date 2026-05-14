"""
Fit a Language-Induced Prior (LIP) pi over the K=10 sources from LLM
elicitation responses, via the paper's conditional-logit-with-null MLE.

Reads queries from `cc_workdir/results.json` (produced by lip_query_gemini.py)
where each query is a dict {subgroup: [source_NN, ...], matching: [source_MM]
or []}. Maps source_NN -> integer gravity g via SOURCE_TO_G (calibrated by
matching the source CSVs' z-statistics to the integer-gravity replay buffers).

Usage:
    py fit_lip.py                                       # defaults p0=0.01, eps=1.0
    py fit_lip.py --p0 0.05                             # higher prior
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch


# Source -> integer gravity g (verified by matching cc_workdir CSV z-stats
# against data/sources_replay/g{N}.hdf5).
SOURCE_TO_G = {
    "source_01": 6,
    "source_02": 7,
    "source_03": 1,
    "source_04": 8,
    "source_05": 4,
    "source_06": 3,
    "source_07": 5,
    "source_08": 10,
    "source_09": 2,
    "source_10": 9,
}
G_TO_INDEX = {g: g - 1 for g in range(1, 11)}


def load_queries(results_path: Path) -> list[tuple[list[int], int]]:
    """Returns [(subgroup_indices_0..9, choice_idx_or_-1), ...]."""
    with open(results_path) as f:
        r = json.load(f)
    out = []
    for q in r["queries"]:
        sub_idx = [G_TO_INDEX[SOURCE_TO_G[s]] for s in q["subgroup"]]
        if not q["matching"]:
            out.append((sub_idx, -1))
        else:
            for s in q["matching"]:
                out.append((sub_idx, G_TO_INDEX[SOURCE_TO_G[s]]))
    return out


def fit_lip(queries, K: int = 10, p0: float = 0.01, eps: float = 1.0,
            max_iters: int = 5000, tol: float = 1e-8, verbose: bool = True):
    """Fit the conditional-logit-with-null model.

    π_k = σ(α_k); parameters α_0 (null worth), α_1..α_K (source worths).
    Loss = -Σ log p(k_m | S_m) + eps Σ_{k≥1} (α_k - α_default)^2,
    where α_default = log(p_0/(1-p_0)). Regularization is on α_1..α_K only.
    """
    alpha_default = np.log(p0 / (1 - p0))
    alpha_t = torch.tensor(np.full(K + 1, alpha_default, dtype=np.float64),
                           requires_grad=True)
    opt = torch.optim.LBFGS([alpha_t], max_iter=max_iters, history_size=50,
                             tolerance_grad=tol, tolerance_change=tol,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        nll = torch.tensor(0.0, dtype=torch.float64)
        for sub_idx, k_idx in queries:
            terms = [alpha_t[0]] + [alpha_t[j + 1] for j in sub_idx]
            log_denom = torch.logsumexp(torch.stack(terms), dim=0)
            log_num = alpha_t[0] if k_idx == -1 else alpha_t[k_idx + 1]
            nll = nll - (log_num - log_denom)
        reg = eps * ((alpha_t[1:] - alpha_default) ** 2).sum()
        loss = nll + reg
        loss.backward()
        if verbose:
            print(f"  loss={loss.item():.4f}  nll={nll.item():.4f}  "
                  f"reg={reg.item():.4f}  alpha_0={alpha_t[0].item():.3f}  "
                  f"alpha_max(src)={alpha_t[1:].max().item():.3f}")
        return loss

    opt.step(closure)

    alpha_final = alpha_t.detach().cpu().numpy().copy()
    pi = 1.0 / (1.0 + np.exp(-alpha_final[1:]))  # sources only
    return alpha_final, pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str,
                   default="elicitation/results.json",
                   help="LLM elicitation results file (from lip_query_gemini.py).")
    p.add_argument("--p0", type=float, default=0.01,
                   help="Prior probability for an arbitrary source (default 0.01).")
    p.add_argument("--eps", type=float, default=1.0,
                   help="L2 regularization on alpha_k (default 1.0).")
    p.add_argument("--out", type=str,
                   default="results/lip.json")
    args = p.parse_args()

    queries = load_queries(Path(args.results))
    n_null = sum(1 for _, k in queries if k == -1)
    print(f"Loaded {len(queries)} (subgroup, choice) observations from {args.results}")
    print(f"  null picks: {n_null}, source picks: {len(queries) - n_null}")

    alpha, pi = fit_lip(queries, K=10, p0=args.p0, eps=args.eps)

    print(f"\n=== Fitted LIP (eps={args.eps}, p0={args.p0}) ===")
    print(f"  alpha_0 (null worth) = {alpha[0]:.3f}")
    for g in range(1, 11):
        print(f"  g={g:<2}: alpha = {alpha[g]:+.3f}  pi = {pi[g - 1]:.4f}")

    out = {
        "p0": args.p0,
        "eps": args.eps,
        "alpha_0": float(alpha[0]),
        "alpha": [float(a) for a in alpha[1:]],
        "pi": [float(x) for x in pi],
        "gravities": list(range(1, 11)),
        "source_to_g": SOURCE_TO_G,
        "n_queries": len(queries),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
