"""LIP query-complexity simulation (appendix: scaling to many sources).

How must the number of subgroup queries ``N_M`` grow when a repository holds
``K >= 1000`` sources, and how many queries are needed to stabilize the latent
coefficients ``alpha_k``?

The claim under test is that the cost is **linear in K, not combinatorial in
the number of subsets**: the conditional logit has one coefficient per source,
each query of size ``s`` supplies evidence about ``s`` of them, so a source is
seen ``E[n_k] = N_M * s / K`` times and accuracy should be governed by that
appearance count rather than by ``K`` itself.

Design.
  * Ground-truth worths are resampled (with replacement, seeded) from the
    *fitted* C-MAPSS LIP coefficients ``alpha_k - alpha_0`` (target machine 80,
    cached Claude responses), so the simulated signal-to-noise of the
    elicitation matches the real one rather than an optimistic synthetic one.
  * Subgroups are drawn uniformly at random; choices are sampled from the
    conditional-logit-with-outside-option model (paper eq:lip_choice).
  * The fit is the paper objective (eq:LIP_opt) optimized with the same
    L-BFGS/strong-Wolfe setup as ``lip.fit_lip``, vectorized so that K = 1000
    with 20,000 queries is tractable.  Every run first checks the vectorized
    fit against ``lip.fit_lip`` on the real cached responses.
  * ``eps`` (lambda_LIP) is scaled with votes per source so the
    prior/likelihood balance is the same in every cell; it equals the paper's
    ``eps = 0.1`` at the C-MAPSS operating point (K = 99, N_M = 200).
  * Because the conditional-logit likelihood is invariant to a common shift of
    all worths, the headline coefficient metric is the RMSE of the *centred*
    worth vector.  The raw contrast ``alpha_k - alpha_0`` is reported too, but
    it is identified only when the response set contains at least one null
    choice (see the paper's strong-convexity condition for eq:LIP_opt).

Two sweeps are run:
  1. ``K in {10, 100, 1000}`` at fixed subgroup size ``s = 5``, over
     ``N_M / K``.  Tests linearity in K.
  2. ``s in {2, 5, 10}`` at fixed ``K = 100``, over appearances per source.
     Tests whether appearances per source, not queries, is the governing
     quantity.

Usage (from this folder):
    python lip_scaling_simulation.py [--responses PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lip.lip import fit_lip, load_responses  # noqa: E402

# Cached Claude subgroup responses of the C-MAPSS case study (global pool of
# 100 machines; the held-out target is filtered out at load time).
DEFAULT_RESPONSES = (HERE.parent / "cmapss" / "data"
                     / "llm_subgroup_queries_claude.json")
TARGET = "80.csv"

# --- LIP setting of the C-MAPSS case study ------------------------------------
P_0 = 0.01
EPS_LIP = 0.1
CMAPSS_K = 99              # 100 machines minus the held-out target
CMAPSS_N_M = 200
CMAPSS_MEAN_S = 4.0        # |S_m| ~ Uniform{3, 4, 5}

# `eps` per source-appearance, matched to the C-MAPSS setting so that the
# regularizer does not silently strengthen or weaken as N_M/K changes.
# lip.fit_lip: "a suitable value scales roughly with n_queries / K".
EPS_PER_VOTE = EPS_LIP / (CMAPSS_N_M / CMAPSS_K)

K_GRID = [10, 100, 1000]
RATIO_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]      # N_M / K
S_GRID = [2, 5, 10]
APPEARANCE_GRID = [2.0, 5.0, 10.0, 20.0, 50.0]     # N_M * s / K, sweep 2
N_SEEDS = 5
BASE_SEED = 42

METRICS = ["coef_rmse", "coef_rmse_centred", "coef_nrmse", "spearman",
           "recall_top10pct", "recall_top5", "null_rate", "alpha0_identified",
           "appearances"]
RUN_COLUMNS = ["K", "n_queries", "s", "seed", "appearances", "null_rate",
               "n_null_choices", "coef_rmse", "coef_rmse_centred",
               "coef_nrmse", "spearman", "recall_top10pct", "recall_top5"]


# ==============================================================================
# Vectorized conditional logit with outside option (paper eq:lip_choice)
# ==============================================================================

def fit_lip_fast(S_idx: np.ndarray, S_mask: np.ndarray, choice: np.ndarray,
                 K: int, p0: float = P_0, eps: float = EPS_LIP,
                 n_iter: int = 200) -> tuple[np.ndarray, float]:
    """Same objective and optimizer as ``lip.fit_lip``, vectorized.

    S_idx  : (M, s_max) int, source indices; padded entries ignored via mask.
    S_mask : (M, s_max) bool, True where the entry is a real member.
    choice : (M,) int, index into 0..K-1 of the chosen source, or -1 for null.
    Returns (alpha, alpha_0).
    """
    anchor = math.log(p0 / (1.0 - p0))
    alpha = torch.full((K,), anchor, dtype=torch.float64, requires_grad=True)
    alpha_0 = torch.tensor([anchor], dtype=torch.float64, requires_grad=True)

    idx = torch.as_tensor(S_idx, dtype=torch.long)
    mask = torch.as_tensor(S_mask, dtype=torch.bool)
    ch = torch.as_tensor(choice, dtype=torch.long)
    M = idx.shape[0]
    is_null = ch < 0
    ch_safe = torch.where(is_null, torch.zeros_like(ch), ch)

    optimizer = torch.optim.LBFGS([alpha, alpha_0], lr=0.5, max_iter=n_iter,
                                  history_size=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        member = torch.where(mask, alpha[idx],
                             torch.full_like(alpha[idx], -float("inf")))
        allv = torch.cat([alpha_0.expand(M, 1), member], dim=1)
        Z = torch.logsumexp(allv, dim=1)
        picked = torch.where(is_null, alpha_0.expand(M), alpha[ch_safe])
        nll = (Z - picked).sum()
        loss = nll + eps * ((alpha - anchor) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return alpha.detach().numpy(), float(alpha_0.detach())


# ==============================================================================
# Cached C-MAPSS responses
# ==============================================================================

def cached_responses(path: Path, target: str = TARGET):
    """Cached subgroup responses for one target, as a list and in padded form."""
    with open(path, encoding="utf-8") as f:
        K = len(json.load(f)["source_basenames"]) - 1
    responses = load_responses(str(path), target_basename=target)

    s_max = max(len(S) for S, _ in responses)
    S_idx = np.zeros((len(responses), s_max), dtype=np.int64)
    S_mask = np.zeros((len(responses), s_max), dtype=bool)
    choice = np.full(len(responses), -1, dtype=np.int64)
    for m, (S, C) in enumerate(responses):
        assert len(C) <= 1, "eq:lip_choice needs single-choice responses"
        S_idx[m, :len(S)] = S
        S_mask[m, :len(S)] = True
        if C:
            choice[m] = int(C[0])
    return responses, K, S_idx, S_mask, choice


def check_fast_fit(path: Path) -> dict:
    """Check the vectorized fit against lip.fit_lip on the cached responses."""
    responses, K, S_idx, S_mask, choice = cached_responses(path)
    pi_ref = fit_lip(responses, K=K, p0=P_0, eps=EPS_LIP)
    a, a0 = fit_lip_fast(S_idx, S_mask, choice, K=K, eps=EPS_LIP)
    pi_fast = 1.0 / (1.0 + np.exp(-a))
    return {
        "K": K, "n_responses": len(responses),
        "max_abs_pi_diff": float(np.max(np.abs(pi_ref - pi_fast))),
        "max_pi_ref": float(pi_ref.max()),
        "alpha_0": a0,
        "empirical_null_rate": float((choice < 0).mean()),
    }


def reference_worths(path: Path) -> np.ndarray:
    """Fitted worths relative to the null option (alpha_k - alpha_0).

    These set the ground-truth signal level of the simulation, so the
    simulated elicitation is exactly as separable as the real one.
    """
    _, K, S_idx, S_mask, choice = cached_responses(path)
    alpha, alpha_0 = fit_lip_fast(S_idx, S_mask, choice, K=K, eps=EPS_LIP)
    return alpha - alpha_0


# ==============================================================================
# Simulation
# ==============================================================================

def sample_queries(alpha_true: np.ndarray, alpha0_true: float, K: int,
                   n_queries: int, s: int, rng: np.random.Generator):
    """Draw subgroups uniformly and sample choices from paper eq:lip_choice."""
    S_idx = np.empty((n_queries, s), dtype=np.int64)
    for m in range(n_queries):
        S_idx[m] = rng.choice(K, size=s, replace=False)
    util = np.concatenate([np.full((n_queries, 1), alpha0_true),
                           alpha_true[S_idx]], axis=1)
    util = util - util.max(axis=1, keepdims=True)
    p = np.exp(util)
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random((n_queries, 1))
    pick = (p.cumsum(axis=1) < u).sum(axis=1)          # 0 = null option
    choice = np.where(pick == 0, -1, S_idx[np.arange(n_queries),
                                           np.clip(pick - 1, 0, s - 1)])
    return S_idx, np.ones_like(S_idx, dtype=bool), choice


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def recall_at(true_a: np.ndarray, fit_a: np.ndarray, m: int) -> float:
    m = max(1, min(m, len(true_a)))
    t = set(np.argsort(-true_a)[:m].tolist())
    f = set(np.argsort(-fit_a)[:m].tolist())
    return len(t & f) / m


def run_cell(K: int, n_queries: int, s: int, seed: int,
             ref_worths: np.ndarray) -> dict:
    rng = np.random.default_rng(seed)
    alpha_true = rng.choice(ref_worths, size=K, replace=True)
    alpha0_true = 0.0                                   # gauge fixed at null
    S_idx, S_mask, choice = sample_queries(alpha_true, alpha0_true, K,
                                           n_queries, s, rng)
    eps = EPS_PER_VOTE * (n_queries / K)
    a_hat, a0_hat = fit_lip_fast(S_idx, S_mask, choice, K=K, eps=eps)
    contrast = a_hat - a0_hat
    # The choice likelihood is invariant to a common shift of all worths, so
    # the estimable object is the centred worth vector.  The raw contrast is
    # also reported: it is only identified when M contains at least one null
    # choice (paper eq:LIP_opt discussion), which a small M may not.
    centred_err = (a_hat - a_hat.mean()) - (alpha_true - alpha_true.mean())
    sd_true = float(np.std(alpha_true))
    rmse_c = float(np.sqrt(np.mean(centred_err ** 2)))
    n_null = int((choice < 0).sum())
    return {
        "K": K, "n_queries": n_queries, "s": s, "seed": seed,
        "appearances": n_queries * s / K,
        "null_rate": float((choice < 0).mean()),
        "n_null_choices": n_null,
        "alpha0_identified": float(n_null > 0),
        "coef_rmse": float(np.sqrt(np.mean((contrast - alpha_true) ** 2))),
        "coef_rmse_centred": rmse_c,
        "coef_nrmse": rmse_c / sd_true if sd_true > 0 else float("nan"),
        "spearman": spearman(alpha_true, a_hat),
        "recall_top10pct": recall_at(alpha_true, a_hat, max(1, K // 10)),
        "recall_top5": recall_at(alpha_true, a_hat, 5),
    }


def aggregate(cells: list[dict], keys: tuple[str, ...]) -> list[dict]:
    out = {}
    for c in cells:
        out.setdefault(tuple(c[k] for k in keys), []).append(c)
    rows = []
    for key, group in sorted(out.items()):
        row = dict(zip(keys, key))
        row["n_seeds"] = len(group)
        for metric in METRICS:
            vals = [g[metric] for g in group]
            row[metric] = float(np.mean(vals))
            row[metric + "_se"] = float(np.std(vals) / math.sqrt(len(vals)))
        rows.append(row)
    return rows


# ==============================================================================
# Tables (appendix)
# ==============================================================================

def format_tables(summary: dict, cells: list[dict]) -> str:
    """Markdown tables of the two sweeps (means over seeds)."""
    lines = [f"# LIP query-complexity simulation ({N_SEEDS} seeds per cell)", ""]

    ref = summary["config"]["reference_worth_stats"]
    chk = summary["fast_fit_check"]
    null_sim = 100 * np.mean([c["null_rate"] for c in cells])
    lines += [
        f"- Reference worths (alpha_k - alpha_0, C-MAPSS target 80): "
        f"n = {ref['n']}, mean {ref['mean']:.2f}, sd {ref['sd']:.2f}, "
        f"range [{ref['min']:.2f}, {ref['max']:.2f}].",
        f"- Null-choice rate: simulated {null_sim:.2f}% "
        f"(mean over all {len(cells)} runs), empirical "
        f"{100 * chk['empirical_null_rate']:.2f}%.",
        f"- Vectorized fit vs lip.fit_lip on the cached responses: "
        f"max |pi_ref - pi_fast| = {chk['max_abs_pi_diff']:.1e}.",
        "",
        "## Sweep 1: query cost vs K (subgroup size s = 5)",
        "",
        "| K | `N_M` | `N_M/K` | appearances | nRMSE | Spearman | Recall@10% "
        "| alpha_0 identified |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(summary["sweep_k"],
                    key=lambda r: (r["n_queries"] / r["K"], r["K"])):
        lines.append(
            f"| {r['K']} | {r['n_queries']} | {r['n_queries'] / r['K']:g} "
            f"| {r['appearances']:g} | {r['coef_nrmse']:.2f} "
            f"| {r['spearman']:.3f} | {r['recall_top10pct']:.2f} "
            f"| {r['alpha0_identified']:.1f} |")

    lines += [
        "",
        "## Sweep 2: subgroup size (K = 100)",
        "",
        "| `s` | `N_M` | appearances | nRMSE | Spearman | Recall@10% "
        "| alpha_0 identified |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(summary["sweep_s"],
                    key=lambda r: (r["appearances"], r["s"])):
        lines.append(
            f"| {r['s']} | {r['n_queries']} | {r['appearances']:g} "
            f"| {r['coef_nrmse']:.2f} | {r['spearman']:.3f} "
            f"| {r['recall_top10pct']:.2f} | {r['alpha0_identified']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--responses", type=str, default=str(DEFAULT_RESPONSES),
                    help="cached C-MAPSS Claude subgroup responses (JSON)")
    ap.add_argument("--out-dir", type=str, default=str(HERE / "results"))
    args = ap.parse_args()
    responses_path = Path(args.responses)
    if not responses_path.is_file():
        ap.error(f"cached responses not found: {responses_path} "
                 f"(see README, section Data)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"config": {
        "p_0": P_0, "eps_lip": EPS_LIP, "eps_per_vote": EPS_PER_VOTE,
        "K_grid": K_GRID, "ratio_grid": RATIO_GRID, "s_grid": S_GRID,
        "appearance_grid": APPEARANCE_GRID, "n_seeds": N_SEEDS,
        "cmapss_setting": {"K": CMAPSS_K, "N_M": CMAPSS_N_M,
                           "mean_s": CMAPSS_MEAN_S,
                           "appearances": CMAPSS_N_M * CMAPSS_MEAN_S
                                          / CMAPSS_K},
    }}

    summary["fast_fit_check"] = check_fast_fit(responses_path)
    print("vectorized fit vs lip.fit_lip:",
          json.dumps(summary["fast_fit_check"], indent=2), flush=True)

    ref = reference_worths(responses_path)
    summary["config"]["reference_worth_stats"] = {
        "n": int(ref.size), "mean": float(ref.mean()), "sd": float(ref.std()),
        "min": float(ref.min()), "max": float(ref.max()),
    }
    print(f"reference worths (alpha_k - alpha_0): n={ref.size} "
          f"mean={ref.mean():.2f} sd={ref.std():.2f} "
          f"range [{ref.min():.2f}, {ref.max():.2f}]", flush=True)

    # ---- Sweep 1: K x N_M/K at fixed s = 5 ----
    cells = []
    for K in K_GRID:
        for ratio in RATIO_GRID:
            n_q = max(1, int(round(ratio * K)))
            for j in range(N_SEEDS):
                cells.append(run_cell(K, n_q, 5, BASE_SEED + 1000 * j + K, ref))
            print(f"  K={K:5d} N_M={n_q:6d} done", flush=True)
    summary["sweep_k"] = aggregate(cells, ("K", "n_queries"))

    # ---- Sweep 2: subgroup size at fixed K = 100 ----
    cells_s = []
    for s in S_GRID:
        for appear in APPEARANCE_GRID:
            n_q = max(1, int(round(appear * 100 / s)))
            for j in range(N_SEEDS):
                cells_s.append(run_cell(100, n_q, s, BASE_SEED + 1000 * j + s,
                                        ref))
            print(f"  s={s:3d} N_M={n_q:6d} (appearances {appear}) done",
                  flush=True)
    summary["sweep_s"] = aggregate(cells_s, ("s", "n_queries"))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "runs.csv", "w") as f:
        f.write(",".join(RUN_COLUMNS) + "\n")
        for c in cells + cells_s:
            f.write(",".join(str(c[k]) for k in RUN_COLUMNS) + "\n")
    tables = format_tables(summary, cells + cells_s)
    with open(out_dir / "tables.md", "w") as f:
        f.write(tables)

    print("\n" + tables)
    print(f"wrote {out_dir / 'summary.json'}, {out_dir / 'runs.csv'} "
          f"and {out_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
