"""LIP-EM with exact Bernoulli and Poisson likelihoods, and controlled
prior corruption (paper App. "Non-Gaussian Likelihoods and Controlled Prior
Corruption", Tables tab:nongauss and tab:corruption).

Each replication has K = 20 sources with N_k = 80 observations. Five sources
are relevant and share the target parameter theta_0, drawn per replication
from {0.2, 0.8} (Bernoulli) or {2, 4} (Poisson); the other 15 share the null
parameter (0.5 or 3). EM is Algorithm 2 with d = 1 and tau = 0: the E-step
uses the exact likelihood ratio against the true null model, the M-step is
the exact weighted MLE of the mean parameter, and the tempering scale is
eps_k^2 = N_k / N_0. Priors:

  * uniform:   p_0 = 0.01 for every source;
  * aligned:   a hand-specified informative prior (0.15-0.35 on the relevant
               sources, 0.004-0.018 on the others; no LLM is queried);
  * shuffled:  a random permutation of the aligned prior per replication;
  * oracle:    0.99 on the relevant sources and 0.01 on the others;
  * corrupt_q: the oracle with a share q of its five high-prior assignments
               swapped onto random irrelevant sources.

Permutations and swaps keep the multiset of prior values. The metric is the
excess test negative log-likelihood, KL(truth || estimate).

Usage:
    python run.py      # 2 families x 6 N_0 x 1000 replications, ~5 min
                       # -> results/replicates.csv.gz, results/summary.csv, results/settings.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cmapss"))

from em import EMConfig  # noqa: E402


HERE = Path(__file__).resolve().parent
K_SOURCES = 20
RELEVANT_SOURCES = 5
SOURCE_SIZE = 80
TARGET_SIZES = (2, 4, 8, 16, 32, 64)
CORRUPTION_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
FAMILIES = {  # target parameters, null parameter
    "bernoulli": ((0.2, 0.8), 0.5),
    "poisson": ((2.0, 4.0), 3.0),
}
UNIFORM_P0 = 0.01
# EM settings of this study: nu as for C-MAPSS, a tighter tolerance and a
# 500-iteration cap. Patience is counted from t_0 = ceil(ln 2 / nu), as in
# ../../cmapss/em.py.
NU = 0.05
MAX_ITERATIONS = 500
CONVERGENCE_TOLERANCE = 1e-8
CONVERGENCE_PATIENCE = 5
T_0 = EMConfig(nu=NU).stop_count_start


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p) - np.log1p(-p)


def pairwise_auc(weights: np.ndarray, relevant: np.ndarray) -> float:
    pos, neg = weights[relevant], weights[~relevant]
    return float(((pos[:, None] > neg[None, :])
                  + 0.5 * (pos[:, None] == neg[None, :])).mean())


def aligned_prior() -> np.ndarray:
    """Informative, heterogeneous and directionally correct, but not an oracle."""
    high = np.asarray([0.35, 0.30, 0.25, 0.20, 0.15])
    low = np.linspace(0.004, 0.018, K_SOURCES - RELEVANT_SOURCES)
    return np.concatenate((high, low))


def oracle_prior() -> np.ndarray:
    return np.concatenate((np.full(RELEVANT_SOURCES, 0.99),
                           np.full(K_SOURCES - RELEVANT_SOURCES, 0.01)))


def corrupted_oracle(level: float, rng: np.random.Generator) -> np.ndarray:
    """Swap a share `level` of the oracle's high-prior assignments onto
    irrelevant sources."""
    prior = oracle_prior()
    swaps = int(round(level * RELEVANT_SOURCES))
    relevant = rng.choice(RELEVANT_SOURCES, size=swaps, replace=False)
    irrelevant = rng.choice(np.arange(RELEVANT_SOURCES, K_SOURCES), size=swaps,
                            replace=False)
    prior[relevant], prior[irrelevant] = (prior[irrelevant].copy(),
                                          prior[relevant].copy())
    return prior


def priors_for_replicate(rng: np.random.Generator) -> dict[str, np.ndarray]:
    priors = {"uniform": np.full(K_SOURCES, UNIFORM_P0),
              "aligned": aligned_prior(),
              "shuffled": rng.permutation(aligned_prior()),
              "oracle": oracle_prior()}
    for level in CORRUPTION_LEVELS:
        priors[f"corrupt_{int(round(100 * level))}"] = corrupted_oracle(level, rng)
    return priors


def log_lik(family: str, sums: np.ndarray, size: int, theta: float) -> np.ndarray:
    """Log-likelihood of each source's data at theta, up to terms that cancel
    in the likelihood ratio (the Poisson log-factorials)."""
    if family == "bernoulli":
        return sums * math.log(theta) + (size - sums) * math.log1p(-theta)
    return sums * math.log(theta) - size * theta


def em_fit(family: str, target_sum: float, target_size: int,
           source_sums: np.ndarray, prior: np.ndarray, null: float
           ) -> tuple[float, np.ndarray, int]:
    """tau = 0 LIP-EM for the mean parameter. Returns (theta, w, n_iter)."""
    lo, hi = (1e-8, 1.0 - 1e-8) if family == "bernoulli" else (1e-10, np.inf)
    if family == "bernoulli":
        theta = float(np.clip(target_sum / target_size, 1e-6, 1 - 1e-6))
    else:
        theta = max(target_sum / target_size, 1e-8)
    null = float(np.clip(null, lo, hi))
    log_null = log_lik(family, source_sums, SOURCE_SIZE, null)
    weights = np.asarray(prior, dtype=float).copy()
    prior_logit = logit(weights)
    inv_eps = math.sqrt(target_size / SOURCE_SIZE)   # eps_k^2 = N_k / N_0

    stable = 0
    for t in range(MAX_ITERATIONS):
        previous = weights
        theta = float(np.clip(theta, lo, hi))
        log_rel = log_lik(family, source_sums, SOURCE_SIZE, theta)
        temper = inv_eps * (1.0 - math.exp(-NU * t))
        weights = stable_sigmoid(temper * (log_rel - log_null) + prior_logit)
        theta = float((target_sum + weights @ source_sums)
                      / (target_size + SOURCE_SIZE * weights.sum()))

        # ||Δw||_inf <= tol for `patience` consecutive t >= t_0.
        if t >= T_0 and np.max(np.abs(weights - previous)) <= CONVERGENCE_TOLERANCE:
            stable += 1
        else:
            stable = 0
        if stable >= CONVERGENCE_PATIENCE:
            return theta, weights, t + 1
    return theta, weights, MAX_ITERATIONS


def excess_nll(family: str, truth: float, estimate: float) -> float:
    """KL(truth || estimate) for one observation."""
    if family == "bernoulli":
        p = float(np.clip(truth, 1e-12, 1 - 1e-12))
        q = float(np.clip(estimate, 1e-12, 1 - 1e-12))
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    p, q = max(truth, 1e-12), max(estimate, 1e-12)
    return q - p + p * math.log(p / q)


def draw_replicate(family: str, target_size: int, rng: np.random.Generator
                   ) -> tuple[float, float, float, np.ndarray]:
    """Returns (theta_0, null parameter, target sum, source sums)."""
    choices, null = FAMILIES[family]
    truth = float(rng.choice(choices))
    params = np.concatenate((np.full(RELEVANT_SOURCES, truth),
                             np.full(K_SOURCES - RELEVANT_SOURCES, null)))
    if family == "bernoulli":
        target_sum = float(rng.binomial(target_size, truth))
        source_sums = rng.binomial(SOURCE_SIZE, params).astype(float)
    else:
        target_sum = float(rng.poisson(target_size * truth))
        source_sums = rng.poisson(SOURCE_SIZE * params).astype(float)
    return truth, null, target_sum, source_sums


def run(replicates: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    relevant = np.arange(K_SOURCES) < RELEVANT_SOURCES
    rows = []
    for family in FAMILIES:
        for target_size in TARGET_SIZES:
            for rep in range(replicates):
                truth, null, target_sum, source_sums = draw_replicate(
                    family, target_size, rng)
                base = dict(family=family, target_size=target_size,
                            replicate=rep, truth=truth)
                if family == "bernoulli":
                    target_only = float(np.clip(target_sum / target_size,
                                                1e-6, 1 - 1e-6))
                else:
                    target_only = max(target_sum / target_size, 1e-8)
                pooled = float((target_sum + source_sums.sum())
                               / (target_size + K_SOURCES * SOURCE_SIZE))
                for method, estimate in (("target_only", target_only),
                                         ("pooled", pooled)):
                    rows.append(dict(base, method=method, estimate=estimate,
                                     excess_nll=excess_nll(family, truth, estimate)))
                for method, prior in priors_for_replicate(rng).items():
                    estimate, w, n_iter = em_fit(family, target_sum, target_size,
                                                 source_sums, prior, null)
                    rows.append(dict(
                        base, method=method, estimate=estimate,
                        excess_nll=excess_nll(family, truth, estimate),
                        source_auc=pairwise_auc(w, relevant),
                        source_brier=float(np.mean((w - relevant) ** 2)),
                        n_iter=n_iter,
                        max_abs_w_minus_pi=float(np.max(np.abs(w - prior)))))
    return rows


def se(v: np.ndarray) -> float:
    return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else math.nan


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["target_size"], r["method"])].append(r)
    out = []
    for (family, target_size, method), local in groups.items():
        nll = np.array([r["excess_nll"] for r in local])
        sq = np.array([(r["estimate"] - r["truth"]) ** 2 for r in local])
        rec = dict(family=family, target_size=target_size, method=method,
                   replicates=len(local), excess_nll=float(nll.mean()),
                   excess_nll_se=se(nll),
                   parameter_rmse=math.sqrt(float(sq.mean())))
        if "n_iter" in local[0]:
            auc = np.array([r["source_auc"] for r in local])
            rec.update(source_auc=float(auc.mean()), source_auc_se=se(auc),
                       source_brier=float(np.mean([r["source_brier"] for r in local])),
                       mean_iter=float(np.mean([r["n_iter"] for r in local])),
                       n_at_cap=sum(r["n_iter"] >= MAX_ITERATIONS for r in local))
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if path.suffix == ".gz":
        # mtime=0 keeps the archive byte-identical across runs.
        with gzip.GzipFile(path, "wb", mtime=0) as f:
            f.write(buf.getvalue().encode())
    else:
        path.write_text(buf.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    t_start = time.time()
    rows = run(args.replicates, args.seed)
    summary = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "replicates.csv.gz", rows)
    write_csv(args.out_dir / "summary.csv", summary)
    settings = dict(seed=args.seed, replicates=args.replicates,
                    sources=K_SOURCES, relevant_sources=RELEVANT_SOURCES,
                    source_size=SOURCE_SIZE, target_sizes=list(TARGET_SIZES),
                    families={f: dict(theta_0=list(c), null=n)
                              for f, (c, n) in FAMILIES.items()},
                    uniform_p0=UNIFORM_P0, aligned_prior=aligned_prior().tolist(),
                    oracle_prior=oracle_prior().tolist(),
                    corruption_levels=list(CORRUPTION_LEVELS), tau=0, nu=NU,
                    convergence_tol=CONVERGENCE_TOLERANCE,
                    convergence_patience=CONVERGENCE_PATIENCE,
                    stop_count_start=T_0, max_iter=MAX_ITERATIONS)
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=1))
    print(f"{len(rows)} rows in {time.time() - t_start:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
