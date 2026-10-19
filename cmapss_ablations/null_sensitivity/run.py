"""Sensitivity of C-MAPSS LIP-EM to the null likelihood
(paper App. "Sensitivity to the Null Likelihood", Table tab:null_sweep).

The null enters EM only through p(D_k | c_k = 0). Multiplying it by e^c per
observation maps

    log LR_k  ->  log LR_k - N_k * c,

so c > 0 raises the null (more conservative) and c < 0 lowers it. We sweep
c over NULL_LEVELS (nats per observation) at the three cold-start cutoffs, and
also replace the empirical Bayes null (eq. eb-null) by the parametric pooled
null of App. null-heuristic: one Gaussian GLM fitted to all 99 sources pooled.

Everything else is the C-MAPSS setup of ../../cmapss (targets, spline basis,
LIPs, EM settings). With the EB null and c = 0 the loop below is exactly
em.em_train and reproduces Table 1; `--check` verifies this.

Usage:
    python run.py                  # 900 EM runs, ~40 s -> results/runs.json
    python run.py --targets 4,80   # a subset of targets
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cmapss"))

import experiment as E  # noqa: E402
from dataset import load_machine  # noqa: E402
from em import (  # noqa: E402
    _log_gauss_pdf,
    _m_step_closed_form,
    _ols,
    _precompute_ll_matrix,
    _precompute_source_meta,
    compute_weights,
    em_train,
    predict,
    stable_sigmoid,
    tempering_scale,
)


HERE = Path(__file__).resolve().parent
RUL_LEVELS = [0.9, 0.7, 0.5]          # cold-start cutoffs
# Null-level shifts c in nats per observation (0 = the EB null of Table 1).
NULL_LEVELS = [-10.0, -3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 10.0]
EM_METHODS = ["LIP-C", "LIP-G", "Uniform"]


def em_train_null(X_target, y_target, X_sources, y_sources, pi, metas,
                  ll_matrix, null_ll=None, c=0.0, cfg=E.EM_CONFIG):
    """em.em_train with a shifted and optionally replaced null.

    null_ll = None uses the EB null (eq. eb-null) that em.compute_weights
    rebuilds from the previous weights; an array uses that fixed
    log p(D_k | c_k = 0) instead (pooled null). The null log-likelihood is then
    raised by N_k * c. Returns (beta, weights, n_iter, converged, log-LR per
    observation at the last E-step, 1 / eps_k).
    """
    beta_0, sigma_0 = _ols(X_target, y_target)
    H_0 = (X_target.T @ X_target) / (sigma_0 ** 2)
    inv_eps = tempering_scale(H_0, metas)
    prior_logit = np.array([math.log(p + cfg.pi_clip)
                            - math.log(max(1.0 - p, cfg.pi_clip)) for p in pi])
    N_k = np.array([m["N"] for m in metas], dtype=np.float64)

    weights = pi.copy()
    beta = beta_0.copy()
    n_converged = 0
    converged = False
    for t in range(cfg.max_iter):
        w_prev = weights
        _, diag = compute_weights(beta, X_sources, y_sources, metas,
                                  prior_logit, inv_eps, weights, ll_matrix,
                                  t, cfg)
        log_p_null = diag["log_p_null"] if null_ll is None else null_ll
        log_lr = diag["log_p_rel"] - log_p_null - N_k * c
        weights = np.array([stable_sigmoid(x)
                            for x in diag["temper"] * log_lr + prior_logit])
        beta = _m_step_closed_form(beta_0, H_0, metas, weights)

        # Same stopping rule as em.em_train (counted from t_0 = ceil(ln 2 / nu)).
        w_change = float(np.max(np.abs(w_prev - weights)))
        if t >= cfg.stop_count_start and w_change <= cfg.convergence_tol:
            n_converged += 1
        else:
            n_converged = 0
        if n_converged >= cfg.convergence_patience:
            converged = True
            break
    return beta, weights, t + 1, converged, log_lr / N_k, inv_eps


def run_target(target: int, check: bool) -> tuple[list[dict], list[dict], float]:
    pis = E.priors_for(target)
    X_sources, y_sources = E.load_sources(target)
    cfg = E.EM_CONFIG

    # Source-only quantities: the same for every cutoff and null level.
    metas = _precompute_source_meta(X_sources, y_sources, cfg.tau)
    ll_matrix = _precompute_ll_matrix(X_sources, y_sources, metas)
    beta_pool, sigma_pool = _ols(np.vstack(X_sources), np.concatenate(y_sources))
    pooled_null_ll = np.array([_log_gauss_pdf(X_k, y_k, beta_pool, sigma_pool)
                               for X_k, y_k in zip(X_sources, y_sources)])

    target_file = E.SEQ_DIR / f"{target}.csv"
    _, X_grid, y_full = load_machine(target_file)
    runs, baselines, max_check_diff = [], [], 0.0
    for rul in RUL_LEVELS:
        co = E.cutoff_for(len(y_full), rul)
        _, X_t, y_t = load_machine(target_file, cutoff=co)
        future = y_full[co:]

        settings = [("eb", c) for c in NULL_LEVELS] + [("pooled", 0.0)]
        for method in EM_METHODS:
            pi = pis[method]
            for null, c in settings:
                beta, w, n_iter, converged, llr, inv_eps = em_train_null(
                    X_t, y_t, X_sources, y_sources, pi, metas, ll_matrix,
                    null_ll=pooled_null_ll if null == "pooled" else None, c=c)
                runs.append(dict(
                    target=target, rul=rul, method=method, null=null, c=c,
                    rmse=E.rmse(future, predict(X_grid, beta)[co:]),
                    n_iter=n_iter, converged=converged,
                    sum_w=float(w.sum()), max_w=float(w.max()),
                    max_abs_w_minus_pi=float(np.max(np.abs(w - pi))),
                    llr_p5=float(np.percentile(llr, 5)),
                    llr_median=float(np.median(llr)),
                    llr_p95=float(np.percentile(llr, 95)),
                    llr_max=float(llr.max()),
                    inv_eps_median=float(np.median(inv_eps))))
                if check and null == "eb" and c == 0.0:
                    beta_ref, hist = em_train(X_t, y_t, X_sources, y_sources,
                                              pi=pi, cfg=E.EM_CONFIG)
                    assert hist.n_iter == n_iter
                    max_check_diff = max(
                        max_check_diff,
                        float(np.max(np.abs(beta_ref - beta))),
                        float(np.max(np.abs(hist.weights[-1] - w))))

        X_pool = np.vstack([X_t] + list(X_sources))
        y_pool = np.concatenate([y_t] + list(y_sources))
        for method, beta in (
                ("Pooled", E.ridge_fit(X_pool, y_pool, E.RIDGE_LAMBDA)),
                ("Target-Only", E.ridge_fit(X_t, y_t, E.RIDGE_LAMBDA))):
            baselines.append(dict(target=target, rul=rul, method=method,
                                  rmse=E.rmse(future, predict(X_grid, beta)[co:])))
    return runs, baselines, max_check_diff


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--targets", default=",".join(map(str, E.TARGETS)),
                    help="comma-separated target engine numbers")
    ap.add_argument("--check", action="store_true",
                    help="also run em.em_train and assert that the EB null at "
                         "c = 0 gives the same fit")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "runs.json")
    args = ap.parse_args()
    targets = [int(s) for s in args.targets.split(",")]

    cfg = E.EM_CONFIG
    t_start = time.time()
    runs, baselines, max_check_diff = [], [], 0.0
    for target in targets:
        r, b, d = run_target(target, args.check)
        runs += r
        baselines += b
        max_check_diff = max(max_check_diff, d)
        print(f"  engine {target:>3d}: done ({time.time() - t_start:.0f}s)",
              flush=True)
    if args.check:
        print(f"EB null, c = 0 vs em.em_train: same iteration counts, "
              f"max |difference| in beta and w = {max_check_diff:.1e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(rul_levels=RUL_LEVELS, null_levels=NULL_LEVELS,
                    null_modes=["eb", "pooled"], tau=cfg.tau, nu=cfg.nu,
                    convergence_tol=cfg.convergence_tol,
                    convergence_patience=cfg.convergence_patience,
                    stop_count_start=cfg.stop_count_start,
                    max_iter=cfg.max_iter, p_0=E.P_0, eps_lip=E.EPS_LIP,
                    ridge_lambda=E.RIDGE_LAMBDA,
                    lip_files={k: v.name for k, v in E.LIP_FILES.items()})
    with open(args.out, "w") as f:
        json.dump(dict(settings=settings, targets=targets, runs=runs,
                       baselines=baselines), f, indent=1)
    print(f"{len(runs)} EM runs in {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
