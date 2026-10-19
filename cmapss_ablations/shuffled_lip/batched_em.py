"""em.em_train of ../../cmapss/em.py, vectorised over many priors.

The shuffled-LIP control runs EM for hundreds of priors on the same
(target, cutoff). This module runs them together: every row of `priors` gets
exactly the update of em.em_train (Algorithm 1: E-step eq. rel_prox with the
tau^2 correction, EB null eq. eb-null with normalised (1 - w_j), tempering
beta_k(t) = (1 - exp(-nu t)) / eps_k with the pseudo-inverse scale eps_k^2 = Tr(H_0^+ H_k), closed-form M-step,
stopping counted from t_0 = ceil(ln 2 / nu)). The per-source quantities, the
tempering scale and the settings come from em.py itself; only the E- and
M-steps are vectorised over the rows. `run.py` checks the result against
em.em_train for the unshuffled priors.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cmapss"))

from em import (  # noqa: E402
    EMConfig,
    _ols,
    _precompute_ll_matrix,
    _precompute_source_meta,
    tempering_scale,
)


def source_stats(X_sources: Sequence[np.ndarray],
                 y_sources: Sequence[np.ndarray], cfg: EMConfig) -> dict:
    """Source-only quantities (em._precompute_source_meta and the EB-null
    log-likelihood matrix), stacked over sources."""
    metas = _precompute_source_meta(X_sources, y_sources, cfg.tau)
    stack = lambda key: np.stack([m[key] for m in metas])  # noqa: E731
    return dict(metas=metas,
                ll_matrix=_precompute_ll_matrix(X_sources, y_sources, metas),
                beta=stack("beta"), sigma=stack("sigma"),
                inv_term=stack("inv_term"), M=stack("M"),
                log_det=stack("log_det"), X=list(X_sources), y=list(y_sources))


def _e_step(beta: np.ndarray, weights_prev: np.ndarray, stats: dict,
            cfg: EMConfig) -> tuple[np.ndarray, np.ndarray]:
    """log p_rel and log p_null of em.compute_weights for every row; shape (B, K)."""
    B, K = weights_prev.shape
    log_p_rel = np.empty((B, K))
    for k, (X_k, y_k) in enumerate(zip(stats["X"], stats["y"])):
        sigma = stats["sigma"][k]
        resid = y_k[None, :] - beta @ X_k.T                       # (B, N_k)
        log_L = (-0.5 * len(y_k) * math.log(2.0 * math.pi * sigma ** 2)
                 - np.sum(resid ** 2, axis=1) / (2.0 * sigma ** 2))
        g = resid @ X_k / sigma ** 2                              # (B, d)
        quad = np.einsum("bi,ij,bj->b", g, stats["inv_term"][k], g)
        log_p_rel[:, k] = (log_L + 0.5 * cfg.tau ** 2 * quad
                           - 0.5 * stats["log_det"][k])

    inv_w = np.clip(1.0 - weights_prev, cfg.null_min_inv_weight, 1.0)
    log_p_null = np.empty((B, K))
    for k in range(K):
        iw = inv_w.copy()
        iw[:, k] = cfg.null_min_inv_weight                         # exclude self
        log_iw = np.log(iw / iw.sum(axis=1, keepdims=True))
        log_p_null[:, k] = logsumexp(stats["ll_matrix"][k][None, :] + log_iw,
                                     axis=1)
    return log_p_rel, log_p_null


def em_batch(X_target: np.ndarray, y_target: np.ndarray, priors: np.ndarray,
             stats: dict, cfg: EMConfig | None = None):
    """Run em.em_train for every row of `priors` (shape (B, K)).

    Returns (beta (B, d), final weights (B, K), n_iter (B,), converged (B,),
    1 / eps_k (K,)).
    """
    if cfg is None:
        cfg = EMConfig()
    priors = np.asarray(priors, dtype=np.float64)
    B, K = priors.shape
    beta_0, sigma_0 = _ols(X_target, y_target)
    H_0 = (X_target.T @ X_target) / (sigma_0 ** 2)
    inv_eps = tempering_scale(H_0, stats["metas"])
    prior_logit = (np.log(priors + cfg.pi_clip)
                   - np.log(np.maximum(1.0 - priors, cfg.pi_clip)))
    M_beta = np.einsum("kij,kj->ki", stats["M"], stats["beta"])
    H_0_beta_0 = H_0 @ beta_0
    t_0 = cfg.stop_count_start

    weights = priors.copy()
    beta = np.repeat(beta_0[None, :], B, axis=0)
    n_iter = np.zeros(B, dtype=int)
    patience = np.zeros(B, dtype=int)
    converged = np.zeros(B, dtype=bool)
    for t in range(cfg.max_iter):
        rows = np.flatnonzero(~converged)
        if rows.size == 0:
            break
        w_prev = weights[rows]
        log_p_rel, log_p_null = _e_step(beta[rows], w_prev, stats, cfg)
        temper = inv_eps * (1.0 - math.exp(-cfg.nu * t))
        w_new = expit(temper[None, :] * (log_p_rel - log_p_null)
                      + prior_logit[rows])

        # M-step (eq. m_step_update), one closed-form solve per row.
        left = H_0[None] + np.einsum("bk,kij->bij", w_new, stats["M"])
        right = H_0_beta_0[None] + w_new @ M_beta
        beta[rows] = np.einsum("bij,bj->bi", np.linalg.pinv(left), right)
        weights[rows] = w_new
        n_iter[rows] = t + 1

        # ||Δw||_inf <= tol for `patience` consecutive t >= t_0.
        change = np.max(np.abs(w_new - w_prev), axis=1)
        ok = (t >= t_0) & (change <= cfg.convergence_tol)
        patience[rows] = np.where(ok, patience[rows] + 1, 0)
        converged[rows] = patience[rows] >= cfg.convergence_patience
    return beta, weights, n_iter, converged, inv_eps
