"""
LIP-EM for the C-MAPSS case study (paper Algorithm 1, no approximation).

Backbone: GLM with Gaussian noise on a user-supplied design matrix
`(X_target, y_target, X_sources, y_sources)`. The driver picks the basis;
the paper uses the natural cubic spline in ``dataset.py``.

Key choices (paper main text §3 and Algorithm 1):

* theta-hat: per-domain maximum-likelihood estimates (OLS coefficients and
  sigma^2 = RSS / N).
* M-step: paper eq:m_step_update, the closed-form precision-weighted average
  of the source MLEs with M_k = (I + tau^2 H_k)^-1 H_k.
* E-step relevant likelihood: paper eq:rel_prox, with the exact tau-squared
  correction ``log L(theta;D_k) + (tau^2/2) g^T (I+tau^2 H_k)^-1 g
  - 1/2 log|I+tau^2 H_k|``.
* Null likelihood: empirical Bayes mixture of the other sources' MLE fits
  (paper eq:eb-null), weighted by the normalised (1 - w_j) and excluding the
  source itself.
* Tempering (paper eq:tempering): the factor beta_k(t) = (1 - exp(-nu*t)) / eps_k
  with eps_k^2 = Tr(H_0^+ H_k), where H_0^+ is the pseudo-inverse (H_0 is singular
  when the observed target prefix ends before some of the spline knots). (In the
  code, `beta` denotes GLM coefficients; the tempering factor is `temper`.)
* E-step weights: w_k = sigmoid(beta_k(t) * (log p_rel - log p_null) + logit pi_k).
* Convergence: ||w^(t) - w^(t-1)||_inf <= tol for `convergence_patience`
  consecutive iterations, counted only from t >= t_0 = max(1, ceil(ln 2 / nu))
  (t_0 = 14 for nu = 0.05), so EM cannot stop before the tempering ramp
  1 - exp(-nu*t) has reached 1/2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ============================================================================
# Numerically stable helpers
# ============================================================================

def np_logsumexp(values) -> float:
    """log(sum(exp(values))) without scipy."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    m = a.max()
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.exp(a - m).sum()))


def stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ============================================================================
# GLM helpers
# ============================================================================

def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns the MLE (beta_hat, sigma_hat) with sigma_hat = sqrt(RSS / N)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rss = float(np.sum((y - X @ beta) ** 2))
    sigma = max(math.sqrt(rss / max(len(y), 1)), 1e-6)
    return beta, sigma


def _log_gauss_pdf(X: np.ndarray, y: np.ndarray,
                   beta: np.ndarray, sigma: float) -> float:
    """log N(y | X beta, sigma^2 I)."""
    safe = max(sigma, 1e-6)
    rss = float(np.sum((y - X @ beta) ** 2))
    return -0.5 * len(y) * math.log(2.0 * math.pi * safe ** 2) \
           - rss / (2.0 * safe ** 2)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EMConfig:
    # --- iteration budget and convergence ---
    max_iter: int = 1000
    # Stop when ||w_t - w_{t-1}||_inf <= convergence_tol for
    # `convergence_patience` consecutive iterations t >= `stop_count_start`.
    convergence_tol: float = 1e-3
    convergence_patience: int = 5

    # --- tempering (paper eq:tempering) ---
    # factor beta_k(t) = (1 - exp(-nu*t)) / eps_k,
    # eps_k^2 = Tr(H_0^+ H_k), with the pseudo-inverse H_0^+.
    nu: float = 0.05

    # --- model ---
    tau: float = 1e-3             # prior std on theta_k around theta^(t)

    # --- numerical stability ---
    pi_clip: float = 1e-9         # floor on pi_k when computing logit(pi_k)
    null_min_inv_weight: float = 1e-9   # floor on (1 - w_j) in the EB null

    verbose: bool = False

    @property
    def stop_count_start(self) -> int:
        """t_0 = max(1, ceil(ln 2 / nu)): the ramp 1 - exp(-nu*t) has reached 1/2."""
        return max(1, math.ceil(math.log(2.0) / self.nu))


# ============================================================================
# Pre-compute per-source quantities
# ============================================================================

def _precompute_source_meta(X_sources: Sequence[np.ndarray],
                            y_sources: Sequence[np.ndarray],
                            tau: float) -> list[dict]:
    """For each source k, returns:
        beta_k_hat, sigma_k_hat,
        H_k = X_k^T X_k / sigma_k^2,
        inv_term = (I + tau^2 H_k)^-1,
        M_k = inv_term @ H_k       (precision in eq. m_step_update),
        log_det = log |I + tau^2 H_k|,
        N = N_k.
    """
    metas = []
    for X_s, y_s in zip(X_sources, y_sources):
        beta_s, sigma_s = _ols(X_s, y_s)
        d = X_s.shape[1]
        H_k = (X_s.T @ X_s) / (sigma_s ** 2)
        I = np.eye(d)
        inv_term = np.linalg.pinv(I + tau ** 2 * H_k)
        M_k = inv_term @ H_k

        sign, log_det = np.linalg.slogdet(I + tau ** 2 * H_k)
        if sign <= 0:
            log_det = -1e6  # numerical fallback

        metas.append({
            "beta": beta_s, "sigma": sigma_s, "H": H_k,
            "inv_term": inv_term, "M": M_k, "log_det": log_det,
            "N": len(y_s),
        })
    return metas


def _precompute_ll_matrix(X_sources: Sequence[np.ndarray],
                          y_sources: Sequence[np.ndarray],
                          metas: list[dict]) -> np.ndarray:
    """ll_matrix[k, j] = log p(D_k | beta_j_hat, sigma_j_hat).

    The diagonal is -inf because the EB null excludes the source itself;
    otherwise its in-sample MLE log-likelihood would dominate the mixture.
    """
    K = len(metas)
    ll = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            if k == j:
                ll[k, j] = -np.inf
            else:
                ll[k, j] = _log_gauss_pdf(X_sources[k], y_sources[k],
                                          metas[j]["beta"],
                                          metas[j]["sigma"])
    return ll


def tempering_scale(H_0: np.ndarray, metas: list[dict]) -> np.ndarray:
    """1 / eps_k for every source (paper Algorithm 1):

        eps_k^2 = Tr(H_0^+ H_k), with the pseudo-inverse H_0^+.
    """
    H_0_pinv = np.linalg.pinv(H_0)
    inv_eps = np.zeros(len(metas))
    for k, meta in enumerate(metas):
        eps2 = float(np.trace(H_0_pinv @ meta["H"]))
        inv_eps[k] = 1.0 / math.sqrt(max(eps2, 1e-12))
    return inv_eps


# ============================================================================
# E-step
# ============================================================================

def compute_weights(beta: np.ndarray,
                    X_sources: Sequence[np.ndarray],
                    y_sources: Sequence[np.ndarray],
                    metas: list[dict],
                    prior_logit: np.ndarray,
                    inv_eps: np.ndarray,
                    weights_prev: np.ndarray,
                    ll_matrix: np.ndarray,
                    iter_idx: int,
                    cfg: EMConfig) -> tuple[np.ndarray, dict]:
    """One E-step. Returns (new_w, diagnostics)."""
    K = len(metas)

    # ------ Relevant likelihood with the tau-squared correction (eq:rel_prox) ------
    log_p_rel = np.zeros(K)
    for k in range(K):
        meta = metas[k]
        X_k, y_k = X_sources[k], y_sources[k]
        log_L = _log_gauss_pdf(X_k, y_k, beta, meta["sigma"])
        g_k = X_k.T @ (y_k - X_k @ beta) / (meta["sigma"] ** 2)
        quad_form = float(g_k @ meta["inv_term"] @ g_k)
        log_p_rel[k] = log_L + (0.5 * cfg.tau ** 2 * quad_form
                                - 0.5 * meta["log_det"])

    # ------ EB null (eq:eb-null): normalised (1 - w_j), self excluded ------
    log_p_null = np.zeros(K)
    inv_w_full = np.clip(1.0 - weights_prev, cfg.null_min_inv_weight, 1.0)
    for k in range(K):
        iw = inv_w_full.copy()
        iw[k] = cfg.null_min_inv_weight  # exclude self
        iw_sum = iw.sum()
        if iw_sum <= 0:
            log_p_null[k] = -np.inf
            continue
        log_iw = np.log(iw / iw_sum)
        log_p_null[k] = np_logsumexp(ll_matrix[k, :] + log_iw)

    # ------ Tempered logit and weight per source ------
    ramp = 1.0 - math.exp(-cfg.nu * iter_idx)
    new_w = np.zeros(K)
    diag = {"log_p_rel": log_p_rel, "log_p_null": log_p_null,
            "temper": np.zeros(K), "logit": np.zeros(K)}
    for k in range(K):
        temper = inv_eps[k] * ramp
        logit = temper * (log_p_rel[k] - log_p_null[k]) + prior_logit[k]
        new_w[k] = stable_sigmoid(logit)
        diag["temper"][k] = temper
        diag["logit"][k] = logit

    return new_w, diag


# ============================================================================
# M-step
# ============================================================================

def _m_step_closed_form(beta_0: np.ndarray, H_0: np.ndarray,
                        metas: list[dict],
                        weights: np.ndarray) -> np.ndarray:
    """Paper main-text eq. m_step_update:

        theta^(t+1) = (H_0 + sum_k w_k M_k)^-1 (H_0 beta_0 + sum_k w_k M_k beta_k_hat)

    where M_k = (I + tau^2 H_k)^-1 H_k.
    """
    left = np.copy(H_0)
    right = H_0 @ beta_0
    for k, w in enumerate(weights):
        if w > 0:
            meta = metas[k]
            left += w * meta["M"]
            right += w * (meta["M"] @ meta["beta"])
    return np.linalg.pinv(left) @ right


# ============================================================================
# Full EM loop
# ============================================================================

@dataclass
class EMHistory:
    weights: list[np.ndarray] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    converged_at: int | None = None

    @property
    def n_iter(self) -> int:
        """Number of EM iterations (E-steps) run."""
        return len(self.weights)


def em_train(X_target: np.ndarray, y_target: np.ndarray,
             X_sources: Sequence[np.ndarray],
             y_sources: Sequence[np.ndarray],
             pi: np.ndarray,
             cfg: EMConfig | None = None) -> tuple[np.ndarray, EMHistory]:
    """Run LIP-EM on a Gaussian-noise GLM.

    Returns (beta_final, history).
    """
    if cfg is None:
        cfg = EMConfig()
    K = len(X_sources)
    pi = np.asarray(pi, dtype=np.float64)
    assert pi.shape == (K,), f"pi must have shape ({K},), got {pi.shape}"

    beta_0, sigma_0 = _ols(X_target, y_target)
    H_0 = (X_target.T @ X_target) / (sigma_0 ** 2)
    d = X_target.shape[1]
    N_0 = len(y_target)

    if cfg.verbose:
        print(f"[em] N_0={N_0}  K={K}  d={d}  tau={cfg.tau}  nu={cfg.nu}")

    metas = _precompute_source_meta(X_sources, y_sources, cfg.tau)
    ll_matrix = _precompute_ll_matrix(X_sources, y_sources, metas)
    inv_eps = tempering_scale(H_0, metas)
    prior_logit = np.array([math.log(p + cfg.pi_clip)
                            - math.log(max(1.0 - p, cfg.pi_clip)) for p in pi])
    t_0 = cfg.stop_count_start

    # Initialization. The tempering factor is 0 at t=0, so w^(0) = pi whatever
    # the starting coefficients; the target MLE beta_0 is convenient.
    weights = pi.copy()
    beta = beta_0.copy()

    history = EMHistory()
    n_converged = 0   # patience counter for ||Δw||_inf convergence

    for t in range(cfg.max_iter):
        w_prev = weights.copy()
        weights, diag = compute_weights(beta, X_sources, y_sources, metas,
                                        prior_logit, inv_eps, weights,
                                        ll_matrix, t, cfg)
        history.weights.append(weights.copy())
        history.diagnostics.append(diag)

        beta = _m_step_closed_form(beta_0, H_0, metas, weights)

        # Convergence: ||Δw||_inf <= tol for `patience` consecutive t >= t_0.
        w_change = float(np.max(np.abs(w_prev - weights)))
        if cfg.verbose and (t < 3 or (t + 1) % max(1, cfg.max_iter // 10) == 0):
            print(f"[em it {t}] w_change={w_change:.2e}  w[:3]={weights[:3]}")
        if t >= t_0 and w_change <= cfg.convergence_tol:
            n_converged += 1
        else:
            n_converged = 0
        if n_converged >= cfg.convergence_patience:
            history.converged_at = t
            if cfg.verbose:
                print(f"[em] converged at iter {t} (w_change={w_change:.2e})")
            break

    return beta, history


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta
