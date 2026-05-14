"""
LIP-aided EM for the C-MAPSS case study.

Backbone: GLM with Gaussian noise on a user-supplied design matrix
`(X_target, y_target, X_sources, y_sources)`. The driver picks the basis;
the published config uses natural cubic spline (see ``data.py`` for the
basis builder ``basis()`` and the K_KNOTS / X_MAX defaults).

Key choices (paper main text §3.1.2 / §3.2 / §3.3):

* M-step: paper main-text eq:m_step_update — closed-form weighted average of
  source MLEs in the precision-weighted form. Supports the §3.3.1 small-tau
  approximation via ``EMConfig.approximation`` for the appendix table.
* E-step relevant likelihood: paper eq:rel_prox — exact tau-squared correction
  ``log L(theta;D_k) + (tau^2/2) g^T (I+tau^2 H_k)^-1 g - 1/2 log|I+tau^2 H_k|``.
  Skipped when ``e_step_approx=True`` (small-tau).
* Tempering schedule: paper eq:tempering ``lambda(t) = ratio * (1 - exp(-nu*t))``
  with ratio = ``1/sqrt(Tr(H_0^-1 H_k))`` (the exact 1/eps_k from §3.3.2).
* Null likelihood: empirical-Bayes (paper eq:eb-null) with self-exclusion
  to avoid the in-sample MLE log-lik dominating the mixture for partially-
  relevant sources.
* Convergence: ``||Δw||_inf <= conv_tol`` for `convergence_patience` consecutive
  iterations (matches the publish/NN reference).
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

def _ols(X: np.ndarray, y: np.ndarray,
         rcond: float | None = None) -> tuple[np.ndarray, float]:
    """Returns (beta_hat, sigma_hat). sigma_hat is sqrt(RSS / N)."""
    beta = np.linalg.lstsq(X, y, rcond=rcond)[0]
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
    # --- iteration budget ---
    max_iter: int = 1000
    # Stop when ||w_t - w_{t-1}||_inf <= convergence_tol for `convergence_patience`
    # consecutive iterations.
    convergence_tol: float = 1e-3
    convergence_patience: int = 5

    # --- tempering schedule (paper eq:tempering) ---
    # lambda(t) = ratio * (1 - exp(-nu*t)), ratio = 1/sqrt(Tr(H_0^-1 H_k)).
    # nu=0.05 -> ~95% saturated at t=60.
    nu: float = 0.05

    # --- model ---
    tau: float = 1e-3             # prior std on theta_k around theta^(t)

    # --- Hessian approximation (paper §3.3.1, eq. prox_update) ---
    # Setting `approximation=True` reproduces the paper's appendix table:
    #   * E-step drops the tau-squared correction terms (small-tau limit)
    #   * M-step uses theta^(t+1) ≈ (N_0 theta_0 + sum_k w_k N_k theta_k_hat)
    #                                  / (N_0 + sum_k w_k N_k)
    approximation: bool = False
    e_step_approx: bool | None = None  # None -> defer to `approximation`
    m_step_approx: bool | None = None  # None -> defer to `approximation`

    @property
    def use_e_step_approx(self) -> bool:
        return self.approximation if self.e_step_approx is None else self.e_step_approx

    @property
    def use_m_step_approx(self) -> bool:
        return self.approximation if self.m_step_approx is None else self.m_step_approx

    # --- Numerical stability ---
    pi_clip: float = 1e-9         # floor on pi_k when computing log(pi_k)
    null_min_inv_weight: float = 1e-9   # floor on (1 - w_j) in EB null

    verbose: bool = True


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

    Diagonal is -inf because the EB null sum excludes self (paper §C.2 caveat:
    EB is "fragile when sources are few or contaminated by relevant ones" — so
    we drop the in-sample MLE term that would otherwise dominate the mixture
    for any source with pi_k not already ~1).
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


# ============================================================================
# E-step
# ============================================================================

def _tempering_lambda(t: int, ratio: float, nu: float) -> float:
    """Paper eq:tempering: lambda(t) = ratio * (1 - exp(-nu * t))."""
    return ratio * (1.0 - math.exp(-nu * t))


def compute_weights(beta: np.ndarray,
                    X_sources: Sequence[np.ndarray],
                    y_sources: Sequence[np.ndarray],
                    metas: list[dict],
                    pi: np.ndarray,
                    weights_prev: np.ndarray,
                    ll_matrix: np.ndarray,
                    iter_idx: int,
                    cfg: EMConfig,
                    H_0: np.ndarray) -> tuple[np.ndarray, dict]:
    """One E-step. Returns (new_w, diagnostics)."""
    K = len(metas)
    log_L_rel = np.zeros(K)
    log_p_rel_terms = np.zeros(K)  # tau-squared correction (eq:rel_prox)

    for k in range(K):
        meta = metas[k]
        X_k, y_k = X_sources[k], y_sources[k]
        log_L_rel[k] = _log_gauss_pdf(X_k, y_k, beta, meta["sigma"])
        if not cfg.use_e_step_approx:
            g_k = X_k.T @ (y_k - X_k @ beta) / (meta["sigma"] ** 2)
            quad_form = float(g_k @ meta["inv_term"] @ g_k)
            log_p_rel_terms[k] = (0.5 * cfg.tau ** 2 * quad_form
                                  - 0.5 * meta["log_det"])

    log_p_rel = log_L_rel + log_p_rel_terms

    # ------ EB null (paper eq:eb-null) with self-exclusion ------
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

    # ------ Logit and weight per source ------
    new_w = np.zeros(K)
    diag = {"log_L_rel": log_L_rel.copy(), "log_p_rel": log_p_rel.copy(),
            "log_p_null": log_p_null.copy(), "diff": np.zeros(K),
            "lambda": np.zeros(K), "prior_logit": np.zeros(K),
            "logit": np.zeros(K)}

    for k in range(K):
        meta = metas[k]
        # Exact 1/eps from §3.3.2: trace(H_0^-1 H_k).
        trace_term = float(np.trace(np.linalg.pinv(H_0) @ meta["H"]))
        ratio = 1.0 / math.sqrt(max(trace_term, 1e-12))
        lam = _tempering_lambda(iter_idx, ratio, cfg.nu)

        diff = log_p_rel[k] - log_p_null[k]
        prior_logit = (math.log(pi[k] + cfg.pi_clip)
                       - math.log(max(1.0 - pi[k], cfg.pi_clip)))
        logit = lam * diff + prior_logit
        new_w[k] = stable_sigmoid(logit)

        diag["diff"][k] = diff
        diag["lambda"][k] = lam
        diag["prior_logit"][k] = prior_logit
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


def _m_step_closed_form_approx(beta_0: np.ndarray, N_0: int,
                               metas: list[dict],
                               weights: np.ndarray) -> np.ndarray:
    """Paper §3.3.1 small-tau approximation (eq. prox_update, appendix Table):

        theta^(t+1) = (N_0 beta_0 + sum_k w_k N_k beta_k_hat)
                       / (N_0 + sum_k w_k N_k).
    """
    weight_sum = float(N_0)
    weighted = N_0 * beta_0.astype(np.float64)
    for k, w in enumerate(weights):
        if w > 0:
            N_k = metas[k]["N"]
            weight_sum += w * N_k
            weighted = weighted + w * N_k * metas[k]["beta"].astype(np.float64)
    return weighted / weight_sum


# ============================================================================
# Full EM loop
# ============================================================================

@dataclass
class EMHistory:
    weights: list[np.ndarray] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    converged_at: int | None = None


def em_train(X_target: np.ndarray, y_target: np.ndarray,
             X_sources: Sequence[np.ndarray],
             y_sources: Sequence[np.ndarray],
             pi: np.ndarray,
             cfg: EMConfig | None = None) -> tuple[np.ndarray, EMHistory]:
    """Run LIP-aided EM on a Gaussian-noise GLM.

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
        print(f"[em] N_0={N_0}  K={K}  d={d}  tau={cfg.tau}  "
              f"approximation={cfg.approximation}")

    metas = _precompute_source_meta(X_sources, y_sources, cfg.tau)
    ll_matrix = _precompute_ll_matrix(X_sources, y_sources, metas)

    # Initialization. lambda(0)=0 makes w^(0)=pi independent of beta, so
    # any starting beta is fine; using beta_0 (target MLE) is convenient.
    weights = pi.copy()
    beta = beta_0.copy()

    history = EMHistory()
    n_converged = 0   # patience counter for ||Δw||_inf convergence

    for t in range(cfg.max_iter):
        w_prev = weights.copy()
        weights, diag = compute_weights(beta, X_sources, y_sources, metas,
                                        pi, weights, ll_matrix, t, cfg, H_0)
        history.weights.append(weights.copy())
        history.diagnostics.append(diag)

        # M-step
        if cfg.use_m_step_approx:
            beta = _m_step_closed_form_approx(beta_0, N_0, metas, weights)
        else:
            beta = _m_step_closed_form(beta_0, H_0, metas, weights)

        # Convergence: ||Δw||_inf <= conv_tol for `patience` consecutive iters.
        w_change = float(np.max(np.abs(w_prev - weights)))
        if cfg.verbose and (t < 3 or (t + 1) % max(1, cfg.max_iter // 10) == 0):
            print(f"[em it {t}] w_change={w_change:.2e}  w[:3]={weights[:3]}")
        if w_change <= cfg.convergence_tol:
            n_converged += 1
        else:
            n_converged = 0
        if n_converged >= cfg.convergence_patience:
            history.converged_at = t
            if cfg.verbose:
                print(f"[em] converged at iter {t} (w_change={w_change:.2e})")
            break

    return beta, history


# ============================================================================
# Convenience: predict y at new design points
# ============================================================================

def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta
