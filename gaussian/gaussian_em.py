"""
LIP-EM for Gaussian estimation: paper Algorithm 1 (closed-form, Hessian-reuse version).

Both the mean and the (co)variance are estimated. The parameter vector is
    1-d:  theta = (mu, sigma^2)
    2-d:  theta = (mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy)
and theta_hat_k are maximum-likelihood estimates (the covariance divides by n).

Setup (once):
  * theta_hat_0, H_0 on the target data; theta_hat_k, H_k on each source (H_k frozen and reused).
  * Tempering scale eps_k^2 = Tr(H_0^+ H_k), with the pseudo-inverse H_0^+ (here H_0 has full rank).
  * Empirical-Bayes null components log L(theta_hat_j; D_k) for j != k.

Iteration t = 0, 1, ...:
  * E-step (paper eq:tempering):
        w_k = sigmoid(beta_k(t) * log[p(D_k | c_k=1, theta) / p(D_k | c_k=0)] + logit(pi_k)),
        beta_k(t) = (1 - exp(-nu t)) / eps_k,
    relevant likelihood (eq:rel_prox, H_k frozen):
        log L(theta; D_k) + tau^2/2 g^T (I + tau^2 H_k)^-1 g - 1/2 log|I + tau^2 H_k|,  g = score at theta,
    null likelihood (eq:eb-null): mixture of L(theta_hat_j; D_k) over j != k with weights
        (1 - w_j) / sum_{i != k} (1 - w_i), using the previous iteration's w.
  * M-step (eq:m_step_update), with M_k = (I + tau^2 H_k)^-1 H_k:
        theta = (H_0 + sum_k w_k M_k)^-1 (H_0 theta_hat_0 + sum_k w_k M_k theta_hat_k).

Stopping rule: ||w(t) - w(t-1)||_inf <= tol for `patience` consecutive iterations, counted only
from t >= t0 = max(1, ceil(ln 2 / nu)) (t0 = 14 for nu = 0.05), or at most `max_iter` iterations.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, logit, logsumexp
from scipy.stats import multivariate_normal

VAR_FLOOR = 1e-6       # lower bound on a variance (keeps theta a valid Gaussian)
COV_JITTER = 1e-6      # added to the diagonal of the 2-d MLE covariance
NULL_FLOOR = 1e-9      # floor on (1 - w_j) in the EB null, so its log stays finite


class GaussianEM:
    """Algorithm 1 for a Gaussian family. Subclasses define the parameterisation:
    `mle`, `loglik`, `score`, `information` (= -Hessian of log L), `project` and `unpack`."""

    n_params: int

    def __init__(self, pi, tau: float = 1e-3, nu: float = 0.05,
                 tol: float = 1e-3, patience: int = 5, max_iter: int = 1000):
        self.pi = np.asarray(pi, dtype=float)
        self.tau, self.nu = tau, nu
        self.tol, self.patience, self.max_iter = tol, patience, max_iter
        self.t0 = max(1, math.ceil(math.log(2) / nu))
        self.weights = self.pi.copy()
        self.n_iter = 0
        self.converged = False

    def fit(self, target: np.ndarray, sources: list[np.ndarray]):
        """Run EM; returns (mean, covariance) of the target estimate. Final weights in `self.weights`."""
        K, eye, tau2 = len(sources), np.eye(self.n_params), self.tau ** 2

        theta_0 = self.mle(target)
        H_0 = self.information(target, theta_0)
        theta_k = [self.mle(x) for x in sources]
        H_k = [self.information(x, th) for x, th in zip(sources, theta_k)]
        A_inv = [np.linalg.pinv(eye + tau2 * H) for H in H_k]                  # (I + tau^2 H_k)^-1
        log_det = [np.linalg.slogdet(eye + tau2 * H)[1] for H in H_k]          # log|I + tau^2 H_k|
        M = [A @ H for A, H in zip(A_inv, H_k)]

        # Tempering scale (Algorithm 1)
        H_0_pinv = np.linalg.pinv(H_0)
        eps = np.array([math.sqrt(float(np.trace(H_0_pinv @ H))) for H in H_k])

        # EB null components: ll[k, j] = log L(theta_hat_j; D_k), j != k
        ll = np.full((K, K), -np.inf)
        for k in range(K):
            for j in range(K):
                if j != k:
                    ll[k, j] = self.loglik(sources[k], theta_k[j])

        def m_step(w):
            L, r = H_0.copy(), H_0 @ theta_0
            for k in range(K):
                L += w[k] * M[k]
                r += w[k] * (M[k] @ theta_k[k])
            return self.project(np.linalg.pinv(L) @ r)

        logit_pi = logit(self.pi)
        w, theta = self.pi.copy(), theta_0.copy()     # beta(0) = 0, so w(0) = pi whatever theta is
        n_small = 0
        for t in range(self.max_iter):
            beta = (1.0 - math.exp(-self.nu * t)) / eps
            keep = np.clip(1.0 - w, NULL_FLOOR, 1.0)
            w_new = np.empty(K)
            for k in range(K):
                x = sources[k]
                g = self.score(x, theta)
                log_rel = self.loglik(x, theta) + 0.5 * tau2 * (g @ A_inv[k] @ g) - 0.5 * log_det[k]
                others = [j for j in range(K) if j != k]
                log_null = logsumexp(ll[k, others] + np.log(keep[others] / keep[others].sum()))
                w_new[k] = expit(beta[k] * (log_rel - log_null) + logit_pi[k])
            dw = float(np.max(np.abs(w_new - w)))
            w, theta = w_new, m_step(w_new)
            n_small = n_small + 1 if (t >= self.t0 and dw <= self.tol) else 0
            if n_small >= self.patience:
                break
        self.weights, self.theta = w, theta
        self.n_iter, self.converged = t + 1, n_small >= self.patience
        return self.unpack(theta)


class Gaussian1DEM(GaussianEM):
    """theta = (mu, sigma^2); H is the observed information."""

    n_params = 2

    def mle(self, x):
        return np.array([np.mean(x), max(np.var(x), VAR_FLOOR)])

    def loglik(self, x, theta):
        mu, s2 = theta
        return -0.5 * len(x) * np.log(2 * np.pi * s2) - np.sum((x - mu) ** 2) / (2 * s2)

    def score(self, x, theta):
        mu, s2 = theta
        n, diff = len(x), x - mu
        return np.array([np.sum(diff) / s2, -n / (2 * s2) + np.sum(diff ** 2) / (2 * s2 ** 2)])

    def information(self, x, theta):
        mu, s2 = theta
        n, diff = len(x), x - mu
        hess = np.array([[-n / s2, -np.sum(diff) / s2 ** 2],
                         [-np.sum(diff) / s2 ** 2, n / (2 * s2 ** 2) - np.sum(diff ** 2) / s2 ** 3]])
        return -hess

    def project(self, theta):
        theta[1] = max(theta[1], VAR_FLOOR)
        return theta

    def unpack(self, theta):
        return np.array([theta[0]]), np.array([[theta[1]]])


class Gaussian2DEM(GaussianEM):
    """theta = (mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy); H is the analytic Fisher information
    (equal to the observed information at the MLE)."""

    n_params = 5
    _dS = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]

    @staticmethod
    def pack(mu, cov):
        return np.array([mu[0], mu[1], cov[0, 0], cov[1, 1], cov[0, 1]])

    def unpack(self, theta):
        return theta[0:2], np.array([[theta[2], theta[4]], [theta[4], theta[3]]])

    def mle(self, x):
        return self.pack(np.mean(x, axis=0), np.cov(x, rowvar=False, ddof=0) + np.eye(2) * COV_JITTER)

    def loglik(self, x, theta):
        mu, cov = self.unpack(theta)
        return np.sum(multivariate_normal.logpdf(x, mean=mu, cov=cov))

    def score(self, x, theta):
        mu, cov = self.unpack(theta)
        ic, n, diff = np.linalg.pinv(cov), len(x), x - mu
        g_mu = ic @ np.sum(diff, axis=0)
        g_S = -0.5 * n * ic + 0.5 * ic @ (diff.T @ diff) @ ic
        return np.array([g_mu[0], g_mu[1], g_S[0, 0], g_S[1, 1], 2 * g_S[0, 1]])

    def information(self, x, theta):
        _, cov = self.unpack(theta)
        ic, n = np.linalg.pinv(cov), len(x)
        F = np.zeros((5, 5))
        F[0:2, 0:2] = ic * n
        for i in range(3):
            for j in range(3):
                F[2 + i, 2 + j] = 0.5 * n * np.trace(ic @ self._dS[i] @ ic @ self._dS[j])
        return F

    def project(self, theta):
        """Keep the covariance positive definite: variances >= 1e-4, |corr| <= 0.99."""
        theta[2], theta[3] = max(theta[2], 1e-4), max(theta[3], 1e-4)
        m = np.sqrt(theta[2] * theta[3]) * 0.99
        theta[4] = np.clip(theta[4], -m, m)
        return theta
