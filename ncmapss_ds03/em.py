"""
LIP-EM for the N-CMAPSS DS03 joint-sensor prognostics model.

Backbone: a multiple-output Gaussian regression Y = X B + E that shares one
design X (the raw-cycle natural-spline basis) across the q = 14 sensor
outputs, with a single relevance weight w_k per source engine. In vectorised
form vec(Y) ~ N((I_q kron X) vec(B), sigma^2 I), so every Hessian is
I_q kron (X^T X / sigma^2) and only the d x d block is ever formed.

Key choices (paper Algorithm 1, exact Gaussian E- and M-steps):

* Estimates: maximum-likelihood OLS fits; sigma^2 = RSS / (n q).
* E-step relevant likelihood (paper eq:rel_prox): log L(theta; D_k)
  + (tau^2/2) g^T (I + tau^2 H_k)^-1 g - (q/2) log|I + tau^2 H_k|.
* Null likelihood: empirical Bayes mixture of the other sources' MLE fits
  with normalised weights (1 - w_j) / sum_{i != k} (1 - w_i) (eq:eb-null).
* Tempering: beta_k(t) = (1 - exp(-nu t)) / eps_k with
  eps_k^2 = q Tr(H_0^+ H_k), with the pseudo-inverse H_0^+ (the target Hessian
  is singular when the observed prefix does not reach every spline knot).
* E-step: w_k = sigmoid(beta_k(t) logLR_k + logit pi_k).
* M-step (paper eq:m_step_update): precision-weighted average of the target
  MLE and the source MLEs.
* Stopping: ||w^(t) - w^(t-1)||_inf <= 1e-3 for 5 consecutive iterations,
  counted only for t >= t_0 = ceil(log(2) / nu), with a 1000-iteration cap.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


TAU = 1e-3
NU = 0.05
MAX_ITERATIONS = 1000
CONVERGENCE_TOLERANCE = 1e-3
CONVERGENCE_PATIENCE = 5
PI_CLIP = 1e-9                  # floor on pi_k when computing logit(pi_k)
NULL_MIN_INVERSE_WEIGHT = 1e-9  # floor on (1 - w_j) in the EB null


def stopping_start(nu: float) -> int:
    """First iteration t_0 = ceil(log(2) / nu) at which patience may count.

    For t < t_0 the tempering beta(t) is below half its final value, so
    small weight changes there do not indicate convergence.
    """
    return max(1, math.ceil(math.log(2.0) / nu))


# ============================================================================
# Numerically stable helpers
# ============================================================================

def logsumexp(values: np.ndarray) -> float:
    """log(sum(exp(values))) without scipy."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return -np.inf
    maximum = values.max()
    if not np.isfinite(maximum):
        return float(maximum)
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


# ============================================================================
# Multiple-output Gaussian helpers
# ============================================================================

def multi_ols(
    design: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the multiple-output OLS coefficient matrix and shared MLE sigma."""
    coefficients = np.linalg.lstsq(design, outcomes, rcond=None)[0]
    residuals = outcomes - design @ coefficients
    sigma = max(
        math.sqrt(float(np.sum(residuals**2)) / max(outcomes.size, 1)),
        1e-6,
    )
    return coefficients, sigma


def multi_log_gauss_pdf(
    design: np.ndarray,
    outcomes: np.ndarray,
    coefficients: np.ndarray,
    sigma: float,
) -> float:
    """Log N(vec(Y) | (I_q kron X) vec(B), sigma^2 I)."""
    safe = max(float(sigma), 1e-6)
    residuals = outcomes - design @ coefficients
    rss = float(np.sum(residuals**2))
    observations = outcomes.size
    return (
        -0.5 * observations * math.log(2.0 * math.pi * safe**2)
        - rss / (2.0 * safe**2)
    )


def source_metadata(
    source_designs: Sequence[np.ndarray],
    source_outcomes: Sequence[np.ndarray],
    tau: float,
) -> list[dict[str, Any]]:
    """Per-source MLE, Hessian block H_k, (I + tau^2 H_k)^-1 and log-det."""
    metadata = []
    for design, outcomes in zip(source_designs, source_outcomes):
        coefficients, sigma = multi_ols(design, outcomes)
        dimension = design.shape[1]
        hessian = design.T @ design / sigma**2
        inverse_term = np.linalg.pinv(
            np.eye(dimension) + tau**2 * hessian
        )
        sign, block_log_det = np.linalg.slogdet(
            np.eye(dimension) + tau**2 * hessian
        )
        if sign <= 0:
            block_log_det = -1e6  # numerical fallback
        metadata.append(
            {
                "coefficients": coefficients,
                "sigma": sigma,
                "hessian": hessian,
                "inverse_term": inverse_term,
                "precision": inverse_term @ hessian,
                "block_log_det": float(block_log_det),
            }
        )
    return metadata


def null_likelihood_matrix(
    source_designs: Sequence[np.ndarray],
    source_outcomes: Sequence[np.ndarray],
    metadata: list[dict[str, Any]],
) -> np.ndarray:
    """Entry [k, j] = log L(theta_j_hat; D_k); the diagonal is never used."""
    source_count = len(metadata)
    likelihoods = np.full((source_count, source_count), -np.inf)
    for left in range(source_count):
        for right in range(source_count):
            if left != right:
                likelihoods[left, right] = multi_log_gauss_pdf(
                    source_designs[left],
                    source_outcomes[left],
                    metadata[right]["coefficients"],
                    metadata[right]["sigma"],
                )
    return likelihoods


def m_step(
    target_coefficients: np.ndarray,
    target_hessian: np.ndarray,
    metadata: list[dict[str, Any]],
    weights: np.ndarray,
) -> np.ndarray:
    """Paper eq:m_step_update for one shared Hessian block."""
    left = target_hessian.copy()
    right = target_hessian @ target_coefficients
    for weight, source in zip(weights, metadata):
        if weight <= 0.0:
            continue
        left += weight * source["precision"]
        right += weight * source["precision"] @ source["coefficients"]
    return np.linalg.pinv(left) @ right


def tempering_ratios(
    target_hessian: np.ndarray,
    metadata: list[dict[str, Any]],
    output_count: int,
) -> np.ndarray:
    """1 / eps_k with eps_k^2 = q Tr(H_0^+ H_k), H_0^+ the pseudo-inverse.

    The factor q is the trace of the I_q Kronecker block.
    """
    target_hessian_inverse = np.linalg.pinv(target_hessian)
    return np.asarray(
        [
            1.0
            / math.sqrt(
                max(
                    output_count
                    * float(
                        np.trace(
                            target_hessian_inverse @ source["hessian"]
                        )
                    ),
                    1e-12,
                )
            )
            for source in metadata
        ]
    )


# ============================================================================
# EM
# ============================================================================

def em_train(
    target_design: np.ndarray,
    target_outcomes: np.ndarray,
    source_designs: Sequence[np.ndarray],
    source_outcomes: Sequence[np.ndarray],
    prior: np.ndarray,
    tau: float = TAU,
    nu: float = NU,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run LIP-EM.

    Returns (coefficients, final weights, number of EM iterations run).
    """
    output_count = target_outcomes.shape[1]
    target_coefficients, target_sigma = multi_ols(
        target_design, target_outcomes
    )
    target_hessian = target_design.T @ target_design / target_sigma**2
    metadata = source_metadata(source_designs, source_outcomes, tau)
    null_matrix = null_likelihood_matrix(
        source_designs, source_outcomes, metadata
    )
    ratios = tempering_ratios(target_hessian, metadata, output_count)
    source_count = len(metadata)
    prior = np.asarray(prior, dtype=float)
    prior_logit = np.log(prior + PI_CLIP) - np.log(
        np.maximum(1.0 - prior, PI_CLIP)
    )
    first_counted = stopping_start(nu)

    weights = prior.copy()
    coefficients = target_coefficients.copy()
    stable_iterations = 0
    iterations = 0
    for iteration in range(MAX_ITERATIONS):
        iterations = iteration + 1
        previous = weights.copy()

        # E-step: relevant likelihood with the exact tau^2 correction.
        relevant = np.zeros(source_count)
        for index, (design, outcomes, source) in enumerate(
            zip(source_designs, source_outcomes, metadata)
        ):
            relevant[index] = multi_log_gauss_pdf(
                design, outcomes, coefficients, source["sigma"]
            )
            residuals = outcomes - design @ coefficients
            gradient = design.T @ residuals / source["sigma"] ** 2
            quadratic = float(
                np.sum(gradient * (source["inverse_term"] @ gradient))
            )
            relevant[index] += (
                0.5 * tau**2 * quadratic
                - 0.5 * output_count * source["block_log_det"]
            )

        # Empirical-Bayes null over the other sources, normalised weights.
        inverse_weights = np.clip(
            1.0 - previous, NULL_MIN_INVERSE_WEIGHT, 1.0
        )
        null = np.zeros(source_count)
        for index in range(source_count):
            others = np.arange(source_count) != index
            log_mixing = np.log(
                inverse_weights[others] / inverse_weights[others].sum()
            )
            null[index] = logsumexp(null_matrix[index, others] + log_mixing)

        tempering = ratios * (1.0 - np.exp(-nu * iteration))
        logits = tempering * (relevant - null) + prior_logit
        weights = np.asarray([stable_sigmoid(float(v)) for v in logits])

        coefficients = m_step(
            target_coefficients, target_hessian, metadata, weights
        )

        change = float(np.max(np.abs(previous - weights)))
        if iteration >= first_counted and change <= CONVERGENCE_TOLERANCE:
            stable_iterations += 1
        else:
            stable_iterations = 0
        if stable_iterations >= CONVERGENCE_PATIENCE:
            break
    return coefficients, weights, iterations
