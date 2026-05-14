"""
EM algorithm with NN dynamics for the LIP-aided negative-transfer correction.

For NN dynamics, the closed-form weighted-average M-step over per-source MLE
parameters is meaningless (parameter convex combinations are nonsensical
across NN re-parameterizations). Instead we use the gradient-based M-step
from the paper appendix: at iterate theta^t, approximate the Hessian by
N * I and step Adam on the weighted negative log-likelihood

    L(theta) = -log p(D_0 | theta) - sum_k w_k * log p(D_k | theta).

This is generalized EM (each M-step monotonically increases Q without fully
maximizing it).

Null model: p(D_k | c_k=0) = log p(D_k | theta_pool) — the pooled-dynamics
baseline. diff_k = log p(D_k | theta^t) - log p(D_k | theta_pool) measures
whether theta^t fits D_k better than the unconditional pool.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import S as STATE_DIM, A as ACTION_DIM, DynamicsModel


def stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ============================================================================
# Per-sample / total log-likelihood under a DynamicsModel
# ============================================================================

def log_lik_per_sample(model: DynamicsModel,
                       s: torch.Tensor, a: torch.Tensor,
                       delta: torch.Tensor) -> torch.Tensor:
    """log p(delta | s, a, model) in raw (un-normalized) delta space.

    Includes the Jacobian -sum log(out_std) so the result is a density over
    actual delta values; this matters because different models have different
    normalization buffers.
    """
    in_norm = (torch.cat([s, a], -1) - model.in_mean) / model.in_std
    delta_norm = (delta - model.out_mean) / model.out_std
    mean_norm, lv = model.forward_normalized(in_norm)

    quad = (delta_norm - mean_norm).pow(2) * (-lv).exp()
    log_p_norm = -0.5 * (lv + quad).sum(-1) \
                 - 0.5 * model.S * math.log(2.0 * math.pi)
    log_jac = model.out_std.log().sum()
    return log_p_norm - log_jac


@torch.no_grad()
def total_log_lik(model: DynamicsModel, data: dict,
                  batch_size: int = 8192) -> float:
    """Sum_i log p(delta_i | s_i, a_i, model) over the dataset."""
    model.eval()
    s_all, a_all, d_all = data["states"], data["actions"], data["deltas"]
    N = s_all.shape[0]
    total = 0.0
    for i in range(0, N, batch_size):
        total += log_lik_per_sample(
            model, s_all[i:i+batch_size], a_all[i:i+batch_size],
            d_all[i:i+batch_size]
        ).sum().item()
    return total


def mean_nll_batch(model: DynamicsModel,
                   s: torch.Tensor, a: torch.Tensor,
                   delta: torch.Tensor) -> torch.Tensor:
    """Per-sample training NLL on one minibatch (drops constants)."""
    in_norm = (torch.cat([s, a], -1) - model.in_mean) / model.in_std
    delta_norm = (delta - model.out_mean) / model.out_std
    mean_norm, lv = model.forward_normalized(in_norm)
    return (lv + (delta_norm - mean_norm).pow(2) * (-lv).exp()).sum(-1).mean()


# ============================================================================
# EM config
# ============================================================================

@dataclass
class EMConfig:
    # ---- Outer EM loop ----
    n_em_iter: int = 100
    # Stop when ||w_t - w_{t-1}||_inf <= thresh for `weight_convergence_patience`
    # consecutive iterations.
    weight_convergence_thresh: float = 1e-3
    weight_convergence_patience: int = 5

    # ---- M-step ----
    m_steps_per_iter: int = 100
    batch_size: int = 1024
    # Constant LR matching the pool's terminal LR. With the pool already trained
    # to convergence, peaking back to 3e-4 every M-step yanks theta off the basin
    # and corrupts the diff = log L(D_k|theta^t) - log L(D_k|theta_pool) signal.
    lr: float = 1e-5
    weight_decay: float = 1e-4

    # ---- E-step (paper's tempering) ----
    # beta = (1 - exp(-nu * t)) / sqrt(d_eff * N_k / N_0)
    nu: float = 0.1
    # d_eff = NN parameter count (set automatically in em_train from pool model).
    d_eff: int = 0

    # ---- Pool null ----
    # Required: path to a pretrained pool dynamics model whose log-lik on each
    # source dataset is the null baseline.
    pool_model_path: str = ""

    # ---- Misc ----
    eval_batch_size: int = 8192
    verbose: bool = True


@dataclass
class EMHistory:
    weights: list[np.ndarray] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    test_nll: list[float] = field(default_factory=list)


# ============================================================================
# E-step
# ============================================================================

def compute_weights(theta: DynamicsModel,
                    source_data: list[dict],
                    pi: np.ndarray,
                    log_L_pool: np.ndarray,
                    iter_idx: int,
                    cfg: EMConfig,
                    N_0: int,
                    N_k: list[int]) -> tuple[np.ndarray, dict]:
    """One E-step. Returns (new weights, diagnostics)."""
    K = len(source_data)

    # log p(D_k | theta^t)
    log_L_rel = np.array([
        total_log_lik(theta, source_data[k], cfg.eval_batch_size)
        for k in range(K)
    ])

    new_w = np.zeros(K)
    diag = {"log_L_rel": log_L_rel, "log_L_null": log_L_pool.copy(),
            "diff": np.zeros(K), "beta": np.zeros(K),
            "prior_logit": np.zeros(K), "logit": np.zeros(K)}

    for k in range(K):
        eps_sq = cfg.d_eff * N_k[k] / max(N_0, 1)
        ratio = 1.0 / math.sqrt(max(eps_sq, 1e-12))
        warmup = 1.0 - math.exp(-cfg.nu * iter_idx)
        beta = ratio * warmup

        diff = log_L_rel[k] - log_L_pool[k]
        prior_logit = (math.log(pi[k] + 1e-12)
                       - math.log(max(1.0 - pi[k], 1e-12)))
        logit = beta * diff + prior_logit
        new_w[k] = stable_sigmoid(logit)

        diag["diff"][k] = diff
        diag["beta"][k] = beta
        diag["prior_logit"][k] = prior_logit
        diag["logit"][k] = logit

    return new_w, diag


# ============================================================================
# M-step
# ============================================================================

def m_step(theta: DynamicsModel,
           target_data: dict,
           source_data: list[dict],
           weights: np.ndarray,
           optimizer: torch.optim.Optimizer,
           cfg: EMConfig,
           N_0: int, N_k: list[int]) -> float:
    """One M-step: gradient descent on the weighted negative log-likelihood.

        loss = (N_0 * mean_nll_target + sum_k w_k * N_k * mean_nll_source_k)
               / (N_0 + sum_k w_k * N_k).
    """
    K = len(source_data)
    total_w = N_0 + sum(weights[k] * N_k[k] for k in range(K))
    p_target = N_0 / total_w
    p_source = [weights[k] * N_k[k] / total_w for k in range(K)]

    theta.train()

    # Constant LR (no per-M-step reset / no decay). cfg.lr matches the pool's
    # terminal training LR (1e-5), so theta evolves slowly within the pool basin
    # — empirically this gives a much cleaner diff signal than re-peaking each
    # M-step (which yanks theta off the basin and degrades log L(D_k | theta^t)
    # for every k, including the right one).

    # Without-replacement (per-dataset permutation) sampling.
    dev = target_data["states"].device

    def _new_perm(N): return torch.randperm(N, device=dev)

    target_perm, target_pos = _new_perm(N_0), 0
    source_perms = [_new_perm(N_k[k]) if p_source[k] > 0 else None
                    for k in range(K)]
    source_poss = [0] * K

    def _next_idx(perm, pos, N):
        if pos + cfg.batch_size > N:
            perm = _new_perm(N); pos = 0
        idx = perm[pos:pos + cfg.batch_size]
        return perm, pos + cfg.batch_size, idx

    losses = []
    for _ in range(cfg.m_steps_per_iter):
        target_perm, target_pos, idx0 = _next_idx(target_perm, target_pos, N_0)
        loss = p_target * mean_nll_batch(
            theta, target_data["states"][idx0],
            target_data["actions"][idx0], target_data["deltas"][idx0])

        for k in range(K):
            if p_source[k] <= 0:
                continue
            source_perms[k], source_poss[k], idxk = _next_idx(
                source_perms[k], source_poss[k], N_k[k])
            loss = loss + p_source[k] * mean_nll_batch(
                theta, source_data[k]["states"][idxk],
                source_data[k]["actions"][idxk], source_data[k]["deltas"][idxk])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(theta.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


# ============================================================================
# Full EM loop
# ============================================================================

def em_train(target_data: dict,
             source_data: list[dict],
             pi: np.ndarray,
             cfg: EMConfig,
             theta_init: DynamicsModel,
             target_test: dict | None = None
             ) -> tuple[DynamicsModel, EMHistory]:
    """LIP-aided EM with NN dynamics, pool null, and weight-convergence stopping.

    `theta_init` is the pretrained pool dynamics model (used both as the EM
    initialization AND to compute the null log-likelihoods log p(D_k | theta_pool)).
    """
    K = len(source_data)
    assert pi.shape == (K,)

    dev = target_data["states"].device
    N_0 = target_data["states"].shape[0]
    N_k = [d["states"].shape[0] for d in source_data]

    if cfg.verbose:
        print(f"[em] N_0={N_0:,}  N_k={N_k}  K={K}  pi={pi}")

    theta = copy.deepcopy(theta_init).to(dev)
    optimizer = optim.AdamW(theta.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    # Precompute pool null: log p(D_k | theta_pool) for each source.
    pool_model = copy.deepcopy(theta_init).to(dev).eval()
    log_L_pool = np.array([
        total_log_lik(pool_model, source_data[k], cfg.eval_batch_size)
        for k in range(K)
    ])
    if cfg.verbose:
        log_L_pool_ps = log_L_pool / np.array(N_k)
        print(f"[em] pool null log_L per-sample: "
              f"min={log_L_pool_ps.min():.2f}  "
              f"median={np.median(log_L_pool_ps):.2f}  "
              f"max={log_L_pool_ps.max():.2f}")

    # ----- EM iterations -----
    weights = pi.copy()
    history = EMHistory()
    prev_weights = None
    n_converged_iters = 0

    for it in range(cfg.n_em_iter):
        # E-step
        weights, diag = compute_weights(
            theta, source_data, pi, log_L_pool, it, cfg, N_0, N_k)
        history.weights.append(weights.copy())

        weight_delta = (float(np.max(np.abs(weights - prev_weights)))
                        if prev_weights is not None else float("inf"))
        prev_weights = weights.copy()

        if cfg.verbose:
            with np.printoptions(formatter={"float_kind": "{:.4f}".format}):
                print(f"[em it {it}] w = {weights}  "
                      f"||dw||_inf={weight_delta:.2e}")

        # M-step
        avg_loss = m_step(theta, target_data, source_data, weights,
                          optimizer, cfg, N_0, N_k)
        history.train_loss.append(avg_loss)

        # Optional held-out target test NLL (for diagnostics only).
        if target_test is not None:
            n_test = target_test["states"].shape[0]
            test_nll = -total_log_lik(theta, target_test, cfg.eval_batch_size) / n_test
            history.test_nll.append(test_nll)
            if cfg.verbose:
                print(f"[em it {it}] train_loss={avg_loss:.4f}  "
                      f"test_nll={test_nll:.4f}")
        elif cfg.verbose:
            print(f"[em it {it}] train_loss={avg_loss:.4f}")

        # Weight-convergence stopping
        if weight_delta <= cfg.weight_convergence_thresh:
            n_converged_iters += 1
        else:
            n_converged_iters = 0
        if n_converged_iters >= cfg.weight_convergence_patience:
            if cfg.verbose:
                print(f"[em] converged at iter {it}: "
                      f"||dw||_inf <= {cfg.weight_convergence_thresh:.0e} "
                      f"for {cfg.weight_convergence_patience} consecutive iters.")
            break

    return theta, history
