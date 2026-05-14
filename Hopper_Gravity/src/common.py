"""
Shared components: Hopper environment, neural networks, dynamics model,
dataset loading.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn

# Hopper state and action dimensions
S, A = 11, 3


# ============================================================================
# Environment
# ============================================================================

def make_hopper(g: float) -> gym.Env:
    """Create Hopper-v5 with custom gravity (m/s^2)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        env = gym.make("Hopper-v5", render_mode=None)
    env.unwrapped.model.opt.gravity[2] = -abs(g)
    return env


# ============================================================================
# Policy networks (used by collect_target_sac and weighted_iql)
# ============================================================================

class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy for IQL."""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, state_dim: int = S, action_dim: int = A,
                 hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, s: torch.Tensor):
        h       = self.trunk(s)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, s: torch.Tensor):
        """Reparameterized sample with tanh squashing. Returns (action, log_prob)."""
        mean, log_std = self(s)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        x   = mean + std * eps
        a   = torch.tanh(x)
        logp = (-0.5 * eps.pow(2) - log_std - 0.9189).sum(-1) \
               - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

    def log_prob(self, s: torch.Tensor, a: torch.Tensor):
        mean, log_std = self(s)
        std = log_std.exp()
        x = a.clamp(-0.999, 0.999).atanh()
        logp = (-0.5 * ((x - mean) / std).pow(2) - log_std - 0.9189).sum(-1) \
               - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return logp


class WidePolicy(GaussianPolicy):
    """SAC data-collection policy with a higher log_std floor for exploration."""
    LOG_STD_MIN = -2.0


class QNetwork(nn.Module):
    """State-action value Q(s, a). Used by SAC during target data collection."""
    def __init__(self, state_dim: int = S, action_dim: int = A,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1))


# ============================================================================
# Dynamics model
# ============================================================================

class DynamicsModel(nn.Module):
    """Probabilistic dynamics model: s_{t+1} = s_t + delta, delta ~ N(mu, sigma).

    A probabilistic MLP with input/output normalization buffers, LayerNorm + SiLU
    trunk, and mean/log-variance heads.
    """
    def __init__(self, state_dim: int = S, action_dim: int = A,
                 hidden: int = 512, n_layers: int = 4,
                 max_log_var: float = 0.5, min_log_var: float = -10.0):
        super().__init__()
        self.S, self.A = state_dim, action_dim
        self.max_lv, self.min_lv = max_log_var, min_log_var

        layers = []
        for i in range(n_layers):
            in_dim = state_dim + action_dim if i == 0 else hidden
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
        self.trunk     = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden, state_dim)
        self.lv_head   = nn.Linear(hidden, state_dim)

        self.register_buffer("in_mean",  torch.zeros(state_dim + action_dim))
        self.register_buffer("in_std",   torch.ones(state_dim + action_dim))
        self.register_buffer("out_mean", torch.zeros(state_dim))
        self.register_buffer("out_std",  torch.ones(state_dim))

    def set_normalization(self, states, actions, deltas):
        inp = torch.cat([states, actions], -1)
        self.in_mean.copy_(inp.mean(0))
        self.in_std.copy_(inp.std(0).clamp(min=1e-6))
        self.out_mean.copy_(deltas.mean(0))
        self.out_std.copy_(deltas.std(0).clamp(min=1e-6))

    def _normalize_input(self, s, a):
        return (torch.cat([s, a], -1) - self.in_mean) / self.in_std

    def forward(self, s, a):
        x_norm = self._normalize_input(s, a)
        h = self.trunk(x_norm)
        mean_norm = self.mean_head(h)
        lv = self.max_lv - nn.functional.softplus(self.max_lv - self.lv_head(h))
        lv = self.min_lv + nn.functional.softplus(lv - self.min_lv)
        mean = mean_norm * self.out_std + self.out_mean
        return mean, lv

    def forward_normalized(self, x_norm):
        h = self.trunk(x_norm)
        mean_norm = self.mean_head(h)
        lv = self.max_lv - nn.functional.softplus(self.max_lv - self.lv_head(h))
        lv = self.min_lv + nn.functional.softplus(lv - self.min_lv)
        return mean_norm, lv

    @staticmethod
    def gaussian_nll(mean_norm, lv, target_norm):
        """Training-loss form (drops constants; not a proper log-density)."""
        return (lv + (target_norm - mean_norm).pow(2) * (-lv).exp()).mean()


# ============================================================================
# Dataset loading
# ============================================================================

def load_dataset(path: str, device: str) -> dict:
    """Load offline dataset from HDF5.

    Returns dict {states, actions, deltas} as torch tensors on `device`.
    Terminal transitions are filtered out (dynamics undefined at episode boundaries).
    """
    with h5py.File(path, "r") as f:
        s  = np.array(f["observations"])
        a  = np.array(f["actions"])
        ns = np.array(f["next_observations"])
        t  = np.array(f["terminals"], dtype=bool)

    mask = ~t
    s, a, ns = s[mask], a[mask], ns[mask]
    d = ns - s

    dev = torch.device(device)
    to = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    print(f"[dataset] {len(s):,} transitions  ({path})")
    return dict(states=to(s), actions=to(a), deltas=to(d))
