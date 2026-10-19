"""Shared setup for the C-MAPSS case study (paper §5.2 and App. C-MAPSS).

Targets, cutoffs, hyperparameters and LIP files, plus the helpers every
driver uses to build one target's problem and fit each method.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip.lip import fit_lip, load_responses  # noqa: E402

from dataset import load_machine  # noqa: E402
from em import EMConfig, em_train  # noqa: E402


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SEQ_DIR = DATA_DIR / "Physical_Core_Speed_Sequences"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

# 10 target engines, sampled with np.random.default_rng(42).choice(candidates,
# size=10, replace=False) from the 12 fast-degrading candidates
# [4, 26, 27, 53, 80, 97, 9, 18, 48, 51, 55, 82].
TARGETS = [4, 9, 18, 26, 27, 48, 51, 53, 55, 80]
RUL_LEVELS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# LIP fit (paper eq:LIP_opt)
P_0 = 0.01            # uniform default prior p_0
EPS_LIP = 0.1         # L2 regularizer lambda_LIP

# Ridge baselines (intercept un-penalized) keep the truncated-power basis
# numerically stable at intermediate cutoffs.
RIDGE_LAMBDA = 1e4

# Cached LLM responses, one LIP per oracle (paper column order).
LIP_FILES = {
    "LIP-C": DATA_DIR / "llm_subgroup_queries_claude.json",
    "LIP-G": DATA_DIR / "llm_subgroup_queries_gemini.json",
}
METHODS = ["LIP-C", "LIP-G", "Uniform", "Pooled", "Target-Only"]

# EM: tau = 1e-3, nu = 0.05, ||Δw||_inf <= 1e-3 for 5 iterations counted from
# t_0 = ceil(ln 2 / nu) = 14, at most 1000 iterations (the EMConfig defaults).
EM_CONFIG = EMConfig()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge OLS, intercept (column 0) un-penalized."""
    P = np.eye(X.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * P, X.T @ y)


def cutoff_for(n_cycles: int, rul: float) -> int:
    """Number of observed target cycles at remaining-useful-life fraction `rul`."""
    return int(n_cycles * (1.0 - rul))


def source_files(target: int) -> list[Path]:
    """The other 99 engines, in the (lexicographic) order of the LIP files."""
    files = sorted(SEQ_DIR.glob("*.csv"), key=lambda f: f.name)
    return [f for f in files if f.name != f"{target}.csv"]


def load_sources(target: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X_sources, y_sources = [], []
    for f in source_files(target):
        _, X_s, y_s = load_machine(f)
        X_sources.append(X_s)
        y_sources.append(y_s)
    return X_sources, y_sources


def lip_prior(lip_file: Path, target: int) -> np.ndarray:
    """Fit the LIP for one target from the shared response set.

    Queries whose subgroup contains the target are dropped and the remaining
    indices are re-indexed to the 99 sources (`lip.load_responses`).
    """
    with open(lip_file) as f:
        basenames = json.load(f)["source_basenames"]
    assert basenames == sorted(p.name for p in SEQ_DIR.glob("*.csv")), \
        f"{lip_file.name}: source order does not match {SEQ_DIR.name}/"
    responses = load_responses(str(lip_file), target_basename=f"{target}.csv")
    return fit_lip(responses, K=len(basenames) - 1, p0=P_0, eps=EPS_LIP)


def priors_for(target: int) -> dict[str, np.ndarray]:
    """Prior inclusion probabilities pi for the three EM methods."""
    pis = {name: lip_prior(path, target) for name, path in LIP_FILES.items()}
    K = len(next(iter(pis.values())))
    pis["Uniform"] = np.full(K, P_0)
    return pis


def fit_all(X_t, y_t, X_sources, y_sources, pis, cfg: EMConfig = EM_CONFIG):
    """Fit every method on one (target, cutoff). Returns {method: (beta, history)};
    history is None for the ridge baselines."""
    fits = {}
    for name, pi in pis.items():
        fits[name] = em_train(X_t, y_t, X_sources, y_sources, pi=pi, cfg=cfg)
    X_pool = np.vstack([X_t] + list(X_sources))
    y_pool = np.concatenate([y_t] + list(y_sources))
    fits["Pooled"] = (ridge_fit(X_pool, y_pool, RIDGE_LAMBDA), None)
    fits["Target-Only"] = (ridge_fit(X_t, y_t, RIDGE_LAMBDA), None)
    return fits
