"""
1-d and 2-d Gaussian estimation (paper Sec. 5.1, Fig. gaus; hyperparameters in App. "Gaussian Estimation").

Design (data seed 42; np.random.seed, then the target draw, then the sources in order):
  1-d: target N(2, 5^2); sources N(2, 5^2) (relevant), N(0, 5^2), N(-5, 5^2).
       Target description: "The target domain has a larger mean."
  2-d: target N((55, 120), S0), S0 = [[20, 5], [5, 20]]; sources N((55, 120), S0) (relevant),
       N((51, 120), S0), N((49, 120), S0).
       Target description: "The target domain has a larger x mean."
  N_0 = 4 target points, N_k = 200 points per source.
LIP: the subgroup judge of lip/lip.py (gemini-3-flash-preview, temperature 0, pick one source or none) on all
  3 source pairs, each attachment preceded by a text label, then fit_lip with p0 = 0.01, eps = 0.01.
  Uniform EM uses pi_k = p0 for every source.
EM: Algorithm 1 (gaussian_em.py) with tau = 1e-3, nu = 0.05, at most 1000 iterations.
Baselines: Target Only (MLE on the 4 target points) and Pooled (MLE on target and all sources).

Usage (from this folder):
  GEMINI_API_KEY=... python gaussian.py query --dim 1   # 3 subgroup queries -> data/lip_responses_1d.json
  python gaussian.py plot --dim 1                        # LIP fit, EM, baselines -> results/1dgaussian.{pdf,json}
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3 (conference PDF requirement)
import numpy as np
from scipy.stats import multivariate_normal, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lip.lip import LLMSubgroupJudge, fit_lip, load_responses, save_responses  # noqa: E402
from gaussian_em import Gaussian1DEM, Gaussian2DEM  # noqa: E402

SEED, N0, NK = 42, 4, 200
SIGMA_1D, MEANS_1D = 5.0, [2.0, 0.0, -5.0]                 # target = source 0
COV_2D, MEANS_2D = [[20, 5], [5, 20]], [[55, 120], [51, 120], [49, 120]]
K = 3

MODEL, P0, EPS_LIP = "gemini-3-flash-preview", 0.01, 0.01
TAU, NU, MAX_ITER = 1e-3, 0.05, 1000
EM_CLASS = {1: Gaussian1DEM, 2: Gaussian2DEM}

REPORT = HERE / "data" / "technical_report.txt"           # context file uploaded to the judge
DESC = {1: "The target domain has a larger mean.", 2: "The target domain has a larger x mean."}


def responses_path(dim: int) -> Path:
    return HERE / "data" / f"lip_responses_{dim}d.json"


def make_data(dim: int):
    """Target sample (N0 points) and the three source samples (NK points each)."""
    np.random.seed(SEED)
    if dim == 1:
        target = np.random.normal(MEANS_1D[0], SIGMA_1D, N0)
        return target, [np.random.normal(m, SIGMA_1D, NK) for m in MEANS_1D]
    target = np.random.multivariate_normal(MEANS_2D[0], COV_2D, N0)
    return target, [np.random.multivariate_normal(m, COV_2D, NK) for m in MEANS_2D]


def truth(dim: int):
    if dim == 1:
        return np.array([MEANS_1D[0]]), np.array([[SIGMA_1D ** 2]])
    return np.array(MEANS_2D[0], float), np.array(COV_2D, float)


# ============================================================================
# LIP elicitation
# ============================================================================

PROMPT = {2: """
You are a statistical reasoning engineer. I have attached a brief Technical Report (the file).

[TASK]
1. Read the attached Report (defines the system / what affects the underlying distribution).
2. Read the 'Target Description' below.
3. Examine the {n_labels} attached Source Dataset files, labeled {labels_str}. Each file contains 2D points formatted as `x, y` (one row per sample).
4. Pick exactly ONE of:
   - the single label whose source dataset BEST matches the distribution implied by the Target Description; or
   - an empty list, if none of the sources is a credible match.
5. NEVER return more than one label.

[TARGET DESCRIPTION]
"{target_env_desc}"

Respond strictly in JSON with two fields:
  - "selected": one label from {labels_str} (as a single-element list), or an empty list.
  - "reasoning": 1-3 sentences citing summary statistics (per-dimension means, spread, correlation) that drove your decision.

Example formats:
{{
    "selected": ["A"],
    "reasoning": "Source A's x-coordinate mean is clearly higher, matching the target description."
}}
or
{{
    "selected": [],
    "reasoning": "None of the sources show a clearly higher x-coordinate mean."
}}
""", 1: """
You are a statistical reasoning engineer. I have attached a brief Technical Report (the file).

[TASK]
1. Read the attached Report (defines the system / what affects the underlying distribution).
2. Read the 'Target Description' below.
3. Examine the {n_labels} attached Source Dataset files, labeled {labels_str}. Each file contains 1D values (one per row).
4. Pick exactly ONE of:
   - the single label whose source dataset BEST matches the distribution implied by the Target Description; or
   - an empty list, if none of the sources is a credible match.
5. NEVER return more than one label.

[TARGET DESCRIPTION]
"{target_env_desc}"

Respond strictly in JSON with two fields:
  - "selected": one label from {labels_str} (as a single-element list), or an empty list.
  - "reasoning": 1-3 sentences citing summary statistics (mean, spread) that drove your decision.

Example formats:
{{
    "selected": ["A"],
    "reasoning": "Source A's mean is clearly higher, matching the target description."
}}
or
{{
    "selected": [],
    "reasoning": "None of the sources show a clearly higher mean."
}}
"""}


def query(dim: int, path: Path) -> None:
    """Ask the judge about every source pair and save the (S, C, reasoning) records. A failed query aborts."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY in the environment.")
    _, sources = make_data(dim)
    judge = LLMSubgroupJudge(api_key=key, manual_path=str(REPORT), model_name=MODEL,
                             prompt_template=PROMPT[dim], label_attachments=True)
    files = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, x in enumerate(sources):
            p = Path(tmp) / f"source_{i}.txt"
            rows = (f"{v:.4f}\n" for v in x) if dim == 1 else (f"{v[0]:.4f}, {v[1]:.4f}\n" for v in x)
            p.write_text("".join(rows))
            files.append(judge.client.files.upload(file=str(p), config={"mime_type": "text/plain"}))
    responses = []
    for S in itertools.combinations(range(K), 2):
        S = list(S)
        res = judge.query(DESC[dim], [files[k] for k in S])
        if res is None:
            sys.exit(f"query on subgroup {S} failed; nothing saved (rerun the command)")
        chosen, reasoning = res
        C = sorted(S[c] for c in chosen)
        responses.append((S, C, reasoning))
        print(f"  S={S} -> C={C} | {reasoning[:100]}")
        time.sleep(1)
    for fo in files:
        try:
            judge.client.files.delete(name=fo.name)
        except Exception:
            pass
    meta = dict(model=MODEL, temperature=0.0, dim=dim, data_seed=SEED, query_format="labeled attachments",
                queried_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    save_responses(str(path), DESC[dim], responses, meta=meta)
    print("saved", os.path.relpath(path))


# ============================================================================
# Estimation
# ============================================================================

def moments(x: np.ndarray):
    """Sample mean and MLE (co)variance (divides by n)."""
    if x.ndim == 1:
        return np.array([x.mean()]), np.array([[x.var()]])
    return x.mean(axis=0), np.cov(x, rowvar=False, ddof=0)


def kl(mu0, cov0, mu1, cov1) -> float:
    """KL(N(mu0, cov0) || N(mu1, cov1))."""
    ic1, d = np.linalg.inv(cov1), mu1 - mu0
    return float(0.5 * (np.trace(ic1 @ cov0) + d @ ic1 @ d - len(mu0)
                        + np.log(np.linalg.det(cov1) / np.linalg.det(cov0))))


def estimate(dim: int, pi_lip: np.ndarray):
    """Uniform EM, LIP-EM, Target Only and Pooled estimates on the seed-42 draw."""
    target, sources = make_data(dim)
    est = {}
    for name, pi in (("Uniform", np.full(K, P0)), ("LIP", pi_lip)):
        em = EM_CLASS[dim](pi, tau=TAU, nu=NU, max_iter=MAX_ITER)
        mu, cov = em.fit(target, sources)
        est[name] = dict(mu=mu, cov=cov, w=em.weights, n_iter=em.n_iter, converged=em.converged)
    for name, x in (("Target", target), ("Pooled", np.concatenate([target] + sources))):
        mu, cov = moments(x)
        est[name] = dict(mu=mu, cov=cov)
    return target, sources, est


# ============================================================================
# Figure
# ============================================================================

def plot(dim: int, target, sources, est, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.25, 3.2))
    colors = ['green', 'red', 'blue']
    if dim == 1:
        x = np.linspace(-40, 50, 2000)
        for i, a in enumerate(sources):
            ax.hist(a, bins=30, density=True, alpha=0.1, color=colors[i], label=f"Source {i + 1}")
        ax.plot(target, np.zeros_like(target) + 0.002, 'kx', markersize=10, markeredgewidth=2, label='Target Data')
        ax.plot(x, norm.pdf(x, MEANS_1D[0], SIGMA_1D), 'k--', linewidth=2.5, label='True Target')

        def curve(name, **kw):
            ax.plot(x, norm.pdf(x, est[name]["mu"][0], math.sqrt(est[name]["cov"][0, 0])), **kw)
        curve("Uniform", color='orange', linewidth=2, linestyle='--', label='Uniform EM')
        curve("LIP", color='purple', linewidth=4, label='LIP-EM')
        curve("Target", color='gray', linestyle=':', label='Target Only')
        curve("Pooled", color='brown', linestyle='-.', linewidth=2, label='Pooled')
        ax.set_xlim(-20, 20)
    else:
        xg, yg = np.mgrid[25:75:300j, 100:140:300j]
        pos = np.dstack((xg, yg))
        for i, a in enumerate(sources):
            ax.scatter(a[:, 0], a[:, 1], alpha=0.15, color=colors[i], label=f"Source {i + 1}", s=20)
        ax.scatter(target[:, 0], target[:, 1], color='black', marker='x', s=100, linewidths=2, label='Target Data')

        def ellipse(mu, cov, color, label, linestyle='solid', linewidth=2):
            ax.contour(xg, yg, multivariate_normal(mu, cov).pdf(pos), levels=[0.001, 0.005], colors=color,
                       linestyles=linestyle, linewidths=linewidth)
            ax.plot([], [], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        ellipse(est["Uniform"]["mu"], est["Uniform"]["cov"], 'orange', 'Uniform EM', linewidth=3)
        ellipse(est["LIP"]["mu"], est["LIP"]["cov"], 'purple', 'LIP-EM', linewidth=4)
        ellipse(est["Target"]["mu"], est["Target"]["cov"], 'gray', 'Target Only', linestyle='dotted', linewidth=3)
        ellipse(est["Pooled"]["mu"], est["Pooled"]["cov"], 'brown', 'Pooled', linestyle='dashdot', linewidth=3)
        ellipse(*truth(2), 'black', 'True Target', linestyle='dashed', linewidth=2)
        ax.set_xlim(35, 65)
        ax.set_ylim(110, 130)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(dim == 1 and side == 'bottom')
    ax.legend(loc='upper left', fontsize=9, frameon=False, handlelength=3.0, handletextpad=0.4, borderaxespad=0.2)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0, metadata={'CreationDate': None})
    plt.close(fig)


def run(dim: int, path: Path, outdir: Path) -> None:
    pi = fit_lip(load_responses(str(path)), K=K, p0=P0, eps=EPS_LIP)
    target, sources, est = estimate(dim, pi)
    mu0, cov0 = truth(dim)
    result = dict(dim=dim, seed=SEED, responses=os.path.relpath(path, HERE), pi=pi.tolist())
    print(f"{dim}-d   pi_LIP = {np.round(pi, 4)}")
    for name, e in est.items():
        r = dict(mu=e["mu"].tolist(), cov=e["cov"].tolist(), kl=kl(mu0, cov0, e["mu"], e["cov"]))
        line = f"  {name:8s} mu={np.round(e['mu'], 3)} cov={np.round(e['cov'], 2).tolist()} KL={r['kl']:.3f}"
        if "w" in e:
            r.update(w=e["w"].tolist(), n_iter=e["n_iter"], converged=e["converged"])
            line += f" w={np.round(e['w'], 4)} n_iter={e['n_iter']}"
        result[name] = r
        print(line)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf, js = outdir / f"{dim}dgaussian.pdf", outdir / f"{dim}dgaussian.json"
    plot(dim, target, sources, est, pdf)
    with open(js, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", os.path.relpath(pdf), "and", os.path.relpath(js))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["query", "plot"])
    ap.add_argument("--dim", type=int, choices=[1, 2], required=True)
    ap.add_argument("--responses", type=Path, default=None,
                    help="LIP response file (default: data/lip_responses_{dim}d.json)")
    ap.add_argument("--outdir", type=Path, default=HERE / "results", help="where plot writes the PDF and JSON")
    a = ap.parse_args()
    path = a.responses or responses_path(a.dim)
    if a.cmd == "query":
        query(a.dim, path)
    else:
        run(a.dim, path, a.outdir)
