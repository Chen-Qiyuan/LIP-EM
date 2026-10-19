"""Sensitivity of C-MAPSS LIP-EM to the target description
(paper App. "Sensitivity to the Target Description", Table tab:prompt).

For every cached response file in conditions.py, the LIP is fitted per target
(queries that contain the target are dropped, as for Table 1) and the C-MAPSS
EM of ../../cmapss is run at all nine cutoffs, next to Uniform EM, Pooled and
Target-Only. No LLM is called.

Usage:
    python run.py      # 19 LIPs + Uniform, 10 targets, 9 cutoffs; ~2.5 min
                       # -> results/runs.json, results/priors.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cmapss"))

import experiment as E  # noqa: E402
from dataset import load_machine  # noqa: E402
from em import predict  # noqa: E402

from conditions import CONDITIONS  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--targets", default=",".join(map(str, E.TARGETS)),
                    help="comma-separated target engine numbers")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    targets = [int(s) for s in args.targets.split(",")]

    t_start = time.time()
    runs, priors = [], {key: {} for key in CONDITIONS}
    for target in targets:
        X_sources, y_sources = E.load_sources(target)
        pis = {key: E.lip_prior(path, target)
               for key, (_, path) in CONDITIONS.items()}
        for key, pi in pis.items():
            priors[key][str(target)] = pi.tolist()
        pis["Uniform"] = np.full(len(X_sources), E.P_0)
        target_file = E.SEQ_DIR / f"{target}.csv"
        _, X_grid, y_full = load_machine(target_file)

        for rul in E.RUL_LEVELS:
            co = E.cutoff_for(len(y_full), rul)
            _, X_t, y_t = load_machine(target_file, cutoff=co)
            for method, (beta, hist) in E.fit_all(X_t, y_t, X_sources,
                                                  y_sources, pis).items():
                rec = dict(method=method, target=target, rul=rul,
                           rmse=E.rmse(y_full[co:], predict(X_grid, beta)[co:]))
                if hist is not None:
                    w = hist.weights[-1]
                    rec.update(n_iter=hist.n_iter,
                               converged=hist.converged_at is not None,
                               sum_w=float(w.sum()),
                               max_abs_w_minus_pi=float(
                                   np.max(np.abs(w - pis[method]))))
                runs.append(rec)
        print(f"  engine {target:>3d}: done ({time.time() - t_start:.0f}s)",
              flush=True)

    cfg = E.EM_CONFIG
    settings = dict(tau=cfg.tau, nu=cfg.nu,
                    convergence_tol=cfg.convergence_tol,
                    convergence_patience=cfg.convergence_patience,
                    stop_count_start=cfg.stop_count_start,
                    max_iter=cfg.max_iter, p_0=E.P_0, eps_lip=E.EPS_LIP,
                    ridge_lambda=E.RIDGE_LAMBDA)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "runs.json", "w") as f:
        json.dump(dict(settings=settings, targets=targets,
                       rul_levels=E.RUL_LEVELS,
                       conditions={k: dict(label=label,
                                           file=str(path.relative_to(REPO)))
                                   for k, (label, path) in CONDITIONS.items()},
                       runs=runs), f, indent=1)
    with open(args.out_dir / "priors.json", "w") as f:
        json.dump(priors, f)
    print(f"{len(runs)} runs in {time.time() - t_start:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
