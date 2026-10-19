"""Shape-matched shuffled-LIP control on C-MAPSS
(paper App. "Shuffled LIP Control (C-MAPSS)", Table tab:shuffled).

For each target engine, the fitted LIP-C and LIP-G vectors are permuted over
the 99 source identities (the same permutation for both). A permutation keeps
the multiset of prior values, and hence its mean, entropy and concentration,
and destroys only the assignment of prior mass to sources. Each shuffled prior
is run through the C-MAPSS EM of ../../cmapss with all other settings
unchanged, next to the real LIPs, Uniform EM, Pooled and Target-Only.

The permutations for target m come from np.random.default_rng(seed + 10000 m).
EM is vectorised over the priors (batched_em.py); for every (target, cutoff)
the unshuffled LIP-C, LIP-G and Uniform rows are checked against em.em_train.

Usage:
    python run.py                                                # paper: 100 shuffles, seed 42,
                                                                 #   all 9 cutoffs -> results/shuffles100/
    python run.py --shuffles 500 --seed 314159 --rul 0.9,0.7,0.5 # additional check (not in the paper): 500 shuffles -> results/shuffles500/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cmapss"))

import experiment as E  # noqa: E402
from batched_em import em_batch, source_stats  # noqa: E402
from dataset import load_machine  # noqa: E402
from em import em_train, predict  # noqa: E402


HERE = Path(__file__).resolve().parent
LIPS = ["LIP-G", "LIP-C"]
# Tolerances for the check against em.em_train (the batched E-step sums in a
# different order, so agreement is to rounding, not bit for bit).
CHECK_TOL_W = 1e-6
CHECK_TOL_RMSE = 1e-6


def prior_batch(pis: dict[str, np.ndarray], shuffles: int, seed: int,
                target: int) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Rows: LIP-G, LIP-C, Uniform, then (shuffled LIP-G, shuffled LIP-C) per
    permutation. Labels are (method, shuffle index), with -1 for no shuffle."""
    priors = [pis["LIP-G"], pis["LIP-C"], pis["Uniform"]]
    labels = [("LIP-G", -1), ("LIP-C", -1), ("Uniform", -1)]
    rng = np.random.default_rng(seed + 10_000 * target)
    for s in range(shuffles):
        perm = rng.permutation(len(pis["LIP-G"]))
        for name in LIPS:
            priors.append(pis[name][perm])
            labels.append((f"Shuffled {name}", s))
    return np.asarray(priors), labels


def run_target(target: int, ruls: list[float], shuffles: int, seed: int
               ) -> tuple[list[dict], dict]:
    cfg = E.EM_CONFIG
    pis = E.priors_for(target)
    X_sources, y_sources = E.load_sources(target)
    stats = source_stats(X_sources, y_sources, cfg)
    priors, labels = prior_batch(pis, shuffles, seed, target)

    target_file = E.SEQ_DIR / f"{target}.csv"
    _, X_grid, y_full = load_machine(target_file)
    rows, check = [], dict(max_w_diff=0.0, max_rmse_diff=0.0, iter_mismatch=0)
    for rul in ruls:
        co = E.cutoff_for(len(y_full), rul)
        _, X_t, y_t = load_machine(target_file, cutoff=co)
        future = y_full[co:]
        beta, w, n_iter, converged, inv_eps = em_batch(X_t, y_t, priors,
                                                       stats, cfg)
        rmse = [E.rmse(future, predict(X_grid, b)[co:]) for b in beta]
        for i, (method, shuffle) in enumerate(labels):
            rows.append(dict(
                target=target, rul=rul, method=method, shuffle=shuffle,
                rmse=rmse[i], mean_w=float(w[i].mean()),
                max_w=float(w[i].max()), n_iter=int(n_iter[i]),
                converged=bool(converged[i]),
                max_abs_w_minus_pi=float(np.max(np.abs(w[i] - priors[i]))),
                inv_eps_median=float(np.median(inv_eps))))

        # The unshuffled rows must reproduce em.em_train.
        for i in range(3):
            beta_ref, hist = em_train(X_t, y_t, X_sources, y_sources,
                                      pi=priors[i], cfg=cfg)
            check["max_w_diff"] = max(check["max_w_diff"], float(
                np.max(np.abs(hist.weights[-1] - w[i]))))
            check["max_rmse_diff"] = max(check["max_rmse_diff"], abs(
                E.rmse(future, predict(X_grid, beta_ref)[co:]) - rmse[i]))
            check["iter_mismatch"] += int(hist.n_iter != n_iter[i])

        X_pool = np.vstack([X_t] + list(X_sources))
        y_pool = np.concatenate([y_t] + list(y_sources))
        for method, b in (
                ("Pooled", E.ridge_fit(X_pool, y_pool, E.RIDGE_LAMBDA)),
                ("Target-Only", E.ridge_fit(X_t, y_t, E.RIDGE_LAMBDA))):
            rows.append(dict(target=target, rul=rul, method=method, shuffle=-1,
                             rmse=E.rmse(future, predict(X_grid, b)[co:])))
    return rows, check


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rul", default=",".join(map(str, E.RUL_LEVELS)),
                    help="comma-separated RUL levels")
    ap.add_argument("--targets", default=",".join(map(str, E.TARGETS)),
                    help="comma-separated target engine numbers")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: results/shuffles<N>")
    args = ap.parse_args()
    targets = [int(s) for s in args.targets.split(",")]
    ruls = [float(s) for s in args.rul.split(",")]
    out_dir = args.out_dir or HERE / "results" / f"shuffles{args.shuffles}"

    t_start = time.time()
    rows, check = [], dict(max_w_diff=0.0, max_rmse_diff=0.0, iter_mismatch=0)
    for target in targets:
        r, c = run_target(target, ruls, args.shuffles, args.seed)
        rows += r
        check = dict(max_w_diff=max(check["max_w_diff"], c["max_w_diff"]),
                     max_rmse_diff=max(check["max_rmse_diff"], c["max_rmse_diff"]),
                     iter_mismatch=check["iter_mismatch"] + c["iter_mismatch"])
        print(f"  engine {target:>3d}: done ({time.time() - t_start:.0f}s)",
              flush=True)
    print(f"Unshuffled rows vs em.em_train: max |w diff| {check['max_w_diff']:.1e}, "
          f"max |RMSE diff| {check['max_rmse_diff']:.1e}, "
          f"{check['iter_mismatch']} iteration-count mismatches")
    assert (check["max_w_diff"] < CHECK_TOL_W
            and check["max_rmse_diff"] < CHECK_TOL_RMSE
            and check["iter_mismatch"] == 0), "batched EM differs from em.em_train"

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["target", "rul", "method", "shuffle", "rmse", "mean_w", "max_w",
              "n_iter", "converged", "max_abs_w_minus_pi", "inv_eps_median"]
    with open(out_dir / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cfg = E.EM_CONFIG
    settings = dict(shuffles=args.shuffles, seed=args.seed, targets=targets,
                    rul_levels=ruls, tau=cfg.tau, nu=cfg.nu,
                    convergence_tol=cfg.convergence_tol,
                    convergence_patience=cfg.convergence_patience,
                    stop_count_start=cfg.stop_count_start,
                    max_iter=cfg.max_iter, p_0=E.P_0, eps_lip=E.EPS_LIP,
                    ridge_lambda=E.RIDGE_LAMBDA,
                    lip_files={k: v.name for k, v in E.LIP_FILES.items()},
                    check_against_em_train=check)
    with open(out_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=1)
    print(f"{len(rows)} rows in {time.time() - t_start:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
