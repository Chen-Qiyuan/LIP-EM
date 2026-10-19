"""Summarise the prompt ablation (paper Table tab:prompt and the text of
App. "Sensitivity to the Target Description").

Metrics, per condition:
  * all cutoffs / 90-40%: per target, the mean RMSE over the nine cutoffs
    (or the six cold-start cutoffs 90%..40%); then mean (SE) over the targets,
    SE = std / sqrt(n) with the sample std (ddof = 1),
  * Delta: paired difference of these per-target values from a reference
    (the cached LIP-C, or Uniform EM), with a 95% percentile bootstrap over
    target engines (20,000 resamples, np.random.default_rng(20260802));
  * rho: mean over targets of the Spearman correlation of the fitted LIP with
    the cached LIP-C;
  * nulls: queries answered with the null option; positive slope: share of the
    selected sources whose full core-speed trajectory has a positive slope.

Usage:
    python analyze.py                  # reads results/runs.json, results/priors.json
    python analyze.py --per-cutoff     # also print mean (SE) at every cutoff
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cmapss"))

import experiment as E  # noqa: E402
from dataset import load_machine  # noqa: E402

from conditions import CACHED, CONDITIONS, INFORMATIVE_KEYS, TABLE_KEYS  # noqa: E402


HERE = Path(__file__).resolve().parent
COLD = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
N_BOOT = 20_000
BOOT_SEED = 20260802


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def mean_se(v: np.ndarray) -> tuple[float, float]:
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def response_stats(path: Path, slopes: np.ndarray) -> dict:
    responses = json.loads(path.read_text())["responses"]
    chosen = [r["C"][0] for r in responses if r["C"]]
    return dict(selections=len(chosen), nulls=len(responses) - len(chosen),
                positive_slope=(float(np.mean(slopes[chosen] > 0))
                                if chosen else None))


def source_slopes() -> np.ndarray:
    """Least-squares slope of every engine's core speed against the cycle,
    in the source order of the response files."""
    files = sorted(E.SEQ_DIR.glob("*.csv"), key=lambda f: f.name)
    slopes = []
    for f in files:
        _, _, y = load_machine(f)
        slopes.append(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])
    return np.asarray(slopes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--per-cutoff", action="store_true")
    args = ap.parse_args()
    payload = json.loads((args.results / "runs.json").read_text())
    priors = json.loads((args.results / "priors.json").read_text())
    targets, ruls = payload["targets"], payload["rul_levels"]
    runs = payload["runs"]
    rmse = {(r["method"], r["target"], r["rul"]): r["rmse"] for r in runs}
    labels = {k: label for k, (label, _) in CONDITIONS.items()}
    labels.update({m: m for m in ("Uniform", "Pooled", "Target-Only")})
    methods = list(CONDITIONS) + ["Uniform", "Pooled", "Target-Only"]

    def per_target(method, levels):
        return np.array([np.mean([rmse[(method, t, l)] for l in levels])
                         for t in targets])

    boot_idx = np.random.default_rng(BOOT_SEED).integers(
        0, len(targets), size=(N_BOOT, len(targets)))

    def paired(a, b):
        d = a - b
        lo, hi = np.percentile(d[boot_idx].mean(axis=1), [2.5, 97.5])
        return float(d.mean()), float(lo), float(hi)

    def fmt_diff(t):
        # * = the interval excludes 0 at the printed precision.
        m, lo, hi = t
        return (f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                + ("*" if lo >= 0.005 or hi <= -0.005 else ""))

    all_ = {m: per_target(m, ruls) for m in methods}
    cold = {m: per_target(m, COLD) for m in methods}
    slopes = source_slopes()
    stats = {k: response_stats(path, slopes) for k, (_, path) in CONDITIONS.items()}
    rho = {}
    for k in CONDITIONS:
        vals = [spearman(np.array(priors[CACHED][str(t)]),
                         np.array(priors[k][str(t)])) for t in targets]
        vals = [v for v in vals if v is not None]
        rho[k] = float(np.mean(vals)) if vals else None

    # ---- Table tab:prompt ----
    print("=" * 112)
    print(f"Prompt ablation: future RMSE, mean (SE) over {len(targets)} targets; "
          "Delta = paired difference in all-cutoff RMSE from the cached LIP-C "
          "[95% bootstrap]; * = interval excludes 0")
    print("=" * 112)
    head = (f"{'Description':<38s}{'Nulls':>6s}{'All cutoffs':>15s}{'90%-40%':>15s}"
            f"{'Delta [95% CI]':>28s}{'rho':>8s}")
    print(head)
    for k in TABLE_KEYS + ["Uniform"]:
        nulls = "-" if k == "Uniform" else str(stats[k]["nulls"])
        delta = "-" if k == CACHED else fmt_diff(paired(all_[k], all_[CACHED]))
        r = "-" if rho.get(k) is None else f"{rho[k]:.3f}"
        print(f"{labels[k]:<38s}{nulls:>6s}{'%.2f (%.2f)' % mean_se(all_[k]):>15s}"
              f"{'%.2f (%.2f)' % mean_se(cold[k]):>15s}{delta:>28s}{r:>8s}")

    print("\nLaTeX rows (Description, Nulls, All cutoffs, 10%-60% observed life, Delta, rho):")
    for k in TABLE_KEYS + ["Uniform"]:
        nulls = "--" if k == "Uniform" else str(stats[k]["nulls"])
        if k == CACHED:
            delta = "--"
        else:
            m, lo, hi = paired(all_[k], all_[CACHED])
            delta = f"${m:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$]"
        r = "--" if rho.get(k) is None else f"{rho[k]:.3f}".replace("-", "$-$")
        print(f"    {labels[k]:<38s} & {nulls:>3s} & "
              + " & ".join("%.2f\\,(%.2f)" % mean_se(v[k]) for v in (all_, cold))
              + f" & {delta} & {r} \\\\")

    # ---- Every condition ----
    print("\nEvery condition: sel/nulls, positive-slope share of the selections, "
          "rho, all-cutoff and 90%-40% RMSE,\npaired differences from the cached "
          "LIP-C and from Uniform EM (negative = better than the reference)")
    print(f"{'Condition':<38s}{'sel/nulls':>10s}{'pos.':>7s}{'rho':>8s}"
          f"{'All':>15s}{'90-40%':>15s}{'all vs cached':>26s}{'90-40% vs cached':>26s}"
          f"{'all vs Uniform':>26s}{'90-40% vs Uniform':>26s}")
    for k in methods:
        s = stats.get(k)
        sel = f"{s['selections']}/{s['nulls']}" if s else "-"
        pos = ("-" if not s or s["positive_slope"] is None
               else f"{100 * s['positive_slope']:.1f}%")
        r = "-" if rho.get(k) is None else f"{rho[k]:.3f}"
        d_c = ["-", "-"] if k == CACHED else [fmt_diff(paired(v[k], v[CACHED]))
                                             for v in (all_, cold)]
        d_u = ["-", "-"] if k == "Uniform" else [fmt_diff(paired(v[k], v["Uniform"]))
                                                for v in (all_, cold)]
        print(f"{labels[k]:<38s}{sel:>10s}{pos:>7s}{r:>8s}"
              f"{'%.2f (%.2f)' % mean_se(all_[k]):>15s}"
              f"{'%.2f (%.2f)' % mean_se(cold[k]):>15s}"
              + "".join(f"{x:>26s}" for x in d_c + d_u))

    # ---- Headline numbers ----
    print("\nHeadline numbers")
    n_neg = int((slopes < 0).sum())
    gem = response_stats(E.LIP_FILES["LIP-G"], slopes)
    print(f"  {n_neg} of {len(slopes)} engines have a negative core-speed slope; "
          f"positive-slope share of the selections: cached LIP-C "
          f"{100 * stats[CACHED]['positive_slope']:.1f}% of "
          f"{stats[CACHED]['selections']}, LIP-G (Gemini) "
          f"{100 * gem['positive_slope']:.1f}% of {gem['selections']}")
    gain_all = [paired(all_["Uniform"], all_[k]) for k in INFORMATIVE_KEYS]
    gain_cold = [paired(cold["Uniform"], cold[k]) for k in INFORMATIVE_KEYS]
    excl = all(lo > 0 for _, lo, _ in gain_all + gain_cold)
    print(f"  informative descriptions improve on Uniform EM by "
          f"{min(g[0] for g in gain_all):.1f}-{max(g[0] for g in gain_all):.1f} "
          f"(all cutoffs) and {min(g[0] for g in gain_cold):.1f}-"
          f"{max(g[0] for g in gain_cold):.1f} (90%-40%); every interval "
          f"excludes 0: {excl}")
    rhos = [rho[k] for k in INFORMATIVE_KEYS if k != CACHED]
    print(f"  rho of the informative re-elicited LIPs: {min(rhos):.3f}-{max(rhos):.3f}")
    for k in ("opus/no_information", "cli_default/no_information"):
        diff = max(abs(rmse[(k, t, l)] - rmse[("Uniform", t, l)])
                   for t in targets for l in ruls)
        print(f"  {labels[k]}: {stats[k]['nulls']} nulls; max per-run |RMSE "
              f"difference| from Uniform EM {diff:.1e}")
    print(f"  cached LIP-C minus no-information control (all cutoffs): "
          f"{fmt_diff(paired(all_[CACHED], all_['opus/no_information']))}")

    if args.per_cutoff:
        print("\nMean (SE) at every cutoff")
        print(f"{'Condition':<38s}" + "".join(f"{int(round(100 * l)):>12d}%"
                                              for l in ruls))
        for k in methods:
            cells = ["%.1f (%.1f)" % mean_se(np.array([rmse[(k, t, l)]
                                                       for t in targets]))
                     for l in ruls]
            print(f"{labels[k]:<38s}" + "".join(f"{c:>13s}" for c in cells))


if __name__ == "__main__":
    main()
