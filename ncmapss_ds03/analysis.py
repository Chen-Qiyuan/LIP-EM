"""Collect the DS03 tables: per-fleet results, EM diagnostics, pooled analysis.

Reads the prognostics runs and engine bootstraps written by `run_all.sh` and
writes `tables.md` and `analysis.json` into the same results directory.

* Per fleet (five target engines): RUL RMSE with engine-bootstrap SE, paired
  RMSE(LIP) - RMSE(Uniform) with 95% percentile CI, and the complete
  nine-cutoff trajectory comparison. The long-haul fleet is reported with
  both Claude LIPs (independent queries, and
  the coding-agent session).
* EM diagnostics: runs whose final weights stay at the prior
  (max_k |w_k - pi_k| < 0.02) and iteration counts.
* Pooled secondary analysis over all 15 targets (Claude = coding-agent LIP in
  every fleet): engine bootstrap stratified within fleet (B = 20000, seed 42,
  fleet draw order long, medium, short) and the exact paired sign-flip test
  over all 2^15 swaps of LIP and Uniform errors (two-sided). The 10-30%
  observed-life band pools the first three cutoffs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import common


FRACTIONS = list(common.PROGNOSTIC_FRACTIONS)
FLEETS = ("long", "medium", "short")
METHODS = ("claude_lip_em", "gemini_lip_em", "uniform_em", "global_pool", "target_only")
LIPS = ("claude_lip_em", "gemini_lip_em")
LABELS = {
    "claude_lip_em": "Claude LIP-EM",
    "gemini_lip_em": "Gemini LIP-EM",
    "uniform_em": "Uniform EM",
    "global_pool": "Global Pool",
    "target_only": "Target-Only",
}
BOOTSTRAPS = {
    "bootstrap_long_claude_independent": "Long-haul, Claude LIP from independent queries",
    "bootstrap_long_claude_agent": "Long-haul, Claude LIP from the coding-agent session",
    "bootstrap_medium_claude_agent": "Medium-haul, Claude LIP from the coding-agent session",
    "bootstrap_short_claude_agent": "Short-haul, Claude LIP from the coding-agent session",
}
POOLED_BOOTSTRAPS = {fleet: f"bootstrap_{fleet}_claude_agent" for fleet in FLEETS}
EM_RUNS = {
    "long": [("Uniform", "long_claude_agent", "uniform_em"),
             ("Claude agent", "long_claude_agent", "llm_lip_em"),
             ("Claude independent", "long_claude_independent", "llm_lip_em"),
             ("Gemini", "long_gemini", "llm_lip_em")],
    "medium": [("Uniform", "medium_claude_agent", "uniform_em"),
               ("Claude agent", "medium_claude_agent", "llm_lip_em"),
               ("Gemini", "medium_gemini", "llm_lip_em")],
    "short": [("Uniform", "short_claude_agent", "uniform_em"),
              ("Claude agent", "short_claude_agent", "llm_lip_em"),
              ("Gemini", "short_gemini", "llm_lip_em")],
}
AT_PRIOR_TOLERANCE = 0.02
REPLICATES = 20_000
SEED = 42
BAND = slice(0, 3)  # 10%, 20%, 30% observed life


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def cutoff(value: str) -> float:
    return round(float(value), 1)


def percent(fraction: float) -> str:
    return f"{round(fraction * 100)}%"


def star(lower: float, upper: float) -> str:
    return "*" if lower > 0 or upper < 0 else ""


def engine_errors(directory: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """method -> (9 cutoffs, 5 engines) error array, engines sorted."""
    rows = read_rows(directory / "individual_engine_errors.csv")
    engines = sorted({row["target_engine"] for row in rows})
    errors = {method: np.full((9, 5), np.nan) for method in METHODS}
    for row in rows:
        errors[row["method"]][
            FRACTIONS.index(cutoff(row["observed_fraction"])),
            engines.index(row["target_engine"]),
        ] = float(row["error"])
    for method in METHODS:
        if np.isnan(errors[method]).any():
            raise RuntimeError(f"missing errors for {method} in {directory}")
    return engines, errors


# ------------------------------------------------------------------ per fleet
def per_fleet(results: Path, lines: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    lines.append("# Per-fleet results (5 target engines, engine bootstrap B = 20000, seed 42)\n")
    for name, title in BOOTSTRAPS.items():
        directory = results / name
        table = {
            (cutoff(row["observed_fraction"]), row["method"]): row
            for row in read_rows(directory / "engine_bootstrap_results.csv")
        }
        paired = {
            (cutoff(row["observed_fraction"]), row["method"]): row
            for row in read_rows(directory / "paired_bootstrap_vs_uniform.csv")
        }
        trajectory = read_rows(directory / "complete_trajectory_bootstrap.csv")
        output[name] = {
            "rmse_se": {
                f"{fraction}|{method}": [
                    float(table[(fraction, method)]["rmse"]),
                    float(table[(fraction, method)]["bootstrap_standard_error"]),
                ]
                for fraction in FRACTIONS
                for method in METHODS
            },
            "paired_vs_uniform": {
                f"{fraction}|{method}": {
                    "difference": float(paired[(fraction, method)]["difference"]),
                    "ci95": [
                        float(paired[(fraction, method)]["ci95_lower"]),
                        float(paired[(fraction, method)]["ci95_upper"]),
                    ],
                    "probability_better": float(
                        paired[(fraction, method)][
                            "bootstrap_probability_better_than_uniform"
                        ]
                    ),
                    "wins_ties_losses": [
                        int(paired[(fraction, method)]["engine_wins_out_of_5"]),
                        int(paired[(fraction, method)]["engine_ties_out_of_5"]),
                        int(paired[(fraction, method)]["engine_losses_out_of_5"]),
                    ],
                }
                for fraction in FRACTIONS
                for method in LIPS
            },
            "complete_trajectory": trajectory,
        }

        lines.append(f"## {title}\n")
        lines.append("RUL RMSE (engine-bootstrap SE).\n")
        lines.append(
            "| Observed target life | "
            + " | ".join(LABELS[method] for method in METHODS)
            + " |"
        )
        lines.append("|---:|" + "---:|" * len(METHODS))
        for fraction in FRACTIONS:
            cells = [
                f"{float(table[(fraction, method)]['rmse']):.2f} "
                f"({float(table[(fraction, method)]['bootstrap_standard_error']):.2f})"
                for method in METHODS
            ]
            lines.append(f"| {percent(fraction)} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "Paired RMSE(LIP) - RMSE(Uniform) [95% CI] (* = CI excludes 0); "
            "bootstrap P(LIP better); engine wins/ties/losses on |error|.\n"
        )
        lines.append("| Observed | Claude | Gemini |")
        lines.append("|---:|---|---|")
        for fraction in FRACTIONS:
            cells = []
            for method in LIPS:
                row = paired[(fraction, method)]
                lower, upper = float(row["ci95_lower"]), float(row["ci95_upper"])
                cells.append(
                    f"{float(row['difference']):+.2f} [{lower:+.2f}, {upper:+.2f}]"
                    f"{star(lower, upper)}; "
                    f"{float(row['bootstrap_probability_better_than_uniform']):.3f}; "
                    f"{row['engine_wins_out_of_5']}/{row['engine_ties_out_of_5']}/"
                    f"{row['engine_losses_out_of_5']}"
                )
            lines.append(f"| {percent(fraction)} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Complete trajectories (5 engines x 9 cutoffs, engines resampled).\n")
        lines.append("| Comparison | RMSE first | RMSE second | Difference [95% CI] | P(first better) |")
        lines.append("|---|---:|---:|---|---:|")
        for row in trajectory:
            lower, upper = float(row["ci95_lower"]), float(row["ci95_upper"])
            lines.append(
                f"| {LABELS[row['first_method']]} vs {LABELS[row['second_method']]} | "
                f"{float(row['first_rmse']):.3f} | {float(row['second_rmse']):.3f} | "
                f"{float(row['difference']):+.3f} [{lower:+.3f}, {upper:+.3f}]"
                f"{star(lower, upper)} | "
                f"{float(row['bootstrap_probability_first_better']):.3f} |"
            )
        lines.append("")
    return output


# ------------------------------------------------------------------ EM diagnostics
def em_outcomes(run: Path) -> dict[tuple[float, str, str], tuple[bool, int]]:
    """(cutoff, method, target) -> (weights ended at the prior, iterations)."""
    groups: dict[tuple[float, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(run / "source_weights.csv"):
        groups[(cutoff(row["cutoff"]), row["method"], row["target_engine"])].append(row)
    outcomes = {}
    for key, rows in groups.items():
        deviation = max(
            abs(float(row["final_weight"]) - float(row["initial_prior"]))
            for row in rows
        )
        iterations = {int(row["iterations"]) for row in rows}
        if len(iterations) != 1:
            raise RuntimeError(f"inconsistent iteration counts for {key}")
        outcomes[key] = (deviation < AT_PRIOR_TOLERANCE, iterations.pop())
    return outcomes


def diagnostics(results: Path, lines: list[str]) -> dict[str, Any]:
    runs = {
        run: em_outcomes(results / run)
        for specs in EM_RUNS.values()
        for _, run, _ in specs
    }
    lines.append("# EM diagnostics\n")
    lines.append(
        "Per fleet and cutoff: number of the 5 target engines whose final "
        f"weights stay at the prior (max_k |w_k - pi_k| < {AT_PRIOR_TOLERANCE}), "
        "and the median number of EM iterations.\n"
    )
    output: dict[str, Any] = {}
    for fleet, specs in EM_RUNS.items():
        lines.append(f"## {fleet.capitalize()}-haul\n")
        lines.append("| Observed | " + " | ".join(label for label, _, _ in specs) + " |")
        lines.append("|---:|" + "---|" * len(specs))
        output[fleet] = {}
        for fraction in FRACTIONS:
            cells = []
            for label, run, method in specs:
                values = [
                    value
                    for (local_cutoff, local_method, _), value in runs[run].items()
                    if local_cutoff == fraction and local_method == method
                ]
                at_prior = sum(value[0] for value in values)
                median = float(np.median([value[1] for value in values]))
                output[fleet][f"{fraction}|{label}"] = {
                    "at_prior": at_prior,
                    "median_iterations": median,
                }
                cells.append(f"{at_prior}/5, {median:.0f} it")
            lines.append(f"| {percent(fraction)} | " + " | ".join(cells) + " |")
        lines.append("")
    # The earliest possible stop runs t_0 + patience iterations (t = 0, ...).
    settings = json.loads(
        (results / EM_RUNS["long"][0][1] / "summary.json").read_text()
    )["prognostics"]
    start = settings["stopping_start_iteration"]
    patience = settings["convergence_patience"]
    cap = settings["em_max_iterations"]
    every = [value for outcomes in runs.values() for value in outcomes.values()]
    output["earliest_stop_iterations"] = start + patience
    output["runs"] = len(every)
    output["runs_at_earliest_stop"] = sum(
        value[1] == start + patience for value in every
    )
    output["runs_at_iteration_cap"] = sum(value[1] >= cap for value in every)
    lines.append(
        f"Across all {len(every)} EM runs (7 prognostics runs; Uniform EM "
        f"counted in each): {output['runs_at_earliest_stop']} stop at the "
        f"earliest permitted iteration count {start + patience} (t_0 = "
        f"{start} plus {patience} patience iterations); "
        f"{output['runs_at_iteration_cap']} reach the {cap}-iteration cap.\n"
    )
    return output


# ------------------------------------------------------------------ pooled
SIGNS = np.array(list(itertools.product([0, 1], repeat=15)), dtype=bool)


def rmse_by_cutoff(errors: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(errors**2, axis=-1))


def stratified_indices(rng: np.random.Generator, fleets: int, replicates: int) -> np.ndarray:
    """One (B, 5) resampling matrix per fleet, offset into the 15 engines."""
    return np.concatenate(
        [rng.integers(0, 5, size=(replicates, 5)) + 5 * k for k in range(fleets)],
        axis=1,
    )


def bootstrap_rmse_by_cutoff(errors: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """errors (9, 15), indices (B, 15) -> (B, 9)."""
    return np.sqrt((errors**2)[:, indices].mean(axis=2)).T


def bootstrap_rmse_all(errors: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """RMSE over all rows of `errors` (cutoffs x engines) -> (B,)."""
    return np.sqrt((errors**2)[:, indices].mean(axis=(0, 2)))


def interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5], axis=0)
    return float(lower), float(upper)


def sign_flip(lip_squared: np.ndarray, uniform_squared: np.ndarray, count: int) -> tuple[float, float]:
    """Exact paired sign-flip test of RMSE(LIP) - RMSE(Uniform), two-sided.

    Inputs are per-engine sums of squared errors; every one of the 2^15
    relabelings swaps the LIP and Uniform values of a subset of engines.
    """
    observed = np.sqrt(lip_squared.sum() / count) - np.sqrt(uniform_squared.sum() / count)
    statistic = np.sqrt(
        np.where(SIGNS, uniform_squared, lip_squared).sum(1) / count
    ) - np.sqrt(np.where(SIGNS, lip_squared, uniform_squared).sum(1) / count)
    return float(observed), float(np.mean(np.abs(statistic) >= abs(observed) - 1e-12))


def pooled(results: Path, lines: list[str]) -> dict[str, Any]:
    engines: list[tuple[str, str]] = []
    per_fleet_errors = {}
    for fleet in FLEETS:
        fleet_engines, per_fleet_errors[fleet] = engine_errors(
            results / POOLED_BOOTSTRAPS[fleet]
        )
        engines += [(fleet, engine) for engine in fleet_engines]
    errors = {
        method: np.concatenate(
            [per_fleet_errors[fleet][method] for fleet in FLEETS], axis=1
        )
        for method in METHODS
    }  # (9 cutoffs, 15 engines), fleets in the order long, medium, short
    indices = stratified_indices(np.random.default_rng(SEED), 3, REPLICATES)
    boot = {method: bootstrap_rmse_by_cutoff(errors[method], indices) for method in METHODS}

    output: dict[str, Any] = {"rmse_se": {}, "paired_by_cutoff": {}, "band_10_30": {}, "all_cutoffs": {}}
    for method in METHODS:
        output["rmse_se"][method] = [
            [float(rmse_by_cutoff(errors[method])[j]), float(boot[method][:, j].std(ddof=1))]
            for j in range(9)
        ] + [[
            float(np.sqrt(np.mean(errors[method] ** 2))),
            float(bootstrap_rmse_all(errors[method], indices).std(ddof=1)),
        ]]
    uniform = errors["uniform_em"]
    for method in LIPS:
        lip = errors[method]
        output["paired_by_cutoff"][method] = []
        for j, fraction in enumerate(FRACTIONS):
            differences = boot[method][:, j] - boot["uniform_em"][:, j]
            lower, upper = interval(differences)
            absolute = np.abs(lip[j]) - np.abs(uniform[j])
            _, p_value = sign_flip(lip[j] ** 2, uniform[j] ** 2, 15)
            output["paired_by_cutoff"][method].append(
                {
                    "observed_fraction": fraction,
                    "difference": float(rmse_by_cutoff(lip)[j] - rmse_by_cutoff(uniform)[j]),
                    "ci95": [lower, upper],
                    "probability_better": float(np.mean(differences < 0)),
                    "wins_ties_losses": [
                        int(np.sum(absolute < -1e-12)),
                        int(np.sum(np.abs(absolute) <= 1e-12)),
                        int(np.sum(absolute > 1e-12)),
                    ],
                    "sign_flip_p": p_value,
                }
            )
        for key, rows, count in (("band_10_30", BAND, 45), ("all_cutoffs", slice(0, 9), 135)):
            point = float(np.sqrt(np.mean(lip[rows] ** 2)) - np.sqrt(np.mean(uniform[rows] ** 2)))
            differences = bootstrap_rmse_all(lip[rows], indices) - bootstrap_rmse_all(uniform[rows], indices)
            lower, upper = interval(differences)
            _, p_value = sign_flip((lip[rows] ** 2).sum(0), (uniform[rows] ** 2).sum(0), count)
            output[key][method] = {
                "difference": point,
                "ci95": [lower, upper],
                "probability_better": float(np.mean(differences < 0)),
                "sign_flip_p": p_value,
            }
    output["band_10_30_engine_contributions"] = [
        {
            "fleet": fleet,
            "target_engine": engine,
            **{
                method: float(np.mean(errors[method][BAND, index] ** 2 - uniform[BAND, index] ** 2))
                for method in LIPS
            },
        }
        for index, (fleet, engine) in enumerate(engines)
    ]

    lines.append("# Pooled secondary analysis (15 targets; Claude = coding-agent LIP in every fleet)\n")
    lines.append(
        "Engine bootstrap stratified within fleet (B = 20000, seed 42); exact "
        "paired sign-flip test over 2^15 = 32768 relabelings, two-sided.\n"
    )
    lines.append("RUL RMSE (stratified bootstrap SE) over the 15 targets.\n")
    lines.append("| Observed | " + " | ".join(LABELS[method] for method in METHODS) + " |")
    lines.append("|---:|" + "---:|" * len(METHODS))
    for j in range(10):
        label = percent(FRACTIONS[j]) if j < 9 else "all"
        cells = [
            f"{output['rmse_se'][method][j][0]:.2f} ({output['rmse_se'][method][j][1]:.2f})"
            for method in METHODS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "RMSE(LIP) - RMSE(Uniform) [95% CI] (* = CI excludes 0); bootstrap "
        "P(LIP better); wins/ties/losses on |error|; sign-flip p.\n"
    )
    lines.append("| Observed | Claude LIP-EM | Gemini LIP-EM |")
    lines.append("|---:|---|---|")
    for j, fraction in enumerate(FRACTIONS):
        cells = []
        for method in LIPS:
            row = output["paired_by_cutoff"][method][j]
            lower, upper = row["ci95"]
            cells.append(
                f"{row['difference']:+.2f} [{lower:+.2f}, {upper:+.2f}]{star(lower, upper)}; "
                f"P={row['probability_better']:.3f}; "
                f"{'/'.join(str(n) for n in row['wins_ties_losses'])}; "
                f"p={row['sign_flip_p']:.3f}"
            )
        lines.append(f"| {percent(fraction)} | " + " | ".join(cells) + " |")
    for key, label in (("band_10_30", "10-30% band"), ("all_cutoffs", "all cutoffs")):
        cells = []
        for method in LIPS:
            row = output[key][method]
            lower, upper = row["ci95"]
            cells.append(
                f"{row['difference']:+.2f} [{lower:+.2f}, {upper:+.2f}]{star(lower, upper)}; "
                f"P={row['probability_better']:.3f}; p={row['sign_flip_p']:.3f}"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Per-engine contribution in the 10-30% band: mean over the three "
        "cutoffs of e_LIP^2 - e_Uniform^2 (negative favours LIP).\n"
    )
    lines.append("| Fleet | Target engine | Claude LIP-EM | Gemini LIP-EM |")
    lines.append("|---|---|---:|---:|")
    for row in output["band_10_30_engine_contributions"]:
        lines.append(
            f"| {row['fleet']} | {row['target_engine']} | "
            f"{row['claude_lip_em']:+.1f} | {row['gemini_lip_em']:+.1f} |"
        )
    lines.append("")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=common.HERE / "results")
    arguments = parser.parse_args()
    lines: list[str] = []
    output = {
        "per_fleet": per_fleet(arguments.results, lines),
        "em_diagnostics": diagnostics(arguments.results, lines),
        "pooled": pooled(arguments.results, lines),
    }
    text = "\n".join(lines) + "\n"
    (arguments.results / "tables.md").write_text(text)
    (arguments.results / "analysis.json").write_text(json.dumps(output, indent=1))
    print(text)


if __name__ == "__main__":
    main()
