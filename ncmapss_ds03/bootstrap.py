"""Engine-level bootstrap for one DS03 fleet (five held-out target engines).

Reads the `prognostics_predictions.csv` of a Claude-LIP run and a Gemini-LIP
run for the same flight class. Uniform EM, Global Pool and Target Only do not
depend on the LIP and are taken from the Claude run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import common


FRACTIONS = common.PROGNOSTIC_FRACTIONS
METHOD_SPECS = (
    ("claude_lip_em", "Claude LIP-EM", "claude", "llm_lip_em"),
    ("gemini_lip_em", "Gemini LIP-EM", "gemini", "llm_lip_em"),
    ("uniform_em", "Uniform EM", "claude", "uniform_em"),
    ("global_pool", "Global pool", "claude", "global_pool"),
    ("target_only", "Target only", "claude", "target_only"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def error_vector(
    rows: list[dict[str, str]], fraction: float, method: str
) -> tuple[list[str], np.ndarray]:
    local = [
        row
        for row in rows
        if float(row["observed_fraction"]) == fraction
        and row["method"] == method
    ]
    local.sort(key=lambda row: row["target_engine"])
    engines = [row["target_engine"] for row in local]
    errors = np.asarray([float(row["error"]) for row in local])
    if len(engines) != 5 or len(set(engines)) != 5:
        raise RuntimeError(
            f"expected five engine errors for {fraction}, {method}"
        )
    return engines, errors


def cap_hits(
    rows: list[dict[str, str]], fraction: float, method: str
) -> int:
    local = [
        row
        for row in rows
        if float(row["observed_fraction"]) == fraction
        and row["method"] == method
    ]
    if len(local) != 5:
        raise RuntimeError(
            f"expected five cap indicators for {fraction}, {method}"
        )
    return sum(int(row["rul_cap_hit"]) for row in local)


def rmse(errors: np.ndarray, axis: Any = None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(errors), axis=axis))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-class", choices=common.FLIGHT_CLASSES, required=True
    )
    arguments = parser.parse_args()
    common.prepare_output(arguments.output)
    datasets = {
        "claude": read_rows(arguments.claude / "prognostics_predictions.csv"),
        "gemini": read_rows(arguments.gemini / "prognostics_predictions.csv"),
    }
    paired_methods = tuple(
        method_name
        for method_name, _, _, raw_method in METHOD_SPECS
        if raw_method == "llm_lip_em"
    )
    rng = np.random.default_rng(arguments.seed)
    indices = rng.integers(0, 5, size=(arguments.replicates, 5))
    result_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    individual_rows: list[dict[str, Any]] = []
    errors_by_fraction: dict[float, dict[str, np.ndarray]] = {}

    for fraction in FRACTIONS:
        error_by_method: dict[str, np.ndarray] = {}
        engine_order: list[str] | None = None
        for method_name, label, dataset_name, raw_method in METHOD_SPECS:
            engines, errors = error_vector(
                datasets[dataset_name], fraction, raw_method
            )
            if engine_order is None:
                engine_order = engines
            elif engines != engine_order:
                raise RuntimeError("target-engine order differs across runs")
            error_by_method[method_name] = errors
            samples = rmse(errors[indices], axis=1)
            lower, upper = np.quantile(samples, [0.025, 0.975])
            result_rows.append(
                {
                    "observed_fraction": fraction,
                    "method": method_name,
                    "label": label,
                    "target_engines": 5,
                    "rmse": float(rmse(errors)),
                    "bootstrap_standard_error": float(samples.std(ddof=1)),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "rul_cap_hits_out_of_5": cap_hits(
                        datasets[dataset_name], fraction, raw_method
                    ),
                    "bootstrap_replicates": arguments.replicates,
                }
            )
            for engine, error in zip(engines, errors):
                individual_rows.append(
                    {
                        "observed_fraction": fraction,
                        "target_engine": engine,
                        "method": method_name,
                        "error": float(error),
                        "absolute_error": abs(float(error)),
                    }
                )
        errors_by_fraction[fraction] = error_by_method

        uniform_errors = error_by_method["uniform_em"]
        uniform_samples = rmse(uniform_errors[indices], axis=1)
        for method_name in paired_methods:
            errors = error_by_method[method_name]
            differences = rmse(errors[indices], axis=1) - uniform_samples
            lower, upper = np.quantile(differences, [0.025, 0.975])
            absolute_difference = np.abs(errors) - np.abs(uniform_errors)
            paired_rows.append(
                {
                    "observed_fraction": fraction,
                    "method": method_name,
                    "comparison": "RMSE(method) - RMSE(uniform_em)",
                    "difference": float(rmse(errors) - rmse(uniform_errors)),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "bootstrap_probability_better_than_uniform": float(
                        np.mean(differences < 0.0)
                    ),
                    "engine_wins_out_of_5": int(
                        np.sum(absolute_difference < -1e-12)
                    ),
                    "engine_ties_out_of_5": int(
                        np.sum(np.abs(absolute_difference) <= 1e-12)
                    ),
                    "engine_losses_out_of_5": int(
                        np.sum(absolute_difference > 1e-12)
                    ),
                    "bootstrap_replicates": arguments.replicates,
                }
            )

    common.write_csv(
        arguments.output / "engine_bootstrap_results.csv", result_rows
    )
    common.write_csv(
        arguments.output / "paired_bootstrap_vs_uniform.csv", paired_rows
    )
    common.write_csv(
        arguments.output / "individual_engine_errors.csv", individual_rows
    )

    trajectory_rows: list[dict[str, Any]] = []
    trajectory_samples: dict[str, np.ndarray] = {}
    trajectory_estimates: dict[str, float] = {}
    for method_name in (*paired_methods, "uniform_em"):
        # Rows are engines and columns are observation cutoffs. Reusing the
        # same sampled row indices preserves each engine's complete error
        # trajectory across all nine cutoffs.
        matrix = np.column_stack(
            [errors_by_fraction[fraction][method_name] for fraction in FRACTIONS]
        )
        trajectory_estimates[method_name] = float(rmse(matrix))
        trajectory_samples[method_name] = np.sqrt(
            np.mean(np.square(matrix[indices]), axis=(1, 2))
        )

    trajectory_comparisons = [
        (method_name, "uniform_em") for method_name in paired_methods
    ]
    trajectory_comparisons.append((paired_methods[0], paired_methods[1]))
    for first, second in trajectory_comparisons:
        differences = trajectory_samples[first] - trajectory_samples[second]
        lower, upper = np.quantile(differences, [0.025, 0.975])
        trajectory_rows.append(
            {
                "first_method": first,
                "second_method": second,
                "comparison": f"RMSE({first}) - RMSE({second})",
                "first_rmse": trajectory_estimates[first],
                "second_rmse": trajectory_estimates[second],
                "difference": (
                    trajectory_estimates[first] - trajectory_estimates[second]
                ),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "bootstrap_probability_first_better": float(
                    np.mean(differences < 0.0)
                ),
                "bootstrap_replicates": arguments.replicates,
            }
        )
    common.write_csv(
        arguments.output / "complete_trajectory_bootstrap.csv",
        trajectory_rows,
    )

    result_index = {
        (row["observed_fraction"], row["method"]): row for row in result_rows
    }
    display_methods = [specification[0] for specification in METHOD_SPECS]
    display_labels = {
        method_name: label for method_name, label, _, _ in METHOD_SPECS
    }
    lines = [
        "# DS03 engine-bootstrap prognostics",
        "",
        "The independent sampling unit is the engine. At each cutoff, the five "
        f"held-out {arguments.target_class}-haul engine errors are resampled "
        "with replacement, "
        f"and RUL RMSE is recomputed for {arguments.replicates:,} bootstrap "
        "replicates. Flight rows are not resampled because observations from "
        "one run-to-failure engine are serially dependent.",
        "",
        "| Observed life | "
        + " | ".join(specification[1] for specification in METHOD_SPECS)
        + " |",
        "|---:|" + "---:|" * len(METHOD_SPECS),
    ]
    for fraction in FRACTIONS:
        cells = []
        for method in display_methods:
            row = result_index[(fraction, method)]
            cells.append(
                f"{row['rmse']:.2f} "
                f"[{row['ci95_lower']:.2f}, {row['ci95_upper']:.2f}]"
            )
        lines.append(f"| {100 * fraction:.0f}% | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Intervals are percentile 95% engine-bootstrap intervals. With "
            "only five engines, they quantify finite-fleet variability but "
            "will be wide and should not be interpreted as high-powered "
            "hypothesis tests.",
            "",
            "## Paired comparison with uniform EM",
            "",
            "The same resampled target engines are used for each LIP-EM versus "
            "uniform-EM comparison. Negative differences favor LIP-EM.",
            "",
            "| Observed life | LIP | RMSE difference [95% CI] | "
            "Bootstrap P(LIP better) | Engine W/T/L |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {100 * row['observed_fraction']:.0f}% | "
            f"{display_labels[row['method']].removesuffix(' LIP-EM')} | "
            f"{row['difference']:.2f} "
            f"[{row['ci95_lower']:.2f}, {row['ci95_upper']:.2f}] | "
            f"{row['bootstrap_probability_better_than_uniform']:.3f} | "
            f"{row['engine_wins_out_of_5']}/"
            f"{row['engine_ties_out_of_5']}/"
            f"{row['engine_losses_out_of_5']} |"
        )
    lines.extend(
        [
            "",
            "## Complete-trajectory paired comparison",
            "",
            "This summary pools all 45 engine/cutoff predictions. Each "
            "bootstrap replicate resamples complete nine-cutoff engine "
            "trajectories, preserving within-engine dependence. Negative "
            "differences favor the first method.",
            "",
            "| First method | Second method | First RMSE | Second RMSE | "
            "Difference [95% CI] | P(first better) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in trajectory_rows:
        lines.append(
            f"| {display_labels[row['first_method']]} | "
            f"{display_labels[row['second_method']]} | "
            f"{row['first_rmse']:.3f} | {row['second_rmse']:.3f} | "
            f"{row['difference']:.3f} "
            f"[{row['ci95_lower']:.3f}, {row['ci95_upper']:.3f}] | "
            f"{row['bootstrap_probability_first_better']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## RUL-cap diagnostics",
            "",
            "A cap hit means the fitted path did not cross the source-learned "
            "failure threshold within the maximum source RUL horizon. These "
            "are valid capped predictions, but they explain the abrupt RMSE "
            "changes at some cutoffs.",
            "",
            "| Observed life | "
            + " | ".join(specification[1] for specification in METHOD_SPECS)
            + " |",
            "|---:|" + "---:|" * len(METHOD_SPECS),
        ]
    )
    for fraction in FRACTIONS:
        cells = [
            f"{result_index[(fraction, method)]['rul_cap_hits_out_of_5']}/5"
            for method in display_methods
        ]
        lines.append(f"| {100 * fraction:.0f}% | " + " | ".join(cells) + " |")
    lines.append("")
    (arguments.output / "bootstrap_report.md").write_text("\n".join(lines))
    (arguments.output / "bootstrap_metadata.json").write_text(
        json.dumps(
            {
                "sampling_unit": "target engine",
                "target_class": arguments.target_class,
                "claude_run": os.path.basename(os.path.normpath(arguments.claude)),
                "gemini_run": os.path.basename(os.path.normpath(arguments.gemini)),
                "target_engines": 5,
                "replicates": arguments.replicates,
                "seed": arguments.seed,
                "metric": "sqrt(mean(engine prediction error squared))",
                "paired_comparison": (
                    "same resampled engine indices used for LIP and uniform"
                ),
                "complete_trajectory_comparison": (
                    "complete nine-cutoff engine trajectories resampled"
                ),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "output": arguments.output.name,
                "rows": len(result_rows),
                "paired_rows": len(paired_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
