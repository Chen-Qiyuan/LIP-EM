"""N-CMAPSS DS03 leave-one-engine-out joint-sensor prognostics.

For one flight class, each of its five engines is in turn the target and the
other 14 engines are anonymous sources. At every observed-life cutoff
(10%, ..., 90%) the target prefix is fit by Target Only, Global Pool,
Uniform EM and LIP-EM; the fitted 14-sensor paths are mapped to a
source-only composite health index, and the predicted RUL is the first
crossing of the source-terminal health threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import common
import em


BASELINE_RIDGE_LAMBDA = 1e4


def multi_ridge(
    design: np.ndarray,
    outcomes: np.ndarray,
    penalty: float = BASELINE_RIDGE_LAMBDA,
) -> np.ndarray:
    """Baseline ridge fit with an unpenalized intercept (as for C-MAPSS)."""
    dimension = design.shape[1]
    regularizer = np.eye(dimension)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + penalty * regularizer,
        design.T @ outcomes,
    )


def fit_all_methods(
    target_design: np.ndarray,
    target_outcomes: np.ndarray,
    source_designs: Sequence[np.ndarray],
    source_outcomes: Sequence[np.ndarray],
    llm_prior: np.ndarray,
    tau: float,
    nu: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    coefficients = {
        "target_only": multi_ridge(target_design, target_outcomes),
        "global_pool": multi_ridge(
            np.vstack([target_design, *source_designs]),
            np.vstack([target_outcomes, *source_outcomes]),
        ),
    }
    final_weights: dict[str, np.ndarray] = {}
    iterations: dict[str, int] = {}
    uniform = np.full(len(source_designs), common.UNIFORM_PRIOR_VALUE)
    for method, prior in (
        ("uniform_em", uniform),
        ("llm_lip_em", llm_prior),
    ):
        (
            coefficients[method],
            final_weights[method],
            iterations[method],
        ) = em.em_train(
            target_design,
            target_outcomes,
            source_designs,
            source_outcomes,
            prior,
            tau=tau,
            nu=nu,
        )
    return coefficients, final_weights, iterations


def crossing_rul(
    coefficients: np.ndarray,
    current_cycle: float,
    threshold: float,
    rul_cap: float,
    fusion_weights: np.ndarray,
) -> tuple[float, bool, int, float]:
    """First crossing of the health threshold by the predicted sensor paths."""
    future_cycles = np.linspace(
        current_cycle, current_cycle + rul_cap, 4001
    )
    sensor_paths = common.natural_spline_basis(future_cycles) @ coefficients
    contributions = np.abs(sensor_paths) * fusion_weights[None, :]
    log_health = np.log(contributions.sum(axis=1) + common.LOG_OFFSET)
    crossing = np.flatnonzero(log_health >= threshold)
    if len(crossing):
        crossing_index = int(crossing[0])
        prediction = float(future_cycles[crossing_index] - current_cycle)
        cap_hit = False
    else:
        crossing_index = len(future_cycles) - 1
        prediction = rul_cap
        cap_hit = True
    driver = int(np.argmax(contributions[crossing_index]))
    return (
        prediction,
        cap_hit,
        driver,
        float(log_health[crossing_index]),
    )


def run_fold(
    target: dict[str, Any],
    sources: list[dict[str, Any]],
    sensor_columns: list[str],
    llm_prior: np.ndarray,
    tau: float,
    nu: float,
    fusion_weights: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_designs = [source["path_design"] for source in sources]
    source_outcomes = [source["path_sensor_outcome"] for source in sources]
    threshold = float(
        np.mean([source["path_outcome"][-1] for source in sources])
    )
    rul_cap = float(max(source["rul"][0] for source in sources))
    predictions: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    for fraction in common.PROGNOSTIC_FRACTIONS:
        observed = common.prefix_length(target, fraction)
        coefficients, local_weights, iterations = fit_all_methods(
            target["path_design"][:observed],
            target["path_sensor_outcome"][:observed],
            source_designs,
            source_outcomes,
            llm_prior,
            tau,
            nu,
        )
        common.append_weights(
            weights,
            fraction,
            target,
            sources,
            local_weights,
            iterations,
            llm_prior,
        )
        truth = float(target["rul"][observed - 1])
        current_cycle = float(target["cycle"][observed - 1])
        for method, local_coefficients in coefficients.items():
            prediction, cap_hit, driver, predicted_log_health = crossing_rul(
                local_coefficients,
                current_cycle,
                threshold,
                rul_cap,
                fusion_weights,
            )
            error = prediction - truth
            predictions.append(
                {
                    "target_engine": target["domain_id"],
                    "observed_fraction": fraction,
                    "observed_flights": observed,
                    "true_rul": truth,
                    "method": method,
                    "predicted_rul": prediction,
                    "error": error,
                    "squared_error": error**2,
                    "nasa_score": common.nasa_score(error),
                    "rul_cap_hit": int(cap_hit),
                    "source_failure_threshold": threshold,
                    "predicted_crossing_log_health": predicted_log_health,
                    "largest_health_contribution_sensor": (
                        sensor_columns[driver]
                    ),
                }
            )
    return predictions, weights


def write_report(path: Path, summary: dict[str, Any]) -> None:
    labels = {
        "target_only": "Target only",
        "global_pool": "Global pool",
        "uniform_em": "Uniform EM",
        "llm_lip_em": "LIP-EM",
    }
    lines = [
        "# N-CMAPSS DS03 joint-sensor prognostics",
        "",
        f"LIP: {summary['lip']['label']} (`{summary['lip']['file']}`).",
        "",
        "Each source engine has one shared relevance weight. The Gaussian "
        "backbone jointly regresses the 14 signed, operating-adjusted sensor "
        "trajectories on the raw-cycle natural-spline basis. The "
        "source-only fusion map is applied to the predicted sensor "
        "trajectories, after which RUL is the first crossing of the "
        "source-terminal threshold.",
        "",
        f"Exact Gaussian EM uses `tau={summary['tau']}`, "
        f"`nu={summary['nu']}`, and counts convergence patience from "
        f"`t >= {summary['prognostics']['stopping_start_iteration']}`.",
        "The uniform-EM baseline initializes every source-membership "
        f"probability at `p0={summary['uniform_prior']}`.",
        "As in the C-MAPSS experiment, Target Only and Global Pool use ridge "
        f"`lambda={summary['prognostics']['baseline_ridge_lambda']:.0f}` "
        "with an unpenalized intercept.",
        "",
        "| Observed target life | "
        + " | ".join(labels[method] for method in common.METHODS)
        + " |",
        "|---:|" + "---:|" * len(common.METHODS),
    ]
    for fraction in common.PROGNOSTIC_FRACTIONS:
        local = summary["prognostics"]["results"][str(fraction)]
        lines.append(
            f"| {100 * fraction:.0f}% | "
            + " | ".join(
                f"{local[method]['pooled_rmse']:.3f}"
                for method in common.METHODS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Entries are RUL RMSE over the "
            f"{summary['target_engines']} held-out "
            f"{summary['target_class']}-haul target engines.",
            "",
            "## Guardrails",
            "",
            "- No target RUL or future target sensor value is used for fitting.",
            "- Time is raw flight cycle and is not normalized by engine life.",
            "- The health-index weights, failure threshold, and RUL cap are "
            "estimated from source engines only.",
            "- The target flight-class label appears only in the LIP prompt. "
            "Source flight-class labels are used only for scoring source "
            "recovery; neither is passed to EM or the prognostics model.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def relative_path(path: Path, start: Path) -> str:
    """`path` relative to `start`, so outputs never record absolute paths."""
    try:
        return os.path.relpath(path.resolve(), start.resolve())
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=common.DEFAULT_FEATURES
    )
    parser.add_argument(
        "--target-class", choices=common.FLIGHT_CLASSES, required=True
    )
    parser.add_argument(
        "--lip-priors",
        type=Path,
        required=True,
        help="fold-specific priors written by fit_lip_priors.py",
    )
    parser.add_argument("--lip-label", default="LLM LIP")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=em.TAU)
    parser.add_argument("--nu", type=float, default=em.NU)
    arguments = parser.parse_args()
    common.prepare_output(arguments.output)

    all_engines, sensor_columns = common.load_engines(arguments.features)
    prior_payload = json.loads(arguments.lip_priors.read_text())
    if prior_payload["target_class"] != arguments.target_class:
        raise ValueError(
            f"{arguments.lip_priors} holds {prior_payload['target_class']} "
            f"priors, not {arguments.target_class}"
        )
    priors_by_target = prior_payload["priors_by_target"]
    class_targets = [
        engine
        for engine in all_engines
        if engine["flight_class_name"] == arguments.target_class
    ]
    if len(all_engines) != 15 or len(class_targets) != 5:
        raise RuntimeError(
            f"expected 15 DS03 engines and five {arguments.target_class} "
            f"targets; got {len(all_engines)} and {len(class_targets)}"
        )

    predictions: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    for original_target in class_targets:
        engines = common.fold_copy(all_engines)
        target = next(
            engine
            for engine in engines
            if engine["domain_id"] == original_target["domain_id"]
        )
        sources = [
            engine
            for engine in engines
            if engine["domain_id"] != target["domain_id"]
        ]
        llm_prior = np.asarray(
            [
                priors_by_target[target["domain_id"]][source["domain_id"]]
                for source in sources
            ]
        )
        health_metadata = common.add_health_paths(
            engines, sources, len(sensor_columns)
        )
        local_predictions, local_weights = run_fold(
            target,
            sources,
            sensor_columns,
            llm_prior,
            arguments.tau,
            arguments.nu,
            np.asarray(health_metadata["fusion_weights"]),
        )
        predictions.extend(local_predictions)
        weight_rows.extend(local_weights)
        fold_metadata.append(
            {
                "target_engine": target["domain_id"],
                "target_split_scoring_only": target["split"],
                "target_flights": len(target["cycle"]),
                "source_domains": len(sources),
                "matching_sources": sum(
                    common.matching(source, arguments.target_class)
                    for source in sources
                ),
                "health_index": health_metadata,
            }
        )
        print(
            f"completed target {target['domain_id']} "
            f"({len(target['cycle'])} flights)",
            flush=True,
        )

    summary = {
        "dataset": "N-CMAPSS DS03",
        "experiment": f"leave-one-{arguments.target_class}-engine-out",
        "source_domain_unit": "individual engine",
        "target_class": arguments.target_class,
        "target_engines": len(class_targets),
        "sources_per_fold": 14,
        "matching_sources_per_fold": 4,
        "tau": arguments.tau,
        "nu": arguments.nu,
        "uniform_prior": common.UNIFORM_PRIOR_VALUE,
        "lip": {
            "label": arguments.lip_label,
            "file": relative_path(
                arguments.lip_priors, arguments.output.parent
            ),
            "elicitation": prior_payload.get("elicitation"),
            "query_count": prior_payload.get("queries_used"),
            "p0": prior_payload.get("p0"),
            "epsilon": prior_payload.get("epsilon"),
            "target_label_shown_to_llm": True,
            "source_labels_shown_to_llm": False,
        },
        "prognostics": {
            "task": (
                "unlabeled-prefix joint-sensor trajectory regression "
                "followed by health fusion and threshold crossing"
            ),
            "target_rul_used_for_adaptation": False,
            "target_future_used_for_adaptation": False,
            "regression": "multiple-output Gaussian natural-spline regression",
            "outputs": sensor_columns,
            "output_count": len(sensor_columns),
            "shared_source_membership": True,
            "em_max_iterations": em.MAX_ITERATIONS,
            "convergence_tolerance": em.CONVERGENCE_TOLERANCE,
            "convergence_patience": em.CONVERGENCE_PATIENCE,
            "stopping_start_iteration": em.stopping_start(arguments.nu),
            "tempering_scale": "trace of the pseudo-inverse",
            "pi_clip": em.PI_CLIP,
            "null_min_inverse_weight": em.NULL_MIN_INVERSE_WEIGHT,
            "baseline_ridge_lambda": BASELINE_RIDGE_LAMBDA,
            "baseline_intercept_penalized": False,
            "coefficient_dimension": (
                common.natural_spline_basis(np.asarray([1.0])).shape[1]
                * len(sensor_columns)
            ),
            "path_model": common.PATH_MODEL,
            "basis_x_max": common.PATH_X_MAX,
            "knots": list(common.PATH_KNOTS),
            "results": common.summarize_predictions(predictions),
        },
        "selection": common.summarize_selection(weight_rows),
        "folds": fold_metadata,
    }
    common.write_csv(
        arguments.output / "prognostics_predictions.csv", predictions
    )
    common.write_csv(arguments.output / "source_weights.csv", weight_rows)
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    write_report(arguments.output / "report.md", summary)
    print(
        json.dumps(
            {
                "output": arguments.output.name,
                "predictions": len(predictions),
                "weights": len(weight_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
